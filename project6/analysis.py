import pandas as pd
import numpy as np
import os
import requests
from google import genai
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import svds

# ==========================================
# 0. 하이퍼파라미터 및 환경 설정
# ==========================================
TOP_N = 3  # 추천 과목 수
ALPHA = 0.6  # 하이브리드 기본 CF 가중치
N_FACTORS = 5  # SVD 잠재 요인 수

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

if not GOOGLE_API_KEY:
    print("오류: GOOGLE_API_KEY가 설정되어 있지 않습니다.")
    exit(1)

client = genai.Client(api_key=GOOGLE_API_KEY)

def send_to_discord(message):
    if not DISCORD_WEBHOOK_URL:
        print("경고: DISCORD_WEBHOOK_URL이 없습니다.")
        return
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": message[:1990]})
        if response.status_code in [200, 204]:
            print("Discord 전송 성공!")
    except Exception as e:
        print(f"Discord 전송 오류: {e}")

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    excel_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".xlsx")]
    if not excel_files:
        raise FileNotFoundError("데이터 폴더에 엑셀 파일이 없습니다.")
    
    path = os.path.join(data_dir, excel_files[0])
    print(f"분석 대상 파일: {excel_files[0]}")

    sheets = ['courses', 'employees', 'ratings_train', 'ratings_test', 'recommend_target']
    data = {}
    with pd.ExcelFile(path) as xls:
        for sheet in sheets:
            data[sheet] = pd.read_excel(xls, sheet_name=sheet)
    return data, excel_files[0]

def build_rating_matrix(df_ratings, df_employees, df_courses):
    matrix = df_ratings.pivot(index='emp_id', columns='course_id', values='rating').fillna(0)
    all_emps = df_employees['emp_id'].unique()
    all_courses = df_courses['course_id'].unique()
    matrix = matrix.reindex(index=all_emps, columns=all_courses, fill_value=0)
    return matrix

# ==========================================
# 2. 추천 엔진 클래스 (CF, CB, Hybrid)
# ==========================================
class CollaborativeFilter:
    def __init__(self, rating_matrix, n_factors=5):
        self.matrix = rating_matrix
        self.n_factors = n_factors
        self.preds_df = None

    def fit(self):
        user_ratings_mean = np.mean(self.matrix.values, axis=1)
        matrix_centered = self.matrix.values - user_ratings_mean.reshape(-1, 1)
        U, sigma, Vt = svds(matrix_centered, k=min(self.n_factors, min(matrix_centered.shape)-1))
        sigma = np.diag(sigma)
        svd_preds = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        self.preds_df = pd.DataFrame(svd_preds, columns=self.matrix.columns, index=self.matrix.index)

class ContentFilter:
    def __init__(self, rating_matrix):
        self.rating_matrix = rating_matrix
    def get_all_scores(self, emp_id):
        # 데모용: 실제 환경에선 코스 특징 벡터 기반 유사도 계산
        return pd.Series(np.random.rand(len(self.rating_matrix.columns)), index=self.rating_matrix.columns)

class HybridRecommender:
    def __init__(self, cf_model, cb_model, rating_matrix):
        self.cf = cf_model
        self.cb = cb_model
        self.rating_matrix = rating_matrix
        self.scaler = MinMaxScaler()

    def recommend(self, emp_id, top_n=3, base_alpha=0.6):
        rated_items = self.rating_matrix.loc[emp_id]
        unrated_idx = rated_items[rated_items == 0].index
        
        cf_scores = self.cf.preds_df.loc[emp_id][unrated_idx]
        cb_scores = self.cb.get_all_scores(emp_id)[unrated_idx]

        cf_norm = self.scaler.fit_transform(cf_scores.values.reshape(-1, 1)).flatten()
        cb_norm = self.scaler.fit_transform(cb_scores.values.reshape(-1, 1)).flatten()

        num_rated = (self.rating_matrix.loc[emp_id] > 0).sum()
        effective_alpha = 0.2 if num_rated <= 2 else base_alpha
        hybrid_scores = (effective_alpha * cf_norm) + ((1 - effective_alpha) * cb_norm)
        
        return pd.Series(hybrid_scores, index=unrated_idx).sort_values(ascending=False).head(top_n)

# ==========================================
# 3. 메인 실행 및 알림
# ==========================================
def main():
    print("🚀 교육과정 추천 시스템 (Project 6) 시작")
    try:
        data, filename = load_data()
        rating_matrix = build_rating_matrix(data['ratings_train'], data['employees'], data['courses'])
        
        cf = CollaborativeFilter(rating_matrix, n_factors=N_FACTORS)
        cf.fit()
        cb = ContentFilter(rating_matrix)
        hybrid = HybridRecommender(cf, cb, rating_matrix)

        recommend_results = []
        target_emps = data['recommend_target']

        for _, row in target_emps.iterrows():
            emp_id = row['emp_id']
            emp_info = data['employees'][data['employees']['emp_id'] == emp_id].iloc[0]
            hyb_res = hybrid.recommend(emp_id, top_n=TOP_N)
            
            rec_list = []
            for cid, score in hyb_res.items():
                c_name = data['courses'][data['courses']['course_id'] == cid]['name'].values[0]
                rec_list.append(f"- {c_name} (ID: {cid})")
            
            recommend_results.append(f"👤 **{emp_info['dept']} {emp_id}님** 추천:\n" + "\n".join(rec_list))

        # Gemini 해석 요청
        summary_text = "\n\n".join(recommend_results)
        prompt = f"""
        당신은 HR 교육 전문가입니다. 다음은 직원별 하이브리드 추천 시스템 결과입니다.
        이 추천 결과가 직원들의 직무 특성이나 부서에 어떻게 도움이 될지 전문적으로 요약해 주세요.
        
        [추천 결과]
        {summary_text}
        
        [요청]
        1. 전체적인 추천 경향을 분석해 주세요.
        2. 이 교육들이 조직 역량 강화에 어떤 의미가 있는지 경영진에게 보고하는 형식으로 작성해 주세요.
        한국어로 답변해 주세요.
        """
        
        print("Gemini 분석 요청 중...")
        response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        ai_analysis = response.text

        # Discord 전송
        final_msg = f"🎓 **[Project 6] 교육과정 추천 보고서**\n\n**파일명:** `{filename}`\n\n"
        final_msg += summary_text + "\n\n"
        final_msg += f"**AI 인사이트:**\n{ai_analysis}"
        
        send_to_discord(final_msg)
        print("모든 작업 완료!")

    except Exception as e:
        print(f"오류 발생: {e}")
        send_to_discord(f"❌ [Project 6] 분석 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
