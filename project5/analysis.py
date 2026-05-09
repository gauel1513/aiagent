import os
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from google import genai
import requests

# 1. 환경 변수 설정
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

if not GOOGLE_API_KEY:
    print("오류: GOOGLE_API_KEY가 설정되어 있지 않습니다.")
    exit(1)

client = genai.Client(api_key=GOOGLE_API_KEY)

def send_to_discord(message):
    """Discord 웹훅으로 메시지를 전송합니다."""
    if not DISCORD_WEBHOOK_URL:
        print("경고: DISCORD_WEBHOOK_URL이 설정되어 있지 않습니다.")
        return
    
    # 디스코드 메시지 길이 제한(2000자)을 고려하여 분할 전송하거나 자름
    payload = {"content": message[:1990]}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        if response.status_code in [200, 204]:
            print("Discord 전송 성공!")
        else:
            print(f"Discord 전송 실패: {response.status_code}")
    except Exception as e:
        print(f"Discord 전송 중 오류: {e}")

# 2. 데이터 로드 및 전처리
# project5/data 폴더 내의 엑셀 파일을 자동으로 찾습니다.
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

excel_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".xlsx")]

if not excel_files:
    print(f"오류: '{data_dir}' 폴더에 엑셀 파일이 없습니다.")
    exit(1)

file_path = os.path.join(data_dir, excel_files[0])
print(f"분석 대상 파일: {excel_files[0]}")

try:
    df = pd.read_excel(file_path)
    # 결측치 제거
    df = df.dropna(subset=['Employee_Group', 'Pre_Training_Score', 'Post_Training_Score'])
    print("데이터 로드 및 전처리 완료!")

    # 3. ANOVA 분석
    groups = [group['Post_Training_Score'] for name, group in df.groupby('Employee_Group')]
    f_stat_anova, p_val_anova = stats.f_oneway(*groups)
    anova_res = f"ANOVA 결과: F={f_stat_anova:.4f}, p-value={p_val_anova:.4f}"

    # 4. ANCOVA 분석
    model = ols('Post_Training_Score ~ C(Employee_Group) + Pre_Training_Score', data=df).fit()
    ancova_table = sm.stats.anova_lm(model, typ=2)
    
    f_stat_ancova = ancova_table.loc['C(Employee_Group)', 'F']
    p_val_ancova = ancova_table.loc['C(Employee_Group)', 'PR(>F)']
    ancova_res = f"ANCOVA 결과 (사전점수 통제): F={f_stat_ancova:.4f}, p-value={p_val_ancova:.4f}"

    full_result_str = f"[분석 결과 요약]\n1. {anova_res}\n2. {ancova_res}"
    print(full_result_str)

    # 5. Gemini API 비교 해석
    prompt = f"""
    당신은 고급 통계 컨설턴트입니다. 다음 ANOVA와 ANCOVA 분석 결과를 비교하여 해석해 주세요.
    
    [데이터 정보]
    - 종속변수: 교육 후 점수 (Post_Training_Score)
    - 독립변수: 직원 그룹 (Employee_Group)
    - 공변량: 교육 전 점수 (Pre_Training_Score)
    
    [분석 결과]
    {full_result_str}
    
    [요청 사항]
    1. ANOVA와 ANCOVA 결과의 차이점을 설명해 주세요.
    2. 사전 점수(Pre_Training_Score)가 사후 점수에 미치는 영향이 컸는지 추론해 주세요.
    3. 교육 프로그램의 공정성 및 효과성에 대한 인사이트를 제공해 주세요.
    4. 경영진 보고용 요약 문장을 만들어 주세요.
    
    한국어로 전문적이면서도 이해하기 쉽게 설명해 주세요.
    """

    print("Gemini API에 비교 분석을 요청 중입니다...")
    response = client.models.generate_content(
        model="gemini-1.5-flash", 
        contents=prompt
    )
    
    gemini_analysis = response.text
    
    # 6. 최종 결과 Discord 전송
    final_report = f"📊 **[Project 5] 통계 분석 보고서**\n\n"
    final_report += f"**파일명:** `{excel_files[0]}`\n"
    final_report += f"```\n{full_result_str}\n```\n"
    final_report += f"**AI 해석:**\n{gemini_analysis}"
    
    send_to_discord(final_report)

except Exception as e:
    print(f"오류가 발생했습니다: {e}")
    send_to_discord(f"❌ [Project 5] 분석 중 오류 발생: {e}")
