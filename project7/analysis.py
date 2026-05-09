import os
import pandas as pd
import requests
from google import genai

# ==========================================
# 0. 환경 설정
# ==========================================
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
def main():
    print("🕸️ 조직 네트워크 분석(SNA) (Project 7) 시작")
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    excel_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".xlsx")]
    if not excel_files:
        print("오류: 데이터 폴더에 엑셀 파일이 없습니다.")
        return
    
    file_path = os.path.join(data_dir, excel_files[0])
    target_time = '2025Q2'

    try:
        emp_df = pd.read_excel(file_path, sheet_name='employees')
        edge_df = pd.read_excel(file_path, sheet_name='edges')
        
        # 필터링
        filtered_edges = edge_df[
            (edge_df['time_id'] == target_time) & 
            (edge_df['tie_binary'] == 1) & 
            (edge_df['source'] != edge_df['target'])
        ].copy()

        # 지표 계산
        out_degree = filtered_edges.groupby('source')['target'].nunique().rename('out_degree')
        in_degree = filtered_edges.groupby('target')['source'].nunique().rename('in_degree')
        
        sent_int = filtered_edges.groupby('source')['interaction_count'].sum()
        recv_int = filtered_edges.groupby('target')['interaction_count'].sum()
        total_interaction = sent_int.add(recv_int, fill_value=0).rename('total_interaction')

        conn_list = pd.concat([
            filtered_edges[['source', 'target']].rename(columns={'source': 'me', 'target': 'other'}),
            filtered_edges[['target', 'source']].rename(columns={'target': 'me', 'source': 'other'})
        ])
        total_unique_connections = conn_list.groupby('me')['other'].nunique().rename('total_unique_connections')

        # 통합
        metrics_df = pd.concat([total_unique_connections, out_degree, in_degree, total_interaction], axis=1)
        final_df = emp_df.merge(metrics_df, left_on='employee_id', right_index=True, how='left')
        
        cols_to_fill = ['total_unique_connections', 'out_degree', 'in_degree', 'total_interaction']
        final_df[cols_to_fill] = final_df[cols_to_fill].fillna(0).astype(int)

        # 정렬 및 순위
        final_df = final_df.sort_values(
            by=['total_unique_connections', 'total_interaction', 'out_degree'], 
            ascending=False
        ).reset_index(drop=True)
        
        top_10 = final_df.head(10)
        top_1 = final_df.iloc[0]

        # Gemini 해석용 데이터 준비
        top_5_summary = top_10.head(5)[['name', 'department', 'team', 'total_unique_connections', 'total_interaction']].to_string(index=False)
        
        prompt = f"""
        당신은 조직 문화 및 네트워크 분석 전문가입니다. 다음 SNA 분석 결과를 바탕으로 우리 조직의 소통 현황을 진단해 주세요.
        
        [분석 데이터 요약 - 상위 5명]
        {top_5_summary}
        
        [최고 핵심 인물(Hub)]
        {top_1['department']} {top_1['team']}팀 {top_1['name']}님 (연결: {top_1['total_unique_connections']}명)
        
        [요청 사항]
        1. 핵심 인물(Hub)의 역할이 조직 내에서 가지는 전략적 중요성을 설명해 주세요.
        2. 상위권 인물들의 분포를 볼 때, 특정 부서의 소통이 활발한지 아니면 고르게 분포되어 있는지 분석해 주세요.
        3. 이 네트워크를 더 활성화하기 위한 조직 차원의 제언을 포함해 주세요.
        한국어로 전문적이고 통찰력 있게 답변해 주세요.
        """
        
        print("Gemini 분석 요청 중...")
        response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        ai_analysis = response.text

        # Discord 전송
        final_msg = f"🕸️ **[Project 7] 조직 네트워크 분석(SNA) 보고서**\n\n"
        final_msg += f"**기준 시점:** `{target_time}`\n"
        final_msg += f"**분석 파일:** `{excel_files[0]}`\n\n"
        final_msg += f"🏆 **최고 핵심 인물:** {top_1['department']} {top_1['name']}님\n"
        final_msg += f"**AI 인사이트:**\n{ai_analysis}"
        
        send_to_discord(final_msg)
        print("모든 작업 완료!")

    except Exception as e:
        print(f"오류 발생: {e}")
        send_to_discord(f"❌ [Project 7] 분석 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
