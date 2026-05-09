import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import koreanize_matplotlib # 한글 폰트 자동 설정
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

def send_to_discord(message, file_path=None):
    if not DISCORD_WEBHOOK_URL:
        print("경고: DISCORD_WEBHOOK_URL이 없습니다.")
        return
    
    payload = {"content": message[:1990]}
    try:
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f)}
                response = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
        else:
            response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
            
        if response.status_code in [200, 204]:
            print("Discord 전송 성공!")
    except Exception as e:
        print(f"Discord 전송 오류: {e}")

# ==========================================
# 1. 메인 분석 및 시각화
# ==========================================
def main():
    print("🕸️ 조직 네트워크 시각화 분석(SNA) (Project 7) 시작")
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    excel_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".xlsx")]
    if not excel_files:
        print("오류: 데이터 폴더에 엑셀 파일이 없습니다.")
        return
    
    file_path = os.path.join(data_dir, excel_files[0])
    target_time = '2025Q2'

    try:
        # 데이터 로드
        df_employees = pd.read_excel(file_path, sheet_name='employees')
        df_edges = pd.read_excel(file_path, sheet_name='edges')

        # 데이터 전처리
        df_filtered = df_edges[(df_edges['time_id'] == target_time) & 
                               (df_edges['source'] != df_edges['target'])].copy()

        df_filtered['node_a'] = df_filtered.apply(lambda x: min(x['source'], x['target']), axis=1)
        df_filtered['node_b'] = df_filtered.apply(lambda x: max(x['source'], x['target']), axis=1)
        df_grouped = df_filtered.groupby(['node_a', 'node_b'], as_index=False)['interaction_count'].sum()

        # NetworkX 그래프 생성
        G = nx.Graph()
        for _, row in df_employees.iterrows():
            G.add_node(row['employee_id'], name=row['name'], department=row['department'], team=row['team'])

        for _, row in df_grouped.iterrows():
            G.add_edge(row['node_a'], row['node_b'], weight=row['interaction_count'])

        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)

        # 시각화 설정
        pos = nx.spring_layout(G, k=0.5, seed=42)
        departments = list(df_employees['department'].unique())
        color_map_palette = plt.cm.get_cmap('Set3', len(departments))
        dept_color_dict = {dept: color_map_palette(i) for i, dept in enumerate(departments)}
        node_colors = [dept_color_dict[G.nodes[n]['department']] for n in G.nodes()]
        
        degrees = dict(G.degree())
        node_sizes = [v * 100 for v in degrees.values()]
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_widths = [w * 0.5 for w in weights] 

        top_15_nodes = sorted(degrees, key=degrees.get, reverse=True)[:15]
        labels = {n: G.nodes[n]['name'] for n in top_15_nodes}

        # 그래프 그리기
        plt.figure(figsize=(15, 10))
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

        legend_handles = [mpatches.Patch(color=color, label=dept) for dept, color in dept_color_dict.items()]
        plt.legend(handles=legend_handles, title="Departments", loc='upper right', bbox_to_anchor=(1.2, 1))

        plt.title(f"{target_time} Social Network Analysis Map", fontsize=15)
        plt.axis('off')
        plt.tight_layout()

        # 이미지 파일로 저장
        output_image = os.path.join(data_dir, "sna_network_map.png")
        plt.savefig(output_image)
        plt.close()
        print(f"네트워크 지도 저장 완료: {output_image}")

        # Gemini 해석 요청
        top_5_summary = "\n".join([f"- {G.nodes[n]['name']} ({G.nodes[n]['department']}): 연결성 {degrees[n]}" for n in top_15_nodes[:5]])
        
        prompt = f"""
        당신은 조직 분석 전문가입니다. 다음 네트워크 분석 데이터와 함께 생성된 그래프를 바탕으로 인사이트를 제공해 주세요.
        
        [네트워크 상위 5인]
        {top_5_summary}
        
        [요청]
        1. 시각화된 지도에서 노드 크기가 큰 인물들이 조직 내 정보 흐름에 어떤 영향을 주는지 설명해 주세요.
        2. 부서별 색상 분포를 볼 때, 부서 간 협업(Cross-functional)이 활발해 보이는지 분석해 주세요.
        3. 이 네트워크 맵을 통해 발견할 수 있는 조직의 강점과 약점을 한 문장씩 요약해 주세요.
        한국어로 전문성 있게 답변해 주세요.
        """
        
        print("Gemini 분석 요청 중...")
        response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        ai_analysis = response.text

        # Discord 전송 (이미지 포함)
        final_msg = f"🕸️ **[Project 7] 조직 네트워크 시각화 분석 보고서**\n\n"
        final_msg += f"**분석 파일:** `{excel_files[0]}`\n\n"
        final_msg += f"**AI 인사이트:**\n{ai_analysis}"
        
        send_to_discord(final_msg, output_image)
        print("모든 작업 완료!")

    except Exception as e:
        print(f"오류 발생: {e}")
        send_to_discord(f"❌ [Project 7] 시각화 분석 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
