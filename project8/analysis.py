import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import koreanize_matplotlib
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
    print("🕸️ 조직 네트워크 시각화 분석 (Project 8) 시작")
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    excel_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".xlsx")]
    if not excel_files:
        print("오류: 데이터 폴더에 엑셀 파일이 없습니다.")
        return
    
    file_path = os.path.join(data_dir, excel_files[0])
    target_time = '2025Q2'

    try:
        df_employees = pd.read_excel(file_path, sheet_name='employees')
        df_edges = pd.read_excel(file_path, sheet_name='edges')

        df_filtered = df_edges[(df_edges['time_id'] == target_time) & 
                               (df_edges['source'] != df_edges['target'])].copy()

        df_filtered['node_a'] = df_filtered.apply(lambda x: min(x['source'], x['target']), axis=1)
        df_filtered['node_b'] = df_filtered.apply(lambda x: max(x['source'], x['target']), axis=1)
        df_grouped = df_filtered.groupby(['node_a', 'node_b'], as_index=False)['interaction_count'].sum()

        G = nx.Graph()
        for _, row in df_employees.iterrows():
            G.add_node(row['employee_id'], name=row['name'], department=row['department'], team=row['team'])

        for _, row in df_grouped.iterrows():
            G.add_edge(row['node_a'], row['node_b'], weight=row['interaction_count'])

        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)

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

        plt.figure(figsize=(15, 10))
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

        legend_handles = [mpatches.Patch(color=color, label=dept) for dept, color in dept_color_dict.items()]
        plt.legend(handles=legend_handles, title="Departments", loc='upper right', bbox_to_anchor=(1.2, 1))

        plt.title(f"{target_time} Social Network Analysis Map (Project 8)", fontsize=15)
        plt.axis('off')
        plt.tight_layout()

        output_image = os.path.join(data_dir, "sna_network_map_project8.png")
        plt.savefig(output_image)
        plt.close()

        top_5_summary = "\n".join([f"- {G.nodes[n]['name']} ({G.nodes[n]['department']}): 연결 {degrees[n]}" for n in top_15_nodes[:5]])
        
        prompt = f"""
        당신은 조직 분석 전문가입니다. 다음 네트워크 분석 데이터와 그래프를 바탕으로 인사이트를 제공해 주세요.
        
        [네트워크 상위 5인]
        {top_5_summary}
        
        한국어로 전문적이고 통찰력 있게 답변해 주세요.
        """
        
        print("Gemini 분석 요청 중...")
        response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        ai_analysis = response.text

        final_msg = f"🕸️ **[Project 8] 조직 네트워크 시각화 분석 보고서**\n\n"
        final_msg += f"**분석 파일:** `{excel_files[0]}`\n\n"
        final_msg += f"**AI 인사이트:**\n{ai_analysis}"
        
        send_to_discord(final_msg, output_image)
        print("project8 작업 완료!")

    except Exception as e:
        print(f"오류 발생: {e}")
        send_to_discord(f"❌ [Project 8] 시각화 분석 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
