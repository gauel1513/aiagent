import os
import time
import requests
from google import genai

# 1. API 키 설정 (GitHub Secrets의 GOOGLE_API_KEY 환경변수에서 가져옵니다)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("오류: 환경변수 GOOGLE_API_KEY가 설정되어 있지 않습니다.")
    exit(1)

client = genai.Client(api_key=GOOGLE_API_KEY)

# Discord 웹훅 설정
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

def send_to_discord(message, file_path=None):
    """Discord 웹훅으로 메시지와 파일을 전송합니다."""
    if not DISCORD_WEBHOOK_URL:
        print("경고: DISCORD_WEBHOOK_URL 환경변수가 설정되어 있지 않아 메시지를 보낼 수 없습니다.")
        return

    payload = {"content": message}
    
    try:
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f)}
                response = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
        else:
            response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
            
        if response.status_code == 204 or response.status_code == 200:
            print("Discord 메시지 전송 성공!")
        else:
            print(f"Discord 전송 실패 (상태 코드: {response.status_code}): {response.text}")
    except Exception as e:
        print(f"Discord 전송 중 오류 발생: {e}")

# 2. 동영상 파일 경로 설정 (스크립트 기준 data/ 폴더에서 .mp4 파일을 자동으로 찾습니다)
# project5 폴더 내의 data/ 폴더를 참조합니다.
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"알림: '{data_dir}' 폴더가 생성되었습니다. 분석할 MP4 파일을 이 폴더에 넣어주세요.")
    exit(0)

video_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".mp4")]

if not video_files:
    print(f"오류: '{data_dir}' 폴더 안에 분석할 MP4 파일이 없습니다.")
    exit(1)

# 첫 번째 영상 파일을 대상으로 분석 진행
video_filename = video_files[0]
video_path = os.path.join(data_dir, video_filename)
print(f"분석 대상 파일: {video_filename}")

video_file = None

try:
    # 3. 파일 업로드
    print("동영상을 업로드하는 중...")
    video_file = client.files.upload(file=video_path)
    print(f"업로드 완료: {video_file.uri}")

    # 4. 동영상 처리 대기
    print("Google 서버에서 동영상을 처리 중입니다. 잠시만 기다려주세요...")
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)
    print()

    if video_file.state.name == "FAILED":
        raise ValueError("동영상 처리에 실패했습니다. 파일 형식을 확인해주세요.")
    print("동영상 처리 완료!")

    # 5. 모델 선택 및 분석 요청
    model_id = "gemini-2.0-flash" # 최신 안정화 모델 사용

    prompt = "이 동영상에서 인물이 어떤 행동을 하고 있는지 시간 흐름에 따라 상세히 분석해줘."

    print(f"AI({model_id})가 동영상을 분석하고 있습니다...")
    response = client.models.generate_content(
        model=model_id,
        contents=[video_file, prompt]
    )

    # 6. 결과 출력 및 파일 저장
    analysis_text = response.text
    print("\n==========[분석 결과] ==========")
    print(analysis_text)
    print("=================================")

    result_path = os.path.join(data_dir, "analysis_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"[분석 대상 파일] {video_filename}\n\n")
        f.write(analysis_text)
    print(f"\n분석 결과가 '{result_path}'에 저장되었습니다.")

    # 7. Discord로 결과 전송
    discord_msg = f"🚀 **[Project 5] 동영상 분석 보고서**\n\n**파일명:** `{video_filename}`\n\n{analysis_text[:1800]}"
    send_to_discord(discord_msg, result_path)

finally:
    # 8. 파일 삭제
    if video_file:
        try:
            client.files.delete(name=video_file.name)
            print("\nGoogle AI Studio 서버에서 동영상 파일이 안전하게 삭제되었습니다.")
        except Exception as e:
            print(f"\n파일 삭제 중 오류 발생: {e}")
