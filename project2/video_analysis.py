import os
import time
from google import genai

# 1. API 키 설정 (GitHub Secrets의 GOOGLE_API_KEY 환경변수에서 가져옵니다)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("오류: 환경변수 GOOGLE_API_KEY가 설정되어 있지 않습니다.")
    exit(1)

client = genai.Client(api_key=GOOGLE_API_KEY)

# 2. 동영상 파일 경로 설정 (스크립트 기준 data/ 폴더에서 .mp4 파일을 자동으로 찾습니다)
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
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

    # 4. 동영상 처리 대기 (Google 서버에서 동영상을 분석할 수 있게 처리하는 시간)
    print("Google 서버에서 동영상을 처리 중입니다. 잠시만 기다려주세요...")
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(5)  # 5초마다 상태 확인
        video_file = client.files.get(name=video_file.name)
    print()

    # 처리에 실패한 경우
    if video_file.state.name == "FAILED":
        raise ValueError("동영상 처리에 실패했습니다. 파일 형식을 확인해주세요.")
    print("동영상 처리 완료!")

    # 5. 최신 모델 선택 및 분석 요청
    model_id = "gemini-2.5-flash"

    # AI에게 내릴 명령(프롬프트) 작성
    prompt = "이 동영상에서 인물이 어떤 행동을 하고 있는지 시간 흐름에 따라 상세히 분석해줘."

    print(f"AI({model_id})가 동영상을 분석하고 있습니다...")
    response = client.models.generate_content(
        model=model_id,
        contents=[video_file, prompt]
    )

    # 6. 결과 출력 및 파일 저장 (이메일 첨부를 위해 저장합니다)
    analysis_text = response.text
    print("\n==========[분석 결과] ==========")
    print(analysis_text)
    print("=================================")

    result_path = os.path.join(data_dir, "analysis_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"[분석 대상 파일] {video_filename}\n\n")
        f.write(analysis_text)
    print(f"\n분석 결과가 '{result_path}'에 저장되었습니다.")

finally:
    # 7. 보안 및 용량 관리를 위해 분석이 끝난 파일을 서버에서 삭제
    if video_file:
        try:
            client.files.delete(name=video_file.name)
            print("\nGoogle AI Studio 서버에서 동영상 파일이 안전하게 삭제되었습니다.")
        except Exception as e:
            print(f"\n파일 삭제 중 오류 발생: {e}")
