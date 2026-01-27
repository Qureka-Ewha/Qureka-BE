import whisper
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

# 모델 로드 (base보다 훨씬 강력한 turbo 혹은 small 추천)
# 사양에 따라 "small" 또는 "turbo"를 선택하세요.
model = whisper.load_model("turbo") 

def transcribe_long_audio(file_path: str, subject_hint: str = None):
    # 1. 파일 불러오기 및 정규화(소리 크기 최적화)
    print(f"파일 분석 중: {file_path}")
    audio = AudioSegment.from_file(file_path)
    audio = audio.normalize() # 전체 음량을 일정하게 맞춰 인식률 향상
    
    # 2. 무음 구간 기준 자르기
    print("음성 조각 내는 중... (잠시만 기다려주세요)")
    chunks = split_on_silence(
        audio, 
        min_silence_len=1000, 
        silence_thresh=audio.dBFS-14, 
        keep_silence=500
    )

    full_text = ""
    temp_chunk_path = "app/temp/temp_chunk.wav"

    # 3. 파일명을 활용한 문맥 프롬프트 생성
    # 파일명 내 특수문자 제거 후 프롬프트에 주입
    clean_hint = subject_hint.replace("_", " ").replace("-", " ") if subject_hint else "수업 내용"
    context_prompt = f"이 녹음의 주제는 '{clean_hint}'입니다. 전공 수업의 전문 용어를 정확하게 변환해주세요."

    # 4. 각 조각별로 텍스트 변환 실행
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] 조각 변환 중...")
        chunk.export(temp_chunk_path, format="wav")
        
        # Whisper 옵션 최적화: language, initial_prompt 추가
        result = model.transcribe(
            temp_chunk_path, 
            language="ko", 
            initial_prompt=context_prompt,
            fp16=False # CPU 환경에서 에러 방지
        )
        full_text += result["text"] + " "

    # 5. 사용한 임시 파일 및 원본 파일 삭제
    if os.path.exists(temp_chunk_path):
        os.remove(temp_chunk_path)
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return full_text.strip()