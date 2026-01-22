import whisper
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

# 모델 로드 (가장 가벼운 base 모델 사용)
model = whisper.load_model("base")

def transcribe_long_audio(file_path: str):
    # 1. 파일 불러오기
    print(f"파일 분석 중: {file_path}")
    audio = AudioSegment.from_file(file_path)
    
    # 2. 무음 구간 기준 자르기 (1시간 파일을 한꺼번에 처리하면 메모리가 부족하기 때문)
    # min_silence_len: 최소 1초(1000ms) 무음일 때 자름
    # silence_thresh: 평균 데시벨보다 14dB 낮으면 무음으로 간주
    print("음성 조각 내는 중... (잠시만 기다려주세요)")
    chunks = split_on_silence(
        audio, 
        min_silence_len=1000, 
        silence_thresh=audio.dBFS-14, 
        keep_silence=500
    )

    full_text = ""
    # 조각들을 저장할 임시 경로
    temp_chunk_path = "app/temp/temp_chunk.wav"

    # 3. 각 조각별로 텍스트 변환 실행
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] 조각 변환 중...")
        chunk.export(temp_chunk_path, format="wav")
        
        # Whisper로 해당 조각 변환
        result = model.transcribe(temp_chunk_path, language="ko")
        full_text += result["text"] + " "

    # 4. 사용한 임시 파일 및 원본 파일 삭제 (서버 용량 관리)
    if os.path.exists(temp_chunk_path):
        os.remove(temp_chunk_path)
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return full_text.strip()