import fitz  # PyMuPDF: PDF에서 텍스트와 레이아웃을 정밀하게 추출합니다.
import whisper  # OpenAI Whisper: 음성을 텍스트로 변환합니다.
import os
import re  # 정규표현식: 텍스트를 문장 단위로 깔끔하게 자를 때 사용합니다.
import openai  # OpenAI API: 텍스트를 벡터(숫자)로 변환하는 Embedding용입니다.
from pydub import AudioSegment  # 오디오 파일을 다루는 도구입니다.
from pydub.silence import split_on_silence  # 음성 중 무음 구간을 찾아 자르는 기능입니다.
from typing import List
import openai
from openai import OpenAI  # 이 줄을 추가하세요
import os

# 클라이언트 생성 (기존에 openai.api_key = ... 하던 방식 대신 사용)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Whisper 모델 중 'turbo' 버전을 메모리에 올립니다.
# 'base'보다 크지만 'large'보다 빠르며, 한국어 전공 용어 인식률이 매우 좋습니다.
STT_MODEL = whisper.load_model("turbo")

def extract_from_pdf(file_bytes: bytes) -> str:
    # 1. 메모리에 저장된 PDF 바이너리 데이터를 PyMuPDF로 엽니다.
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_result = ""
    
    # 2. PDF의 모든 페이지를 한 장씩 순회합니다.
    for page in doc:
        # 3. 해당 페이지의 텍스트를 뽑아 결과 변수에 추가합니다.
        # 뒤에 \n(줄바꿈)을 넣어 페이지 간 텍스트가 엉키지 않게 합니다.
        text_result += page.get_text() + "\n"
        
    # 4. 양 끝의 불필요한 공백을 제거하고 최종 텍스트를 반환합니다.
    return text_result.strip()

def extract_from_audio(file_path: str, subject_hint: str = None) -> str:
    # 1. 파일을 불러오고 음량을 일정하게 맞춥니다(normalize). 
    # 너무 작게 녹음된 소리도 잘 들리게 보정하여 인식률을 높입니다.
    audio = AudioSegment.from_file(file_path).normalize()
    
    # 2. 1초(1000ms) 이상 무음인 구간을 찾아 파일을 여러 조각으로 자릅니다.
    # 긴 파일을 통째로 Whisper에 넣으면 서버 메모리가 부족해지는 현상을 방지합니다.
    chunks = split_on_silence(audio, min_silence_len=1000, silence_thresh=audio.dBFS-14, keep_silence=500)
    
    # 3. 파일명(예: "운영체제_3강")을 힌트로 활용하여 AI에게 문맥을 알려줍니다.
    clean_hint = subject_hint.replace("_", " ") if subject_hint else "전공 수업"
    context_prompt = f"이 내용은 '{clean_hint}'에 관한 수업입니다. 전문 용어를 정확하게 변환하세요."
    
    full_text = ""
    temp_path = "app/temp/chunk.wav" # 조각난 음성을 잠시 저장할 경로
    
    # 4. 자른 조각들을 하나씩 반복하며 Whisper로 변환합니다.
    for chunk in chunks:
        chunk.export(temp_path, format="wav") # 조각을 파일로 저장
        # initial_prompt에 아까 만든 문맥 힌트를 넣어 전문 용어 오타를 줄입니다.
        result = STT_MODEL.transcribe(temp_path, language="ko", initial_prompt=context_prompt, fp16=False)
        full_text += result["text"] + " " # 변환된 텍스트를 하나로 합칩니다.
        
    return full_text.strip()

def chunk_text(text: str, max_words: int = 150) -> List[str]:
    # 1. 마침표(.), 물음표(?), 느낌표(!) 뒤에 공백이 오는 지점을 찾아 문장 단위로 분리합니다.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk, current_len = [], [], 0
    
    for s in sentences:
        words = s.split()
        # 2. 현재 조각의 단어 수가 150개를 넘어가면 하나의 Chunk로 확정 짓습니다.
        if current_len + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = words, len(words)
        else:
            # 3. 150개 미만이면 다음 문장을 현재 조각에 계속 붙입니다.
            current_chunk.extend(words)
            current_len += len(words)
            
    # 4. 마지막에 남은 문장들도 잊지 않고 추가합니다.
    if current_chunk: chunks.append(" ".join(current_chunk))
    return chunks

def get_embeddings(text_list):
    # 기존: response = openai.Embedding.create(...) -> 에러 발생
    # 수정: 최신 v1.0+ 방식
    response = client.embeddings.create(
        input=text_list,
        model="text-embedding-3-small"
    )
    # 데이터 추출 방식도 살짝 바뀌었습니다.
    return [data.embedding for data in response.data]