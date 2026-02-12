import time
import fitz  # PyMuPDF
import whisper
import os
import re
import io
import random
import numpy as np
from typing import List, Tuple
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Gemini 설정
client = genai.Client(api_key="AIzaSyBe_xJF3lsPCEe6Jnjt7420cCOXAqExoo8")
STT_MODEL = whisper.load_model("turbo")

# -------------------------------------------------
# 1. 스마트 OCR 엔진 (오직 Vision으로만 분석)
# -------------------------------------------------
def ocr_with_gemini(page) -> str:
    """PDF 텍스트 레이어를 무시하고 이미지를 직접 시각 분석"""
    # 해상도를 3.5배로 높여 필기 인식률 극대화 (작은 글씨도 잘 보이게 함)
    pix = page.get_pixmap(matrix=fitz.Matrix(3.5, 3.5)) 
    img_data = pix.tobytes("png")
    
    prompt = """
    이미지 속의 모든 내용을 텍스트로 변환하세요.
    특히, 사람이 직접 펜으로 적은 '손글씨 필기'나 '메모', 화살표 옆의 낙서 등을 절대로 빼놓지 마세요.
    손글씨가 있다면 해당 내용 앞에 반드시 [손글씨] 라고 붙여주세요.
    예시: [손글씨] 시험에 나옴! 
    """
    
    # 모델명은 라이브러리 버전에 맞게 'gemini-1.5-flash'로 설정
    response = client.models.generate_content(
        model='gemini-1.5-flash', 
        contents=[
            types.Part.from_bytes(data=img_data, mime_type='image/png'),
            prompt
        ]
    )
    return response.text

def extract_from_pdf(file_bytes: bytes) -> List[Tuple[str, int]]:
    """모든 로컬 텍스트 추출을 금지하고 오직 Gemini Vision만 사용하여 손글씨를 잡아냄"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_data = []

    for page_num, page in enumerate(doc, start=1):
        # 텍스트 레이어를 아예 안 읽고 바로 이미지 분석으로 들어갑니다.
        print(f"🔥 {page_num}페이지: 텍스트 레이어 무시, 오직 이미지로만 분석 중 (10초 대기)...")
        
        # 무료 티어 안정성(429 에러 방지)을 위해 10초 대기
        time.sleep(10) 
        
        try:
            # Gemini가 눈으로 직접 보고 텍스트와 손글씨를 모두 읽어옵니다.
            text = ocr_with_gemini(page)
            print(f"--- {page_num}페이지 Gemini 추출 성공 ---")
            print(text[:150] + "...") # 터미널 로그 확인용
            
        except Exception as e:
            print(f"❌ API 에러 발생: {e}")
            # API 에러 시에만 어쩔 수 없이 텍스트 레이어 호출 (보험용)
            text = page.get_text().strip()
        
        if text:
            pages_data.append((text, page_num))
            
    return pages_data

# -------------------------------------------------
# 2. 텍스트 처리 및 임베딩
# -------------------------------------------------
def chunk_text_with_page(pages_data: List[Tuple[str, int]], max_words=150):
    """페이지 정보를 유지하며 텍스트 분할"""
    all_chunks = []
    for text, page_num in pages_data:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk, current_len = [], 0
        for s in sentences:
            words = s.split()
            if current_len + len(words) > max_words:
                all_chunks.append((" ".join(current_chunk), page_num))
                current_chunk, current_len = words, len(words)
            else:
                current_chunk.extend(words)
                current_len += len(words)
        if current_chunk:
            all_chunks.append((" ".join(current_chunk), page_num))
    return all_chunks

def get_embeddings(text_list: List[str]):
    """텍스트 리스트를 벡터로 변환 (현재는 테스트용 랜덤 벡터)"""
    return [[random.uniform(-1, 1) for _ in range(1536)] for _ in text_list]

def transcribe_audio(audio_path: str):
    """Whisper를 이용한 음성 인식"""
    print(f"--- 음성 인식 시작: {audio_path} ---")
    result = STT_MODEL.transcribe(audio_path)
    return result['text']