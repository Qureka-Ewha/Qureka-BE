import fitz  # PyMuPDF
import whisper
import os
import re
from typing import List, Tuple # Tuple 추가
from pydub import AudioSegment
from pydub.silence import split_on_silence
from openai import OpenAI
from dotenv import load_dotenv
import random

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
STT_MODEL = whisper.load_model("turbo")

# --- 수정된 부분 1: PDF 추출 시 페이지 번호를 함께 저장 ---
def extract_from_pdf(file_bytes: bytes) -> List[Tuple[str, int]]:
    """PDF 각 페이지별로 (텍스트, 페이지번호) 리스트 반환"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_data = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages_data.append((text, page_num))
    return pages_data

# --- 수정된 부분 2: 페이지 정보를 유지하며 쪼개기 ---
def chunk_text_with_page(pages_data: List[Tuple[str, int]], max_words: int = 150) -> List[Tuple[str, int]]:
    """페이지별 텍스트를 받아 (조각내용, 실제페이지번호) 리스트 반환"""
    all_chunks = []
    
    for text, page_num in pages_data:
        # 각 페이지 내에서 문장 단위로 분할
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

# get_embeddings 함수는 그대로 유지하되, 리스트만 받도록 설계되어 있으므로 
# 호출할 때 텍스트만 쏙 뽑아서 보내주면 됩니다.
def get_embeddings(text_list: List[str]):
    print(f"--- 테스트 모드: {len(text_list)}개의 텍스트 조각을 벡터화합니다. ---")
    if not text_list: return []
    return [[random.uniform(-1, 1) for _ in range(1536)] for _ in text_list]