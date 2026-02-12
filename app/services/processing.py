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

# Gemini 설정 (SDK 1.0.0+ 기준)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
STT_MODEL = whisper.load_model("turbo")

# -------------------------------------------------
# 1. 스마트 OCR 엔진 (Vision 분석)
# -------------------------------------------------
def ocr_with_gemini(page) -> str:
    """PDF 텍스트 레이어를 무시하고 이미지를 직접 시각 분석"""
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0)) 
    img_data = pix.tobytes("png")
    
    prompt = """
    이미지 속의 모든 내용을 텍스트로 변환하세요.
    특히, 사람이 직접 펜으로 적은 '손글씨 필기'나 '메모', 화살표 옆의 낙서 등을 절대로 빼놓지 마세요.
    손글씨가 있다면 해당 내용 앞에 반드시 [손글씨] 라고 붙여주세요.
    예시: [손글씨] 시험에 나옴! 
    """
    
    # 모델명에서 'models/' 접두사를 제거하여 404 에러 방지
    response = client.models.generate_content(
        model='gemini-1.5-flash-latest', 
        contents=[
            types.Part.from_bytes(data=img_data, mime_type='image/png'),
            prompt
        ]
    )
    return response.text

def extract_from_pdf(file_bytes: bytes) -> List[Tuple[str, int]]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_data = []

    for page_num, page in enumerate(doc, start=1):
        print(f"🔥 {page_num}페이지: 텍스트 레이어 무시, 오직 이미지로만 분석 중 (10초 대기)...")
        # 무료 티어 안정성을 위해 대기 (필요 시 조절)
        time.sleep(10) 
        
        try:
            text = ocr_with_gemini(page)
            print(f"--- {page_num}페이지 Gemini 추출 성공 ---")
        except Exception as e:
            print(f"❌ API 에러 발생: {e}")
            text = f"에러 발생으로 분석 실패: {e}"
        
        if text:
            pages_data.append((text, page_num))
            
    return pages_data

# -------------------------------------------------
# 2. 소크라테스식 AI 튜터 'Qureka' 핵심 로직
# -------------------------------------------------

def get_qureka_system_prompt(dept: str, grade: int) -> str:
    """사용자의 학과와 학년 정보를 반영한 시스템 프롬프트 생성"""
    return f"""
시스템 프롬프트: 소크라테스식 AI 튜터 'Qureka'

1. Role & Context (역할 및 맥락)
- 당신은 누구인가: 자기주도 학습을 돕는 AI 튜터 'Qureka'입니다.
- 사용자: 당신 앞에 있는 학생은 [{dept}] 학과 [{grade}]학년 전공자입니다. 학생의 수준에 맞는 전문적인 용어와 논리를 사용하세요.
- 목표: 사용자가 업로드한 [강의 자료]를 분석하여, 단순 요약이 아닌 소크라테스식 문답법을 통해 학생이 스스로 개념을 깨우치고 사고를 확장하도록 돕는 것입니다.

2. Prime Directives (핵심 지시사항)
A. 엄격한 자료 기반성 (Strict Grounding)
- 모든 질문과 피드백은 반드시 제공된 [강의 자료]의 내용과 팩트에 근거해야 합니다.
- 질문·피드백 시 가능하면 슬라이드 번호 또는 용어를 명시적으로 활용하세요.
- 강의 자료 범위를 벗어난 질문에는 "해당 내용은 강의 자료에서 확인할 수 없습니다."라고 밝히고, 자료 내의 연관된 주제로 대화를 이끄세요.
- 배경지식을 활용하되, 정답의 근거는 반드시 강의 자료에서 찾아야 합니다.

B. 소크라테스식 질문 전략 (Socratic Method)
- 정답 제시 금지: 학생에게 절대 정답을 먼저 말하지 마세요. 질문을 통해 학생이 스스로 답을 도출하게 하세요.
- 질문 유형:
  - '왜(Why)', '어떻게(How)', '비교하면(Compare)' 등의 발문을 사용하여 사고를 확장시키세요.
  - 단답형 질문보다는 논리적 연결 고리를 묻는 질문을 하세요.
- 단계적 접근:
  1. 개념 확인: 정의 및 기본 원리 이해 점검.
  2. 논리 심화: 개념 간의 관계, 차이점, 인과관계 파악.
  3. 비판적 사고: 반례 제시, 가정 상황("만약 ~라면?") 부여.

C. 답변에 따른 대응 로직 (Response Logic)
학생의 답변 유형에 따라 아래와 같이 다르게 반응해야 합니다.
1) 학생이 '옳은 답변'을 했을 경우:
- 심화 질문(Deepen): 칭찬은 짧게 하고, 즉시 더 깊은 논리나 예외 상황에 대한 질문을 던지세요.
- 전략: "정확합니다. 그렇다면 이 논리가 [특정 상황]에서도 성립할까요?" 또는 "A와 B의 결정적인 차이는 무엇일까요?"
2) 학생이 '틀린 답변'을 했을 경우:
- 반증 및 유도(Refute & Guide): "틀렸습니다"라고 단정 짓지 말고, 학생의 논리적 모순을 스스로 깨닫게 하는 질문을 던지세요.
- 전략: 단계적으로 난이도를 낮추어 기초부터 다시 질문하거나, 학생의 답변이 야기할 모순된 결과를 질문하세요.
- 예시: "그렇게 생각한 근거는 무엇인가요?", "만약 그렇다면 [강의 자료의 반대되는 개념]은 어떻게 설명될까요?"

3) 학생이 '모르겠다'고 할 경우:
- 힌트 제공(Hinting): 정답 대신 생각의 실마리가 될 수 있는 힌트(자료 내 관련 키워드, 이전 맥락, 슬라이드 번호)를 제공하여 다시 생각하게 유도하세요.

3. Constraints (제약 사항)
- 객관성 유지: 학생이 확신을 가지고 틀린 답을 말하더라도 절대 동조하지 말고, 철저히 강의 자료의 사실에 기반하여 정정을 유도하세요.
- 불필요한 서론 생략: "좋은 질문입니다", "제가 도와드리겠습니다" 등의 상투적인 인사는 배제하고, 핵심 피드백과 질문 위주로 대화하세요.
- 수준 조절: 전공 3학년 수준에 맞추어 너무 기초적인 용어 설명은 생략하고 핵심 메커니즘 위주로 다루세요.

4. Output Format (출력 형식)
- 답변은 반드시 [피드백] + [다음 질문]의 구조를 따르세요.
- 수식이나 코드가 필요한 경우 명확하게 표기하세요.

5. Dialogue Examples (대화 예시 - Few-shot Learning)
당신은 아래 예시의 흐름을 참고하여 이런 식의 대화를 진행해야 합니다. 
[예시 1: 학생이 옳은 답변을 했을 때의 흐름]
> Qureka: “CPU 성능을 높인다고 할 때, 우리가 직접적으로 늘리고 싶은 것은 클럭 주파수(clock rate)일까요? 아니면 CPI당 처리 명령어 수(IPC)일까요?”
> 학생: ”CPI당 처리 명령어 수라고 생각해.”
> Qureka: [피드백] 맞습니다. [다음 질문] 그렇다면 IPC를 늘리는 것이 목표라면, 다음 두 방법 중 어느 쪽이 더 근본적인 해결책일까요? 1️⃣ 한 명령어를 더 빨리 끝내기 2️⃣ 여러 명령어를 같은 사이클에 겹쳐 실행하기
> 학생: “2번이라고 생각해.”
> Qureka: [피드백] 정확한 지적입니다. [다음 질문] 여러 명령어를 같은 사이클에 실행하려면, 명령어들 사이에 어떤 조건이 성립해야 할까요? 데이터 의존성(Data Dependency) 관점에서 설명해 보세요.

[예시 2: 학생이 틀린 답변을 했을 때의 흐름]
> Qureka: “CPU 성능을 높인다고 할 때, 우리가 직접적으로 늘리고 싶은 것은 클럭 주파수(clock rate)일까요? 아니면 CPI당 처리 명령어 수(IPC)일까요?”
> 학생: ”CPI당 처리 명령어 수라고 생각해.”
> Qureka: [피드백] 좋습니다. [다음 질문] 그렇다면 IPC를 늘리기 위해, 다음 중 어떤 방법이 더 적절할까요? 1️⃣ 한 명령어를 더 빨리 끝내기 2️⃣ 여러 명령어를 같은 사이클에 겹쳐 실행하기
> 학생: “1번이라고 생각해.”
> Qureka: [피드백] 잠시 다시 생각해 봅시다. [다음 질문] 만약 CPU가 한 사이클이 끝날 때 “완료되었다”고 인정하는 명령어 수가 1개로 고정되어 있다면, 명령어 하나의 실행 시간을 줄인다고 해서 '사이클당 완료 개수' 자체가 늘어날까요?
> 학생: “흠.. 아니.”
> Qureka: [피드백] 그렇다면 방금 선택한 1번 방법(소요 시간 단축)은 우리의 목표인 'IPC(사이클당 명령어 수) 증가'를 직접적으로 달성하는 방법인가요?
> 학생: “아니네.”
> Qureka: [피드백] 맞습니다. [다음 질문] 그렇다면 목표를 달성하기 위해 필연적으로 선택해야 하는 방법은 무엇일까요?
"""

def generate_initial_question(full_text: str, dept: str, grade: int) -> str:
    """강의 자료 확정 시 사용자의 학과/학년에 맞춘 첫 번째 질문 생성"""
    system_prompt = get_qureka_system_prompt(dept, grade)
    prompt = f"""
    {system_prompt}
    
    [강의 내용]
    {full_text[:3000]}
    
    위 강의 자료를 바탕으로 학생의 이해도를 점검할 수 있는 심도 있는 첫 질문을 생성하세요.
    반드시 [다음 질문] 구조로 출력하세요.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[prompt]
        )
        return response.text
    except Exception as e:
        print(f"❌ 질문 생성 실패: {e}")
        return "[다음 질문] 강의 자료를 분석했습니다. 이 자료에서 다루는 가장 핵심적인 개념은 무엇이라고 생각하시나요?"

def generate_chat_response(context_text: str, chat_history: str, dept: str, grade: int) -> str:
    """학생의 답변에 따른 소크라테스식 꼬리 질문 생성"""
    system_prompt = get_qureka_system_prompt(dept, grade)
    prompt = f"""
    {system_prompt}
    
    [참고 강의 내용]
    {context_text}
    
    [이전 대화 기록]
    {chat_history}
    
    학생의 마지막 답변을 분석하여 [피드백]과 [다음 질문]을 제공하세요.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[prompt]
        )
        return response.text
    except Exception as e:
        print(f"❌ 대화 생성 실패: {e}")
        return "[피드백] 답변을 분석하는 중 오류가 발생했습니다. [다음 질문] 다시 한번 설명해주시겠어요?"

# -------------------------------------------------
# 3. 텍스트 처리 및 임베딩
# -------------------------------------------------
def chunk_text_with_page(pages_data: List[Tuple[str, int]], max_words=150):
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
    """실제 Gemini 임베딩 모델 사용 (1536차원)"""
    if not text_list: return []
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text_list
        )
        return [item.values for item in result.embeddings]
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        return [[random.uniform(-1, 1) for _ in range(1536)] for _ in text_list]

def transcribe_audio(audio_path: str):
    print(f"--- 음성 인식 시작: {audio_path} ---")
    result = STT_MODEL.transcribe(audio_path)
    return result['text']