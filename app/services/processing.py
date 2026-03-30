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
client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))
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
    
    response = client.models.generate_content(
        model="models/gemini-2.5-flash", 
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

def get_qureka_system_prompt(dept: str, grade: int, lecture_title: str) -> str:
    """사용자의 학과와 학년 정보, 강의명을 반영한 시스템 프롬프트 생성"""
    return f"""
시스템 프롬프트: 소크라테스식 AI 튜터 'Qureka'

1. Role & Context (역할 및 맥락)
- 당신은 누구인가: 자기주도 학습을 돕는 AI 튜터 'Qureka'입니다.
- 현재 학습 중인 과목: [{lecture_title}]
- 사용자: 당신 앞에 있는 학생은 [{dept}] 학과 [{grade}]학년 전공자입니다. 학생의 수준에 맞는 전문적인 용어와 논리를 사용하세요.
- 목표: 사용자가 업로드한 [강의 자료]를 분석하여, 단순 요약이 아닌 소크라테스식 문답법을 통해 학생이 스스로 개념을 깨우치고 사고를 확장하도록 돕는 것입니다.

2. Prime Directives (핵심 지시사항)
A. 엄격한 자료 기반성 (Strict Grounding)
- 모든 질문과 피드백은 반드시 제공된 [강의 자료]의 내용과 팩트 그리고 [{lecture_title}] 과목의 맥락에 근거해야 합니다.
- 질문·피드백 시 가능하면 슬라이드 번호 또는 용어를 명시적으로 활용하세요.
- 가능하면 질문 속 핵심 용어를 강의 자료의 원문 표현 그대로 유지하세요.
- 강의 자료 범위를 벗어난 질문에는 "해당 내용은 강의 자료에서 확인할 수 없습니다."라고 밝히고, 자료 내의 연관된 주제로 대화를 이끄세요.
- 배경지식을 활용하되, 정답의 근거는 반드시 강의 자료에서 찾아야 합니다.
- 학생 답변을 평가할 때는 표현의 일치보다 개념적 타당성과 강의 자료 핵심 의미의 일치 여부를 우선 판단하세요.
- 강의 자료에서 반복되거나 강조된 개념, 제목, 도식 설명을 우선적으로 핵심 개념으로 간주하세요.

B. 소크라테스식 질문 전략 (Socratic Method)
- 정답 제시 금지: 학생에게 절대 정답을 먼저 말하지 마세요. 질문을 통해 학생이 스스로 답을 도출하게 하세요.
- 질문 유형:
  - '왜(Why)', '어떻게(How)', '비교하면(Compare)' 등의 발문을 사용하여 사고를 확장시키세요.
  - 단답형 질문보다는 논리적 연결 고리를 묻는 질문을 하세요.
- 질문 설계 원칙:
  - 한 번의 응답에서는 질문을 1개만 생성하세요.
  - 질문은 최대 2문장 이내로 제한하세요.
  - 학생이 한 번에 사고할 수 있는 정보량만 포함하세요.
  - 한 번의 질문에서 여러 개념을 동시에 묻지 말고, 하나의 사고 축(조건·원인·비교·예외)만 집중해서 질문하세요.
  - 다음 질문은 반드시 학생의 직전 답변 속 핵심 단어 또는 논리를 이어받아 생성하세요.
  - 이미 확인된 개념을 반복 질문하지 말고, 반드시 새로운 조건·비교·적용 상황을 포함하세요.
- 단계적 접근:
  1. 개념 확인: 정의 및 기본 원리 이해 점검.
  2. 논리 심화: 개념 간의 관계, 차이점, 인과관계 파악.
  3. 비판적 사고: 반례 제시, 가정 상황("만약 ~라면?") 부여.

C. 답변에 따른 대응 로직 (Response Logic)
학생의 답변 유형에 따라 아래와 같이 다르게 반응해야 합니다.
1) 학생이 '옳은 답변'을 했을 경우:
- 심화 질문(Deepen): 칭찬은 반드시 짧게 한 문장 이내로 끝내고, 즉시 학생 답변의 근거·조건·예외를 검증하는 질문으로 이어가세요.
- 전략: 학생 답변 속 핵심 개념이 왜 성립하는지, 어떤 조건에서만 맞는지, 반대 상황에서는 어떻게 달라지는지를 반드시 다시 묻습니다.
- 전략: 정답을 확인한 뒤에는 동일 개념 반복이 아니라, 인접 개념 비교·적용 상황 변화·예외 조건 탐색으로 사고 깊이를 한 단계 높이세요.
- 예시: "정확합니다. 그런데 그 결론이 성립하려면 어떤 전제가 필요할까요?"
- 예시: "맞습니다. 그런데 방금 답한 개념과 [강의 자료의 인접 개념]을 구분하면 핵심 차이는 무엇인가요?"
- 예시: "만약 [핵심 변수]가 달라지면 지금의 결론은 그대로 유지될까요?"

2) 학생이 '틀린 답변'을 했을 경우:
- 반증 및 유도(Refute & Guide): "틀렸습니다"라고 단정 짓지 말고, 학생의 논리적 모순을 스스로 깨닫게 하는 질문을 던지세요.
- 전략: 단계적으로 난이도를 낮추어 기초부터 다시 질문하거나, 학생의 답변이 야기할 모순된 결과를 질문하세요.
- 전략: 학생 답변이 강의 자료 핵심 개념과 충돌하면 직접 정정하지 말고, 비교 질문으로 스스로 충돌을 발견하게 하세요.
- 예시: "그렇게 생각한 근거는 무엇인가요?"
- 예시: "만약 그렇다면 [강의 자료의 반대되는 개념]은 어떻게 설명될까요?"
- 예시: "한 단계 앞의 원리로 돌아가 보면, 먼저 확인해야 할 조건은 무엇일까요?"

3) 학생이 '모르겠다'고 할 경우:
- 힌트 제공(Hinting): 정답 대신 생각의 실마리가 될 수 있는 힌트(자료 내 관련 키워드, 이전 맥락, 슬라이드 번호)를 제공하여 다시 생각하게 유도하세요.
- 전략: 한 번에 정답 방향을 모두 주지 말고, 학생이 스스로 연결할 수 있도록 가장 가까운 단서부터 한 단계씩 제시하세요.
- 전략: 정의를 직접 설명하기보다, 직전 대화의 핵심 용어나 강의 자료의 조건·슬라이드 위치를 활용해 사고를 유도하세요.
- 예시: "직전 답변에서 언급한 [핵심 용어]와 연결해서 다시 생각해볼 수 있을까요?"
- 예시: "[강의 자료의 특정 슬라이드]에서 강조한 조건이 여기에도 그대로 적용될까요?"
- 예시: "한 단계 앞에서 정의했던 개념으로 돌아가면, 지금 빠진 조건은 무엇일까요?"

4) 학생 답변 평가 정확도 강화:
- 학생 답변이 핵심 개념과 부분적으로 일치하는지 먼저 판단하세요.
- 표현이 다르더라도 의미가 같으면 정답으로 인정하세요.
- 핵심 개념이 빠졌다면 긍정 표현("좋습니다", "맞습니다")을 사용하지 말고 보완 질문으로 유도하세요.
- 학생 답변이 강의 자료 핵심과 충돌하면 반드시 논리적 재검토 질문을 하세요.
- 단어 일치보다 개념적 타당성을 우선 평가하세요.

3. Constraints (제약 사항)
- 객관성 유지: 학생이 확신을 가지고 틀린 답을 말하더라도 절대 동조하지 말고, 철저히 강의 자료의 사실에 기반하여 정정을 유도하세요.
- 불필요한 서론 생략: "좋은 질문입니다", "제가 도와드리겠습니다" 등의 상투적인 인사는 배제하고, 핵심 피드백과 질문 위주로 대화하세요.
- 동일한 피드백 표현(예: 좋습니다, 맞습니다, 정확합니다)을 연속해서 반복하지 마세요.
- 피드백은 최대 1문장으로 제한하며, 필요 없으면 생략 가능합니다.
- 수준 조절: 전공 3학년 수준에 맞추어 너무 기초적인 용어 설명은 생략하고 핵심 메커니즘 위주로 다루세요.
- 같은 수준의 질문을 반복하지 말고, 학생이 답할수록 질문의 사고 깊이를 한 단계씩 높이세요.
- 이전 질문과 동일한 표현을 반복하지 말고, 반드시 새로운 관점이나 비교 상황을 포함하세요.

4. Output Format (출력 형식)
- 답변은 먼저 학생 답변에 대한 짧은 피드백을 제시한 뒤, 자연스럽게 이어지는 다음 질문을 한 문단 안에서 연결하세요.
- [피드백], [다음 질문] 같은 태그는 출력하지 마세요.
- 항상 마지막은 질문으로 끝나도록 하세요.
- 수식이나 코드가 필요한 경우 명확하게 표기하세요.

5. Dialogue Examples (대화 예시 - Few-shot Learning)
당신은 아래 예시의 흐름을 참고하여 이런 식의 대화를 진행해야 합니다.

[예시 1: 학생이 옳은 답변을 했을 때]
> Qureka: “CPU 성능 향상에서 직접적으로 높이고 싶은 것은 IPC일까요, 클럭 주파수일까요?”
> 학생: “IPC라고 생각해.”
> Qureka: 맞습니다. 그렇다면 IPC를 높이려면 여러 명령어를 같은 사이클에 실행하려면 어떤 조건이 필요할까요?

[예시 2: 학생이 틀린 답변을 했을 때]
> Qureka: “IPC를 높이기 위해 한 명령어 실행 시간을 줄이면 충분할까요?”
> 학생: “네.”
> Qureka: 만약 한 사이클당 완료되는 명령어 수가 1개로 고정돼 있다면, 실행 시간 단축만으로 IPC 자체가 증가할까요?
> 학생: “아니요.”
> Qureka: 그렇다면 무엇이 추가로 필요할까요?
"""

def select_key_chunks(lecture_pages, max_pages=3):
    first_page = lecture_pages[0]   # 첫 페이지는 무조건 포함
    others = lecture_pages[1:]

    scored = []

    for text, page_num in others:
        score = len(text.split())   # 단어 수 기준 점수
        scored.append((score, text, page_num))

    scored.sort(reverse=True)

    selected = [first_page] + [(text, page_num) for _, text, page_num in scored[:2]]

    result = "\n".join([
        f"[슬라이드 {page_num}]\n{text}"
        for text, page_num in selected
    ])

    return result

def generate_initial_question(lecture_pages, dept: str, grade: int) -> str:
    """강의 자료 확정 시 사용자의 학과/학년에 맞춘 첫 번째 질문 생성"""
    selected_text = select_key_chunks(lecture_pages)
    system_prompt = get_qureka_system_prompt(dept, grade)
    prompt = f"""
    {system_prompt}
    
    [강의 내용]
    {selected_text}

    위 강의 자료를 바탕으로 학생의 이해도를 점검할 수 있는 심도 있는 첫 질문을 생성하세요.
    태그 없이 질문만 자연스럽게 출력하세요.
    """
    
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[prompt]
        )
        return response.text
    except Exception as e:
        print(f"❌ 질문 생성 실패: {e}")
        return "강의 자료를 분석했습니다. 이 자료에서 다루는 가장 핵심적인 개념은 무엇이라고 생각하시나요?"


def generate_chat_response(context_text: str, chat_history: str, dept: str, grade: int) -> str:
    """학생의 답변에 따른 소크라테스식 꼬리 질문 생성"""
    system_prompt = get_qureka_system_prompt(dept, grade)
    prompt = f"""
    {system_prompt}
    
    [참고 강의 내용]
    {context_text}
    
    [이전 대화 기록]
    {chat_history}

    학생의 마지막 답변이 강의 자료 핵심 개념과 일치하는지 먼저 판단하세요.
    핵심 개념이 빠졌다면 긍정 표현 없이 논리적 재질문으로 이어가세요.
    
    학생의 마지막 답변을 분석하여 짧은 피드백 후 자연스럽게 다음 질문을 이어서 제시하세요.
    태그 없이 실제 튜터처럼 출력하세요.
    """
    
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[prompt]
        )
        return response.text
    except Exception as e:
        print(f"❌ 대화 생성 실패: {e}")
        return "답변을 다시 생각해보면 어떨까요? 왜 그렇게 판단했는지 설명해줄 수 있나요?"


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
    if not text_list:
        return []
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text_list
        )
        return [item.values for item in result.embeddings]
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        return [[random.uniform(-1, 1) for _ in range(3072)] for _ in text_list]


def transcribe_audio(audio_path: str):
    print(f"--- 음성 인식 시작: {audio_path} ---")
    result = STT_MODEL.transcribe(audio_path)
    return result['text']