from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
import os
import re
import io
import mimetypes
import random
import requests  
import json     
from typing import Dict, List, Literal, Tuple
from google import genai
from google.genai import types
from dotenv import load_dotenv

from app.services import academic_concepts as ac

load_dotenv()

SourceKind = Literal["pdf", "transcript"]


def is_audio_file_url(file_url: str | None) -> bool:
    """업로드 경로 기준 음성 파일(.mp3/.wav/.m4a)."""
    if not file_url:
        return False
    return file_url.lower().endswith((".mp3", ".wav", ".m4a"))


def is_transcript_style_file(file_url: str | None) -> bool:
    """PDF가 아닌 소스(음성 전사·txt). 슬라이드/페이지 참조 없이 튜터링."""
    if not file_url:
        return False
    return file_url.lower().endswith((".mp3", ".wav", ".m4a", ".txt"))


def read_uploaded_txt_file(path: str) -> str:
    """로컬 txt 바이트를 UTF-8( BOM )·CP949 등으로 디코딩."""
    with open(path, "rb") as f:
        raw = f.read()
    if not raw:
        return ""
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# Gemini 설정 (SDK 1.0.0+ 기준)
client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

_PDF_MATRIX_SCALE = float(os.getenv("PDF_RENDER_SCALE", "1.75"))
_NATIVE_TEXT_MIN_CHARS = int(os.getenv("PDF_NATIVE_TEXT_MIN_CHARS", "50"))
PDF_OCR_MAX_WORKERS = max(1, int(os.getenv("PDF_OCR_MAX_WORKERS", "4")))
_FORCE_FULL_VISION = os.getenv("PDF_FORCE_FULL_VISION", "").lower() in ("1", "true", "yes")

_GEMINI_SLIDE_PROMPT = """
이미지 속의 모든 내용을 텍스트로 변환하세요.
특히, 사람이 직접 펜으로 적은 '손글씨 필기'나 '메모', 화살표 옆의 낙서 등을 절대로 빼놓지 마세요.
손글씨가 있다면 해당 내용 앞에 반드시 [손글씨] 라고 붙여주세요.
예시: [손글씨] 시험에 나옴!
"""


def _slide_image_to_png_bytes(page) -> bytes:
    pix = page.get_pixmap(matrix=fitz.Matrix(_PDF_MATRIX_SCALE, _PDF_MATRIX_SCALE))
    return pix.tobytes("png")


def ocr_with_gemini_png(img_data: bytes) -> str:
    """슬라이드/페이지 렌더 이미지를 Gemini Vision으로 읽음."""
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=img_data, mime_type="image/png"),
            _GEMINI_SLIDE_PROMPT,
        ],
    )
    text = getattr(response, "text", None)
    return (text or "").strip()


def extract_from_pdf(file_bytes: bytes) -> List[Tuple[str, int]]:
    """
    디지털 PDF는 PyMuPDF 텍스트 레이어를 우선 사용(즉시)하고,
    텍스트가 거의 없는 페이지만 Gemini Vision 처리합니다.
    Vision 호출은 ThreadPoolExecutor로 병렬화합니다.

    모든 페이지를 무조건 Gemini로 할 때는 환경 변수 PDF_FORCE_FULL_VISION=true
    병렬 수: PDF_OCR_MAX_WORKERS (기본 4)
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    native_pages: List[Tuple[str, int]] = []
    ocr_jobs: List[Tuple[int, bytes]] = []
    ocr_fallback: Dict[int, str] = {}

    for page_num, page in enumerate(doc, start=1):
        plain = (page.get_text("text") or "").strip()

        need_vision = _FORCE_FULL_VISION or (
            len(plain) < _NATIVE_TEXT_MIN_CHARS
        )

        if need_vision:
            try:
                ocr_fallback[page_num] = plain
                ocr_jobs.append((page_num, _slide_image_to_png_bytes(page)))
            except Exception as e:
                print(f"⚠ 페이지 {page_num} 렌더 실패: {e}")
                if plain:
                    native_pages.append((plain, page_num))
        elif plain:
            native_pages.append((plain, page_num))

    doc.close()

    vision_results: List[Tuple[str, int]] = []
    if ocr_jobs:
        workers = min(PDF_OCR_MAX_WORKERS, len(ocr_jobs))

        def _run(item: Tuple[int, bytes]) -> Tuple[str, int]:
            pn, png = item
            fb = ocr_fallback.get(pn, "")
            try:
                txt = (ocr_with_gemini_png(png) or "").strip()
                print(f"--- {pn}페이지 Gemini 추출 성공 ---")
                merged = txt or fb
                return (
                    merged if merged else "(빈 응답)",
                    pn,
                )
            except Exception as e:
                print(f"{pn}페이지 API 에러: {e}")
                return (
                    fb or f"에러 발생으로 분석 실패: {e}",
                    pn,
                )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_run, job): job[0] for job in ocr_jobs}
            for fut in as_completed(future_map):
                text, pn = fut.result()
                if text:
                    vision_results.append((text, pn))

    combined = native_pages + vision_results
    combined.sort(key=lambda x: x[1])

    print(f"PDF 추출 완료 - 총 {len(combined)}페이지")

    return combined


# -------------------------------------------------
# 2. 소크라테스식 AI 튜터 'Qureka' 핵심 로직
# -------------------------------------------------

def _matches_structural_heading(text: str) -> bool:
    return ac.matches_structural_heading(text)


def _is_structural_keyword(term: str) -> bool:
    return ac.is_structural_keyword(term)


def _slide_heading(text: str) -> str:
    return ac.slide_heading(text)


def _is_structural_slide_text(text: str) -> bool:
    return ac.is_structural_slide_text(text)


def _document_meta_exclusion_prompt_block(is_transcript: bool = False) -> str:
    return ac.academic_extraction_prompt_block(is_transcript)


def get_qureka_system_prompt(
    dept: str | None,
    grade: int | None,
    lecture_title: str,
    source_kind: SourceKind = "pdf",
) -> str:
    """사용자의 학과와 학년 정보, 강의명을 반영한 시스템 프롬프트 생성"""
    dept_disp = (dept or "").strip() or "학과 미입력"
    grade_disp = f"{grade}학년" if grade is not None else "학년 미입력"
    is_transcript = source_kind == "transcript"
    material_label = (
        "녹음으로부터 변환된 강의 텍스트(전사)"
        if is_transcript
        else "강의 자료(PDF 슬라이드)"
    )
    grounding_location_rule = (
        "- 질문·피드백 시 전사 텍스트에 나온 용어·표현을 우선 활용하세요.\n"
        "- 슬라이드·페이지·장 번호는 존재하지 않습니다. 번호나 '[참조]'류 표기를 쓰지 마세요.\n"
        "- 위치를 짚을 때는 '앞에서 언급된 부분', '바로 다음 문맥'처럼 서술만 사용하세요.\n"
        if is_transcript
        else "- 질문·피드백 시 가능하면 슬라이드 번호 또는 용어를 명시적으로 활용하세요.\n"
    )
    hint_slide_rule = (
        "- 힌트 제공(Hinting): 정답 대신 생각의 실마리가 될 수 있는 힌트(전사 속 키워드, 이전 맥락, 문맥상 인접한 표현)를 제공하여 다시 생각하게 유도하세요.\n"
        "- 전략: 정의를 직접 설명하기보다, 직전 대화의 핵심 용어나 전사 텍스트의 조건·접속 문맥을 활용해 사고를 유도하세요.\n"
        '- 예시: "전사에서 강조한 조건이 여기에도 그대로 적용될까요?"\n'
        if is_transcript
        else "- 힌트 제공(Hinting): 정답 대신 생각의 실마리가 될 수 있는 힌트(자료 내 관련 키워드, 이전 맥락, 슬라이드 번호)를 제공하여 다시 생각하게 유도하세요.\n"
        "- 전략: 정의를 직접 설명하기보다, 직전 대화의 핵심 용어나 강의 자료의 조건·슬라이드 위치를 활용해 사고를 유도하세요.\n"
        '- 예시: "[강의 자료의 특정 슬라이드]에서 강조한 조건이 여기에도 그대로 적용될까요?"\n'
    )
    dont_know_slide_rule = (
        "- 학생이 두 번 이상 \"모르겠다\"고 할 경우, 이전보다 훨씬 더 구체적인 '키워드'나 전사 속 표현을 언급하며 직접적인 힌트를 제공하세요.\n"
        if is_transcript
        else "- 학생이 두 번 이상 \"모르겠다\"고 할 경우, 이전보다 훨씬 더 구체적인 '키워드'나 '슬라이드 번호'를 언급하며 직접적인 힌트를 제공하세요.\n"
    )
    example3 = (
        '> Qureka: 괜찮습니다. 전사 앞부분에서 CPU의 사이클 타임과 IPC의 관계를 다루었는데, IPC가 고정되었을 때 성능을 높일 수 있는 다른 변수는 무엇이었나요?'
        if is_transcript
        else "> Qureka: 괜찮습니다. [슬라이드 4]를 보면 CPU의 사이클 타임과 IPC의 관계가 나오는데, 여기서 IPC가 고정되었을 때 성능을 높일 수 있는 다른 변수는 무엇이었나요?"
    )
    meta_exclusion_block = _document_meta_exclusion_prompt_block(is_transcript)
    meta_few_shot_block = f"""
[Few-shot: 질문 주제 선정 — 반드시 준수]

❌ 절대 금지:
> Qureka: "Goal의 의미와 중요성을 설명해보세요."
> Qureka: "Chapter 1이 무엇인가요?"
> Qureka: "Objective와 Target의 차이는?"
> Qureka: "Introduction이 왜 중요한가요?"

✅ 올바름:
> (제목: Learning Goal / 본문: 프로토콜, 라우팅) → Goal 무시
> Qureka: "강의안에서 언급된 '프로토콜'의 핵심 역할이 무엇인지, 왜 필요한지 직관적으로 설명해 줄 수 있나요?"
> (제목: Overview / 본문: Internet, Protocol, Network core 나열) → Overview 무시, 질문 Skip
> (좌측 "Get feel..." 문장의 Get 단독 추출 금지 → 본문의 Protocol, Network edge 사용)
> (Goal·목차만 있음 → 질문 없음, 다음 {('슬라이드' if not is_transcript else '구간')}로 Skip)
"""
    return f"""
시스템 프롬프트: 소크라테스식 AI 튜터 'Qureka'

0. 최우선 절대 규칙 — 질문 주제는 Core Academic Concept Only
{meta_exclusion_block}
- 이 규칙은 Role, 소크라테스 전략, Few-shot 예시 등 모든 하위 지침보다 항상 우선합니다. 충돌 시 무조건 이 규칙을 따르세요.
- 출력 직전 자가 검증: 질문의 핵심 명사가 Goal/Chapter/Slide/Introduction/목표/목차 등 메타 단어면 즉시 폐기하고, 본문의 전공 개념으로만 다시 작성하세요.
{meta_few_shot_block}

1. Role & Context (역할 및 맥락)
- 당신은 누구인가: 자기주도 학습을 돕는 AI 튜터 'Qureka'입니다.
- 현재 학습 중인 과목: [{lecture_title}]
- 사용자: 당신 앞에 있는 학생은 [{dept_disp}] 학과 [{grade_disp}] 전공자입니다. 학생의 수준에 맞는 전문적인 용어와 논리를 사용하세요.
- 목표: 사용자가 업로드한 [{material_label}]를 분석하여, 단순 요약이 아닌 소크라테스식 문답법을 통해 학생이 스스로 개념을 깨우치고 사고를 확장하도록 돕는 것입니다.

2. Prime Directives (핵심 지시사항)
A. 엄격한 자료 기반성 (Strict Grounding)
- 모든 질문과 피드백은 반드시 제공된 [{material_label}]의 내용과 팩트 그리고 [{lecture_title}] 과목의 맥락에 근거해야 합니다.
{grounding_location_rule}
- 가능하면 질문 속 핵심 용어를 강의 자료의 원문 표현 그대로 유지하세요.
- 강의 자료 범위를 벗어난 질문에는 "해당 내용은 강의 자료에서 확인할 수 없습니다."라고 밝히고, 자료 내의 연관된 주제로 대화를 이끄세요.
- 배경지식을 활용하되, 정답의 근거는 반드시 강의 자료에서 찾아야 합니다.
- 학생 답변을 평가할 때는 표현의 일치보다 개념적 타당성과 강의 자료 핵심 의미의 일치 여부를 우선 판단하세요.
- 강의 자료에서 반복되거나 강조된 개념, 도식·본문 설명을 우선적으로 핵심 개념으로 간주하세요.
- Section 0의 메타 단어(Goal, Chapter, Slide, Objective, 목표, 목차 등)는 어떤 경우에도 핵심 개념이 아닙니다.

B. 소크라테스식 질문 전략 (Socratic Method)
- 새 개념에 대한 첫 질문은 반드시 기초 정의·직관 수준에서 시작하고, 이후 응답에 따라 점진적으로 심화하세요.
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
{hint_slide_rule}
- 전략: 한 번에 정답 방향을 모두 주지 말고, 학생이 스스로 연결할 수 있도록 가장 가까운 단서부터 한 단계씩 제시하세요.
- 예시: "직전 답변에서 언급한 [핵심 용어]와 연결해서 다시 생각해볼 수 있을까요?"
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
- 동일한 격려나 권유(예: 다시 생각해보세요, 조금 더 고민해보세요)를 2회 이상 반복하지 마세요.
{dont_know_slide_rule}
- "모르겠다"는 답변에 대해 단순히 다시 생각하라고 말하는 것은 금지하며, 반드시 사고의 출발점이 될 수 있는 자료 내의 구체적 사실을 한 가지 언급하세요.

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

[예시 3: 학생이 모르겠다고 할 때]
> 학생: "잘 모르겠어요."
{example3}
"""

def _valid_slide_number(page_num) -> bool:
    return isinstance(page_num, int) and page_num > 0


def select_key_chunks(lecture_pages, max_pages=3, include_page_tag=True):
    clean_pages = [
        ((text or "").strip(), page_num)
        for text, page_num in lecture_pages
        if (text or "").strip()
    ]
    if not clean_pages:
        return ""

    if include_page_tag:
        slide_pages = [
            (text, page_num)
            for text, page_num in clean_pages
            if _valid_slide_number(page_num)
        ]
        # PDF 질문에서는 "슬라이드 None"이 생기지 않도록 실제 슬라이드 번호가 있는 페이지만 우선 사용.
        pages_for_selection = slide_pages or clean_pages
    else:
        pages_for_selection = clean_pages

    substantive_pages = [
        (text, page_num)
        for text, page_num in pages_for_selection
        if not _is_structural_slide_text(text) and not ac.is_overview_roadmap_slide(text)
    ]
    ordered_pages = substantive_pages or [
        (text, page_num)
        for text, page_num in pages_for_selection
        if not _is_structural_slide_text(text)
    ] or pages_for_selection

    first_page = ordered_pages[0]
    others = ordered_pages[1:]

    scored = []

    for text, page_num in others:
        score = len(text.split())   # 단어 수 기준 점수
        scored.append((score, text, page_num))

    scored.sort(reverse=True)

    selected = [first_page] + [
        (text, page_num)
        for _, text, page_num in scored[:max_pages - 1]
    ]

    if include_page_tag:
        result = "\n".join([
            f"[슬라이드 {page_num}]\n{text}" if _valid_slide_number(page_num) else text
            for text, page_num in selected
        ])
    else:
        result = "\n".join([
            text
            for text, _ in selected
        ])

    return result


def _extract_initial_question_topic(selected_text: str, lecture_title: str) -> str:
    return ac.pick_best_academic_concept(selected_text, lecture_title)


def _first_valid_slide_number(lecture_pages) -> int | None:
    for text, page_num in lecture_pages:
        if (
            _valid_slide_number(page_num)
            and not _is_structural_slide_text(text)
            and not ac.is_overview_roadmap_slide(text)
        ):
            return page_num
    for text, page_num in lecture_pages:
        if _valid_slide_number(page_num) and not _is_structural_slide_text(text):
            return page_num
    for _, page_num in lecture_pages:
        if _valid_slide_number(page_num):
            return page_num
    return None


def _first_question_source_pages(lecture_pages) -> tuple[str, int | None]:
    """질문 생성용: Overview/목차 슬라이드를 건너뛴 텍스트와 슬라이드 번호."""
    chunks = []
    slide_num = None
    for text, page_num in lecture_pages:
        if not (text or "").strip():
            continue
        if ac.is_overview_roadmap_slide(text) or _is_structural_slide_text(text):
            continue
        if _valid_slide_number(page_num):
            chunks.append(f"[슬라이드 {page_num}]\n{text.strip()}")
            if slide_num is None:
                slide_num = page_num
        else:
            chunks.append(text.strip())
        if len(chunks) >= 3:
            break
    return "\n\n".join(chunks), slide_num

def generate_initial_question(
    lecture_pages,
    dept: str | None,
    grade: int | None,
    lecture_title: str,
    source_kind: SourceKind = "pdf",
) -> str:
    """강의 자료 확정 시 사용자의 학과/학년에 맞춘 첫 번째 질문 생성"""
    if source_kind == "pdf":
        question_text, slide_number = _first_question_source_pages(lecture_pages)
        if not question_text:
            question_text = select_key_chunks(
                lecture_pages,
                include_page_tag=True,
            )
    else:
        question_text = select_key_chunks(
            lecture_pages,
            include_page_tag=False,
        )
        slide_number = None

    if os.getenv("INITIAL_QUESTION_USE_GEMINI", "").lower() not in ("1", "true", "yes"):
        topic = ac.pick_best_academic_concept(question_text, lecture_title)
        if source_kind == "pdf" and slide_number is not None:
            source_phrase = f"슬라이드 {slide_number}에서"
        elif source_kind == "transcript":
            source_phrase = "녹음 자료에서"
        else:
            source_phrase = "강의 자료에서"
        return ac.format_first_socratic_question(topic, source_phrase)

    system_prompt = get_qureka_system_prompt(dept, grade, lecture_title, source_kind)
    prompt = f"""
    {system_prompt}
    
    [강의 내용]
    {question_text}

    위 강의 자료를 바탕으로 학생의 이해도를 점검할 수 있는 심도 있는 첫 질문을 생성하세요.
    Overview/Roadmap/목차 슬라이드는 Skip하고, 상세 설명 슬라이드의 전공 명사만 주제로 하세요.

    [첫 질문 생성 — 최우선 재확인]
    {_document_meta_exclusion_prompt_block(source_kind == "transcript")}
    - "X의 느낌/이미지" 질문 금지. "X의 핵심 역할과 왜 필요한지" 형태로 작성하세요.

    PDF 슬라이드 번호는 실제 [슬라이드 N] 태그가 있을 때만 언급하고, 절대로 "슬라이드 None" 또는 "None 개념"이라고 말하지 마세요.
    녹음/전사 자료에는 슬라이드 번호를 붙이지 말고 "녹음 자료"라고만 표현하세요.
    태그 없이 질문만 자연스럽게 출력하세요.
    """
    
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[prompt]
        )
        text = getattr(response, "text", None) or ""
        text = text.strip()
        return text or (
            "강의 자료를 분석했습니다. 이 자료에서 다루는 가장 핵심적인 개념은 무엇이라고 생각하시나요?"
        )
    except Exception as e:
        print(f"❌ 질문 생성 실패: {e}")
        return "강의 자료를 분석했습니다. 이 자료에서 다루는 가장 핵심적인 개념은 무엇이라고 생각하시나요?"


def generate_chat_response(
    context_text: str,
    chat_history: str,
    dept: str | None,
    grade: int | None,
    lecture_title: str,
    source_kind: SourceKind = "pdf",
) -> str:
    """학생의 답변에 따른 소크라테스식 꼬리 질문 생성"""
    system_prompt = get_qureka_system_prompt(dept, grade, lecture_title, source_kind)
    context_heading = (
        "[관련 전사 발췌]"
        if source_kind == "transcript"
        else "[참고 강의 내용]"
    )
    prompt = f"""
    {system_prompt}
    
    {context_heading}
    {context_text}
    
    [이전 대화 기록]
    {chat_history}

    학생의 마지막 답변이 강의 자료 핵심 개념과 일치하는지 먼저 판단하세요.
    핵심 개념이 빠졌다면 긍정 표현 없이 논리적 재질문으로 이어가세요.

    [다음 질문 생성 — 최우선 재확인]
    {_document_meta_exclusion_prompt_block(source_kind == "transcript")}

    학생의 마지막 답변을 분석하여 짧은 피드백 후 자연스럽게 다음 질문을 이어서 제시하세요.
    태그 없이 실제 튜터처럼 출력하세요.
    """
    
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[prompt]
        )
        text = getattr(response, "text", None) or ""
        text = text.strip()
        return text or (
            "답변을 다시 생각해보면 어떨까요? 왜 그렇게 판단했는지 설명해줄 수 있나요?"
        )
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
        print(f"임베딩 실패: {e}")
        return [[random.uniform(-1, 1) for _ in range(3072)] for _ in text_list]


def _clova_longspeech_extract_text(data: dict) -> str | None:
    """
    CLOVA Speech Long Sentence(sync) JSON에서 전체 텍스트 추출.
    fullText가 비어 있어도 segments[].text / textEdited를 이어 붙입니다.
    """
    if not isinstance(data, dict):
        return None
    top = data.get("text")
    if isinstance(top, str) and top.strip():
        return top.strip()
    segments = data.get("segments")
    if not isinstance(segments, list):
        return None
    parts: List[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        frag = (seg.get("textEdited") or seg.get("text") or "").strip()
        if frag:
            parts.append(frag)
    if not parts:
        return None
    return " ".join(parts)


def _transcribe_ncp_csr_short(audio_path: str) -> str | None:
    """
    NCP CSR 단문 STT (바이너리 POST). 최대 약 60초 음성.
    CLOVA Long Sentence가 본문을 주지 않을 때 선택 폴백.
    """
    cid = (
        os.getenv("NCP_CSR_CLIENT_ID")
        or os.getenv("NAVER_OPENAPI_CLIENT_ID")
        or ""
    ).strip()
    csec = (
        os.getenv("NCP_CSR_CLIENT_SECRET")
        or os.getenv("NAVER_OPENAPI_CLIENT_SECRET")
        or ""
    ).strip()
    if not cid or not csec:
        return None
    lang = (os.getenv("NCP_CSR_LANG") or "Kor").strip()
    base = (
        os.getenv("NCP_CSR_STT_URL")
        or "https://naveropenapi.apigw.ntruss.com/recog/v1/stt"
    ).rstrip("/")
    url = f"{base}?lang={lang}"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": cid,
        "X-NCP-APIGW-API-KEY": csec,
        "Content-Type": "application/octet-stream",
    }
    try:
        with open(audio_path, "rb") as f:
            raw = f.read()
        r = requests.post(
            url,
            headers=headers,
            data=raw,
            timeout=int(os.getenv("NCP_CSR_TIMEOUT_SECONDS", "60")),
        )
        if r.status_code != 200:
            print(f"NCP CSR STT HTTP {r.status_code}: {(r.text or '')[:500]}")
            return None
        try:
            body = r.json()
        except json.JSONDecodeError:
            t = (r.text or "").strip()
            return t or None
        if isinstance(body, dict):
            txt = body.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
    except requests.RequestException as e:
        print(f"NCP CSR STT 요청 실패: {e}")
    return None


def transcribe_audio(audio_path: str):
    print(f"--- Clova Speech 음성 인식 시작: {audio_path} ---")

    if not os.path.exists(audio_path):
        return "오디오 파일이 존재하지 않습니다."

    invoke_url = os.getenv("CLOVA_SPEECH_INVOKE_URL")
    secret_key = os.getenv("CLOVA_SPEECH_SECRET_KEY")

    if not invoke_url or not secret_key:
        return "Clova Speech API 설정이 없습니다."

    url = f"{invoke_url.rstrip('/')}/recognizer/upload"

    request_body = {
        "language": "ko-KR",
        "completion": "sync",
        "fullText": True,
        "wordAlignment": True,
        "diarization": {"enable": False},
    }

    headers = {
        "Accept": "application/json",
        "X-CLOVASPEECH-API-KEY": secret_key,
    }

    ext = os.path.splitext(audio_path)[1].lower()
    mime_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".wave": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }
    guessed, _ = mimetypes.guess_type(audio_path)
    content_type = (guessed or mime_map.get(ext) or "application/octet-stream")
    media_name = os.path.basename(audio_path) or f"audio{ext or '.bin'}"

    timeout = max(30, int(os.getenv("CLOVA_SPEECH_TIMEOUT_SECONDS", "180")))

    try:
        with open(audio_path, "rb") as f:
            files = {
                "media": (media_name, f, content_type),
                "params": (None, json.dumps(request_body, ensure_ascii=False), "application/json"),
            }
            response = requests.post(
                url,
                headers=headers,
                files=files,
                timeout=timeout,
            )

        response.raise_for_status()

        try:
            data = response.json()
        except json.JSONDecodeError:
            snippet = (response.text or "")[:800]
            return f"Clova Speech JSON 파싱 실패(HTTP {response.status_code}): {snippet}"

        if not isinstance(data, dict):
            return f"Clova Speech 예상치 못한 응답 형식: {type(data).__name__}"

        result_code = str(data.get("result") or "").upper()
        if result_code and result_code not in ("COMPLETED", "SUCCEEDED"):
            msg = data.get("message") or ""
            print(f"Clova Speech result={result_code}, message={msg}")

        full_text = _clova_longspeech_extract_text(data)
        if full_text:
            print(f"--- Clova Speech 추출 성공 ({audio_path}) ---")
            return full_text

        fb = _transcribe_ncp_csr_short(audio_path)
        if fb:
            print(f"--- NCP CSR STT 폴백 성공 ({audio_path}) ---")
            return fb

        print(
            "Clova Speech 응답(본문 없음, 일부 로그):",
            json.dumps(data, ensure_ascii=False)[:2500],
        )
        msg = data.get("message") or result_code or "알 수 없음"
        return (
            "Clova Speech에서 인식 텍스트를 받지 못했습니다. "
            f"(result={data.get('result')}, message={msg}) "
            "음성 형식·길이·Invoke URL을 확인하세요. "
            "단문(60초 이하) 대안으로 환경변수 NCP_CSR_CLIENT_ID, NCP_CSR_CLIENT_SECRET을 설정하면 "
            "NCP CSR STT로 자동 재시도합니다."
        )

    except requests.exceptions.Timeout:
        return "음성 인식 요청 시간이 초과되었습니다."

    except requests.exceptions.RequestException as e:
        return f"Clova API 요청 오류: {e}"

    except Exception as e:
        return f"음성 처리 중 오류 발생: {str(e)}"


def is_stt_service_error(text: str | None) -> bool:
    """Clova/STT 단계 실패 메시지면 True (교정 대상 아님)."""
    if not text or not str(text).strip():
        return True
    s = str(text).strip()
    prefixes = (
        "오디오 파일이 존재하지",
        "Clova Speech API 설정이 없습니다",
        "음성 인식 요청 시간이 초과",
        "Clova API 요청 오류",
        "음성 처리 중 오류",
        "Clova Speech에서 인식 텍스트를 받지 못했습니다",
        "Clova Speech JSON 파싱 실패",
        "Clova Speech 예상치 못한 응답",
        "[텍스트 추출 오류]",
    )
    return any(s.startswith(p) for p in prefixes)


def refine_stt_with_lecture_docs(stt_raw: str, reference_corpus: str) -> str:
    """
    강의 자료(참고 텍스트)만 근거로 STT 오인식을 보수. 새 내용 창작·할루시네이션 금지.
    """
    stt = (stt_raw or "").strip()
    ref = (reference_corpus or "").strip()
    if not stt:
        return ""
    if not ref:
        return stt

    max_ref = max(8000, int(os.getenv("STT_REFINE_REFERENCE_MAX_CHARS", "120000")))
    if len(ref) > max_ref:
        ref = ref[:max_ref] + "\n[... 이하 생략 ...]"

    prompt = f"""당신은 STT(음성 인식) 전사 오류만 바로잡는 교정기입니다.

[절대 규칙 — 위반 시 잘못된 응답]
1. 출력은 "교정된 전사 텍스트" 한 덩어리만. 제목·설명·따옴표·불릿·메타 코멘트 금지.
2. 아래 [강의 자료]에 없는 사실·개념·문장·수치를 새로 추가하지 마세요. 교수가 말했다고 지어낸 내용(할루시네이션) 금지.
3. [STT 원문]에 없는 발화를 요약·추론·보완해서 넣지 마세요.
4. [STT 원문]의 말 순서와 문단 흐름을 최대한 유지하세요.
5. 단어만 강의 자료에 실제로 등장하는 표기·용어로 바꿀 수 있을 때만 바꾸세요. 근거가 애매하면 원문 유지.
6. 강의 자료의 문장/단락을 그대로 복사해 전사에 끼워 넣지 마세요.
7. 알아듣기 어려운 한 구간만 [들리지 않음]으로 표시할 수 있으나 남용 금지.
8. 출력 길이는 STT 원문 길이의 약 80%~120% 범위를 넘지 마세요(불필요한 확장 금지).

[강의 자료 — 참고·근거용. 이 텍스트를 인용해 전사를 대체하지 말 것]
{ref}

[STT 원문]
{stt}
"""
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.1),
        )
        out = getattr(response, "text", None) or ""
        out = out.strip()
        return out if out else stt
    except Exception as e:
        print(f"STT 교정(Gemini) 실패, 원문 유지: {e}")
        return stt

def process_lecture_materials(
    pdf_bytes: bytes | None = None, 
    audio_path: str | None = None,
    dept: str | None = None,
    grade: int | None = None,
    lecture_title: str = "미지정 강의"
) -> dict:
    """
    PDF 자료와 음성 파일이 함께 업로드되었을 때 
    순서를 보장하여 처리하는 마스터 파이프라인 함수
    """
    pdf_text_inside = ""
    transcript_raw = ""
    transcript_refined = ""
    source_kind: SourceKind = "pdf"

    # 1단계: PDF 추출을 무조건 '먼저' 수행 (교정의 기준 정립)
    if pdf_bytes:
        print("▶ [1단계] PDF 텍스트 추출 시작")
        pdf_pages = extract_from_pdf(pdf_bytes)
        
        # 교정기(refine_stt_with_lecture_docs)에 통째로 넣기 위해 하나의 텍스트로 합침
        pdf_text_inside = "\n".join([f"[슬라이드 {pn}] {txt}" for txt, pn in pdf_pages])
        print(f"▶ [1단계] PDF 추출 완료 (총 {len(pdf_pages)}페이지 분량 코퍼스 확보)")

    # 2단계: 음성 파일 STT 추출 진행
    if audio_path and os.path.exists(audio_path):
        print("▶ [2단계] 음성 파일 STT 추출 시작")
        transcript_raw = transcribe_audio(audio_path)
        
        # Clova Speech 에러 메시지인지 검증
        if is_stt_service_error(transcript_raw):
            print(f"STT 서비스 오류 발생: {transcript_raw}")
            transcript_refined = transcript_raw
        else:
            print("▶ [2단계] 음성 파일 STT 추출 성공")
            
            # 3단계: PDF 텍스트가 존재한다면 이를 바탕으로 음성 텍스트 '교정' 진행
            if pdf_text_inside:
                print("▶ [3단계] PDF 자료를 기반으로 STT 오인식 교정 시작")
                transcript_refined = refine_stt_with_lecture_docs(
                    stt_raw=transcript_raw, 
                    reference_corpus=pdf_text_inside
                )
                print("▶ [3단계] STT 교정 완료")
            else:
                transcript_refined = transcript_raw
                source_kind = "transcript"
                
    # [보완] 음성 파일이 아예 안 들어왔거나, 들어왔더라도 STT 서비스 자체에 실패한 경우
    if not transcript_refined and pdf_text_inside:
        source_kind = "pdf"
        transcript_refined = pdf_text_inside

    # 최종 결과 반환
    return {
        "source_kind": source_kind,
        "pdf_corpus": pdf_text_inside,          # 교정의 기준이 된 PDF 텍스트
        "transcript_raw": transcript_raw,      # 교정 전 순수 STT 결과
        "transcript_refined": transcript_refined # 최종 교정 완료된 결과
    }