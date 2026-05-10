import os
import json
from google import genai
from dotenv import load_dotenv
from collections import Counter
import re

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_GEMINI_MODEL = "models/gemini-2.5-flash"


def _generation_text(resp) -> str:
    try:
        t = getattr(resp, "text", None)
        return (t or "").strip()
    except Exception:
        return ""

# 너무 흔한 단어 제거
STOPWORDS = {
    "그리고", "하지만", "그러면", "이건", "그건", "있다", "없다",
    "하는", "이다", "입니다", "잘", "너무", "좀", "그", "저", "것",
    "에서", "으로", "하는데", "모르겠어", "설명", "차이", "통해"
}


def extract_keywords(messages):
    words = []

    for msg in messages:
        if msg["role"] != "user":
            continue

        text = msg["content"]
        tokens = re.findall(r"[가-힣A-Za-z]+", text)

        for token in tokens:
            if len(token) >= 2 and token not in STOPWORDS:
                words.append(token)

    return words


def build_tree(messages, concept_freq):
    if not concept_freq:
        return {}

    # 가장 많이 등장한 핵심 개념 1개 선택
    main_concept = concept_freq.most_common(1)[0][0]

    # 상위 관련 개념 5개 추출 (main 제외)
    related = [
        word for word, count in concept_freq.most_common(6)
        if word != main_concept
    ]

    return {
        main_concept: related[:5]
    }

def build_tree_with_gemini(messages, concept_freq):
    if not concept_freq:
        return {}

    chat_text = "\n".join([
        f'{m["role"]}: {m["content"]}'
        for m in messages
    ])

    prompt = f"""
다음은 학생과 AI 튜터의 대화입니다.

{chat_text}

규칙:
- 학생이 학습한 핵심 개념 1개를 중심 개념으로 선정
- 관련 개념 3~5개를 연결
- 학생 질문 중심으로 판단할 것
- 반드시 JSON만 반환
- 중심 개념은 key 하나만 사용

반드시 아래 형식:

{{
  "운영체제": ["프로세스", "스레드", "메모리관리"]
}}
"""

    response = _client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=[prompt],
    )

    cleaned = _generation_text(response).replace("```json", "").replace("```", "")
    try:
        return json.loads(cleaned)
    except Exception:
        return build_tree(messages, concept_freq)


def summarize_text(text):
    if "?" in text or "뭐" in text or "왜" in text:
        return "핵심 개념 질문"
    elif any(word in text for word in ["설명", "역할", "정의"]):
        return "개념 설명 요청"
    elif any(word in text for word in ["차이", "비교"]):
        return "개념 비교 질문"
    else:
        return text[:20]


def build_timeline(messages):
    timeline = []

    for i, msg in enumerate(messages):
        text = msg["content"]
        role = msg["role"]

        if i == 0:
            state = "start"
        elif any(word in text for word in ["모르겠", "헷갈", "어려워"]):
            state = "confusion"
        elif any(word in text for word in ["알겠", "이해", "오케이"]):
            state = "understanding"
        else:
            state = "progress"

        timeline.append({
            "role": role,
            "type": state,
            "text": summarize_text(text)
        })

    return timeline

def build_timeline_with_gemini(messages):
    chat_text = "\n".join([
        f'{m["role"]}: {m["content"]}'
        for m in messages
    ])

    prompt = f"""
다음은 학생과 AI 튜터의 대화입니다.

{chat_text}

규칙:
- 대화 원문 그대로 복사하지 말 것
- 핵심 개념 중심으로 짧게 요약
- text는 15자~25자 이내
- role 유지
- 입력 메시지 개수와 동일한 개수 반환
- JSON 외 다른 설명 금지
- type은 다음 중 하나:
start, progress, confusion, understanding

반드시 JSON만 반환:

[
  {{
    "role": "user",
    "type": "start",
    "text": "운영체제의 정의를 질문"
  }},
  {{
    "role": "assistant",
    "type": "progress",
    "text": "운영체제의 자원 관리 역할 설명"
  }}
]
"""

    response = _client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=[prompt],
    )

    cleaned = _generation_text(response).replace("```json", "").replace("```", "")
    try:
        return json.loads(cleaned)
    except Exception:
        return build_timeline(messages)


def generate_learning_report(messages):
    words = extract_keywords(messages)

    concept_freq = Counter(words)

    tree = build_tree_with_gemini(messages, concept_freq)

    timeline = build_timeline_with_gemini(messages)

    return {
        "concept_frequency": dict(concept_freq.most_common(10)),
        "tree": tree,
        "timeline": timeline
    }
