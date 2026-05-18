import os
import json
from google import genai
from dotenv import load_dotenv
from collections import Counter
from datetime import datetime
import re

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_GEMINI_MODEL = "models/gemini-2.5-flash"
_LONG_RESPONSE_SECONDS = int(os.getenv("REPORT_LONG_RESPONSE_SECONDS", "90"))


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
    "에서", "으로", "하는데", "모르겠어", "설명", "차이", "통해",
    "이해", "알겠", "모르겠", "헷갈", "어려워", "같아", "대해"
}


TIMELINE_TYPE_META = {
    "start": {
        "label": "학습 시작",
        "color": "gray",
        "severity": 0,
    },
    "understanding": {
        "label": "이해 양호",
        "color": "green",
        "severity": 0,
    },
    "progress": {
        "label": "학습 진행",
        "color": "blue",
        "severity": 1,
    },
    "hesitation": {
        "label": "답변 지연/고민",
        "color": "orange",
        "severity": 2,
    },
    "confusion": {
        "label": "취약/혼란",
        "color": "red",
        "severity": 3,
    },
}


def _clean_json_text(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def _safe_json_loads(text: str):
    cleaned = _clean_json_text(text)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def _parse_datetime(value):
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _elapsed_from_previous(messages, idx: int) -> int | None:
    if idx <= 0:
        return None
    cur = _parse_datetime(messages[idx].get("created_at"))
    prev = _parse_datetime(messages[idx - 1].get("created_at"))
    if not cur or not prev:
        return None
    return max(0, int((cur - prev).total_seconds()))


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

    # 상위 관련 개념 8개 추출 (main 제외)
    related = [
        word for word, count in concept_freq.most_common(12)
        if word != main_concept
    ]

    return {
        main_concept: related[:8]
    }


def build_detailed_tree(messages, concept_freq):
    """Gemini 실패 시에도 프론트가 풍부한 마인드맵을 그릴 수 있는 기본 구조."""
    if not concept_freq:
        return {"root": "", "nodes": [], "edges": []}

    top = concept_freq.most_common(12)
    root = top[0][0]
    nodes = [
        {
            "id": root,
            "label": root,
            "level": 0,
            "weight": top[0][1],
            "summary": "대화에서 가장 자주 다룬 중심 개념",
        }
    ]
    edges = []

    for word, count in top[1:]:
        nodes.append({
            "id": word,
            "label": word,
            "level": 1,
            "weight": count,
            "summary": "중심 개념과 함께 언급된 관련 개념",
        })
        edges.append({
            "from": root,
            "to": word,
            "label": "관련",
        })

    return {
        "root": root,
        "nodes": nodes,
        "edges": edges,
    }

def build_tree_with_gemini(messages, concept_freq):
    if not concept_freq:
        return {"tree": {}, "mindmap": {"root": "", "nodes": [], "edges": []}}

    chat_text = "\n".join([
        f'{m["role"]}: {m["content"]}'
        for m in messages
    ])

    prompt = f"""
다음은 학생과 AI 튜터의 대화입니다.

{chat_text}

목표:
- 프론트에서 마인드맵으로 시각화할 수 있도록 핵심 개념을 충분히 구조화하세요.
- 단순 키워드 나열이 아니라, 중심 개념 → 주요 하위 개념 → 세부 개념의 관계가 드러나야 합니다.

규칙:
- 학생이 실제로 고민하거나 답한 내용을 중심으로 판단하세요.
- 중심 개념 1개, 주요 하위 개념 4~7개, 필요 시 세부 개념 1~3개씩 포함하세요.
- nodes는 8~18개 정도로 구성하세요. 대화가 짧으면 가능한 범위에서만 작성하세요.
- weight는 대화에서의 중요도/빈도를 1~5 정수로 표시하세요.
- level은 중심 0, 주요 개념 1, 세부 개념 2로 표시하세요.
- edges는 개념 간 관계를 label로 짧게 표현하세요. 예: "구성요소", "비교", "원인", "예시", "조건"
- 반드시 JSON만 반환하세요.

반드시 아래 형식:

{{
  "tree": {{
    "운영체제": {{
      "프로세스": ["스레드", "문맥교환"],
      "메모리관리": ["가상메모리", "페이징"]
    }}
  }},
  "mindmap": {{
    "root": "운영체제",
    "nodes": [
      {{"id": "운영체제", "label": "운영체제", "level": 0, "weight": 5, "summary": "컴퓨터 자원을 관리하는 핵심 주제"}},
      {{"id": "프로세스", "label": "프로세스", "level": 1, "weight": 4, "summary": "실행 중인 프로그램 단위"}}
    ],
    "edges": [
      {{"from": "운영체제", "to": "프로세스", "label": "관리 대상"}}
    ]
  }}
}}
"""

    try:
        response = _client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=[prompt],
        )
        data = _safe_json_loads(_generation_text(response))
        if isinstance(data, dict) and "mindmap" in data:
            tree = data.get("tree")
            mindmap = data.get("mindmap")
            if isinstance(tree, dict) and isinstance(mindmap, dict):
                return {"tree": tree, "mindmap": mindmap}
        if isinstance(data, dict):
            return {
                "tree": data,
                "mindmap": build_detailed_tree(messages, concept_freq),
            }
    except Exception:
        pass

    return {
        "tree": build_tree(messages, concept_freq),
        "mindmap": build_detailed_tree(messages, concept_freq),
    }


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
        elapsed = _elapsed_from_previous(messages, i)

        if i == 0:
            state = "start"
            reason = "대화를 시작한 지점"
        elif (
            role == "user"
            and elapsed is not None
            and elapsed >= _LONG_RESPONSE_SECONDS
        ):
            state = "confusion"
            reason = f"{elapsed}초 동안 답변을 고민한 구간"
        elif any(word in text for word in ["모르겠", "헷갈", "어려워", "몰라"]):
            state = "confusion"
            reason = "학습자가 모름/혼란을 직접 표현"
        elif any(word in text for word in ["잠깐", "음", "어...", "생각", "고민"]):
            state = "hesitation"
            reason = "답변을 망설이거나 오래 고민한 표현"
        elif any(word in text for word in ["알겠", "이해", "오케이", "맞아", "네"]):
            state = "understanding"
            reason = "이해 또는 동의를 표현"
        else:
            state = "progress"
            reason = "개념을 탐색하며 학습 진행"

        meta = TIMELINE_TYPE_META[state]

        timeline.append({
            "role": role,
            "type": state,
            "status": meta["label"],
            "color": meta["color"],
            "severity": meta["severity"],
            "text": summarize_text(text),
            "reason": reason,
            "response_delay_seconds": elapsed,
        })

    return timeline

def build_timeline_with_gemini(messages):
    chat_text = "\n".join([
        (
            f'{m["role"]}'
            f'(prev_elapsed_seconds={_elapsed_from_previous(messages, idx)}): '
            f'{m["content"]}'
        )
        for idx, m in enumerate(messages)
    ])

    prompt = f"""
다음은 학생과 AI 튜터의 대화입니다.

{chat_text}

규칙:
- 대화 원문 그대로 복사하지 말 것
- 핵심 개념과 학습자의 이해 상태 중심으로 짧게 요약
- text는 15자~30자 이내
- reason은 왜 해당 이해 상태로 판단했는지 20자~45자 이내로 작성
- role 유지
- 입력 메시지 개수와 동일한 개수 반환
- JSON 외 다른 설명 금지
- type은 다음 중 하나:
start, progress, hesitation, confusion, understanding
- color는 type에 맞춰 반드시 아래 값 사용:
  start=gray, progress=blue, hesitation=orange, confusion=red, understanding=green
- status는 아래 값 중 하나 사용:
  학습 시작, 학습 진행, 답변 지연/고민, 취약/혼란, 이해 양호
- severity는 취약도 점수입니다. start/understanding=0, progress=1, hesitation=2, confusion=3
- 학습자가 "모르겠다", "헷갈린다", 틀린 추론을 반복하거나 튜터가 더 쉬운 힌트를 제공하는 구간은 confusion/red로 표시
- prev_elapsed_seconds가 {_LONG_RESPONSE_SECONDS} 이상인 user 메시지는 오래 고민한 답변이므로 confusion/red로 표시
- 학습자가 바로 답하지 못하고 "음", "잠깐", "생각", "고민"처럼 망설이지만 시간 지연이 길지 않은 구간은 hesitation/orange로 표시
- 학습자가 개념을 맞게 설명하거나 이해를 표현하고 튜터가 긍정적으로 이어간 구간은 understanding/green으로 표시

반드시 JSON만 반환:

[
  {{
    "role": "user",
    "type": "start",
    "status": "학습 시작",
    "color": "gray",
    "severity": 0,
    "text": "운영체제의 정의를 질문",
    "reason": "처음 다룬 핵심 주제"
  }},
  {{
    "role": "assistant",
    "type": "progress",
    "status": "학습 진행",
    "color": "blue",
    "severity": 1,
    "text": "자원 관리 역할 확인",
    "reason": "개념 이해를 확인하는 질문"
  }}
]
"""

    try:
        response = _client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=[prompt],
        )
        timeline = _safe_json_loads(_generation_text(response))
        if not isinstance(timeline, list):
            raise ValueError("timeline must be list")
        normalized = []
        for idx, item in enumerate(timeline[:len(messages)]):
            if not isinstance(item, dict):
                raise ValueError("timeline item must be object")
            t = item.get("type")
            if t not in TIMELINE_TYPE_META:
                t = "start" if idx == 0 else "progress"
            meta = TIMELINE_TYPE_META[t]
            elapsed = _elapsed_from_previous(messages, idx)
            normalized.append({
                "role": item.get("role") or messages[idx]["role"],
                "type": t,
                "status": meta["label"],
                "color": meta["color"],
                "severity": meta["severity"],
                "text": item.get("text") or summarize_text(messages[idx]["content"]),
                "reason": item.get("reason") or meta["label"],
                "response_delay_seconds": elapsed,
            })
        if len(normalized) != len(messages):
            raise ValueError("timeline length mismatch")
        return normalized
    except Exception:
        return build_timeline(messages)


def generate_learning_report(messages):
    words = extract_keywords(messages)

    concept_freq = Counter(words)

    tree_result = build_tree_with_gemini(messages, concept_freq)

    timeline = build_timeline_with_gemini(messages)

    return {
        "concept_frequency": dict(concept_freq.most_common(10)),
        "tree": tree_result["tree"],
        "mindmap": tree_result["mindmap"],
        "timeline": timeline
    }
