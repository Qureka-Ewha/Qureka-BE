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
_REPORT_USE_GEMINI = os.getenv("REPORT_USE_GEMINI", "").lower() in ("1", "true", "yes")


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
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if not compact:
        return "내용 없음"
    suffix = "..." if len(compact) > 34 else ""
    return compact[:34] + suffix


def _next_assistant_text(messages, idx: int) -> str:
    for next_msg in messages[idx + 1:]:
        if next_msg.get("role") == "assistant":
            return next_msg.get("content") or ""
    return ""


def _classify_user_timeline_state(
    messages,
    idx: int,
    suggested_type: str | None = None,
    suggested_reason: str | None = None,
):
    text = messages[idx].get("content") or ""
    next_ai = _next_assistant_text(messages, idx)
    normalized_next_ai = re.sub(r"\s+", "", next_ai)

    explicit_confusion = ["모르겠", "헷갈", "어려워", "몰라", "잘 모르", "이해 안"]
    understanding_words = ["알겠", "이해했", "이해됐", "오케이", "맞아", "그렇군", "아하"]
    negative_feedback = [
        "아니요", "틀렸", "틀린", "맞지 않", "정확하지", "조금 달라",
        "다시 생각", "혼동", "오해", "그건 아니", "아쉬", "보완",
        "정확히는", "다만", "하지만", "좋은 시도", "가까워", "아니라",
        "구분해야", "놓쳤", "부족", "정답은", "왜 그렇게 판단",
        "어떤 점에서", "어떻게 다를", "모순", "충돌",
    ]
    positive_feedback_patterns = [
        r"맞(아|아요|습니다|네요)[.!?]?",
        r"정확(해|합니다|하게).*?(설명|이해|답)",
        r"잘\s*(이해|설명|했)",
        r"좋아요[.!?]?",
        r"훌륭(해|합니다)",
        r"그렇(죠|습니다)",
        r"정답(입니다|이에요)",
    ]
    has_negative_feedback = any(word in next_ai for word in negative_feedback)
    has_negative_feedback = has_negative_feedback or any(
        word in normalized_next_ai
        for word in ["맞지않", "정확하지않", "그건아니", "조금달라"]
    )
    has_positive_feedback = any(
        re.search(pattern, next_ai)
        for pattern in positive_feedback_patterns
    )

    if any(word in text for word in explicit_confusion):
        return "confusion", "학습자가 모름/혼란을 직접 표현"
    if has_negative_feedback:
        return "confusion", "튜터 피드백상 답변 보완이 필요한 구간"
    if has_positive_feedback or any(word in text for word in understanding_words):
        return "understanding", "학습자가 개념을 정확히 이해한 구간"
    if suggested_type in ("understanding", "confusion"):
        return suggested_type, suggested_reason or TIMELINE_TYPE_META[suggested_type]["label"]
    return "confusion", "정답 여부가 명확히 확인되지 않아 보완이 필요한 구간"


def build_timeline(messages):
    timeline = []

    for i, msg in enumerate(messages):
        text = msg["content"]
        role = msg["role"]
        elapsed = _elapsed_from_previous(messages, i)

        if i == 0:
            state = "start"
            reason = "대화를 시작한 지점"
        elif role == "user":
            state, reason = _classify_user_timeline_state(messages, i)
        else:
            state = "progress"
            reason = "튜터가 질문 또는 피드백으로 학습을 진행"

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
- AI 질문은 "무슨 질문이었는지", 학습자 답변은 "어떤 답변을 했는지" 중심으로 짧게 요약
- text는 15자~35자 이내
- reason은 왜 해당 이해 상태로 판단했는지 20자~45자 이내로 작성
- role 유지
- 입력 메시지 개수와 동일한 개수 반환
- JSON 외 다른 설명 금지
- type은 다음 중 하나:
start, progress, confusion, understanding
- color는 type에 맞춰 반드시 아래 값 사용:
  start=gray, progress=blue, confusion=red, understanding=green
- status는 아래 값 중 하나 사용:
  학습 시작, 학습 진행, 취약/혼란, 이해 양호
- severity는 취약도 점수입니다. start/understanding=0, progress=1, confusion=3
- user 메시지만 understanding/confusion으로 분류하세요. assistant 메시지는 start 또는 progress로 분류하세요.
- 학습자가 "모르겠다", "헷갈린다"처럼 모름을 직접 표현하거나, 다음 튜터 응답이 틀렸다고 교정하는 경우에만 confusion/red로 표시
- 학습자가 개념을 맞게 설명하거나 다음 튜터 응답이 긍정적으로 인정하면 understanding/green으로 표시
- 학습자 메시지에는 절대 start/progress/hesitation을 쓰지 마세요. 정답이면 understanding, 모르겠거나 오답이면 confusion입니다.
- prev_elapsed_seconds는 정오 판단 근거가 아닙니다. 답변 시간보다 다음 튜터의 피드백과 개념적 정오를 우선하세요.

반드시 JSON만 반환:

[
  {{
    "role": "assistant",
    "type": "start",
    "status": "학습 시작",
    "color": "gray",
    "severity": 0,
    "text": "운영체제 정의 질문",
    "reason": "튜터가 첫 질문으로 학습을 시작"
  }},
  {{
    "role": "user",
    "type": "understanding",
    "status": "이해 양호",
    "color": "green",
    "severity": 0,
    "text": "자원 관리 역할로 답변",
    "reason": "튜터가 다음 응답에서 정답으로 인정"
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
            role = messages[idx]["role"]
            t = item.get("type")
            if t not in TIMELINE_TYPE_META:
                t = "start" if idx == 0 else "progress"
            if role == "user":
                t, fallback_reason = _classify_user_timeline_state(
                    messages,
                    idx,
                    suggested_type=t,
                    suggested_reason=item.get("reason"),
                )
            else:
                t = "start" if idx == 0 else "progress"
                fallback_reason = "튜터가 질문 또는 피드백으로 학습을 진행"
            meta = TIMELINE_TYPE_META[t]
            elapsed = _elapsed_from_previous(messages, idx)
            normalized.append({
                "role": role,
                "type": t,
                "status": meta["label"],
                "color": meta["color"],
                "severity": meta["severity"],
                "text": item.get("text") or summarize_text(messages[idx]["content"]),
                "reason": fallback_reason if role == "user" else item.get("reason") or fallback_reason,
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

    if _REPORT_USE_GEMINI:
        tree_result = build_tree_with_gemini(messages, concept_freq)
        timeline = build_timeline_with_gemini(messages)
    else:
        tree_result = {
            "tree": build_tree(messages, concept_freq),
            "mindmap": build_detailed_tree(messages, concept_freq),
        }
        timeline = build_timeline(messages)

    return {
        "concept_frequency": dict(concept_freq.most_common(10)),
        "tree": tree_result["tree"],
        "mindmap": tree_result["mindmap"],
        "timeline": timeline
    }
