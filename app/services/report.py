import os
import json
from google import genai
from dotenv import load_dotenv
from collections import Counter
from datetime import datetime
import re

from app.services import academic_concepts as ac

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_GEMINI_MODEL = "models/gemini-2.5-flash"
_LONG_RESPONSE_SECONDS = int(os.getenv("REPORT_LONG_RESPONSE_SECONDS", "90"))
_REPORT_USE_GEMINI = os.getenv("REPORT_USE_GEMINI", "").lower() in ("1", "true", "yes")
_TIMELINE_USE_GEMINI = os.getenv("TIMELINE_USE_GEMINI", "true").lower() in ("1", "true", "yes")


def _generation_text(resp) -> str:
    try:
        t = getattr(resp, "text", None)
        return (t or "").strip()
    except Exception:
        return ""

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


def _is_gemini_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota exceeded" in msg


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


def extract_concept_freq(messages) -> Counter:
    """튜터(assistant) 발화를 우선해 학술 개념 빈도 집계."""
    freq = Counter()
    for msg in messages:
        content = msg.get("content") or ""
        role = msg.get("role") or "user"
        role_weight = 3 if role == "assistant" else 1
        for phrase, score in ac.extract_academic_noun_phrases(content):
            if score <= 0:
                continue
            freq[phrase] += role_weight
    return freq


def extract_keywords(messages):
    """하위 호환 — extract_concept_freq 결과를 flat list로 반환."""
    freq = extract_concept_freq(messages)
    words = []
    for phrase, count in freq.items():
        words.extend([phrase] * count)
    return words


def _valid_mindmap_label(label: str) -> bool:
    text = re.sub(r"\s+", " ", (label or "").strip())
    if not text:
        return False
    return ac.is_valid_academic_phrase(text)


def _sanitize_nested_tree(tree: dict) -> dict:
    if not isinstance(tree, dict):
        return {}

    def sanitize_branch(key, value):
        label = str(key or "").strip()
        if not _valid_mindmap_label(label):
            return None
        if isinstance(value, list):
            children = [str(v).strip() for v in value if _valid_mindmap_label(str(v))]
            return children
        if isinstance(value, dict):
            nested = {}
            for child_key, child_val in value.items():
                sanitized = sanitize_branch(child_key, child_val)
                if sanitized is None:
                    continue
                if isinstance(sanitized, list) and not sanitized:
                    continue
                if isinstance(sanitized, dict) and not sanitized:
                    continue
                nested[str(child_key).strip()] = sanitized
            return nested
        return value

    sanitized = {}
    for key, value in tree.items():
        branch = sanitize_branch(key, value)
        if branch is None:
            continue
        if isinstance(branch, (dict, list)) and not branch:
            continue
        sanitized[str(key).strip()] = branch
    return sanitized


def _sanitize_mindmap(mindmap: dict, concept_freq: Counter | None = None) -> dict:
    if not isinstance(mindmap, dict):
        return {"root": "", "nodes": [], "edges": []}

    nodes = []
    kept_ids: set[str] = set()

    for raw in mindmap.get("nodes") or []:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or raw.get("id") or "").strip()
        if not _valid_mindmap_label(label):
            continue
        if ac.is_conversational_term(label):
            continue
        node_id = str(raw.get("id") or label)
        kept_ids.add(node_id)
        nodes.append({
            "id": node_id,
            "label": label,
            "level": raw.get("level", 1),
            "weight": raw.get("weight", 1),
            "summary": raw.get("summary") or "",
        })

    edges = []
    for raw in mindmap.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        src = str(raw.get("from") or "").strip()
        dst = str(raw.get("to") or "").strip()
        if src in kept_ids and dst in kept_ids:
            edges.append({
                "from": src,
                "to": dst,
                "label": raw.get("label") or "관련",
            })

    root = str(mindmap.get("root") or "").strip()
    if root not in kept_ids:
        root_node = next((n for n in nodes if n.get("level") == 0), None)
        root = root_node["id"] if root_node else (nodes[0]["id"] if nodes else "")

    return {"root": root, "nodes": nodes, "edges": edges}


def _filter_concept_freq(concept_freq: Counter) -> Counter:
    filtered = Counter()
    for term, count in concept_freq.items():
        if ac.is_valid_academic_phrase(term):
            filtered[term] += count
    return filtered


def build_tree(messages, concept_freq):
    concept_freq = _filter_concept_freq(concept_freq)
    if not concept_freq:
        return {}

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
    concept_freq = _filter_concept_freq(concept_freq)
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


def _finalize_tree_result(tree: dict, mindmap: dict, concept_freq: Counter) -> dict:
    sanitized_tree = _sanitize_nested_tree(tree)
    sanitized_mindmap = _sanitize_mindmap(mindmap, concept_freq)

    if not sanitized_tree:
        sanitized_tree = _sanitize_nested_tree(build_tree([], concept_freq))
    if not sanitized_mindmap.get("nodes"):
        sanitized_mindmap = _sanitize_mindmap(build_detailed_tree([], concept_freq), concept_freq)

    return {
        "tree": sanitized_tree,
        "mindmap": sanitized_mindmap,
    }

def build_tree_with_gemini(messages, concept_freq):
    concept_freq = _filter_concept_freq(concept_freq)
    if not concept_freq:
        return {"tree": {}, "mindmap": {"root": "", "nodes": [], "edges": []}}

    chat_text = "\n".join([
        f'{m["role"]}: {m["content"]}'
        for m in messages
    ])

    suggested = ", ".join(word for word, _ in concept_freq.most_common(15))

    prompt = f"""
다음은 학생과 AI 튜터의 대화입니다.

{chat_text}

{ac.mindmap_keyword_prompt_block()}

[추출 후보 참고 — 아래는 시스템이 선별한 학술 명사구입니다. 이 외 메타 단어·동사는 넣지 마세요]
{suggested}

목표:
- 프론트에서 마인드맵으로 시각화할 수 있도록 핵심 개념을 충분히 구조화하세요.
- 단순 키워드 나열이 아니라, 중심 개념 → 주요 하위 개념 → 세부 개념의 관계가 드러나야 합니다.

규칙:
- 학생이 실제로 고민하거나 답한 내용과 강의에서 학습한 기술 개념을 중심으로 판단하세요.
- 일상어·시간 표현·질문 표현(매일, 뭐, 어떤, 역할, 하나, 공부, 질문 등)은 nodes/tree에 절대 넣지 마세요.
- nodes는 [추출 후보 참고] 목록 안의 학술 명사·명사구 위주로만 구성하세요. 목록에 없는 단어도 학술 개념이면 허용합니다.
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
                return _finalize_tree_result(tree, mindmap, concept_freq)
        if isinstance(data, dict):
            return _finalize_tree_result(
                data if isinstance(data, dict) else {},
                build_detailed_tree(messages, concept_freq),
                concept_freq,
            )
    except Exception:
        pass

    return _finalize_tree_result(
        build_tree(messages, concept_freq),
        build_detailed_tree(messages, concept_freq),
        concept_freq,
    )


def summarize_text(text):
    return _summarize_timeline_message("user", text)


def _summarize_timeline_message(role: str, content: str) -> str:
    compact = re.sub(r"\s+", " ", (content or "").strip())
    if not compact:
        return "내용 없음"

    if role == "assistant":
        for sep in ("?", "？"):
            if sep in compact:
                question = compact.split(sep)[0].strip() + sep
                return question[:35] + ("..." if len(question) > 35 else "")
        return compact[:35] + ("..." if len(compact) > 35 else "")

    return compact[:35] + ("..." if len(compact) > 35 else "")


def _previous_assistant_text(messages, idx: int) -> str:
    for j in range(idx - 1, -1, -1):
        if messages[j].get("role") == "assistant":
            return messages[j].get("content") or ""
    return ""


def _next_assistant_text(messages, idx: int) -> str:
    for next_msg in messages[idx + 1:]:
        if next_msg.get("role") == "assistant":
            return next_msg.get("content") or ""
    return ""


def _ai_feedback_is_positive(next_ai: str) -> bool:
    if not next_ai.strip():
        return False
    head = next_ai[:160]
    positive_patterns = [
        r"^맞(아|아요|습니다|네요)",
        r"^정확(합니다|해요|하게|한)",
        r"^좋(습니다|아요|은\s)",
        r"^훌륭",
        r"^그렇(습니다|죠|네요)",
        r"^잘\s*(이해|설명|했)",
        r"^네[,.\s]",
    ]
    return any(re.search(pattern, head) for pattern in positive_patterns)


def _ai_feedback_is_negative(next_ai: str) -> bool:
    if not next_ai.strip():
        return False

    normalized = re.sub(r"\s+", "", next_ai)
    strong_negative = [
        "틀렸", "틀린", "맞지않", "정확하지않", "오해하고", "오해하신",
        "그건아니", "아니에요", "아닙니다", "잘못", "다시생각해보",
        "다시생각해", "모르겠다고", "헷갈리", "어렵다고",
    ]
    if any(token in normalized for token in strong_negative):
        return True

    # 심화 질문용 접속어(정확히는/하지만)만으로는 오답 처리하지 않음
    if _ai_feedback_is_positive(next_ai):
        return False

    soft_negative = ["다시 생각", "혼동", "보완", "놓쳤", "부족", "정답은"]
    return any(word in next_ai for word in soft_negative)


def _classify_user_timeline_state(
    messages,
    idx: int,
    suggested_type: str | None = None,
    suggested_reason: str | None = None,
):
    text = messages[idx].get("content") or ""
    next_ai = _next_assistant_text(messages, idx)

    explicit_confusion = ["모르겠", "헷갈", "어려워", "몰라", "잘 모르", "이해 안", "모르겠어"]
    understanding_words = ["알겠", "이해했", "이해됐", "오케이", "맞아", "그렇군", "아하", "이해합니다"]

    if any(word in text for word in explicit_confusion):
        return "confusion", "학습자가 모름/혼란을 직접 표현"
    if _ai_feedback_is_positive(next_ai):
        return "understanding", "튜터가 정답으로 인정한 구간"
    if _ai_feedback_is_negative(next_ai):
        return "confusion", "튜터 피드백상 답변 보완이 필요한 구간"
    if any(word in text for word in understanding_words):
        return "understanding", "학습자가 이해를 표현한 구간"
    if suggested_type in ("understanding", "confusion"):
        return suggested_type, suggested_reason or TIMELINE_TYPE_META[suggested_type]["label"]
    return "understanding", "튜터 피드백이 중립적이라 이해로 분류"


def _classify_user_messages_with_gemini(messages):
    """학습자 답변만 AI 튜터 피드백 기준으로 understanding/confusion 분류."""
    user_turns = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        user_turns.append({
            "index": idx,
            "question": _previous_assistant_text(messages, idx)[:600],
            "answer": (msg.get("content") or "")[:900],
            "tutor_feedback": _next_assistant_text(messages, idx)[:900],
        })

    if not user_turns:
        return {}

    prompt = f"""
다음은 학생(user)과 AI 튜터(assistant)의 대화입니다.

{json.dumps(user_turns, ensure_ascii=False)}

각 user 답변마다 아래 기준으로 판단하세요.
- understanding: 튜터의 직후 피드백이 정답/이해를 인정함 (예: 맞습니다, 정확합니다, 잘 이해, 좋습니다)
- confusion: 학생이 모르겠다고 했거나, 튜터가 오답/오해/보완 필요를 명확히 지적함 (예: 틀렸, 맞지 않, 다시 생각, 오해)

주의:
- 튜터가 "맞습니다. 그런데/정확히는"처럼 칭찬 후 심화 질문을 하는 것은 understanding입니다.
- "하지만", "다만", "정확히는"만 있다고 confusion으로 분류하지 마세요.
- text는 학생 답변을 15~35자로 요약 (원문 복사 금지)
- reason은 20~45자

반드시 JSON 배열만 반환:
[
  {{"index": 3, "type": "understanding", "text": "대역폭 정의를 설명함", "reason": "튜터가 정답으로 인정"}}
]
"""

    response = _client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=[prompt],
    )
    parsed = _safe_json_loads(_generation_text(response))
    if not isinstance(parsed, list):
        raise ValueError("user classification must be list")

    result = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        t = item.get("type")
        if idx is None or t not in ("understanding", "confusion"):
            continue
        result[int(idx)] = {
            "type": t,
            "text": item.get("text") or "",
            "reason": item.get("reason") or TIMELINE_TYPE_META[t]["label"],
        }
    return result


def build_timeline(messages, *, use_gemini: bool | None = None):
    if use_gemini is None:
        use_gemini = _TIMELINE_USE_GEMINI and bool(os.getenv("GEMINI_API_KEY"))

    user_ai_labels = {}
    if use_gemini:
        try:
            user_ai_labels = _classify_user_messages_with_gemini(messages)
        except Exception as exc:
            if _is_gemini_rate_limit_error(exc):
                use_gemini = False
            user_ai_labels = {}

    timeline = []

    for i, msg in enumerate(messages):
        text = msg["content"]
        role = msg["role"]
        elapsed = _elapsed_from_previous(messages, i)

        if i == 0:
            state = "start"
            reason = "대화를 시작한 지점"
            summary_text = _summarize_timeline_message(role, text)
        elif role == "user":
            ai_label = user_ai_labels.get(i)
            if ai_label:
                state = ai_label["type"]
                reason = ai_label["reason"]
                summary_text = ai_label["text"] or _summarize_timeline_message("user", text)
            else:
                state, reason = _classify_user_timeline_state(messages, i)
                summary_text = _summarize_timeline_message("user", text)
        else:
            state = "progress"
            reason = "튜터가 질문 또는 피드백으로 학습을 진행"
            summary_text = _summarize_timeline_message("assistant", text)

        meta = TIMELINE_TYPE_META[state]

        timeline.append({
            "role": role,
            "type": state,
            "status": meta["label"],
            "color": meta["color"],
            "severity": meta["severity"],
            "text": summary_text,
            "reason": reason,
            "response_delay_seconds": elapsed,
        })

    return timeline

def build_timeline_with_gemini(messages):
    user_ai_labels = {}
    try:
        user_ai_labels = _classify_user_messages_with_gemini(messages)
    except Exception as exc:
        if _is_gemini_rate_limit_error(exc):
            return build_timeline(messages, use_gemini=False)
        user_ai_labels = {}

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
                ai_label = user_ai_labels.get(idx)
                if ai_label:
                    final_type = ai_label["type"]
                    final_reason = ai_label["reason"]
                    summary_text = ai_label["text"] or item.get("text") or _summarize_timeline_message("user", messages[idx]["content"])
                elif t in ("understanding", "confusion"):
                    final_type = t
                    final_reason = item.get("reason") or TIMELINE_TYPE_META[t]["label"]
                    summary_text = item.get("text") or _summarize_timeline_message("user", messages[idx]["content"])
                else:
                    final_type, final_reason = _classify_user_timeline_state(
                        messages,
                        idx,
                        suggested_type=t,
                        suggested_reason=item.get("reason"),
                    )
                    summary_text = item.get("text") or _summarize_timeline_message("user", messages[idx]["content"])
            else:
                final_type = "start" if idx == 0 else "progress"
                final_reason = item.get("reason") or "튜터가 질문 또는 피드백으로 학습을 진행"
                summary_text = item.get("text") or _summarize_timeline_message("assistant", messages[idx]["content"])
            meta = TIMELINE_TYPE_META[final_type]
            elapsed = _elapsed_from_previous(messages, idx)
            normalized.append({
                "role": role,
                "type": final_type,
                "status": meta["label"],
                "color": meta["color"],
                "severity": meta["severity"],
                "text": summary_text,
                "reason": final_reason,
                "response_delay_seconds": elapsed,
            })
        if len(normalized) != len(messages):
            raise ValueError("timeline length mismatch")
        return normalized
    except Exception:
        return build_timeline(messages, use_gemini=False)


def generate_learning_report(messages):
    concept_freq = _filter_concept_freq(extract_concept_freq(messages))

    if _REPORT_USE_GEMINI:
        tree_result = build_tree_with_gemini(messages, concept_freq)
    else:
        tree_result = _finalize_tree_result(
            build_tree(messages, concept_freq),
            build_detailed_tree(messages, concept_freq),
            concept_freq,
        )

    if _TIMELINE_USE_GEMINI and os.getenv("GEMINI_API_KEY"):
        try:
            timeline = build_timeline_with_gemini(messages)
        except Exception:
            timeline = build_timeline(messages, use_gemini=False)
    else:
        timeline = build_timeline(messages, use_gemini=False)

    return {
        "concept_frequency": dict(concept_freq.most_common(10)),
        "tree": tree_result["tree"],
        "mindmap": tree_result["mindmap"],
        "timeline": timeline
    }
