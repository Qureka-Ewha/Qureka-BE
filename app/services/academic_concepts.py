"""
전공 학술 명사/명사구 추출·필터링 — 질문 생성·마인드맵 공통 로직
"""
from __future__ import annotations

import re
from collections import Counter

# --- 메타/구조 단어 ---
STRUCTURAL_KEYWORDS_EN = (
    "introduction", "overview", "table of contents", "index", "agenda",
    "outlines", "outline", "summary", "conclusion", "q&a", "qa",
    "references", "bibliography", "title", "thank you", "thanks",
    "contents", "toc", "closing", "end",
    "chapter", "chapters", "section", "sections", "part", "parts",
    "unit", "units", "lesson", "lessons", "page", "pages", "slide", "slides",
    "module", "modules", "lecture", "lectures", "week", "appendix",
    "step", "steps", "goal", "goals", "objective", "objectives", "target",
    "targets", "purpose", "purposes", "mission", "missions", "context",
    "learning objectives", "learning outcome", "outcomes",
    "roadmap", "preview", "recap", "wrap up", "wrap-up", "approach",
)

STRUCTURAL_KEYWORDS_KO = (
    "소개", "개요", "목차", "요약", "결론", "참고문헌", "참고 문헌",
    "감사합니다", "감사", "질의응답", "머리말", "표지", "부록",
    "장", "절", "과", "편", "부", "단원", "차시", "회차",
    "목표", "목적", "방향", "학습목표", "학습 목표", "핵심목표",
    "단계", "서론", "마무리", "정리", "안내", "진행", "오늘의", "이번",
)

STRUCTURAL_KEYWORDS_ALL = STRUCTURAL_KEYWORDS_EN + STRUCTURAL_KEYWORDS_KO

STRUCTURAL_SINGLE_TOKEN_BLOCKLIST = frozenset(
    kw.lower() for kw in STRUCTURAL_KEYWORDS_ALL if " " not in kw
)

# 단독 동사·일반어 (기술 약어 GET/POST는 명사구 안에서만 허용)
ENGLISH_VERB_BLOCKLIST = frozenset({
    "get", "got", "post", "put", "set", "run", "use", "make", "feel", "see",
    "say", "go", "come", "take", "give", "find", "know", "think", "want",
    "look", "try", "ask", "work", "need", "become", "leave", "call", "keep",
    "let", "begin", "seem", "help", "show", "hear", "play", "move", "live",
    "believe", "hold", "bring", "happen", "write", "provide", "sit", "stand",
    "learn", "change", "lead", "understand", "watch", "follow", "stop", "create",
    "speak", "read", "spend", "grow", "open", "walk", "win", "teach", "offer",
    "remember", "consider", "appear", "buy", "serve", "die", "send", "build",
    "stay", "fall", "cut", "reach", "kill", "raise", "pass", "sell", "decide",
    "return", "explain", "develop", "carry", "break", "receive", "agree",
    "support", "hit", "produce", "eat", "cover", "catch", "draw", "choose",
})

KOREAN_ADJECTIVE_BLOCKLIST = frozenset({
    "실제", "최대", "가장", "새로운", "중요한", "기본", "전체", "다른", "같은",
    "주요", "핵심", "일반", "특정", "다양", "관련", "필요", "가능", "적절",
    "빠른", "느린", "높은", "낮은", "큰", "작은", "많은", "적은", "좋은", "나쁜",
})

KOREAN_PARTICLE_SUFFIXES = (
    "이라고", "라고", "에서", "으로", "에게", "께서", "처럼", "까지", "부터",
    "에는", "과는", "와는", "이란", "라는", "이라", "이란",
    "입니다", "습니다", "였습니다", "했습니다", "이고", "이며", "인데",
    "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만",
    "요", "음", "함", "임", "됨", "있", "없", "하는", "된", "되는", "이다",
)

GENERIC_STOPWORDS = frozenset({
    "그리고", "하지만", "그러면", "이건", "그건", "있다", "없다", "하는", "이다",
    "입니다", "잘", "너무", "좀", "그", "저", "것", "에서", "으로", "하는데",
    "설명", "차이", "통해", "이해", "알겠", "대해", "경우", "때문", "통한",
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "were",
    "have", "has", "had", "will", "can", "may", "not", "but", "you", "your",
    "link", "links", "실제", "링크가", "처리량은", "최대",
})

# 일상·시간·지시·질문 표현 — 학술 개념 아님
KOREAN_CONVERSATIONAL_BLOCKLIST = frozenset({
    "매일", "매번", "항상", "가끔", "자주", "보통", "역시", "정말", "진짜", "아마", "그냥",
    "아침", "점심", "저녁", "오늘", "내일", "어제", "방금", "지금", "나중", "먼저", "다음",
    "무엇", "뭐", "뭔", "뭔가", "뭐야", "뭐예", "뭐에", "뭐를", "어떤", "어떻", "어떻게",
    "왜", "언제", "어디", "누구", "얼마", "몇", "어느", "무슨",
    "그것", "이것", "저것", "그거", "이거", "저거", "여기", "거기", "저기",
    "그런", "이런", "저런", "그렇", "이렇", "저렇", "그래", "그리", "근데", "하지",
    "하면", "하고", "해서", "인데", "일까", "수도", "같이", "처럼", "정도", "하나",
    "둘", "셋", "번째", "우리", "저희", "여러분", "학생", "튜터", "사람",
    "맞습", "맞아", "틀렸", "모르", "헷갈", "어렵", "쉽", "좋", "나쁘", "그렇군", "아하",
    "공부", "복습", "예습", "수업", "교재", "시험", "문제", "답변", "질문", "대화",
})

# 학습 맥락 메타어 — 기술 개념 자체가 아님
KOREAN_META_DISCOURSE_BLOCKLIST = frozenset({
    "역할", "의미", "개념", "중요", "필요", "가능", "기본", "간단", "자세", "예시",
    "비교", "특징", "장점", "단점", "원리", "과정", "단계", "구조", "방법", "내용",
    "부분", "관련", "핵심", "주요", "전체", "일반", "정의", "요약", "정리", "분석",
    "학습", "강의", "강의안", "언급", "질문해볼", "설명해", "말해", "생각", "느낌",
})

KOREAN_SHORT_ACADEMIC_ALLOWLIST = frozenset({
    "연결", "해시", "캐시", "병렬", "직렬", "동기", "비동", "노드", "링크", "포트",
    "큐", "스택", "힙", "코어", "엣지",
})

_KOREAN_INFLECTED_SUFFIX = re.compile(
    r"(?:"
    r"이고|이며|이니|이다|입니다|합니다|됩니다|세요|어요|아요|게요|"
    r"했|였|겠|거나|든지|라면|도록|면서|지만|"
    r"스러|하게|"
    r"적이|적인|적으|"
    r"형입니다|형이|형적|"
    r"지향적"
    r")$"
)

# 단독으로 허용할 짧은 기술 약어
TECH_ACRONYM_ALLOWLIST = frozenset({
    "tcp", "udp", "ip", "dns", "http", "https", "osi", "api", "rpc", "nat",
    "lan", "wan", "vpn", "ssl", "tls", "ftp", "smtp", "icmp", "arp", "mac",
    "cpu", "gpu", "ipc", "ram", "rom", "ssd", "hdd",
})

_META_HEADING_PATTERNS = (
    re.compile(
        r"^(introduction|overview|chapter|section|part|unit|lesson|module|"
        r"page|slide|appendix|lecture|week|step|goal|objective|target|purpose|"
        r"mission|context|approach|roadmap)\b(\s*\d+|[:\.\-]|\s|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(table\s+of\s+contents|outline|summary|conclusion|references|"
        r"bibliography|agenda|index|title|thank\s*you|thanks|contents|toc|closing|end|"
        r"learning\s+objectives?|learning\s+outcomes?|preview|recap)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^q\s*&\s*a\b", re.IGNORECASE),
    re.compile(
        r"^(소개|개요|목차|요약|결론|참고\s*문헌|감사|질의\s*응답|표지|부록|머리말|단원|차시|"
        r"목표|목적|방향|학습\s*목표|단계|서론|안내)\s*$"
    ),
    re.compile(r"^제?\s*\d+\s*(장|절|과|편|부)\b"),
    re.compile(r"^(장|절|과|편|부)\s*\d+"),
    re.compile(r"^슬라이드\s*\d+", re.IGNORECASE),
)

_EN_PHRASE_PATTERN = re.compile(
    r"\b("
    r"(?:HTTP|HTTPS|TCP|UDP|IP|DNS|OSI|API)\s+(?:GET|POST|PUT|DELETE|PATCH)\s*(?:method|request|response)?|"
    r"(?:GET|POST|PUT|DELETE|PATCH)\s+(?:method|request|response)|"
    r"[A-Za-z]+(?:\s+[a-z]+){1,4}|"
    r"[A-Z][a-z]+(?:\s+[A-Za-z][a-z]+)+|"
    r"[A-Z][a-z]{3,}|"
    r"[A-Z]{2,}"
    r")\b",
    re.IGNORECASE,
)


def _normalize_heading(text: str) -> str:
    cleaned = re.sub(r"[^\w\s가-힣&]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def matches_structural_heading(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    normalized = _normalize_heading(raw)
    if not normalized:
        return False
    for pattern in _META_HEADING_PATTERNS:
        if pattern.search(raw.strip()) or pattern.search(normalized):
            return True
    for keyword in STRUCTURAL_KEYWORDS_ALL:
        kw = _normalize_heading(keyword)
        if not kw:
            continue
        if normalized == kw or normalized.startswith(f"{kw} ") or normalized.endswith(f" {kw}"):
            return True
    return False


def is_structural_keyword(term: str) -> bool:
    token = (term or "").strip()
    if not token:
        return False
    if matches_structural_heading(token):
        return True
    if token.lower() in STRUCTURAL_SINGLE_TOKEN_BLOCKLIST:
        return True
    if re.fullmatch(
        r"(chapter|section|part|unit|lesson|page|slide|module|step|goal|objective|target)\s*\d+",
        token,
        re.I,
    ):
        return True
    if re.fullmatch(r"제?\s*\d+\s*(장|절|과)", token):
        return True
    return False


def normalize_korean_noun(token: str) -> str:
    word = (token or "").strip()
    if not word or not re.search(r"[가-힣]", word):
        return word
    for suffix in sorted(KOREAN_PARTICLE_SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            word = word[: -len(suffix)]
            break
    return word.strip()


def _is_english_verb_only(token: str) -> bool:
    low = token.lower()
    if low in ENGLISH_VERB_BLOCKLIST:
        return True
    if len(low) <= 3 and low not in TECH_ACRONYM_ALLOWLIST:
        return True
    return False


def _is_korean_conversational(norm: str) -> bool:
    if not norm:
        return True
    if norm in KOREAN_CONVERSATIONAL_BLOCKLIST:
        return True
    if norm in KOREAN_META_DISCOURSE_BLOCKLIST:
        return True
    if norm in GENERIC_STOPWORDS:
        return True
    if _KOREAN_INFLECTED_SUFFIX.search(norm):
        return True
    if len(norm) <= 2 and norm.startswith(("뭐", "왜", "무", "몇")):
        return True
    return False


def is_conversational_term(term: str) -> bool:
    """일상어·메타어 등 학술 개념이 아닌 표현인지."""
    raw = (term or "").strip()
    if not raw:
        return True
    if is_valid_academic_phrase(raw):
        return False
    if re.search(r"[가-힣]", raw):
        norm = normalize_korean_noun(raw.split()[0])
        return _is_korean_conversational(norm)
    return False


def is_valid_academic_token(token: str) -> bool:
    """단일 토큰이 학술 명사 후보인지 (동사·형용사·조사·메타 단어 제외)."""
    raw = (token or "").strip()
    if not raw or len(raw) < 2:
        return False
    if raw in GENERIC_STOPWORDS or raw.lower() in GENERIC_STOPWORDS:
        return False
    if is_structural_keyword(raw):
        return False

    if re.fullmatch(r"[A-Za-z]+", raw):
        low = raw.lower()
        if _is_english_verb_only(low):
            return False
        if low in TECH_ACRONYM_ALLOWLIST:
            return True
        if len(raw) >= 4 and raw[0].isupper():
            return True
        return len(raw) >= 5

    if re.search(r"[가-힣]", raw):
        norm = normalize_korean_noun(raw)
        if not norm or len(norm) < 2:
            return False
        if norm in KOREAN_ADJECTIVE_BLOCKLIST or norm in GENERIC_STOPWORDS:
            return False
        if is_structural_keyword(norm):
            return False
        if _is_korean_conversational(norm):
            return False
        if len(norm) == 2 and norm not in KOREAN_SHORT_ACADEMIC_ALLOWLIST:
            return False
        if len(norm) == 3 and _is_korean_conversational(norm[:2]):
            return False
        return True

    return False


def is_valid_academic_phrase(phrase: str) -> bool:
    """명사구 전체가 유효한지."""
    p = re.sub(r"\s+", " ", (phrase or "").strip())
    if not p or len(p) < 2:
        return False
    if is_structural_keyword(p):
        return False

    parts = p.split()
    if len(parts) == 1:
        return is_valid_academic_token(parts[0])

    if any(_is_english_verb_only(w) for w in parts if re.fullmatch(r"[A-Za-z]+", w)):
        if not re.search(r"(method|request|response|protocol|layer|core|edge|network)", p, re.I):
            return False

    valid_parts = sum(1 for part in parts if is_valid_academic_token(part) or len(part) >= 4)
    return valid_parts >= max(1, len(parts) // 2)


def _score_phrase(phrase: str, freq: int = 1) -> float:
    p = phrase.strip()
    score = freq * 10.0
    score += min(len(p.split()), 4) * 8.0
    if re.search(r"\s", p):
        score += 12.0
    if p and p[0].isupper():
        score += 3.0
    if p.lower() in TECH_ACRONYM_ALLOWLIST:
        score += 5.0
    first = p.split()[0] if p.split() else ""
    if re.fullmatch(r"[A-Za-z]+", first) and _is_english_verb_only(first.lower()):
        score -= 50.0
    if re.search(r"[가-힣]", p):
        norm = normalize_korean_noun(p.split()[0])
        if _is_korean_conversational(norm):
            score -= 100.0
        elif len(norm) == 2 and norm not in KOREAN_SHORT_ACADEMIC_ALLOWLIST:
            score -= 40.0
        elif len(norm) >= 4:
            score += 6.0
    return score


def _extract_korean_tokens(line: str) -> list[str]:
    """공백·구두점 단위로 나눈 뒤 학술 명사 후보만 추출 (문장 전체 슬라이싱 금지)."""
    tokens: list[str] = []
    for segment in re.split(r"[\s,.;:!?\"'()\[\]{}·…—–\-/\\|<>]+", line):
        segment = segment.strip()
        if not segment or not re.search(r"[가-힣]", segment):
            continue
        if re.fullmatch(r"[가-힣]{2,}", segment):
            tokens.append(segment)
            continue
        for run in re.findall(r"[가-힣]{3,}", segment):
            tokens.append(run)
    return tokens


def extract_academic_noun_phrases(text: str) -> list[tuple[str, float]]:
    """문맥 기반 전공 명사/명사구 추출."""
    if not text:
        return []

    text_clean = re.sub(r"\[슬라이드\s+\d+\]", " ", text)
    counter: Counter[str] = Counter()

    for raw_line in text_clean.splitlines():
        line = re.sub(r"^[\-\*•·▪]\s*", "", raw_line.strip())
        if not line or len(line) < 2:
            continue

        for match in _EN_PHRASE_PATTERN.finditer(line):
            phrase = re.sub(r"\s+", " ", match.group(1).strip())
            if phrase.upper() in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                continue
            if is_valid_academic_phrase(phrase):
                display = _format_phrase_label(phrase)
                counter[display] += 1

        for token in _extract_korean_tokens(line):
            norm = normalize_korean_noun(token)
            if is_valid_academic_token(norm):
                counter[norm] += 1

    ranked = [
        (phrase, _score_phrase(phrase, count))
        for phrase, count in counter.items()
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [(phrase, score) for phrase, score in ranked if score > 0]


def _format_phrase_label(phrase: str) -> str:
    p = re.sub(r"\s+", " ", phrase.strip())
    if re.fullmatch(r"[A-Z]{2,}", p):
        return p.upper()
    if re.search(r"\s", p):
        return " ".join(w.capitalize() if w.islower() else w for w in p.split())
    if len(p) >= 2 and p.islower():
        return p.capitalize()
    return p


def pick_best_academic_concept(text: str, fallback: str = "핵심 개념") -> str:
    ranked = extract_academic_noun_phrases(text)
    if ranked:
        return ranked[0][0]
    fb = (fallback or "").strip() or "핵심 개념"
    return fb if not is_structural_keyword(fb) else "핵심 개념"


def slide_heading(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def is_structural_slide_text(text: str) -> bool:
    heading = slide_heading(text)
    if not heading:
        return False
    if not matches_structural_heading(heading):
        return False
    word_count = len(re.findall(r"[가-힣A-Za-z0-9]+", text or ""))
    return word_count <= 35


def is_overview_roadmap_slide(text: str) -> bool:
    """목차·로드맵·Overview — 질문 생성 Skip, 마인드맵 대분류만."""
    if is_structural_slide_text(text):
        return True
    heading = slide_heading(text).lower()
    overview_markers = (
        "overview", "roadmap", "agenda", "outline", "approach", "목차", "개요",
        "feel", "learning goal", "objectives",
    )
    if any(m in heading for m in overview_markers):
        return True
    lower = (text or "").lower()
    if 'get "feel"' in lower or "get feel" in lower or "get a feel" in lower:
        return True
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) >= 4:
        short = sum(1 for ln in lines if len(ln.split()) <= 6)
        if short / len(lines) >= 0.55 and len(text) < 900:
            has_definition = bool(re.search(r"(은|는|이다|합니다|means|refers to|defined)", text, re.I))
            if not has_definition:
                return True
    return False


def format_first_socratic_question(topic: str, source_phrase: str) -> str:
    return (
        f"{source_phrase} 강의안에서 언급된 '{topic}'에 대해 질문해볼게요. "
        f"'{topic}'의 핵심 역할이 무엇인지, 그리고 이 개념이 왜 필요한지 직관적으로 설명해 줄 수 있나요?"
    )


def academic_extraction_prompt_block(is_transcript: bool = False) -> str:
    unit = "구간" if is_transcript else "슬라이드"
    return f"""
[절대 최우선 — Academic Noun Phrases Only]
- 추출·질문 주제는 반드시 전공 서적 색인/시험에 [단독 명사]로 나올 수 있는 완전한 학술 명사·명사구만 허용합니다.
  · 허용: Internet, Protocol, Network core, Network edge, HTTP GET method, 프로토콜, 라우팅, 대역폭
- 단독 동사(Get, Post, Set, Run, Feel 등), 형용사(실제, 최대, 가장), 조사가 붙은 형태(링크가, 처리량은)는 키워드 0개.
- 조사/어미가 붙은 한국어는 기본형으로 정제(처리량은 → 처리량 / Throughput).

[메타·레이아웃 단어 — 완전 무시]
- Chapter, Introduction, Goal, Objective, Overview, Roadmap, Approach, Title, Agenda, 목차, 목표, 목적 등은 분석 대상에서 제외.

[문맥 기반 추출 — 단어 단위 분할 금지]
- 공백·줄바꿈으로 무작정 쪼개지 마세요. 의미 단위 명사구로 추출하세요.
- 'Get' 단독 추출 금지. 문맥상 기술 용어면 'HTTP GET method'처럼 완전한 명사구만 허용.

[{unit} Skip]
- Overview/Roadmap/목차 {unit}: 나열된 Internet, Protocol 등은 '앞으로 배울 요약'일 뿐. 억지 질문 금지.
- 질문은 실제 상세 설명이 시작되는 다음 {unit}부터. 목차 {unit}은 마인드맵 대분류만 등록 가능.

[첫 질문 템플릿]
- "느낌/이미지가 떠오르나요?" 같은 표현 금지.
- 올바른 예: "강의안에서 언급된 '프로토콜'의 핵심 역할이 무엇인지, 그리고 이 개념이 왜 필요한지 직관적으로 설명해 줄 수 있나요?"
"""


def mindmap_keyword_prompt_block() -> str:
    return """
[마인드맵 키워드 — Absolute Top Priority]
- nodes/labels/tree 키워드는 Academic Noun Phrases Only (전공 서적 색인 수준 명사·명사구).
- 단독 동사(Get, Run, Set), 형용사(실제, 최대), 조사 포함 형태(링크가) 절대 금지.
- 문맥 없이 토큰 1개씩 쪼개지 말 것. Network core, Network edge, Protocol, Internet처럼 의미 덩어리로 추출.
- Goal, Chapter, Overview, Roadmap, Approach, Title 등 메타 단어는 노드에 넣지 말 것.
- 매일, 아침, 뭐, 어떤, 하나, 역할, 공부, 질문, 그렇군요 같은 일상·대화 표현은 절대 넣지 말 것.
- 학생 답변에서 나온 단어보다, 대화·강의에서 실제로 학습한 기술 개념을 우선하세요.
- 한국어는 조사 제거 후 명사 기본형만 (처리량은 → 처리량).
"""
