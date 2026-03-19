from collections import Counter
import re


# ---------------------------------------
# 조사 제거
# ---------------------------------------
def clean_word(word: str):
    suffixes = ["은", "는", "이", "가", "을", "를", "에", "의", "와", "과"]

    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > 1:
            return word[:-1]

    return word


# ---------------------------------------
# 개념 단어 추출
# ---------------------------------------
def extract_concepts(messages):
    concepts = []

    stopwords = {
        "왜", "어떻게", "무엇", "이게", "그럼", "근데", "그리고",
        "제가", "저는", "그건", "이건"
    }

    for msg in messages:

        words = re.findall(r"[가-힣A-Za-z]+", msg)

        for word in words:

            word = clean_word(word)

            if len(word) >= 2 and word not in stopwords:
                concepts.append(word)

    return Counter(concepts)


# ---------------------------------------
# 사고 흐름 타임라인 생성
# ---------------------------------------
def build_timeline(messages):

    timeline = []

    for i, msg in enumerate(messages):

        if i == 0:
            timeline.append({
                "step": msg,
                "type": "start"
            })

        elif "모르" in msg or "헷갈" in msg:
            timeline.append({
                "step": msg,
                "type": "confusion"
            })

        elif "아" in msg or "이해" in msg:
            timeline.append({
                "step": msg,
                "type": "understood"
            })

        else:
            timeline.append({
                "step": msg,
                "type": "progress"
            })

    return timeline


# ---------------------------------------
# 간단한 트리 구조 생성
# ---------------------------------------
def build_tree(messages):

    if not messages:
        return {}

    root = clean_word(messages[0].split()[0])

    tree = {root: []}

    for msg in messages[1:]:

        concept = clean_word(msg.split()[0])

        tree[root].append(concept)

    return tree


# ---------------------------------------
# 최종 리포트 생성
# ---------------------------------------
def generate_learning_report(messages):

    concept_freq = extract_concepts(messages)

    tree = build_tree(messages)

    timeline = build_timeline(messages)

    return {
        "concept_frequency": concept_freq,
        "tree": tree,
        "timeline": timeline
    }