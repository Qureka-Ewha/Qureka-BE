from collections import Counter
import re


# 너무 흔한 단어 제거
STOPWORDS = {
    "그리고", "하지만", "그러면", "이건", "그건", "있다", "없다",
    "하는", "이다", "입니다", "잘", "너무", "좀", "그", "저", "것",
    "에서", "으로", "하는데", "모르겠어", "설명", "차이", "통해"
}


def extract_keywords(messages):
    words = []

    for msg in messages:
        tokens = re.findall(r"[가-힣A-Za-z]+", msg)

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


def build_timeline(messages):
    timeline = []

    for i, msg in enumerate(messages):
        if i == 0:
            state = "start"
        elif any(word in msg for word in ["모르겠", "헷갈", "이해 안", "어려워"]):
            state = "confusion"
        elif any(word in msg for word in ["알겠", "이해", "오케이"]):
            state = "understanding"
        else:
            state = "progress"

        timeline.append({
            "step": msg,
            "type": state
        })

    return timeline


def generate_learning_report(messages):
    words = extract_keywords(messages)

    concept_freq = Counter(words)

    tree = build_tree(messages, concept_freq)

    timeline = build_timeline(messages)

    return {
        "concept_frequency": dict(concept_freq),
        "tree": tree,
        "timeline": timeline
    }