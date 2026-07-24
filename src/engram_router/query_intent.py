"""Query-intent helpers for EngramRouter recall/gap checks.

Pure functions extracted from ``store.py`` as the first low-risk step of the
store split.  Keep these functions side-effect free: they are used by
``MemoryStore`` as compatibility delegates and can be unit-tested without a DB.
"""

from __future__ import annotations

import re

REASON_MARKERS = ("因为", "原因", "所以", "导致", "出于", "由于")


def asks_brand(query: str) -> bool:
    return any(k in query for k in ("牌子", "品牌", "什么型号", "型号", "哪个牌", "什么款"))


def asks_identity(query: str) -> bool:
    """Identity questions ask for constant base attributes of a subject."""
    return any(k in query for k in (
        "是谁", "叫什么", "叫啥", "名字", "多大", "几岁", "多少岁", "年龄",
        "什么星座", "属相", "什么血型", "哪里人", "哪儿人", "是男是女", "性别",
    ))


def asks_eval(query: str) -> bool:
    """Evaluation/quality questions ask for judgement/sensory content."""
    return any(k in query for k in (
        "怎么样", "好不好", "好吃吗", "好吃不", "如何", "厉害吗", "手艺",
        "性格", "脾气", "好看吗", "漂亮吗",
    ))


def asks_reason(query: str) -> bool:
    return any(marker in query for marker in ("为什么", "原因", "为啥", "为何"))


def asks_person(query: str) -> bool:
    return any(marker in query for marker in ("谁", "哪位", "哪个人"))


def has_reason(text: str) -> bool:
    return any(marker in text for marker in REASON_MARKERS)


def has_person_like(text: str) -> bool:
    """Check for CJK bigram/trigram that looks like a person.

    Excludes known time/topic/object words that would produce false positives.
    """
    non_person_cjk = {
        "礼物", "键盘", "鼠标", "手机", "电脑", "前天", "昨天",
        "今天", "明天", "后天", "上午", "下午", "晚上", "中午",
        "什么", "怎么", "这个", "那个", "哪个", "因为", "所以",
        "好吃", "好看", "厉害", "红烧", "觉得", "喜欢", "可以",
        "没有", "不是", "还是", "但是", "虽然", "如果",
    }
    for m in re.finditer(r"[\u4e00-\u9fff]{2,3}", text):
        if m.group() not in non_person_cjk:
            return True
    return False


def asks_time(query: str) -> bool:
    return any(marker in query for marker in (
        "什么时候", "何时", "几点", "哪天", "哪一天", "几号", "什么时间", "多久", "几点钟",
    ))


def has_time(text: str) -> bool:
    return bool(re.search(
        r"前[两三四五六七八九十0-9]*天|昨天|今天|明天|前天|后天|"
        r"上[周月]|这[周月]|下[周月]|最近|\d{4}年|\d{1,2}月\d{1,2}日|上周|下周|这周|上午|下午|晚上|早上|中午",
        text,
    ))


def asks_location(query: str) -> bool:
    return any(marker in query for marker in (
        "哪里", "哪儿", "在哪", "什么地方", "地点", "位置", "哪个城市", "哪个省",
    ))


def has_location(text: str) -> bool:
    return bool(re.search(
        r"[\u4e00-\u9fff]{2,}(?:市|省|路|街|区|楼|层|室|房间|家附近|公司|办公室|学校|医院|商场|餐厅|公园)",
        text,
    ))


def asks_object(query: str) -> bool:
    """Detect object-focused questions after removing reason/time phrases."""
    cleaned = query
    for phrase in ("为什么", "什么时候", "何时", "几点", "哪天", "多久"):
        cleaned = cleaned.replace(phrase, "")
    return any(marker in cleaned for marker in ("什么东西", "什么", "啥", "哪个", "哪种"))


def has_object(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9\-]{2,}", text)) or any(
        obj in text for obj in ("键盘", "鼠标", "耳机", "礼物", "书", "手机", "电脑", "猫", "狗")
    )


def suggest_question(missing: list[str]) -> str:
    questions: dict[str, str] = {
        "reason": "你之前有说过为什么/出于什么原因吗？",
        "person": "你说的是哪一位？",
        "time": "这大概是什么时候的事？",
        "location": "这发生在哪里？",
        "object": "具体是什么东西？",
    }
    parts = [questions[m] for m in missing if m in questions]
    return " ".join(parts) if parts else "这部分记忆不够完整，你能补充一下吗？"
