"""Configuration system for EngramRouter.

All configurable values live here with sensible defaults.
Users can override via ~/.engram/config.yaml without editing code.

Usage:
    from engram_router.config import config
    config.entities.kinship_words  # list of kinship terms
    config.recall.weights.ascii_base  # scoring weight
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


def _env_config_path() -> Path | None:
    """Return user config path set via ENGRAM_CONFIG env var."""
    p = os.environ.get("ENGRAM_CONFIG")
    return Path(p) if p else None


def _default_config_path() -> Path:
    return Path.home() / ".engram" / "config.yaml"


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EntityConfig:
    """All entity extraction patterns and word lists."""

    # ── person ──
    kinship_words: list[str] = field(default_factory=lambda: [
        "妈妈", "母亲", "爸爸", "父亲", "爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷",
        "哥哥", "弟弟", "姐姐", "妹妹", "老婆", "老公", "妻子", "丈夫", "儿子", "女儿",
        "舅舅", "叔叔", "姑姑", "阿姨", "伯伯", "婶婶", "表哥", "表姐", "堂哥",
        "室友", "同事", "朋友", "同学",
        "咪咪",
    ])
    role_words: list[str] = field(default_factory=lambda: [
        "同事", "前同事", "同学", "朋友", "老板", "领导", "客户", "老师",
    ])
    surname_chars: str = "张李王赵刘陈杨黄周吴徐孙马朱胡林郭何高罗"
    name_breakers: str = (
        "前送是在的说现给最这那今昨明上下个把和与认识喜爱有过来去要会想做"
        "、，。！？ 和跟同当于到脾平每总很也已才就又都还便只"
    )

    # ── object ──
    known_objects: list[str] = field(default_factory=lambda: [
        "机械键盘", "键盘", "鼠标", "显示器", "耳机", "猫", "狗",
        "特斯拉", "车", "电动车", "手机", "房子",
        "咪咪", "英短", "豆豆", "金毛", "宠物",
        "一百二", "六千",
    ])
    food_words: list[str] = field(default_factory=lambda: [
        "红烧肉", "糖醋排骨", "西红柿炒蛋", "番茄炒蛋", "饺子", "包子", "面条", "米饭",
        "炒饭", "粥", "火锅", "烤鱼", "回锅肉", "鱼香肉丝", "宫保鸡丁", "麻婆豆腐",
        "青椒肉丝", "可乐鸡翅", "土豆丝", "炖鸡", "排骨汤", "小笼包",
    ])
    ascii_stop_words: set[str] = field(default_factory=lambda: {"the", "a", "an", "is", "are"})

    # ── company ──
    known_companies: list[str] = field(default_factory=lambda: [
        "腾讯", "阿里", "阿里巴巴", "字节", "字节跳动",
        "百度", "美团", "京东", "华为", "小米", "网易",
    ])
    company_markers: list[str] = field(default_factory=lambda: ["公司", "集团", "科技", "有限"])

    # ── topic ──
    topic_words: list[str] = field(default_factory=lambda: [
        "键盘", "礼物", "生日", "机械键盘",
        "车", "手机", "房子", "职业", "工作",
        "退休教师", "教师", "脾气",
        "钓鱼", "爱好", "菜", "饭", "馆子", "餐厅", "钱", "价格", "人均",
        "狗", "猫",
    ])
    object_topic_aliases: dict[str, str] = field(default_factory=lambda: {
        "HHKB": "键盘", "MX": "键盘", "Keychron": "键盘", "机械键盘": "键盘",
        "特斯拉": "车", "Model 3": "车", "电动车": "车",
        "iPhone": "手机", "iPad": "手机",
        "水煮鱼": "菜", "宫保鸡丁": "菜", "红烧肉": "菜", "糖醋排骨": "菜",
        "毛血旺": "菜", "火锅": "菜",
        "一百二": "钱", "120": "钱", "六千": "钱", "20亿": "钱", "5000万": "钱",
        "豆豆": "狗", "金毛": "狗", "咪咪": "猫", "英短": "猫",
        "钓鱼": "爱好",
    })

    # ── time ──
    time_patterns: list[str] = field(default_factory=lambda: [
        r"前[两三四五六七八九十0-9]*天",
        r"昨天", r"今天", r"明天", r"前天", r"后天",
        r"上[周月]", r"这[周月]", r"下[周月]",
        r"最近",
        r"\d{4}年", r"\d{1,2}月\d{1,2}日",
    ])

    # ── reason ──
    reason_markers: list[str] = field(default_factory=lambda: [
        "因为", "由于", "为了", "原因是", "之所以", "是因为",
    ])

    # ── attribute ──
    attr_patterns: list[str] = field(default_factory=lambda: [
        r"[0-9]{1,3}岁",
        r"[ABO]型血",
        r"(?<![\u4e00-\u9fff])[男女](?![\u4e00-\u9fff])",
        r"[\u4e00-\u9fff]{2,3}人(?![\u4e00-\u9fff])",
        r"(?:属)[鼠牛虎兔龙蛇马羊猴鸡狗猪]",
    ])


@dataclass
class SalienceConfig:
    """Salience classification patterns."""

    base_attr_name_patterns: list[str] = field(default_factory=lambda: [
        r"^[男女]$", r"^\d{1,3}岁$", r"^[ABO]型?血$",
        r"^[\u4e00-\u9fff]{2,3}人$", r"^\d{4}年$",
    ])
    base_attr_context: list[str] = field(default_factory=lambda: [
        r"性别", r"叫什么", r"名字", r"出生", r"籍贯", r"血型",
    ])
    sensory_patterns: list[str] = field(default_factory=lambda: [
        r"(?:做[饭菜]|烧菜|炒菜|烹饪|手艺).{0,4}(?:好吃|难吃|一般|不错|很棒|香)",
        r"(?:性格|脾气|人)(?:很|挺|特别|有点)?(?:好|坏|温柔|暴躁|急|慢|细心|粗心|唠叨)",
        r"(?:很|挺|特别|非常|有点)(?:好看|漂亮|帅|丑|胖|瘦|高|矮|年轻|老|温柔|唠叨|细心)",
        r"(?:很|挺|特别|非常|不太)(?:会|能|擅长|懂)",
        r"(?:想念|想吃|喜欢|讨厌|怀念)",
    ])
    decision_markers: list[str] = field(default_factory=lambda: [
        "决定", "确定", "确认", "选择", "采用", "最终方案",
        "定下来", "就这样", "不改了", "通过", "同意",
    ])
    constraint_markers: list[str] = field(default_factory=lambda: [
        "不能", "不允许", "禁止", "必须", "最多", "最少",
        "不超过", "不低于", "上限", "下限", "硬性要求",
    ])
    event_markers: list[str] = field(default_factory=lambda: [
        "昨天", "今天", "刚才", "前几天", "前两天", "上周", "上个月",
        "那天", "当时", "中午", "早上", "晚上",
        "做了", "干了", "去了", "买了", "吃了", "说了", "写了", "做过",
    ])


@dataclass
class RecallWeightsConfig:
    """All scoring weights and thresholds."""

    # ── token scoring ──
    ascii_base: float = 4.0
    ascii_per_char_cap: int = 6
    ascii_per_char: float = 0.5
    cjk_multi_base: float = 2.0
    cjk_multi_per_char: float = 0.5
    stop_char_weight: float = 0.05
    single_cjk_weight: float = 0.4

    # ── semantic boosts ──
    colleague_boost: float = 1.0
    reason_marker_boost: float = 1.5

    # ── recall pipeline ──
    fts_boost: float = 0.1
    shared_entity_multiplier: float = 1.2
    conflicting_person_penalty: float = 1.5
    person_match_boost: float = 1.5
    entity_tie_break_bonus: float = 0.01

    # ── context boosts ──
    brand_boost: float = 2.0
    occupation_boost: float = 1.5
    identity_base_attr_boost: float = 2.0
    eval_sensory_boost: float = 1.5

    # ── correction ──
    correction_penalty: float = 0.3

    # ── spreading activation ──
    max_recall_hops: int = 5
    recall_decay: float = 0.5
    activation_threshold: float = 0.03

    # ── associative reach ──
    assoc_reach_base_attr: float = 0.15
    assoc_reach_constraint: float = 0.6
    assoc_reach_decision: float = 0.7
    assoc_reach_sensory: float = 1.0
    assoc_reach_event: float = 1.0

    # ── scale protection ──
    full_scan_limit: int = 2000

    # ── stop chars ──
    stop_chars: str = "我你他她它的了是在有和与啊吗呢吧那这个什么牌子哪家为么把了和就都也很"

    # ── occupation topics ──
    occupation_topics: list[str] = field(default_factory=lambda: [
        "退休教师", "教师", "医生", "工程师", "设计师", "护士", "律师", "司机", "会计",
    ])


@dataclass
class PrivacyConfig:
    """Privacy controls — data never leaves the machine without consent."""
    allow_cloud_llm: bool = False
    allow_cloud_embedding: bool = False
    allow_cloud_reranker: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "PrivacyConfig":
        return cls(
            allow_cloud_llm=bool(d.get("allow_cloud_llm", False)),
            allow_cloud_embedding=bool(d.get("allow_cloud_embedding", False)),
            allow_cloud_reranker=bool(d.get("allow_cloud_reranker", False)),
        )


@dataclass
class ExpansionConfig:
    """Query expansion configuration."""

    # ── Synonym table ──
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    """User-defined synonyms merged with built-in defaults.
       Example:
         synonyms:
           HHKB: [机械键盘, 键盘, 静电容]
           Mac: [苹果电脑, MacBook]
    """

    # ── LLM query rewriting ──
    llm_enabled: bool = True
    """Enable LLM query rewriting. Set to False for synonym-only mode."""

    llm_max_variants: int = 4
    """Maximum variants per rewrite call."""

    # ── Cache ──
    cache_size: int = 256
    """LRU cache size in entries."""

    async_llm: bool = True
    """Use async LLM mode (recommended). False = sync wait on first call."""


@dataclass
class EngramConfig:
    """Master configuration for EngramRouter."""

    entities: EntityConfig = field(default_factory=EntityConfig)
    salience: SalienceConfig = field(default_factory=SalienceConfig)
    recall: RecallWeightsConfig = field(default_factory=RecallWeightsConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> EngramConfig:
        """Load config from YAML file, merge with defaults.

        Priority: env ENGRAM_CONFIG > ~/.engram/config.yaml > defaults
        """
        config = cls()

        target = path or _env_config_path() or _default_config_path()
        if not target.exists():
            return config

        try:
            import yaml
        except ImportError:
            return config

        try:
            with open(target) as f:
                user_data = yaml.safe_load(f) or {}
        except Exception:
            return config

        _deep_merge(config, user_data)
        return config


# ═══════════════════════════════════════════════════════════════════════════
# Deep merge helper
# ═══════════════════════════════════════════════════════════════════════════

def _deep_merge(base: Any, override: Any) -> None:
    """Merge override dict into base dataclass/dict in-place."""
    if not isinstance(override, dict):
        return
    for key, val in override.items():
        if not hasattr(base, key):
            continue
        base_val = getattr(base, key)
        if isinstance(base_val, dict) and isinstance(val, dict):
            base_val.update(val)
        elif hasattr(base_val, "__dataclass_fields__") and isinstance(val, dict):
            _deep_merge(base_val, val)
        else:
            setattr(base, key, val)


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════

config = EngramConfig.load()
