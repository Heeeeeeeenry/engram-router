"""L0 agent-traces dataset builder: 100 scenarios × 50-turn conversations.

Unlike semantic_audit (15-turn hand-written) and LongMemEval (real ChatGPT
logs, English), this generates Chinese multi-character agent conversations
with intentional knowledge conflicts, cross-person contamination, and
temporal drift — designed to stress-test engram-router's exclusive Persona /
Causal / Timeline / Forgetting capabilities under realistic agent workloads.

Design
------
- 100 scenarios organized in 5 categories (20 each):
  1. multi_person: 2-3 characters with overlapping facts (names, cars, pets)
  2. temporal_drift: 同一人的信息随时间变化（换工作、搬家、换车）
  3. causal_chain: 事件前后有因果关系
  4. preference_update: 用户偏好随时间变化
  5. mixed: 以上混合
- Each scenario: ~50 user turns + 0-3 assistant confirmations
- QA pairs: 3-5 per scenario, covering recall + isolation + temporal + causal

Output
------
``tests/data/agent_traces.json`` — list of EvalScenario dicts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "data" / "agent_traces.json"

# ── Template scenarios ────────────────────────────────────────────────────


def _make_multi_person(i: int) -> dict[str, Any]:
    """Two characters with distinct cars, pets, jobs.

    Fields per character: (name, gender, car, pet, occupation)
    gender ∈ {"男", "女"} governs pronoun choice (他/她) to avoid contamination.
    """
    seed = i * 7 + 3
    random.seed(seed)
    # 20 unique names covering diverse surnames (H12 fix: prev only 4 names)
    names = [
        ("张伟", "男", "宝马", "金毛", "架构师"),
        ("李娜", "女", "比亚迪", "英短", "设计师"),
        ("王磊", "男", "特斯拉", "柯基", "产品经理"),
        ("陈静", "女", "蔚来", "布偶", "数据科学家"),
        ("刘洋", "男", "奥迪", "拉布拉多", "算法工程师"),
        ("周敏", "女", "小鹏", "泰迪", "运营总监"),
        ("赵磊", "男", "吉利", "边牧", "前端负责人"),
        ("孙丽", "女", "理想", "美短", "客户成功"),
        ("吴强", "男", "凯迪拉克", "哈士奇", "DBA"),
        ("郑雪", "女", "沃尔沃", "银渐层", "产品设计师"),
        ("钱峰", "男", "奔驰", "德牧", "架构评审"),
        ("徐婷", "女", "本田", "萨摩耶", "QA 组长"),
        ("何军", "男", "丰田", "比熊", "SRE"),
        ("罗颖", "女", "MG", "蓝猫", "数据分析师"),
        ("胡涛", "男", "领克", "金毛巡回", "网络工程师"),
        ("彭珊", "女", "极氪", "缅因猫", "视觉设计师"),
        ("邓凯", "男", "长安", "松狮", "内核研发"),
        ("袁玥", "女", "红旗", "折耳猫", "服务端 leader"),
        ("方博", "男", "捷豹", "阿拉斯加", "机器学习平台"),
        ("范丽", "女", "路虎", "英短蓝白", "前端架构"),
    ]
    a, b = random.sample(names, 2)
    # H13 fix: pronoun follows gender to avoid data contamination
    pron_a = "他" if a[1] == "男" else "她"
    pron_b = "他" if b[1] == "男" else "她"
    turns = [
        f"Week 1: {a[0]}是我们部门新来的{a[4]}。",
        f"Week 1: {a[0]}开一辆{a[2]}，养了一只{a[3]}。",
        f"Week 1: {b[0]}是我们隔壁团队的{b[4]}。",
        f"Week 1: {b[0]}开一辆{b[2]}，养了一只{b[3]}。",
        f"Week 2: 今天和{a[0]}开会讨论项目方案，{pron_a}思路很清晰。",
        f"Week 2: {b[0]}请假了，说是家里的{b[3]}生病了要带去看兽医。",
        f"Week 3: {a[0]}说{pron_a}家的{a[3]}最近很活跃，每天要遛两次。",
        f"Week 3: 中午和{b[0]}一起吃饭，{pron_b}推荐了公司附近一家川菜馆。",
        f"Week 4: 项目deadline要到了，{a[0]}加班到很晚。",
        f"Week 4: {b[0]}分享了一个设计工具，说是能让效率翻倍。",
    ]
    while len(turns) < 50:
        turns.append(f"Week {len(turns)//5 + 1}: 日常对话记录 #{len(turns)}。")
    turns = turns[:50]

    return {
        "id": f"agent-multi-person-{i:03d}",
        "source": "agent_traces_multi_person",
        "memories": turns,
        "cases": [
            {"id": f"{i:03d}-a-car", "query": f"{a[0]}开什么车？",
             "expected_ids_or_texts": [a[2]], "forbidden_ids_or_texts": [b[2]],
             "description": f"{a[0]}的车 vs {b[0]}的车"},
            {"id": f"{i:03d}-a-pet", "query": f"{a[0]}养了什么宠物？",
             "expected_ids_or_texts": [a[3]], "forbidden_ids_or_texts": [],
             "description": f"{a[0]}的宠物"},
            {"id": f"{i:03d}-b-car", "query": f"{b[0]}开什么车？",
             "expected_ids_or_texts": [b[2]], "forbidden_ids_or_texts": [a[2]],
             "description": f"{b[0]}的车 vs {a[0]}的车"},
            {"id": f"{i:03d}-isolation", "query": f"{a[0]}养{a[3]}吗？",
             "expected_ids_or_texts": [a[3]], "forbidden_ids_or_texts": [],
             "description": f"{a[0]}的{a[3]}"},
        ],
    }


def _make_temporal_drift(i: int) -> dict[str, Any]:
    seed = i * 11 + 5
    random.seed(seed)
    name, city = random.choice([("周明", "深圳"), ("吴芳", "成都"), ("郑浩", "武汉")])
    turns = [
        f"2024-03: {name}加入了公司，目前base在{city}。",
        f"2024-04: {name}完成了入职培训，导师评价很高。",
        f"2024-06: {name}参与了第一个正式项目，进度很快。",
        f"2024-09: {name}被转到了上海分部，需要relocate。",
        f"2024-10: {name}已经搬到了上海，租了浦东的房子。",
        f"2024-12: 公司年会，{name}获得了最佳新人奖。",
        f"2025-01: {name}升职为高级工程师，工资涨了30%。",
        f"2025-03: {name}开始负责一个小团队，4个人。",
        f"2025-06: {name}的团队完成了Q2目标，拿到了bonus。",
    ]
    while len(turns) < 50:
        turns.append(f"2025-{random.randint(6,12):02d}: {name}的日常记录。")
    turns = turns[:50]

    return {
        "id": f"agent-temporal-{i:03d}",
        "source": "agent_traces_temporal",
        "memories": turns,
        "cases": [
            {"id": f"{i:03d}-first-city", "query": f"{name}最初在哪个城市？",
             "expected_ids_or_texts": [city], "forbidden_ids_or_texts": [],
             "description": "最早的城市"},
            {"id": f"{i:03d}-latest-city", "query": f"{name}现在在哪个城市？",
             "expected_ids_or_texts": ["上海"], "forbidden_ids_or_texts": [],
             "description": "最新的城市（搬迁后）"},
            {"id": f"{i:03d}-role", "query": f"{name}现在的职位是什么？",
             "expected_ids_or_texts": ["高级工程师"], "forbidden_ids_or_texts": [],
             "description": "最新的职位"},
        ],
    }


def _make_causal_chain(i: int) -> dict[str, Any]:
    seed = i * 13 + 7
    random.seed(seed)
    turns = [
        "最近因为需求变了，项目需要延期两周。",
        "因为延期，PM重新安排了sprint planning。",
        "由于新的planning，后端需要重新设计API。",
        "API重新设计导致前端需要改6个页面。",
        "前端改动量太大，不得不招了一个临时外包。",
        "外包进场后速度明显加快，但代码质量参差不齐。",
        "由于代码质量问题，Tech Lead决定引入Code Review机制。",
        "新Code Review流程上线后，bug率下降了40%。",
    ]
    while len(turns) < 50:
        turns.append(f"因果相关的日常记录 #{len(turns)}。")
    return {
        "id": f"agent-causal-{i:03d}",
        "source": "agent_traces_causal",
        "memories": turns[:50],
        "cases": [
            {"id": f"{i:03d}-why-delay", "query": "项目为什么延期了？",
             "expected_ids_or_texts": ["需求变"], "forbidden_ids_or_texts": [],
             "description": "延期原因"},
            {"id": f"{i:03d}-why-review", "query": "为什么引入了Code Review？",
             "expected_ids_or_texts": ["代码质量"], "forbidden_ids_or_texts": [],
             "description": "因果链"},
        ],
    }


def _make_preference_update(i: int) -> dict[str, Any]:
    seed = i * 17 + 9
    random.seed(seed)
    turns = [
        "我最喜欢的颜色是蓝色。",
        "最近在学Python，感觉比Java舒服多了。",
        "尝试了Go语言后发现性能确实更好。",
        "开始觉得Go比Python更适合写后端服务了。",
        "以前每周去三次健身房，现在减到一次了。",
        "现在完全不去健身房了，改成在家做瑜伽。",
        "我原来吃辣很厉害，现在胃不好不敢吃了。",
    ]
    while len(turns) < 50:
        turns.append(f"偏好记录 #{len(turns)}。")
    return {
        "id": f"agent-pref-{i:03d}",
        "source": "agent_traces_preference",
        "memories": turns[:50],
        "cases": [
            {"id": f"{i:03d}-color", "query": "我最喜欢什么颜色？",
             "expected_ids_or_texts": ["蓝色"], "forbidden_ids_or_texts": [],
             "description": "初始偏好"},
            {"id": f"{i:03d}-lang", "query": "我现在最喜欢的编程语言是什么？",
             "expected_ids_or_texts": ["Go"], "forbidden_ids_or_texts": [],
             "description": "最近偏好"},
            {"id": f"{i:03d}-workout", "query": "我现在怎么锻炼？",
             "expected_ids_or_texts": ["瑜伽"], "forbidden_ids_or_texts": ["健身房"],
             "description": "偏好变更"},
        ],
    }


def _make_mixed(i: int) -> dict[str, Any]:
    seed = i * 19 + 11
    random.seed(seed)
    a = random.choice(["赵亮", "孙悦", "钱涛"])
    b = random.choice(["周静", "吴天", "郑琳"])
    car_a = random.choice(["奔驰", "奥迪", "雷克萨斯"])
    car_b = random.choice(["本田", "丰田", "大众"])
    turns = [
        f"{a}刚买了一辆{car_a}。",
        f"{b}是我们团队的{b}，去年刚来。",
        f"{a}之前在北京工作，今年因为家庭原因搬到上海了。",
        f"{b}说她最近在学做菜，专门报了一个烹饪班。",
        f"由于上海办公室的扩建，{a}的团队扩充到了12人。",
        f"{b}开的是{car_b}，每天通勤要一个小时。",
    ]
    while len(turns) < 50:
        turns.append(f"混合场景记录 #{len(turns)}。")
    return {
        "id": f"agent-mixed-{i:03d}",
        "source": "agent_traces_mixed",
        "memories": turns[:50],
        "cases": [
            {"id": f"{i:03d}-a-car", "query": f"{a}开什么车？",
             "expected_ids_or_texts": [car_a], "forbidden_ids_or_texts": [car_b],
             "description": f"{a} vs {b} 的车"},
            {"id": f"{i:03d}-a-move", "query": f"{a}为什么搬到上海？",
             "expected_ids_or_texts": ["家庭原因"], "forbidden_ids_or_texts": [],
             "description": "因果+时间"},
            {"id": f"{i:03d}-b-class", "query": f"{b}在学什么？",
             "expected_ids_or_texts": ["做菜"], "forbidden_ids_or_texts": [],
             "description": "偏好"},
            {"id": f"{i:03d}-team-size", "query": f"{a}的团队现在多少人？",
             "expected_ids_or_texts": ["12人"], "forbidden_ids_or_texts": [],
             "description": "最新信息"},
        ],
    }


def main() -> None:
    generators = {
        "multi_person": _make_multi_person,
        "temporal_drift": _make_temporal_drift,
        "causal_chain": _make_causal_chain,
        "preference_update": _make_preference_update,
        "mixed": _make_mixed,
    }

    scenarios: list[dict[str, Any]] = []
    for cat, fn in generators.items():
        for i in range(20):
            scenarios.append(fn(i))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)

    total_mems = sum(len(s["memories"]) for s in scenarios)
    total_cases = sum(len(s["cases"]) for s in scenarios)
    print(f"Agent traces: {len(scenarios)} scenarios, {total_mems} turns, {total_cases} QA pairs")
    print(f"Written to: {OUTPUT}")


if __name__ == "__main__":
    main()
