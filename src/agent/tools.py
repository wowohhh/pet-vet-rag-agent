"""Tool definitions for the Pet Vet RAG Agent."""

import os
import requests
from src.retrieval.hybrid_search import hybrid_search

# Emergency warning signs that require immediate vet visit
EMERGENCY_SIGNS = [
    "呼吸困难", "张口呼吸", "喘息", "窒息",
    "抽搐", "昏迷", "瘫痪", "站不起来",
    "大出血", "流血不止", "外伤",
    "中毒", "吃了百合", "吃了巧克力", "吃了洋葱", "吃了葡萄",
    "持续呕吐", "呕吐不止", "吐血",
    "持续腹泻", "拉血", "便血",
    "腹部膨胀", "嚎叫", "痛苦",
    "尿不出来", "尿血",
    "眼睛突出", "瞳孔散大",
    "烫伤", "烧伤", "触电",
]


def search_knowledge_base(query: str) -> str:
    """搜索宠物兽医知识库，返回相关学术文献摘录。

    Args:
        query: 用户的自然语言问题。

    Returns:
        格式化的文献摘录，含标题、期刊、年份和内容。
    """
    results = hybrid_search(query, top_k=5)
    if not results:
        return "未找到相关文献。当前知识库可能未涵盖此问题。"

    lines = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        title = meta.get("title", "未知标题")
        journal = meta.get("journal", "未知期刊")
        year = meta.get("year", "")
        text = r.get("text", "")[:300]
        score = r.get("score", 0)

        lines.append(
            f"### 文献 {i} (相关度: {score:.2f})\n"
            f"**标题**: {title}\n"
            f"**期刊**: {journal}" + (f" ({year})" if year else "") + "\n"
            f"**摘录**: {text}\n"
        )
    return "\n".join(lines)


def analyze_symptoms(symptoms: str) -> str:
    """分析症状描述，列出可能的疾病方向（仅供参考，不构成诊断）。

    Args:
        symptoms: 用户描述的宠物症状。

    Returns:
        症状分析结果，含警告信息。
    """
    # First, search for relevant literature
    results = hybrid_search(f"猫咪 {symptoms} 症状 诊断", top_k=3)

    if not results:
        return (
            "无法在知识库中找到与「{symptoms}」相关的症状分析文献。\n"
            "建议您带宠物就医，由兽医进行专业诊断。"
        )

    lines = ["## 症状分析（仅供参考，不构成兽医诊断）", ""]
    lines.append(f"**用户描述症状**: {symptoms}")
    lines.append("")
    lines.append("### 相关文献记载:")

    for i, r in enumerate(results, 1):
        text = r.get("text", "")[:250]
        meta = r.get("metadata", {})
        title = meta.get("title", "未知")
        lines.append(f"{i}. {text} —《{title}》")

    lines.append("")
    lines.append("⚠️ 以上仅为文献记载的可能方向，不代表确诊。请咨询执业兽医。")
    return "\n".join(lines)


def triage_decision(symptoms: str) -> str:
    """根据症状严重程度，给出就医紧迫性建议。

    Args:
        symptoms: 用户描述的宠物症状。

    Returns:
        分级建议：急诊 / 尽快就医 / 居家观察 / 信息不足。
    """
    symptoms_lower = symptoms.lower()

    # Check for emergency signs
    for sign in EMERGENCY_SIGNS:
        if sign in symptoms_lower:
            return (
                f"## 🚨 急诊建议\n\n"
                f"检测到危险信号「**{sign}**」，建议**立即**前往最近的宠物医院急诊科。\n"
                f"途中可先电话联系医院告知情况，以便医院提前准备。\n\n"
                f"### 检测到的危险信号:\n"
                f"- {sign}\n"
            )

    # Check for concerning but non-emergency signs
    concern_signs = [
        "不吃", "不喝", "没精神", "嗜睡", "呕吐", "腹泻", "拉稀",
        "跛行", "瘸", "瘙痒", "掉毛", "消瘦", "咳嗽", "打喷嚏",
        "流鼻涕", "眼屎", "发红", "肿胀", "发热", "发烧",
    ]
    found_concerns = [s for s in concern_signs if s in symptoms_lower]

    if found_concerns:
        return (
            f"## ⚠️ 建议尽快就医\n\n"
            f"检测到以下症状: {', '.join(found_concerns)}\n\n"
            f"这些症状可能需要兽医诊断。建议在 **24-48小时内** 预约兽医。\n"
            f"期间密切观察症状变化，如加重则立即就医。\n"
        )

    # Mild or unclear
    return (
        f"## ℹ️ 居家观察\n\n"
        f"未检测到明显的急诊或紧急症状。\n"
        f"建议继续观察 24-48 小时。如症状持续或加重，请咨询兽医。\n"
        f"如您对症状描述有补充，可以提供更多细节以便更准确判断。\n"
    )


# ── 🏗️ CNKI web search tool (MCP-driven, Phase 7) ──────────────────────────

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "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")


def search_cnki(query: str) -> str:
    """当本地知识库无结果时，通过搜索引擎查找知网论文摘要作为补充。

    Args:
        query: 搜索查询（宠物猫疾病相关）。

    Returns:
        格式化的论文标题和摘要，或提示未找到。
    """
    if not TAVILY_API_KEY:
        return "CNKI 搜索工具未配置（缺少 API key）。"

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "query": f"site:cnki.net 猫 {query}",
                "search_depth": "advanced",
                "max_results": 3,
                "include_domains": ["cnki.net"],
            },
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            timeout=15,
        )
        data = resp.json()
        results = data.get("results", [])
    except Exception as e:
        return f"CNKI 搜索失败: {e}"

    if not results:
        return "未在知网找到相关论文。"

    lines = ["## 🌐 知网检索结果（来自网络搜索，非本地知识库）", ""]
    for i, r in enumerate(results, 1):
        title = r.get("title", "未知标题")[:100]
        content = r.get("content", "")[:250]
        url = r.get("url", "")
        lines.append(f"### 文献 {i}\n**标题**: {title}\n**摘要**: {content}")
        if url:
            lines.append(f"**链接**: {url}")
        lines.append("")

    lines.append("⚠️ 以上结果来自网络检索，未经过本地知识库验证，仅供参考。")
    return "\n".join(lines)
