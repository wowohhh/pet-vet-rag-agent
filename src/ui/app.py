"""Streamlit UI with Human-in-the-Loop for high-risk recommendations.

Phase 5: Human-in-the-loop confirmation dialog
Phase 7: 🏗️ Structured output — citation cards, triage badges, precise confirmation
"""

import json
import streamlit as st
from src.agent.orchestrator import get_agent
from src.retrieval.vector_store import get_chunk_count
from src.api.models import (
    create_conversation, get_conversation, get_messages,
    save_message, list_conversations, delete_conversation,
)
from src.observability.logger import get_recent_traces

st.set_page_config(
    page_title="宠物兽医知识助手",
    page_icon="🐱",
    layout="wide",
)

TRIAGE_COLORS = {
    "EMERGENCY": "🔴",
    "URGENT": "🟠",
    "OBSERVE": "🟢",
    "UNKNOWN": "⚪",
}

TRIAGE_LABELS = {
    "EMERGENCY": "急诊",
    "URGENT": "尽快就医",
    "OBSERVE": "居家观察",
    "UNKNOWN": "无法判断",
}


def init_session():
    if "agent" not in st.session_state:
        st.session_state.agent = get_agent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conv_id" not in st.session_state:
        st.session_state.conv_id = ""
    if "pending_confirmation" not in st.session_state:
        st.session_state.pending_confirmation = None
    if "pet_profile" not in st.session_state:
        st.session_state.pet_profile = {"name": "", "breed": "", "age": ""}


def _build_full_query(prompt: str) -> str:
    """Build context-aware query from pet profile."""
    profile = st.session_state.pet_profile
    parts = []
    if profile["name"]:
        parts.append(f"宠物名: {profile['name']}")
    if profile["breed"]:
        parts.append(f"品种: {profile['breed']}")
    if profile["age"]:
        parts.append(f"年龄: {profile['age']}")
    context = "。".join(parts) + "。" if parts else ""
    return f"[宠物档案] {context}\n\n{prompt}" if context else prompt


def render_sidebar():
    with st.sidebar:
        st.title("🐱 宠物档案")

        st.session_state.pet_profile["name"] = st.text_input(
            "宠物名字", value=st.session_state.pet_profile["name"]
        )
        st.session_state.pet_profile["breed"] = st.selectbox(
            "品种", ["中华田园猫", "英短", "美短", "布偶", "暹罗", "橘猫", "三花", "其他"], index=0
        )
        st.session_state.pet_profile["age"] = st.text_input(
            "年龄", value=st.session_state.pet_profile["age"], placeholder="如: 6个月"
        )

        st.divider()

        chunk_count = get_chunk_count()
        st.metric("知识库片段", chunk_count)

        traces = get_recent_traces(5)
        if traces:
            st.divider()
            st.caption("📊 最近调用")
            for t in traces[-3:]:
                st.caption(
                    f"`{t['query'][:30]}...` "
                    f"— {t['total_duration_ms']}ms, "
                    f"{t['total_tool_calls']} tools"
                )

        st.divider()
        st.caption("💬 历史对话")
        convs = list_conversations(10)
        for c in convs:
            label = c["title"] or f"对话 {c['id'][:8]}"
            if st.button(f"📝 {label}", key=f"conv_{c['id']}", use_container_width=True):
                st.session_state.conv_id = c["id"]
                st.session_state.messages = []
                msgs = get_messages(c["id"])
                for m in msgs:
                    content = m["content"]
                    if m["role"] == "assistant" and m.get("metadata_json"):
                        try:
                            meta = json.loads(m["metadata_json"])
                            content = meta.get("answer", content)
                        except json.JSONDecodeError:
                            pass
                    st.session_state.messages.append({
                        "role": m["role"],
                        "content": content,
                    })
                st.rerun()

        if st.button("🗑 新建对话", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.messages = []
            st.session_state.conv_id = ""
            st.rerun()

        st.caption("⚠️ 免责声明：本助手仅供教育参考，不构成兽医诊断或医疗建议。")


def render_triage_badge(level: str, signal: str = "", reasoning: str = ""):
    """Render a colored triage badge with tooltip info."""
    color = TRIAGE_COLORS.get(level, "⚪")
    label = TRIAGE_LABELS.get(level, level)
    badge = f"{color} **{label}**"
    if signal:
        badge += f" — 信号: `{signal}`"
    st.markdown(badge)
    if reasoning:
        st.caption(f"📋 {reasoning}")


def render_citations(citations: list[dict]):
    """Render citation cards from structured data."""
    if not citations:
        return
    st.divider()
    st.caption(f"📚 **引用来源 ({len(citations)} 篇)**")
    cols = st.columns(min(len(citations), 3))
    for i, c in enumerate(citations[:6]):  # max 6 visible
        with cols[i % 3]:
            with st.container(border=True):
                title = c.get("title", "未知")[:50]
                journal = c.get("journal", "")
                year = c.get("year", "")
                text = c.get("relevant_text", "")[:80]
                st.caption(f"**{title}**")
                if journal:
                    st.caption(f"*{journal}*" + (f" ({year})" if year else ""))
                if text:
                    st.caption(text + "..." if len(text) >= 80 else text)


def render_confirmation_dialog(pending: dict):
    """Show Human-in-the-Loop confirmation for high-risk recommendations."""
    triage = pending.get("triage", {})
    level = triage.get("level", "UNKNOWN")
    label = TRIAGE_LABELS.get(level, level)
    signal = triage.get("signal", "")

    st.warning(
        f"⚠️ Agent 建议分级为「{label}」"
        + (f"（信号: {signal}）" if signal else "")
        + f"。请确认是否采纳此建议。"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 采纳建议", use_container_width=True):
            note = "\n\n> ✅ [用户已确认采纳此就医建议]"
            st.session_state.messages.append({
                "role": "assistant",
                "content": pending["answer"] + note,
                "structured": pending,
            })
            save_message(
                st.session_state.conv_id, "assistant",
                pending["answer"] + note,
                metadata_json=json.dumps(pending, ensure_ascii=False),
            )
            st.session_state.pending_confirmation = None
            st.rerun()
    with col2:
        if st.button("❌ 仅作参考", use_container_width=True):
            note = "\n\n> ⚠️ [用户选择暂不采纳，仅供参考]"
            st.session_state.messages.append({
                "role": "assistant",
                "content": pending["answer"] + note,
                "structured": pending,
            })
            save_message(
                st.session_state.conv_id, "assistant",
                pending["answer"] + note,
                metadata_json=json.dumps(pending, ensure_ascii=False),
            )
            st.session_state.pending_confirmation = None
            st.rerun()


def render_chat():
    st.title("🐱 宠物兽医知识助手")
    st.caption("基于知网学术文献的猫咪健康 RAG Agent · 每句话都有出处 · 高风险建议需人工确认")

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show citation cards if structured data available
            if msg["role"] == "assistant" and msg.get("structured"):
                render_citations(msg["structured"].get("citations", []))
                triage = msg["structured"].get("triage", {})
                if triage.get("level") and triage["level"] != "UNKNOWN":
                    st.divider()
                    render_triage_badge(
                        triage["level"],
                        triage.get("signal", ""),
                        triage.get("reasoning", ""),
                    )

    # 🏗️ Human-in-the-loop: structured confirmation
    if st.session_state.pending_confirmation:
        pending = st.session_state.pending_confirmation
        render_confirmation_dialog(pending)
        return  # Block input while pending confirmation

    # Chat input
    if prompt := st.chat_input("输入您的问题，如「猫咪打喷嚏三天了怎么办」"):
        if not st.session_state.conv_id:
            st.session_state.conv_id = create_conversation(
                pet_name=st.session_state.pet_profile.get("name", ""),
                pet_breed=st.session_state.pet_profile.get("breed", ""),
                pet_age=st.session_state.pet_profile.get("age", ""),
            )["id"]

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.conv_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response — stream tokens for real-time display
        with st.chat_message("assistant"):
            full_query = _build_full_query(prompt)

            # 🏗️ P0: streaming — show tool calls live, then stream answer tokens
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            answer_text = ""
            tool_calls_seen = []

            for event in st.session_state.agent.chat_stream(
                full_query,
                conversation_id=st.session_state.conv_id,
            ):
                if event["type"] == "tool_call":
                    tool_calls_seen.append(event["name"])
                    tool_names = {"search_knowledge_base": "📖 检索本地知识库",
                                  "search_cnki": "🌐 搜索知网论文",
                                  "analyze_symptoms": "🔬 分析症状",
                                  "triage_decision": "🏥 评估就医紧迫性"}
                    label = tool_names.get(event["name"], event["name"])
                    status_placeholder.caption(f"🔄 {label}...")

                elif event["type"] == "token":
                    answer_text += event["text"]
                    answer_placeholder.markdown(answer_text + "▌")

                elif event["type"] == "done":
                    answer_text = event["answer"]
                    status_placeholder.empty()
                    answer_placeholder.markdown(answer_text)

            # 🏗️ Extract structured data from completed stream (no re-run)
            from src.agent.orchestrator import _extract_citations, _extract_triage, _determine_source
            tool_results = st.session_state.agent._last_tool_results
            citations_raw = _extract_citations(tool_results)
            triage = _extract_triage(tool_results, answer_text)
            source = _determine_source(tool_results)
            citations = [{"title": c.title, "journal": c.journal, "year": c.year, "relevant_text": c.relevant_text} for c in citations_raw]
            requires_confirmation = triage.level in ("EMERGENCY", "URGENT")

            # Show source badge
            source_labels = {"local": "📖 本地知识库", "cnki": "🌐 知网网络检索", "fallback": "⚠️ 无文献支持"}
            st.caption(source_labels.get(source, source))

            # Show citations
            if citations:
                render_citations(citations)

            # Show triage badge
            if triage.level != "UNKNOWN":
                st.divider()
                render_triage_badge(triage.level, triage.signal, triage.reasoning)

            # 🏗️ Structured confirmation
            if requires_confirmation:
                st.session_state.pending_confirmation = {
                    "answer": answer_text,
                    "citations": citations,
                    "triage": {"level": triage.level, "signal": triage.signal, "reasoning": triage.reasoning},
                    "source": source,
                }
                st.warning(f"⚠️ 检测到高风险建议，请选择是否采纳（见上方确认框）")
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "structured": {"citations": citations, "triage": {"level": triage.level, "signal": triage.signal, "reasoning": triage.reasoning}},
                })
                save_message(
                    st.session_state.conv_id, "assistant",
                    answer_text,
                    metadata_json=json.dumps({"citations": citations, "answer": answer_text}, ensure_ascii=False),
                )


def main():
    init_session()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
