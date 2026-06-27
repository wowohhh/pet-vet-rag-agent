"""Reusable UI components — citations, triage badges, confirmation dialogs."""

import json
import streamlit as st

TRIAGE_COLORS = {"EMERGENCY": "🔴", "URGENT": "🟠", "OBSERVE": "🟢", "UNKNOWN": "⚪"}
TRIAGE_LABELS = {"EMERGENCY": "急诊", "URGENT": "尽快就医", "OBSERVE": "居家观察", "UNKNOWN": "无法判断"}
SOURCE_LABELS = {"local": "📖 本地知识库", "cnki": "🌐 知网网络检索", "fallback": "⚠️ 无文献支持"}


def render_source_badge(source: str):
    st.caption(SOURCE_LABELS.get(source, source))


def render_triage_badge(level: str, signal: str = "", reasoning: str = ""):
    """Colored triage badge with tooltip info."""
    if level == "UNKNOWN":
        return
    color = TRIAGE_COLORS.get(level, "⚪")
    label = TRIAGE_LABELS.get(level, level)
    badge = f"{color} **{label}**"
    if signal:
        badge += f" — 信号: `{signal}`"
    st.markdown(badge)
    if reasoning:
        st.caption(f"📋 {reasoning}")


def render_citations(citations: list[dict]):
    """Citation cards in 3-column grid."""
    if not citations:
        return
    st.divider()
    st.caption(f"📚 **引用来源 ({len(citations)} 篇)**")
    cols = st.columns(min(len(citations), 3))
    for i, c in enumerate(citations[:6]):
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


def render_confirmation_dialog(pending: dict, conv_id: str):
    """Human-in-the-Loop confirmation for high-risk recommendations."""
    from src.api.models import save_message

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
                "role": "assistant", "content": pending["answer"] + note, "structured": pending,
            })
            save_message(conv_id, "assistant", pending["answer"] + note,
                         metadata_json=json.dumps(pending, ensure_ascii=False))
            st.session_state.pending_confirmation = None
            st.rerun()
    with col2:
        if st.button("❌ 仅作参考", use_container_width=True):
            note = "\n\n> ⚠️ [用户选择暂不采纳，仅供参考]"
            st.session_state.messages.append({
                "role": "assistant", "content": pending["answer"] + note, "structured": pending,
            })
            save_message(conv_id, "assistant", pending["answer"] + note,
                         metadata_json=json.dumps(pending, ensure_ascii=False))
            st.session_state.pending_confirmation = None
            st.rerun()
