"""Streamlit UI — pet veterinary RAG Agent.

Phase 7: Structured output, streaming, resource monitoring, multi-turn context.
"""

import json
import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from src.agent.orchestrator import get_agent, _extract_citations, _extract_triage, _determine_source
from src.retrieval.vector_store import get_chunk_count
from src.api.models import (
    create_conversation, get_conversation, get_messages,
    save_message, list_conversations,
)
from src.observability.logger import get_recent_traces
from src.observability.monitor import get_gpu_info, get_ollama_status, get_latency_stats
from src.ui.components import (
    render_citations, render_triage_badge, render_source_badge,
    render_confirmation_dialog,
)

st.set_page_config(page_title="宠物兽医知识助手", page_icon="🐱", layout="wide")


# ── Session ──────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "agent": get_agent(),
        "messages": [],
        "conv_id": "",
        "pending_confirmation": None,
        "pet_profile": {"name": "", "breed": "", "age": ""},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _build_full_query(prompt: str) -> str:
    p = st.session_state.pet_profile
    parts = [f"宠物名: {p['name']}" if p["name"] else "",
             f"品种: {p['breed']}" if p["breed"] else "",
             f"年龄: {p['age']}" if p["age"] else ""]
    context = "。".join(filter(None, parts))
    return f"[宠物档案] {context}。\n\n{prompt}" if context else prompt


# ── Sidebar ───────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("🐱 宠物档案")
        st.session_state.pet_profile["name"] = st.text_input("宠物名字", value=st.session_state.pet_profile["name"])
        st.session_state.pet_profile["breed"] = st.selectbox(
            "品种", ["中华田园猫", "英短", "美短", "布偶", "暹罗", "橘猫", "三花", "其他"], index=0)
        st.session_state.pet_profile["age"] = st.text_input("年龄", value=st.session_state.pet_profile["age"], placeholder="如: 6个月")

        st.divider()
        st.metric("知识库片段", get_chunk_count())

        # Conversation context
        conv_msgs = st.session_state.agent.messages
        turns = len([m for m in conv_msgs if m['role'] == 'user'])
        if turns:
            from src.agent.context import estimate_message_tokens
            tokens = estimate_message_tokens(conv_msgs)
            st.caption(f"💬 {turns} 轮对话 · ~{tokens} tokens")

        # GPU monitor
        st.divider()
        st.caption("🖥️ 系统状态")
        gpu = get_gpu_info()
        if gpu["vram_total_mb"]:
            pct = min(gpu["vram_used_mb"] / gpu["vram_total_mb"], 1.0)
            st.progress(pct, f"显存: {gpu['vram_used_mb']}/{gpu['vram_total_mb']} MB")
        ollama = get_ollama_status()
        for m in ollama.get("models", []):
            st.caption(f"🤖 {m['name']} ({m['vram_mb']}MB VRAM)")
        lat = get_latency_stats()
        if lat["count"]:
            st.caption(f"⏱️ 延迟: avg {lat['avg_ms']}ms | last {lat['count']} calls")

        # Recent traces
        traces = get_recent_traces(5)
        if traces:
            st.divider()
            st.caption("📊 最近调用")
            for t in traces[-3:]:
                st.caption(f"`{t['query'][:30]}...` — {t['total_duration_ms']}ms, {t['total_tool_calls']} tools")

        # Conversations
        st.divider()
        st.caption("💬 历史对话")
        for c in list_conversations(10):
            label = c["title"] or f"对话 {c['id'][:8]}"
            if st.button(f"📝 {label}", key=f"conv_{c['id']}", use_container_width=True):
                st.session_state.conv_id = c["id"]
                st.session_state.messages = []
                for m in get_messages(c["id"]):
                    st.session_state.messages.append({"role": m["role"], "content": m["content"]})
                st.rerun()

        if st.button("🗑 新建对话", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.messages = []
            st.session_state.conv_id = ""
            st.rerun()

        st.caption("⚠️ 免责声明：本助手仅供教育参考，不构成兽医诊断或医疗建议。")


# ── Chat ──────────────────────────────────────────────────────────────

def render_chat():
    st.title("🐱 宠物兽医知识助手")
    st.caption("基于知网学术文献的猫咪健康 RAG Agent · 每句话都有出处 · 高风险建议需人工确认")

    # History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("structured"):
                s = msg["structured"]
                render_citations(s.get("citations", []))
                render_triage_badge(s.get("triage", {}).get("level", "UNKNOWN"),
                                    s.get("triage", {}).get("signal", ""),
                                    s.get("triage", {}).get("reasoning", ""))

    # Confirmation dialog
    if st.session_state.pending_confirmation:
        render_confirmation_dialog(st.session_state.pending_confirmation, st.session_state.conv_id)
        return

    # Input
    if prompt := st.chat_input("输入您的问题，如「猫咪打喷嚏三天了怎么办」"):
        if not st.session_state.conv_id:
            st.session_state.conv_id = str(uuid.uuid4())[:8]
            create_conversation(st.session_state.conv_id,
                pet_name=st.session_state.pet_profile.get("name", ""),
                pet_breed=st.session_state.pet_profile.get("breed", ""),
                pet_age=st.session_state.pet_profile.get("age", ""))

        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.conv_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            full_query = _build_full_query(prompt)
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            answer_text = ""

            for event in st.session_state.agent.chat_stream(full_query, conversation_id=st.session_state.conv_id):
                if event["type"] == "tool_call":
                    labels = {"search_knowledge_base": "📖 检索本地知识库", "search_cnki": "🌐 搜索知网论文",
                              "analyze_symptoms": "🔬 分析症状", "triage_decision": "🏥 评估就医紧迫性"}
                    status_placeholder.caption(f"🔄 {labels.get(event['name'], event['name'])}...")
                elif event["type"] == "token":
                    answer_text += event["text"]
                    answer_placeholder.markdown(answer_text + "▌")
                elif event["type"] == "done":
                    answer_text = event["answer"]
                    status_placeholder.empty()
                    answer_placeholder.markdown(answer_text)

            # Extract structured data
            tool_results = st.session_state.agent._last_tool_results
            citations_raw = _extract_citations(tool_results)
            triage = _extract_triage(tool_results, answer_text)
            source = _determine_source(tool_results)
            citations = [{"title": c.title, "journal": c.journal, "year": c.year, "relevant_text": c.relevant_text}
                         for c in citations_raw]
            requires_confirmation = triage.level in ("EMERGENCY", "URGENT")

            render_source_badge(source)
            render_citations(citations)
            render_triage_badge(triage.level, triage.signal, triage.reasoning)

            if requires_confirmation:
                st.session_state.pending_confirmation = {
                    "answer": answer_text, "citations": citations,
                    "triage": {"level": triage.level, "signal": triage.signal, "reasoning": triage.reasoning},
                    "source": source}
                st.warning("⚠️ 检测到高风险建议，请选择是否采纳（见上方确认框）")
            else:
                st.session_state.messages.append({
                    "role": "assistant", "content": answer_text,
                    "structured": {"citations": citations, "triage": {"level": triage.level}}})
                save_message(st.session_state.conv_id, "assistant", answer_text,
                             metadata_json=json.dumps({"citations": citations, "answer": answer_text}, ensure_ascii=False))


def main():
    init_session()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
