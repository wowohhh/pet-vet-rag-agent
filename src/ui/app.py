"""Streamlit UI with Human-in-the-Loop for high-risk recommendations.

Phase 5 enhancement: When the Agent recommends "尽快就医" or higher,
the UI shows a confirmation dialog before proceeding.
"""

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

TRIAGE_KEYWORDS = ["尽快就医", "急诊", "立即就医"]


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


def _detect_triage_level(text: str) -> str | None:
    """Check if response contains high-risk triage recommendations."""
    for kw in TRIAGE_KEYWORDS:
        if kw in text:
            return kw
    return None


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

        # Recent traces summary
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

        # Conversations
        st.caption("💬 历史对话")
        convs = list_conversations(10)
        for c in convs:
            label = c["title"] or f"对话 {c['id'][:8]}"
            if st.button(f"📝 {label}", key=f"conv_{c['id']}", use_container_width=True):
                st.session_state.conv_id = c["id"]
                st.session_state.messages = []
                msgs = get_messages(c["id"])
                for m in msgs:
                    st.session_state.messages.append({
                        "role": m["role"],
                        "content": m["content"],
                    })
                st.rerun()

        if st.button("🗑 新建对话", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.messages = []
            st.session_state.conv_id = ""
            st.rerun()

        st.caption("⚠️ 免责声明：本助手仅供教育参考，不构成兽医诊断或医疗建议。")


def render_chat():
    st.title("🐱 宠物兽医知识助手")
    st.caption("基于知网学术文献的猫咪健康 RAG Agent · 每句话都有出处 · 高风险建议需人工确认")

    # Display conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Human-in-the-loop: confirmation dialog for high-risk advice
    if st.session_state.pending_confirmation:
        pending = st.session_state.pending_confirmation
        st.warning(
            f"⚠️ Agent 建议中包含「{pending['triage_level']}」级别的就医建议。"
            f"请确认是否采纳此建议。"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 采纳建议", use_container_width=True):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": pending["response"]
                    + "\n\n> ✅ [用户已确认采纳此就医建议]",
                })
                save_message(st.session_state.conv_id, "assistant",
                             pending["response"] + "\n\n[用户已确认采纳]")
                st.session_state.pending_confirmation = None
                st.rerun()
        with col2:
            if st.button("❌ 仅作参考", use_container_width=True):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": pending["response"]
                    + "\n\n> ⚠️ [用户选择暂不采纳，仅供参考]",
                })
                save_message(st.session_state.conv_id, "assistant",
                             pending["response"] + "\n\n[用户选择不采纳]")
                st.session_state.pending_confirmation = None
                st.rerun()
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

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("检索文献 + Agent 推理中..."):
                profile = st.session_state.pet_profile
                context = ""
                if profile["name"]:
                    context += f"宠物名: {profile['name']}。"
                if profile["breed"]:
                    context += f"品种: {profile['breed']}。"
                if profile["age"]:
                    context += f"年龄: {profile['age']}。"

                full_query = f"[宠物档案] {context}\n\n{prompt}" if context else prompt
                response = st.session_state.agent.chat(
                    full_query,
                    conversation_id=st.session_state.conv_id,
                )

                # Check for high-risk triage
                triage = _detect_triage_level(response)
                if triage:
                    st.session_state.pending_confirmation = {
                        "response": response,
                        "triage_level": triage,
                    }
                    st.warning(f"⚠️ 检测到高风险建议：{triage}")
                    st.markdown("请选择是否采纳（见上方确认框）")
                else:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    save_message(st.session_state.conv_id, "assistant", response)


def main():
    init_session()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
