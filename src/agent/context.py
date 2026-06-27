"""Context engineering: multi-zone context management with token budgets.

Four zones:
  Z1: system_prompt    — Agent identity + rules (fixed, ~500 tokens)
  Z2: retrieved_docs   — RAG results (capped at 60% of context budget)
  Z3: tool_results     — Tool call outputs (capped at 30%)
  Z4: conversation     — Recent chat history (capped at 10%)
"""

from src.config import OLLAMA_MODEL

# Estimated context window per model
CONTEXT_WINDOWS = {
    "qwen3:4b": 4096,   # default context set by ollama
    "qwen3:8b": 4096,
    "qwen2.5:0.5b": 4096,
}

# Zone budget percentages (must sum to <= 100%)
ZONE_BUDGETS = {
    "system": 0.15,       # system prompt
    "retrieved": 0.55,    # RAG documents
    "tool": 0.20,         # tool results
    "conversation": 0.10, # chat history
}


def get_context_window(model: str) -> int:
    return CONTEXT_WINDOWS.get(model, 4096)


def estimate_tokens(text: str) -> int:
    """Rough token estimation for Chinese text (~1.5 chars/token)."""
    return max(1, len(text) // 2)


def build_context(
    system_prompt: str,
    retrieved_docs: list[dict],
    tool_results: list[str],
    conversation_history: list[dict],
    model: str = OLLAMA_MODEL,
) -> list[dict]:
    """Build context-aware messages with token budget management.

    Returns a list of messages respecting the zoned token budgets.
    When a zone overflows, content is truncated from the oldest/least relevant.
    """
    window = get_context_window(model)

    # Fixed: system prompt
    sys_budget = int(window * ZONE_BUDGETS["system"])
    sys_content = _truncate_text(system_prompt, sys_budget)

    # Zone: retrieved docs
    doc_budget = int(window * ZONE_BUDGETS["retrieved"])
    doc_text = _format_docs(retrieved_docs)
    doc_text = _truncate_text(doc_text, doc_budget)

    # Zone: tool results
    tool_budget = int(window * ZONE_BUDGETS["tool"])
    tool_text = "\n\n".join(tool_results[-3:])  # keep last 3 at most
    tool_text = _truncate_text(tool_text, tool_budget)

    # Zone: conversation history
    conv_budget = int(window * ZONE_BUDGETS["conversation"])
    conv_text = _format_conversation(conversation_history)
    conv_text = _truncate_text(conv_text, conv_budget, from_end=False)

    # Build messages
    messages = [
        {"role": "system", "content": sys_content},
    ]

    if conv_text:
        messages.append({"role": "system", "content": f"[对话历史]\n{conv_text}"})

    messages.append({"role": "system", "content": f"[检索到的文献]\n{doc_text}"})

    if tool_text:
        messages.append({"role": "system", "content": f"[工具执行结果]\n{tool_text}"})

    return messages


def _format_docs(docs: list[dict]) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        title = meta.get("title", "未知")
        lines.append(
            f"[文献{i}] {title}\n{doc.get('text', '')[:400]}"
        )
    return "\n\n".join(lines)


def _format_conversation(history: list[dict]) -> str:
    lines = []
    for msg in history[-6:]:  # last 6 messages
        role = "用户" if msg["role"] == "user" else "助手"
        content = msg.get("content", "")[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _truncate_text(text: str, token_budget: int, from_end: bool = True) -> str:
    """Truncate text to fit within token budget."""
    if not text:
        return ""
    current = estimate_tokens(text)
    if current <= token_budget:
        return text
    # Truncate characters proportionally
    target_chars = token_budget * 2
    if from_end:
        return text[-target_chars:]
    else:
        return text[:target_chars]


def estimate_message_tokens(messages: list[dict]) -> int:
    """Estimate total tokens across all messages."""
    total = 0
    for m in messages:
        content = m.get("content", "") or ""
        total += estimate_tokens(content)
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                total += estimate_tokens(tc.get("function", {}).get("arguments", ""))
    return total


def trim_conversation(
    messages: list[dict],
    model: str = OLLAMA_MODEL,
    max_ratio: float = 0.75,
) -> tuple[list[dict], str]:
    """Trim conversation history to fit within token budget.

    Keeps system prompt + most recent exchanges. When trimming,
    summarizes removed messages into a short context note.

    Args:
        messages: Full message list (system + user + assistant + tool...)
        model: Model name for context window lookup
        max_ratio: Max fraction of context window for conversation (default 75%)

    Returns:
        (trimmed_messages, summary_of_removed)
    """
    window = get_context_window(model)
    budget = int(window * max_ratio)
    current = estimate_message_tokens(messages)

    if current <= budget:
        return messages, ""

    # Build from the end: keep most recent messages within budget
    # Always keep system prompt (index 0)
    sys_msg = messages[0] if messages and messages[0]["role"] == "system" else None
    history = messages[1:] if sys_msg else messages

    kept = []
    used = estimate_tokens([sys_msg]) if sys_msg else 0
    removed_count = 0

    # Walk backwards through history, keep what fits
    for m in reversed(history):
        msg_tokens = estimate_message_tokens([m])
        if used + msg_tokens <= budget:
            kept.insert(0, m)
            used += msg_tokens
        else:
            removed_count += 1
            break  # reached budget limit, discard older messages

    # Count all removed messages
    remaining = len(history) - len(kept) - removed_count
    removed_count += max(0, remaining)

    # Build summary of removed exchanges
    summary = ""
    if removed_count > 0:
        # Find the trimmed user messages for summary
        trimmed_users = [
            m.get("content", "")[:50]
            for m in history[:removed_count]
            if m.get("role") == "user"
        ]
        if trimmed_users:
            summary = f"[历史对话摘要 — 共 {removed_count} 条消息已被截断] 用户曾问过: {'; '.join(trimmed_users[:5])}"

    result = [sys_msg] if sys_msg else []
    if summary:
        result.append({"role": "system", "content": summary})
    result.extend(kept)

    return result, summary
