"""Enhanced ReAct Agent — incremental improvements over the working baseline.

Phase 5 additions (marked with 🔧):
  - ③ Failure recovery: retry on tool failure, degrade gracefully
  - ③ Result verification: heuristic self-check on citations
  - ④ Context engineering: truncation when retrieval results overflow
  - ⑤ Observability: per-step tracing via Trace logger
"""

import json
import time
from openai import OpenAI
from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, MAX_ITERATIONS
from src.agent.prompts import SYSTEM_PROMPT, DISCLAIMER
from src.agent.tools import search_knowledge_base, analyze_symptoms, triage_decision
from src.observability.logger import Trace

TOOLS = {
    "search_knowledge_base": {
        "function": search_knowledge_base,
        "description": "搜索宠物兽医知识库，返回相关学术文献摘录。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索查询"}},
            "required": ["query"],
        },
    },
    "analyze_symptoms": {
        "function": analyze_symptoms,
        "description": "分析宠物症状，列出可能的疾病方向。仅供参考，不构成诊断。",
        "parameters": {
            "type": "object",
            "properties": {"symptoms": {"type": "string", "description": "症状描述"}},
            "required": ["symptoms"],
        },
    },
    "triage_decision": {
        "function": triage_decision,
        "description": "根据症状严重程度给出就医紧迫性建议（急诊/尽快就医/居家观察）。",
        "parameters": {
            "type": "object",
            "properties": {"symptoms": {"type": "string", "description": "症状描述"}},
            "required": ["symptoms"],
        },
    },
}


def _build_tools_for_api() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"],
            },
        }
        for name, info in TOOLS.items()
    ]


def _execute_tool_with_retry(name: str, args: dict, max_retries: int = 2) -> str:
    """🔧 Execute tool with retry on failure, then degrade gracefully."""
    if name not in TOOLS:
        return f"错误: 未知工具 '{name}'"

    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            result = TOOLS[name]["function"](**args)
            return str(result)
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))

    return f"工具 '{name}' 执行失败: {last_error}\n[系统提示] 该工具暂时不可用。"


def _truncate_tool_result(result: str, max_chars: int = 2000) -> str:
    """🔧 Truncate oversized tool results to preserve context window."""
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + f"\n\n[... 结果过长，已截断，共 {len(result)} 字符]"


def _verify_response(answer: str, tool_used: bool) -> str:
    """🔧 Self-check: flag unsubstantiated claims."""
    unverified = ["推荐用药", "建议服用", "可以吃", "建议使用"]
    if tool_used:
        for kw in unverified:
            if kw in answer and "来源" not in answer:
                answer += f"\n\n[自检提示] 包含'{kw}'但未标注引用来源，请谨慎对待。"
    return answer


class VetAgent:
    """Enhanced RAG Agent with failure recovery and observability."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.client = OpenAI(
            base_url=base_url or OLLAMA_BASE_URL + "/v1",
            api_key="not-needed",
        )
        self.model = model or OLLAMA_MODEL
        self.messages: list[dict] = []

    def reset(self):
        self.messages = []

    def chat(self, user_message: str, conversation_id: str = "") -> str:
        """Process user message through the ReAct agent loop."""
        trace = Trace(query=user_message, conversation_id=conversation_id)
        self.messages.append({"role": "user", "content": user_message})

        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.messages,
        ]

        tools = _build_tools_for_api()
        tool_used = False

        for iteration in range(MAX_ITERATIONS):
            gen_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                temperature=0.3,
                max_tokens=4096,
            )
            gen_duration = (time.time() - gen_start) * 1000

            choice = response.choices[0]
            msg = choice.message

            # Log token usage if available
            if hasattr(response, "usage") and response.usage:
                trace.log_generation(
                    tokens=response.usage.total_tokens or 0,
                    duration_ms=gen_duration,
                )

            if msg.tool_calls:
                tool_used = True
                for tool_call in msg.tool_calls:
                    name = tool_call.function.name

                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    tool_start = time.time()
                    result = _execute_tool_with_retry(name, args)
                    result = _truncate_tool_result(result)
                    tool_duration = (time.time() - tool_start) * 1000

                    trace.log_tool_call(
                        name,
                        success="错误" not in result[:50],
                        duration_ms=tool_duration,
                    )

                    full_messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": tool_call.function.arguments,
                            },
                        }],
                    })
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
            else:
                answer = msg.content or "抱歉，我无法处理这个问题。"
                answer = _verify_response(answer, tool_used)
                answer += DISCLAIMER

                self.messages.append({"role": "assistant", "content": answer})
                trace.finish(answer)
                return answer

        fallback = "抱歉，处理您的问题超时。请尝试简化问题。" + DISCLAIMER
        trace.finish(fallback)
        return fallback


_agent: VetAgent | None = None


def get_agent() -> VetAgent:
    global _agent
    if _agent is None:
        _agent = VetAgent()
    return _agent
