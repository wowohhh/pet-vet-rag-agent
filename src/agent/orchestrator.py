"""Enhanced ReAct Agent — incremental improvements over the working baseline.

Phase 5 additions (marked with 🔧):
  - ③ Failure recovery: retry on tool failure, degrade gracefully
  - ③ Result verification: heuristic self-check on citations
  - ④ Context engineering: truncation when retrieval results overflow
  - ⑤ Observability: per-step tracing via Trace logger

Phase 7 additions (marked with 🏗️):
  - Structured output: chat_structured() returns dict alongside free text
  - Citation extraction: parse search_knowledge_base results into typed citations
  - Triage extraction: derive level/signal/reasoning from triage_decision output
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from openai import OpenAI
from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, MAX_ITERATIONS
from src.agent.prompts import SYSTEM_PROMPT, DISCLAIMER
from src.agent.tools import search_knowledge_base, analyze_symptoms, triage_decision, search_cnki, EMERGENCY_SIGNS
from src.agent.context import trim_conversation
from src.observability.logger import Trace
from src.observability.monitor import record_latency
from src.mcp.tool_provider import MCPToolProvider


# ── 🏗️ Structured output types ──────────────────────────────────────────

@dataclass
class Citation:
    title: str
    journal: str = ""
    year: str = ""
    relevant_text: str = ""


@dataclass
class TriageInfo:
    level: str = "UNKNOWN"  # EMERGENCY | URGENT | OBSERVE | UNKNOWN
    signal: str = ""
    reasoning: str = ""


@dataclass
class StructuredAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    triage: TriageInfo = field(default_factory=TriageInfo)
    requires_confirmation: bool = False
    source: str = "local"  # local | cnki | fallback
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [asdict(c) for c in self.citations],
            "triage": asdict(self.triage),
            "requires_confirmation": self.requires_confirmation,
            "source": self.source,
            "disclaimer": self.disclaimer,
        }


# ── 🏗️ Parsing helpers ──────────────────────────────────────────────────

def _extract_citations(tool_results: list[dict]) -> list[Citation]:
    """Extract structured citations from search_knowledge_base and search_cnki results."""
    citations = []
    for tr in tool_results:
        if tr["name"] not in ("search_knowledge_base", "search_cnki"):
            continue
        text = tr.get("result", "")
        is_cnki = "🌐" in text or "知网检索结果" in text

        blocks = re.split(r"### 文献 \d+", text)
        for block in blocks[1:]:
            title_match = re.search(r"\*\*标题\*\*:[ \t]*(.*?)(?:\n|$)", block)
            journal_match = re.search(r"\*\*期刊\*\*:[ \t]*(.*?)(?:\n|$)", block)
            year_match = re.search(r"\((\d{4})\)", block)
            excerpt_key = "摘要" if is_cnki else "摘录"
            excerpt_match = re.search(rf"\*\*{excerpt_key}\*\*:[ \t]*(.+?)(?:\n|$)", block)

            title = title_match.group(1).strip() if title_match and title_match.group(1).strip() else ""
            journal = journal_match.group(1).strip() if journal_match and journal_match.group(1).strip() else ""
            if not title and journal:
                title = journal  # fallback: use journal as title
            if is_cnki and not journal:
                journal = "知网网络检索"

            if title or excerpt_match:
                citations.append(Citation(
                    title=title or "未知标题",
                    journal=journal,
                    year=year_match.group(1) if year_match else "",
                    relevant_text=excerpt_match.group(1).strip()[:100] if excerpt_match else "",
                ))
    return citations


def _extract_triage(tool_results: list[dict], answer: str) -> TriageInfo:
    """Extract triage level from triage_decision tool result or answer text."""
    # Priority 1: triage_decision tool result
    for tr in tool_results:
        if tr["name"] != "triage_decision":
            continue
        result_text = tr.get("result", "")

        if "🚨 急诊建议" in result_text:
            signal_match = re.search(r"检测到危险信号「\*\*(.+?)\*\*」", result_text)
            return TriageInfo(
                level="EMERGENCY",
                signal=signal_match.group(1) if signal_match else "",
                reasoning="检测到急诊危险信号",
            )
        elif "⚠️ 建议尽快就医" in result_text:
            concern_match = re.search(r"检测到以下症状:\s*(.+?)(?:\n|$)", result_text)
            return TriageInfo(
                level="URGENT",
                signal=concern_match.group(1).strip() if concern_match else "",
                reasoning="检测到需关注的症状，建议24-48h内就医",
            )
        elif "ℹ️ 居家观察" in result_text:
            return TriageInfo(
                level="OBSERVE",
                reasoning="未检测到急诊或紧急症状，可居家观察",
            )

    # Priority 2: fallback to scanning answer text for emergency keywords
    for sign in EMERGENCY_SIGNS:
        if sign in answer:
            return TriageInfo(
                level="EMERGENCY",
                signal=sign,
                reasoning="从回答文本中检测到危险信号",
            )

    return TriageInfo(level="UNKNOWN", reasoning="无法从工具结果中确定分级")


def _determine_source(tool_results: list[dict]) -> str:
    """Determine knowledge source from tool results."""
    for tr in tool_results:
        if tr["name"] == "search_knowledge_base":
            result = tr.get("result", "")
            if "未找到相关文献" not in result:
                return "local"
    for tr in tool_results:
        if tr["name"] == "search_cnki":
            result = tr.get("result", "")
            if "未在知网找到" not in result and "搜索失败" not in result:
                return "cnki"
    return "fallback"


# ── 🏗️ MCP Tool Provider ──────────────────────────────────────────────────

# Default to MCP provider. Falls back to TOOLS dict if set to None.
_mcp_provider: MCPToolProvider | None = MCPToolProvider()


def use_mcp(enabled: bool = True) -> None:
    """Enable or disable MCP tool provider.

    When disabled, the orchestrator falls back to the direct TOOLS dict.
    This is useful for testing or environments without MCP dependencies.
    """
    global _mcp_provider
    _mcp_provider = MCPToolProvider() if enabled else None


def get_tool_provider_status() -> dict:
    """Return MCP provider status for monitoring / UI display."""
    if _mcp_provider:
        return _mcp_provider.get_status()
    return {"mode": "disabled", "reason": "using direct TOOLS dict"}


# ── Tool registry (direct mode fallback) ────────────────────────────────────

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
    "search_cnki": {
        "function": search_cnki,
        "description": "🏗️ 当本地知识库无结果时，通过网络搜索知网论文摘要作为补充。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索查询"}},
            "required": ["query"],
        },
    },
}


def _build_tools_for_api() -> list[dict]:
    """Build OpenAI-compatible tool definitions.

    Uses MCP provider when available; falls back to direct TOOLS dict.
    """
    if _mcp_provider:
        return _mcp_provider.list_tools_openai()

    # Direct mode fallback
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
    """🔧 Execute tool with retry on failure, then degrade gracefully.

    Routes through MCP provider when available; falls back to TOOLS dict.
    """
    # MCP mode: delegate to provider
    if _mcp_provider and name in _mcp_provider.get_tool_names():
        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                result = _mcp_provider.call_tool(name, args)
                return str(result)
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))
        return f"工具 '{name}' 执行失败 (MCP): {last_error}\n[系统提示] 该工具暂时不可用。"

    # Direct mode fallback
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


# ── Agent class ──────────────────────────────────────────────────────────

class VetAgent:
    """Enhanced RAG Agent with failure recovery, observability, and structured output.

    Uses MCP (Model Context Protocol) for tool discovery and execution by default.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.client = OpenAI(
            base_url=base_url or OLLAMA_BASE_URL + "/v1",
            api_key="not-needed",
        )
        self.model = model or OLLAMA_MODEL
        self.messages: list[dict] = []
        # 🏗️ Track tool results for structured output extraction
        self._last_tool_results: list[dict] = []

    @staticmethod
    def get_mcp_status() -> dict:
        """Return MCP tool provider status for monitoring."""
        return get_tool_provider_status()

    def reset(self):
        self.messages = []
        self._last_tool_results = []

    def chat(self, user_message: str, conversation_id: str = "") -> str:
        """Process user message through the ReAct agent loop.

        Returns free-text markdown answer. Use chat_structured() for typed output.
        """
        trace = Trace(query=user_message, conversation_id=conversation_id)
        self.messages.append({"role": "user", "content": user_message})
        self._last_tool_results = []
        _start_time = time.time()

        full_messages, summary = trim_conversation(
            [{"role": "system", "content": SYSTEM_PROMPT}, *self.messages],
            model=self.model,
        )

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

                    # 🏗️ Record tool result for structured output extraction
                    self._last_tool_results.append({"name": name, "result": result})

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
                record_latency((time.time() - _start_time) * 1000)
                return answer

        fallback = "抱歉，处理您的问题超时。请尝试简化问题。" + DISCLAIMER
        trace.finish(fallback)
        record_latency((time.time() - _start_time) * 1000)
        return fallback

    # ── 🏗️ Streaming output ───────────────────────────────────────────

    def chat_stream(self, user_message: str, conversation_id: str = ""):
        """Process user message with ReAct loop, yield tokens from final answer.

        Tool calls run non-streaming (need full context). When the model
        generates the final answer (no tool_calls), tokens are yielded
        as they arrive via stream=True.

        Yields:
            {"type": "tool_call", "name": str} — tool call started
            {"type": "token", "text": str} — answer token
            {"type": "done", "answer": str} — complete
        """
        trace = Trace(query=user_message, conversation_id=conversation_id)
        self.messages.append({"role": "user", "content": user_message})
        self._last_tool_results = []

        full_messages, _ = trim_conversation(
            [{"role": "system", "content": SYSTEM_PROMPT}, *self.messages],
            model=self.model,
        )

        tools = _build_tools_for_api()
        tool_used = False
        final_answer = ""

        # Phase 1: ReAct loop (non-streaming for tool calls)
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

            if hasattr(response, "usage") and response.usage:
                trace.log_generation(
                    tokens=response.usage.total_tokens or 0,
                    duration_ms=gen_duration,
                )

            if msg.tool_calls:
                tool_used = True
                for tool_call in msg.tool_calls:
                    name = tool_call.function.name
                    yield {"type": "tool_call", "name": name}

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
                    self._last_tool_results.append({"name": name, "result": result})

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
                continue  # next ReAct iteration

            # Phase 2: stream the final answer
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=0.3,
                max_tokens=4096,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    final_answer += token
                    yield {"type": "token", "text": token}

            break  # done

        # Cleanup
        if not final_answer:
            fallback = "抱歉，处理您的问题超时。请尝试简化问题。" + DISCLAIMER
            final_answer = fallback
            yield {"type": "token", "text": fallback}

        final_answer = _verify_response(final_answer, tool_used)
        self.messages.append({"role": "assistant", "content": final_answer})
        trace.finish(final_answer)
        yield {"type": "done", "answer": final_answer, "tool_results": self._last_tool_results}

    # ── 🏗️ Structured output ────────────────────────────────────────────

    def chat_structured(self, user_message: str, conversation_id: str = "") -> StructuredAnswer:
        """Process user message and return typed structured answer.

        Internally calls the same ReAct loop as chat(), then parses
        tool results into structured citations and triage info.
        """
        answer_text = self.chat(user_message, conversation_id)

        citations = _extract_citations(self._last_tool_results)
        triage = _extract_triage(self._last_tool_results, answer_text)
        source = _determine_source(self._last_tool_results)

        return StructuredAnswer(
            answer=answer_text,
            citations=citations,
            triage=triage,
            requires_confirmation=triage.level in ("EMERGENCY", "URGENT"),
            source=source,
        )


_agent: VetAgent | None = None


def get_agent() -> VetAgent:
    global _agent
    if _agent is None:
        _agent = VetAgent()
    return _agent
