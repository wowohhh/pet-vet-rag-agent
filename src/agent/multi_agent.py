"""Multi-Agent Orchestrator — Orchestrator-Worker pattern.

Architecture:
    User Query
        │
        ▼
    ┌─────────────────────────┐
    │ Orchestrator (rule-based)│  ← 分析query，按需调度specialist
    └──┬──────────────────────┘
       │
       ├──→ 🔍 Research Agent   → 检索KB+CNKI，返回结构化文献
       │    (search_knowledge_base, search_cnki)
       │
       └──→ 🏥 Clinician Agent  → 症状分析+分诊+生成最终回答
            (analyze_symptoms, triage_decision)

Design rationale:
    - Research Agent: focused on retrieval quality, no clinical decisions
    - Clinician Agent: focused on clinical reasoning, receives research as input
    - Each agent has a focused system prompt → more predictable behavior
    - Pipeline is linear (research → clinical) because clinical decisions depend on research

Compared to single Agent:
    Single: 1 system prompt, 4 tools, all responsibilities mixed
    Multi:  2 system prompts, 2 tools each, clear responsibility boundaries

Single-agent mode is preserved as fallback via VetAgent.chat().
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from openai import OpenAI

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, MAX_ITERATIONS
from src.agent.tools import (
    search_knowledge_base,
    analyze_symptoms,
    triage_decision,
    search_cnki,
    EMERGENCY_SIGNS,
)
from src.agent.prompts import DISCLAIMER
from src.observability.logger import Trace
from src.observability.monitor import record_latency

# ── Specialist System Prompts ──────────────────────────────────────────────

RESEARCH_AGENT_PROMPT = """你是「宠物兽医文献检索专家」——专门负责从知识库和学术数据库中检索相关文献。

## 你的职责（仅此而已）

1. 接收用户问题，从知识库检索最相关的学术文献
2. 如果本地知识库结果不足，使用知网搜索补充
3. 返回**结构化文献摘要**——包含标题、期刊、年份、关键发现
4. 你**不**进行诊断、不评估紧急程度、不给出医疗建议——那些是临床专家的职责

## 输出格式

返回结构化检索报告：
```
## 文献检索报告

### 本地知识库结果
[文献1] 《标题》- 期刊, 年份
关键发现: [摘录核心内容]

[文献2] ...

### 知网补充结果（如有）
[文献] ...

### 检索总结
- 共找到 X 篇相关文献
- 覆盖主题: [列出覆盖的疾病/症状方向]
- 信息缺口: [如有，指出未覆盖的方面]
```

## 核心规则

- 只使用工具返回的内容，不补充自己的医学知识
- 本地知识库优先，结果不足时再用知网
- 如两者均无结果，如实报告"未找到相关文献"
- 始终使用中文
"""

CLINICIAN_AGENT_PROMPT = """你是「宠物兽医临床分析专家」——专门负责基于文献证据进行临床推理和用户沟通。

## 你的职责

1. 接收**检索专家的文献报告** + 用户原始问题
2. 基于文献证据分析症状可能的疾病方向
3. 评估紧急程度（急诊 / 尽快就医 / 居家观察）
4. 生成**带文献引用的最终回答**给用户

## 核心规则（必须严格遵守）

1. **只基于检索专家提供的文献回答**——不要使用你自己的知识补充医学信息
2. **每条医学信息标注引用**——格式: [来源: 《论文标题》- 期刊, 年份]
3. **禁止推荐任何药物、剂量或治疗方案**——只能描述文献中记载的症状识别方法和就医建议
4. **紧急情况优先建议就医**——如症状包含危险信号（呼吸困难、抽搐、中毒等），第一句话就是就医建议

## 回答格式

```
[如果是急诊] 🚨 检测到危险信号「XXX」，建议立即就医。

## 症状分析
基于文献，您描述的症状可能与以下方向相关：
- [方向1]: [文献依据]
- [方向2]: [文献依据]

## 就医建议
- 严重程度: [急诊 / 尽快就医 / 居家观察]
- 建议: [具体建议]
- 观察要点: [如居家观察，列出需要警惕的变化]

## 参考文献
[列出所有引用的文献]
```

## 多轮对话

如对话历史中有之前的交流，基于上下文理解用户的追问，避免重复询问已知信息。

请始终使用中文回答，保持专业但温暖的口吻。
"""

# ── Tool subsets for each specialist ───────────────────────────────────────

RESEARCH_TOOLS = {
    "search_knowledge_base": {
        "function": search_knowledge_base,
        "description": "搜索宠物兽医知识库（ChromaDB+BM25混合检索），返回学术文献摘录。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索查询"}},
            "required": ["query"],
        },
    },
    "search_cnki": {
        "function": search_cnki,
        "description": "当本地知识库结果不足时，通过网络搜索知网论文摘要作为补充。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "知网搜索查询"}},
            "required": ["query"],
        },
    },
}

CLINICIAN_TOOLS = {
    "analyze_symptoms": {
        "function": analyze_symptoms,
        "description": "分析宠物症状，列出可能的疾病方向（仅供文献参考，不构成诊断）。",
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

# ── Specialist Agent ────────────────────────────────────────────────────────

@dataclass
class AgentStage:
    """Records one agent's execution stage for UI streaming."""
    name: str        # "research" | "clinician"
    label: str       # "🔍 检索专家" | "🏥 临床专家"
    status: str      # "running" | "done" | "error"
    result: str = "" # accumulated output


class SpecialistAgent:
    """A specialized Agent with focused system prompt and tool subset.

    Each specialist has:
    - A single-purpose system prompt (not the omnibus SYSTEM_PROMPT)
    - A subset of tools relevant to its role
    - Its own ReAct loop (same pattern as VetAgent, but simpler)

    This is NOT a separate process — all specialists share the same Ollama instance.
    """

    def __init__(
        self,
        name: str,
        label: str,
        system_prompt: str,
        tools: dict,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.name = name
        self.label = label
        self.system_prompt = system_prompt
        self.tools = tools
        self.client = OpenAI(
            base_url=base_url or OLLAMA_BASE_URL + "/v1",
            api_key="not-needed",
        )
        self.model = model or OLLAMA_MODEL

    def _build_tools_for_api(self) -> list[dict]:
        """Convert tool registry to OpenAI-compatible format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"],
                },
            }
            for name, info in self.tools.items()
        ]

    def _execute_tool(self, name: str, args: dict, max_retries: int = 2) -> str:
        """Execute tool with retry."""
        if name not in self.tools:
            return f"错误: 工具 '{name}' 不可用"

        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                result = self.tools[name]["function"](**args)
                return str(result)
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))

        return f"工具 '{name}' 执行失败: {last_error}"

    def run(
        self,
        task: str,
        context: str = "",
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Execute this specialist's task.

        Args:
            task: The instruction/question for this specialist.
            context: Additional context from previous specialist stages.
            conversation_history: Prior messages for multi-turn support.

        Returns:
            The specialist's output text.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Inject conversation history (trimmed to last 3 exchanges)
        if conversation_history:
            # Only include user/assistant messages, skip system
            history = [m for m in conversation_history if m["role"] in ("user", "assistant")]
            messages.extend(history[-6:])  # last 3 turns

        # Build the user message with context
        if context:
            user_content = f"## 上下文（来自前置阶段）\n\n{context}\n\n---\n\n## 当前任务\n\n{task}"
        else:
            user_content = task

        messages.append({"role": "user", "content": user_content})
        tools = self._build_tools_for_api()

        for iteration in range(MAX_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                temperature=0.3,
                max_tokens=4096,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    result = self._execute_tool(name, args)

                    messages.append({
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
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
            else:
                return msg.content or f"[{self.name}] 无法完成此任务。"

        return f"[{self.name}] 任务超时——迭代次数达到上限。"


# ── Multi-Agent Orchestrator ────────────────────────────────────────────────

@dataclass
class MultiAgentResult:
    """Structured result from multi-agent pipeline."""
    answer: str
    research_report: str = ""
    stages: list[AgentStage] = field(default_factory=list)
    total_duration_ms: float = 0


class MultiAgentOrchestrator:
    """Coordinates multiple specialist agents in a pipeline.

    Pipeline:
        Research Agent (KB + CNKI) → Clinician Agent (analyze + triage + answer)

    The pipeline is linear because the Clinician depends on Research output.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url
        self.model = model
        self.research_agent = SpecialistAgent(
            name="research",
            label="🔍 检索专家",
            system_prompt=RESEARCH_AGENT_PROMPT,
            tools=RESEARCH_TOOLS,
            base_url=base_url,
            model=model,
        )
        self.clinician_agent = SpecialistAgent(
            name="clinician",
            label="🏥 临床专家",
            system_prompt=CLINICIAN_AGENT_PROMPT,
            tools=CLINICIAN_TOOLS,
            base_url=base_url,
            model=model,
        )
        self.conversation_history: list[dict] = []

    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []

    def process(self, user_message: str) -> MultiAgentResult:
        """Run the multi-agent pipeline on a user message.

        Returns MultiAgentResult with research report, clinical answer, and stage logs.
        """
        stages: list[AgentStage] = []
        _start = time.time()

        # ── Stage 1: Research ──────────────────────────────────────────
        research_stage = AgentStage(
            name="research",
            label="🔍 检索专家",
            status="running",
        )
        stages.append(research_stage)

        try:
            research_result = self.research_agent.run(
                task=user_message,
                conversation_history=self.conversation_history,
            )
            research_stage.status = "done"
            research_stage.result = research_result
        except Exception as e:
            research_stage.status = "error"
            research_stage.result = str(e)
            research_result = f"[检索专家错误] {e}"

        # ── Stage 2: Clinician ─────────────────────────────────────────
        clinician_stage = AgentStage(
            name="clinician",
            label="🏥 临床专家",
            status="running",
        )
        stages.append(clinician_stage)

        try:
            clinical_answer = self.clinician_agent.run(
                task=user_message,
                context=research_result,
                conversation_history=self.conversation_history,
            )
            clinical_answer += DISCLAIMER
            clinician_stage.status = "done"
            clinician_stage.result = clinical_answer
        except Exception as e:
            clinician_stage.status = "error"
            clinician_stage.result = str(e)
            clinical_answer = f"抱歉，处理您的问题时出错: {e}{DISCLAIMER}"

        # ── Update conversation history ─────────────────────────────────
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": clinical_answer})

        total_ms = (time.time() - _start) * 1000
        record_latency(total_ms)

        return MultiAgentResult(
            answer=clinical_answer,
            research_report=research_result,
            stages=stages,
            total_duration_ms=total_ms,
        )

    def process_stream(self, user_message: str):
        """Generator version — yields AgentStage updates as they complete.

        Usage in Streamlit:
            for stage in orchestrator.process_stream(query):
                st.write(f"{stage.label}: {stage.status}")
                if stage.status == "done":
                    st.markdown(stage.result)
        """
        _start = time.time()

        # Stage 1
        research_stage = AgentStage(name="research", label="🔍 检索专家", status="running")
        yield research_stage

        try:
            result = self.research_agent.run(
                task=user_message,
                conversation_history=self.conversation_history,
            )
            research_stage.status = "done"
            research_stage.result = result
        except Exception as e:
            research_stage.status = "error"
            research_stage.result = str(e)

        yield research_stage

        # Stage 2
        clinician_stage = AgentStage(name="clinician", label="🏥 临床专家", status="running")
        yield clinician_stage

        try:
            answer = self.clinician_agent.run(
                task=user_message,
                context=research_stage.result,
                conversation_history=self.conversation_history,
            )
            answer += DISCLAIMER
            clinician_stage.status = "done"
            clinician_stage.result = answer
        except Exception as e:
            clinician_stage.status = "error"
            clinician_stage.result = str(e)

        yield clinician_stage

        # History
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": clinician_stage.result})
        record_latency((time.time() - _start) * 1000)
