# 工程日志 — 宠物兽医 RAG Agent

> 记录项目中遇到的每个非预期行为：现象 → 根因 → 解决 → 启示。
> 面试时每条都是一个独立的 debug 故事。

---

## 1. Ollama 中文路径导致 llama.cpp 加载失败

**现象**: qwen3:4b 通过 Ollama 无法加载，服务端日志 `llama_model_loader: failed to load model`。

**排查过程**: 检查 Ollama 日志 → 定位 `C:\Users\芋頭粥粥\.ollama\models\blobs\...` 路径 → 怀疑中文用户名 → 把模型文件移到 `C:\ollama_models` → 设置 `OLLAMA_MODELS=C:/ollama_models` 环境变量 → 重启 ollama serve → 正常加载。

**根因**: llama.cpp 的 Windows 文件 I/O 不支持非 ASCII 路径。

**解决**: `setx OLLAMA_MODELS "C:/ollama_models"` 持久化，避开中文路径。

**启示**: Rust/C++ 底层库在 Windows 中文环境下的编码问题具有共性。后续 ChromaDB 遇到同类问题（见 #5），直接定位到 Rust 后端的路径处理。

**关键词**: llama.cpp, Windows 编码, GGUF, Ollama, 非 ASCII 路径

---

## 2. qwen3:4b thinking 模式 Token 耗尽

**现象**: Agent 返回空内容，但 Ollama 日志显示推理已完成。

**排查过程**: Ollama API 返回 `content: ""`，检查 `response.usage` 发现 `completion_tokens` 等于 `max_tokens=1024` → 怀疑 qwen3:4b 的 `thinking` 模式。

**根因**: qwen3:4b 默认启用内部推理链（thinking），thinking tokens 和 output tokens 共用 `max_tokens` 预算。max_tokens=1024 时全部被 thinking 消耗，output 为空。

**解决**: `max_tokens=4096`，确保 thinking + output 都有空间。

**启示**: 推理模型的 token 分配机制需要理解——thinking 不是"额外"的，是从 output 预算里扣的。生产环境需要对不同模型设置不同的 max_tokens 策略。

**关键词**: qwen3, thinking mode, token budget, Ollama API

---

## 3. Ollama API content=None 拒绝

**现象**: Agent 的 tool call 消息发送到 Ollama 时报错。

**根因**: Ollama 的 OpenAI 兼容 API 不接受 `content: None`，要求至少 `content: ""`。

**解决**: 在构建 tool call 的 assistant 消息时，用 `content: ""` 替代 `content: None`。

**启示**: "兼容 API" 不等于 100% 兼容。测试时要覆盖 tool calling 的完整消息流。

**关键词**: Ollama, OpenAI API 兼容, tool calling, content field

---

## 4. build_context() 过度工程化导致 Agent 超时

**现象**: Phase 5 首次引入上下文工程后，Agent 跑满 MAX_ITERATIONS 无输出。

**排查过程**: 原来的消息流是 `system → user → tool_calls → tool_results`，新增的 build_context() 把所有内容塞成 4 条 system 消息 → 模型在面对单一超长 system prompt 时反复尝试 tool call 但无法收敛 → 5 轮耗尽。

**根因**: 过度设计——把 system/retrieved/tool/conversation 四个 zone 全部塞进 system prompt，破坏了 ReAct 消息循环的结构。

**解决**: 回滚到增量改进方案——保留原始消息流结构，只加 _truncate_tool_result（截断）、_execute_tool_with_retry（重试）、_verify_response（验证）。

**启示**: 系统 prompt 不是越丰富越好。ReAct 的消息角色结构（system → user → assistant[+tool_calls] → tool）是有意义的——不要为了"上下文工程"破坏它。

**关键词**: context engineering, ReAct, system prompt, incremental improvement

---

## 5. ChromaDB Rust 后端中文路径问题

**现象**: 查询 ChromaDB 时报 `Error loading hnsw index`，Agent 所有工具调用失败。

**排查过程**: 清除 ChromaDB 目录重建后仍报错 → 排除数据损坏 → 注意到路径 `C:\Users\芋頭粥粥\Desktop\rag\data\chroma_db` 含中文 → 联想到 Ollama 的中文路径问题（#1）→ ChromaDB 的 Rust 后端同样使用底层文件 API → 确认是同类问题。

**根因**: ChromaDB 的 Rust segment reader 无法处理 Windows 中文路径，与 llama.cpp 同源。

**解决**: `CHROMA_DIR = Path("C:/rag_data/chroma_db")`，完全避开中文路径。

**启示**: 同一类问题在不同组件上以不同表象出现（#1 是 "failed to load model"，#5 是 "Error loading hnsw index"）。底层原因相同——Rust 生态在 Windows 中文环境下的路径处理。排查经验可复用。

**关键词**: ChromaDB, HNSW index, Rust backend, 中文路径, cross-component pattern

---

## 6. CNKI PDF 下载 ≠ curl 能搞定

**现象**: 用户已登录知网机构账号，用 curl 访问论文页面只能拿到 HTML 阅读器页面，拿不到 PDF。

**根因**: CNKI 的 PDF 下载按钮触发 JS 动态生成 `bar.cnki.net` 的临时下载链接，且需要滑块验证码。curl 无 JS 执行能力，无法模拟这一流程。

**解决**: 通过 Chrome DevTools MCP（`chrome-devtools-mcp` + Playwright Chromium）操控浏览器，在已登录的知网会话中自动化下载。20 篇 CNKI 论文成功下载。

**启示**: 国内学术平台的 PDF 获取流程不是简单的 HTTP 请求——JS 渲染 + 验证码 + 动态链接生成。自动化需要 browser-level 工具（CDP/Playwright），不是 curl/requests。

**配置要点**: `.mcp.json` 中配置 `chrome-devtools-mcp --browserUrl http://127.0.0.1:9222`，Chromium 需以 `--remote-debugging-port=9222` 启动。

**关键词**: CNKI, PDF download, Chrome DevTools Protocol, MCP, captcha

---

## 7. 结构化输出引用提取：`\s*` 吞掉换行符

**现象**: `_extract_citations` 解析 `search_knowledge_base` 结果时，提取到的标题字段包含 `**期刊**: (2017)` 而非真实标题。

**根因**: 正则 `\*\*标题\*\*:\s*(.*?)` 中的 `\s*` 匹配了换行符，导致 `(.*?)` 跨越了空标题行，匹配到下一行的 `**期刊**` 字段。

**解决**: 将 `\s*` 替换为 `[ \t]*`（仅匹配水平空白符）。

**启示**: Python 正则中 `\s` 包含 `\n`——在解析逐行结构化文本时必须注意。这是写解析器时的常见陷阱。

**关键词**: regex, \s, newline, citation parsing

---

## 8. Ollama serve env var 重启后丢失

**现象**: Ollama 服务在系统重启或新 shell 中重新以中文路径加载模型，导致 llama.cpp 报错。

**根因**: bash 环境中 `export OLLAMA_MODELS` 只在当前 shell 生效。Ollama app 启动 ollama serve 时继承不到该环境变量。

**解决**: 每次启动 Ollama 前先 `export OLLAMA_MODELS=C:/ollama_models`，或在启动脚本中设置。`setx` 永久设置对 GUI 应用可能不生效（需重启 Windows）。

**启示**: Windows 环境下 GUI 应用和命令行工具的环境变量传递链路不同。持久化环境变量后需确认目标进程实际读取到。

**关键词**: Ollama, env var, Windows, startup, persistence

---

## 9. 流式输出：Streamlit 不支持 SSE 原生的替代方案

**现象**: 需要实现 P0 流式输出降低用户感知延迟，Streamlit 不原生支持 SSE。

**方案**: 不通过 HTTP API→SSE 路径，而是在 Streamlit 进程内直接调用 `chat_stream()` generator，用 `st.empty()` placeholder 逐 token 更新 UI。

**启示**: Streamlit 的架构决定了它不适合做实时流式——但 `st.write_stream()` 和手动 placeholder 方案可以做到近似效果。生产环境会用 FastAPI SSE + 前端 WebSocket。

**关键词**: streaming, SSE, Streamlit, real-time, UI rendering

---

## 10. 结构化输出：`\s` 正则陷阱（编排层）

**现象**: `_extract_citations()` 解析 `search_knowledge_base` 结果时，标题字段被跨行匹配到期刊字段。

**根因**: Python 正则 `\s` 包含 `\n`，`\*\*标题\*\*:\s*(.*?)` 在空标题场景下跳过换行符，捕获到下一行内容。

**解决**: `\s*` → `[ \t]*`，仅匹配水平空白。

**启示**: 逐行结构化文本解析时，必须注意 `\s` 的换行语义。防御性正则习惯：用 `[ \t]` 而非 `\s` 做字段内空白匹配。

**关键词**: regex, \s, structured parsing, Python

---

## 11. MCP 集成：工具标准化与协议解耦（架构演进）

**现象**: 4 个工具的手写 JSON Schema 和调度逻辑分散在 tools.py 和 orchestrator.py 中。新增一个工具需要改 3 处代码（函数定义 + TOOLS 字典 + 参数 schema），外部工具（如 Claude Code）无法直接调用项目工具。

**根因**: 工具定义与调度逻辑紧耦合。OpenAI Function Calling schema 是项目内部格式，不兼容 MCP 协议。

**解决**: 引入 MCP (Model Context Protocol) 作为工具标准化层——
1. `src/mcp/server.py`: 4 个工具包装为 MCP Server，支持 stdio 传输，任何 MCP 客户端可发现并调用
2. `src/mcp/tool_provider.py`: MCPToolProvider 提供统一接口（`list_tools_openai()` + `call_tool()`），orchestrator 不关心工具来自直接调用还是 MCP 协议
3. orchestrator 添加 fallback 机制：MCP provider 可用时走 MCP，不可用时降级回直接 TOOLS 字典
4. Streamlit sidebar 显示 MCP 状态（模式、工具数）

**启示**: MCP 本质是工具层的"接口隔离"——和手写 JSON Schema 的功能完全等价，但架构收益巨大。面试时可以讲两段故事：①"先手写理解底层" ②"再标准化为 MCP 实现解耦"。新增工具只需写一个 MCP Server，核心代码零改动。

**关键词**: MCP, Model Context Protocol, tool standardization, architecture evolution, protocol decoupling

---

## 12. Multi-Agent 拆分：从单 Agent 全能型到双 Agent 协作（架构演进）

**现象**: 单 Agent（VetAgent）承载所有职责——检索、症状分析、分诊、回答生成——共用一个 system prompt。随功能增长，prompt 越来越长，行为边界模糊。面试时 JD 普遍要求 Multi-Agent 经验（6/10 JD）。

**根因**: 单 Agent 的 system prompt 是"全能型"设计——一条 prompt 同时描述检索策略、临床推理规则、用户沟通格式。职责混在一起导致：① prompt 修改时容易引入冲突 ② 工具调用顺序不可控（检索和分析可能同时触发） ③ 无法独立优化单个环节。

**解决**: 引入 Orchestrator-Worker 双 Agent 架构——
1. **Research Agent（检索专家）**: 专注文献检索（KB + CNKI），输出结构化检索报告。只持有 2 个检索类工具。
2. **Clinician Agent（临床专家）**: 接收检索报告 + 用户问题，负责症状分析 + 分诊 + 回答生成。只持有 2 个临床类工具。
3. **MultiAgentOrchestrator**: rule-based 调度器，按 research → clinical 固定流水线执行。
4. Streamlit sidebar 增加 Multi-Agent 模式开关，开启后显示双 Agent 实时工作状态和中间检索报告。

**启示**: Multi-Agent 的价值不在效率（反而多了 1 次 LLM 调用），而在**职责分离**和**可演进性**。检索专家可以独立升级（换 embedding、换检索策略）不影响临床逻辑。面试核心故事："单 Agent prompt 膨胀到一定规模后行为不稳定，拆成专职 Agent 后每个都有清晰的职责边界。"

**关键词**: Multi-Agent, Orchestrator-Worker, specialist agents, responsibility separation, architecture evolution

---

## 总结

| # | 问题 | 根因类型 | 通用度 |
|---|------|---------|--------|
| 1 | Ollama 中文路径 | Rust/C++ Windows 编码 | 高 |
| 2 | thinking token 耗尽 | 模型特性理解不足 | 高 |
| 3 | content=None 拒绝 | API 兼容性细节 | 中 |
| 4 | context 过度工程 | 架构设计失误 | 高 |
| 5 | ChromaDB HNSW 错误 | 同 #1（Rust 路径编码） | 高 |
| 6 | CNKI 无法 curl 下载 | 平台特化 + 自动化策略 | 中 |
| 7 | `\s` 吞换行导致引用解析错误 | 正则语义理解 | 中 |
| 8 | Ollama env var 重启丢失 | Windows 环境变量传递 | 中 |
| 9 | Streamlit 不支持流式 SSE | 框架架构限制 | 低 |
| 10 | 结构化输出字段提取 | 同 #7（正则语义） | 中 |
| 11 | MCP 工具标准化与协议解耦 | 架构演进设计 | 高 |
| 12 | Multi-Agent 双 Agent 职责分离 | 架构演进设计 | 高 |
