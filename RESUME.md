# 简历项目描述 — 宠物兽医知识助手 RAG Agent

> 投递 AI Agent 实习岗时，根据 JD 关键词灵活裁剪以下内容。

---

## 一句话总结

从零构建宠物兽医领域 RAG Agent 系统，涵盖知识库构建、混合检索、ReAct Agent 工具调用、后端API、人机协同完整链路，在消费级 GPU (RTX 4060 8GB) 上本地部署运行。

---

## 简历条目（推荐 3-4 行）

**宠物兽医知识助手 — RAG Agent 系统**  
*个人项目 | Python, Ollama, ChromaDB, FastAPI, Streamlit* | 2026.06  

- 搭建基于 **知网学术论文** 的宠物猫兽医知识 RAG 系统，实现 **ChromaDB 向量检索 + BM25 稀疏检索 + RRF 融合** 混合检索链路，1353 个知识片段覆盖 10 类猫科疾病
- 从零实现 **ReAct Agent 编排引擎**（Plan-Execute-Observe 循环），包含 3 个 Function Calling 工具（知识库检索/症状分析/急诊分级），具备失败重试（指数退避）和回答自检验证能力
- 构建 **FastAPI + SQLite** 后端服务（5 个 REST 端点），实现对话持久化、JSONL 可观测性日志（trace_id / 工具调用耗时 / token 统计）；Streamlit 前端集成 **Human-in-the-Loop** 安全机制（高风险建议弹窗确认）
- 在 **RTX 4060 8GB 消费级 GPU** 上完成模型本地部署：Ollama 推理服务 + qwen3:4b INT4 量化（~2.5GB）+ BGE-small-zh 嵌入模型，解决了 llama.cpp 中文路径编码、ChromaDB Rust 后端路径、thinking tokens 预算等工程问题

---

## JD 关键词匹配表

| 常见 JD 关键词 | 你的项目覆盖 |
|---------------|------------|
| RAG 系统搭建 | ChromaDB + BM25 + RRF 混合检索 |
| Agent 0-1 搭建 | ReAct Agent + 3 Tool + 自研编排引擎 |
| Function Calling / Tool Use | search_knowledge_base / analyze_symptoms / triage_decision |
| 向量数据库 | ChromaDB PersistentClient + score 转换 + 分块策略 |
| Prompt Engineering | 5 条硬约束 System Prompt + 上下文四区管理 + token 预算 |
| 模型推理部署 | Ollama + qwen3:4b INT4 本地 GPU 推理 |
| 混合检索 | Dense (BGE-small-zh) + Sparse (BM25 bigram) + RRF |
| FastAPI 后端 | 5 REST 端点 + SQLite WAL 持久化 |
| 可观测性 | JSONL trace 日志 + 20 题基准测试 |
| Human-in-the-Loop | Streamlit 确认弹窗 + 高风险建议拦截 |
| 人畜共患病知识边界 | 不推荐药物 + 每句话引用来源 + 诚实说"不知道" |

---

## 面试口述模板（1-2 分钟）

> "我做了一个宠物兽医 RAG Agent。起因是我自己养了一只流浪猫，发现网上宠物医疗信息鱼龙混杂，所以想做一个每句话都有论文来源的咨询助手。
>
> 技术栈是 Python + Ollama + ChromaDB + FastAPI + Streamlit。知识库用了知网畜牧兽医期刊的论文，检索层做了向量和 BM25 的混合检索加 RRF 融合。Agent 核心是从零写的 ReAct 循环，不是调 LangChain 高级 API——我理解 tool call 消息的 ID 匹配、结果回传、失败重试这些底层机制。
>
> 后端是 FastAPI + SQLite，前端 Streamlit 做了 Human-in-the-Loop——当 Agent 建议'尽快就医'时弹确认框。还做了 JSONL 可观测性日志和 20 题基准测试。
>
> 过程中踩了不少工程坑：Ollama 的 llama.cpp 不支持中文路径、ChromaDB 的 Rust 后端也有同样问题、qwen3 的 thinking 模式会占满 token 预算导致输出为空——这些都自己排查解决了。
>
> 如果面试官想看的话，我可以当场跑一下 demo。或者聊聊 Agent 架构设计、工具调用的细节。"

---

## 面试可能被追问的点

| 问题 | 准备思路 |
|------|---------|
| "为什么不用 LangChain？" | 我理解 LangChain 的 Agent 封装很好，但我选择手写是为了深入理解 tool calling 的消息机制——tool_call_id 匹配、结果回传格式、消息角色切换，这些被框架藏起来的细节。 |
| "混合检索为什么选 RRF？" | Dense 的余弦相似度和 BM25 的稀疏分数量纲不同，直接加权需要归一化。RRF 天然 scale-free，只关心排序位置。 |
| "ChromaDB vs Milvus？" | ChromaDB 轻量、零配置、Python 原生，适合这个量级（几千 chunks）和消费级 GPU 场景。如果上了百万级会考虑 Milvus。 |
| "qwen3:4b 够用吗？" | 4B 在 RAG 场景下够用——推理任务主要是信息综合和引用，不是生成复杂推理链。实际测试 20 题 65% 有效引用率，延迟平均 75s。用 8B 会更好但显存放不下。 |
| "怎么控制幻觉？" | 三道防线：1) System prompt 硬约束"找不到就说不知道" 2) 回答后自检未标注引用的医学论断 3) Human-in-the-Loop 对高风险回答拦截。 |
