# 🐱 宠物兽医知识助手 — RAG Agent

> **基于知网学术文献的猫咪健康 RAG Agent · 本地量化模型推理 · 全链路工程实现**
>
> 每句话都有出处 · 高风险建议需人工确认 · 不推荐药物

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Ollama](https://img.shields.io/badge/LLM-qwen3:4b-green)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)](https://fastapi.tiangolo.com)

---

## 📋 目录

1. [项目动机](#项目动机)
2. [系统架构](#系统架构)
3. [快速开始](#快速开始)
4. [Agent 核心能力](#agent-核心能力)
5. [面试6标准自查清单](#面试6标准自查清单)
6. [JD 关键词映射](#jd-关键词映射)
7. [基准测试对比](#基准测试对比)
8. [项目结构](#项目结构)
9. [技术亮点与面试故事](#技术亮点与面试故事)
10. [已知限制与改进方向](#已知限制与改进方向)

---

## 项目动机

我是一名应用化学专业大三学生，正在转向 AI Agent 实习生岗位。在调研了 20+ Agent JD 后，我发现几乎所有岗位都要求：

- **RAG 系统搭建经验**
- **Agent 0-1 搭建能力**（Function Calling / Tool Use）
- **模型推理部署**（vLLM / Ollama）
- **向量数据库优化**
- **Prompt Engineering**

这个项目从零开始构建了一个**完整 RAG Agent 系统**，在消费级 GPU（RTX 4060 8GB）上本地运行，聚焦宠物猫兽医知识领域——我自己养了一只流浪猫，有真实的用户痛点。

**这不是又一个 LangChain Demo。** 这是一个具备工程深度的 Agent 系统：

- 后端 API + 数据库持久化
- ReAct Agent + 工具调用 + 失败恢复 + 自检验证
- 混合检索（Dense + Sparse + RRF 融合）
- 结构化可观测性（JSONL trace 日志）
- Human-in-the-Loop 安全机制

---

## 系统架构

```
┌─────────────────────────────────────────────────┐
│                 Streamlit UI                      │
│    Chat Interface · Source Citations · Pet Profile│
│    ⚠️ Human-in-the-Loop Confirmation              │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│              FastAPI Backend                      │
│    POST /api/chat  ·  GET /api/conversations     │
│    DELETE /api/conversations/{id}  ·  /api/health│
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│           🧠 Agent Orchestrator                  │
│                                                    │
│  ReAct Loop (max 5 iterations):                   │
│    1. Think → 2. Act (tool call) → 3. Observe    │
│                                                    │
│  ┌───────────┐ ┌────────────┐ ┌──────────────┐  │
│  │ search_    │ │ analyze_   │ │ triage_      │  │
│  │ knowledge_ │ │ symptoms   │ │ decision     │  │
│  │ base       │ │            │ │              │  │
│  │ Hybrid     │ │ Structured │ │ Emergency    │  │
│  │ Retrieval  │ │ Analysis   │ │ Triage       │  │
│  └─────┬─────┘ └─────┬──────┘ └──────┬───────┘  │
│        │             │              │            │
│  ┌─────▼─────────────▼──────────────▼───────┐   │
│  │  🔧 Failure Recovery (retry + degrade)    │   │
│  │  🔍 Result Verification (citation check)  │   │
│  │  📐 Context Truncation (cap at 2000 char) │   │
│  │  📊 Observability (JSONL trace logging)   │   │
│  └───────────────────────────────────────────┘   │
└────────────┬──────────────────┬───────────────────┘
             │                  │
    ┌────────▼───────┐  ┌──────▼──────────────┐
    │  LLM Service   │  │  Knowledge Base      │
    │  Ollama        │  │  ChromaDB + BM25     │
    │  qwen3:4b Q4   │  │  BGE-small-zh (512d) │
    │  ~2.5GB VRAM   │  │  1356 chunks         │
    └────────────────┘  └─────────────────────┘
```

---

## 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| Python | 3.12+ |
| GPU | NVIDIA RTX 4060 8GB+ (或兼容) |
| Ollama | 已安装运行，加载 qwen3:4b |
| 显存 | 嵌入模型 ~1GB + LLM ~2.5GB ≈ 4GB，留余量 |

### 安装

```bash
# 1. 启动 Ollama 并拉取模型
ollama serve
ollama pull qwen3:4b

# 2. 安装依赖
pip install -e .

# 3. 下载嵌入模型（从 ModelScope，无需 HuggingFace）
python -c "
from modelscope import snapshot_download
snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='data/models')
"

# 4. 导入论文到知识库
# 将 CNKI PDF 论文放入 data/papers/ 目录
python -m src.ingest

# 5. 启动 API 服务
uvicorn src.api.main:app --reload

# 6. 启动 UI
streamlit run src/ui/app.py
```

### 测试命令行对话（无需 UI）

```bash
python -m src.cli
```

---

## Agent 核心能力

### 三个 Tool

| Tool | 功能 | 关键技术 |
|------|------|---------|
| `search_knowledge_base` | 检索知识库 | ChromaDB 向量检索 + BM25 稀疏检索 + RRF 融合 |
| `analyze_symptoms` | 分析症状 | 结构化提取症状 + 可能疾病方向 + 文献依据 |
| `triage_decision` | 就医决策 | 急诊信号检测 → 急诊/尽快就医/居家观察 三级 |

### 安全 Guardrails

- ❌ **不推荐任何药物**（包括人用药、处方药）
- ✅ **每句话都要有文献来源**
- ✅ **知识库没有的信息诚实说不知道**
- ✅ **高风险情况（急诊/尽快就医）触发人工确认**

### Human-in-the-Loop

当 Agent 判断为"尽快就医"或更高风险时，Streamlit UI 会**弹出确认框**，用户必须手动确认采纳或拒绝后，回答才会被记录。

---

## 面试6标准自查清单

> 来源：抖音"什么样的 Agent 项目才有含金量"视频 + 面试官视角

| # | 标准 | 本项目实现 | 文件/位置 |
|---|------|-----------|----------|
| ① | **解决真实问题** | ✅ 宠物猫健康知识助手，自己有猫有痛点；知识库基于真实学术文献 | `data/papers/` |
| ② | **完整后端工程** | ✅ FastAPI + SQLite 持久化，5个REST端点，分层架构 | `src/api/` |
| ③ | **Agent 核心能力** | ✅ ReAct Agent + 3 Tool + 失败重试(指数退避) + 降级策略 + 自检验证 | `src/agent/orchestrator.py` |
| ④ | **上下文工程** | ✅ 四区上下文管理(system/retrieved/tool/conversation) + token预算控制 + 截断策略 | `src/agent/context.py` |
| ⑤ | **可观测性** | ✅ JSONL trace日志(trace_id/每步耗时/token/tool调用) + 20题基准测试对比 | `src/observability/logger.py` |
| ⑥ | **人机协同** | ✅ Streamlit确认框 + 免责声明 + 对话审计日志 | `src/ui/app.py` |

---

## JD 关键词映射

> 从 20+ Agent 实习 JD 中提取的核心关键词，逐项覆盖

| JD 关键词 | 本项目覆盖 | 面试话术要点 |
|-----------|-----------|-------------|
| **RAG 系统搭建** | ChromaDB + BM25 混合检索 + RRF 融合 + chunk 策略调优 | "从 chunk 策略设计到混合检索融合排序，完整的检索链路" |
| **Agent 0-1 搭建** | 完整的 ReAct Agent，从零实现 tool calling 循环 | "基于 OpenAI 兼容 API 手工实现 ReAct 循环，没用 LangChain 高级封装" |
| **Function Calling / Tool Use** | 3 个 Tool 定义 + 自动 tool_choice + 结果处理 | "理解 tool call message 的 ID 匹配和结果回传机制" |
| **向量数据库** | ChromaDB PersistentClient + 自定义 score 转换 | "选择了轻量持久化方案，理解了向量检索的 score 语义" |
| **Prompt Engineering** | 5条硬约束 system prompt + citation 格式 + 多 zone 上下文 | "prompt 是最重要的基础设施——设计了多层约束和上下文预算" |
| **模型推理部署** | Ollama 本地部署 qwen3:4b INT4 | "在消费级 GPU 上跑通了完整的 Agent 推理链路" |
| **混合检索** | Dense(BGE-small-zh) + Sparse(BM25 bigram) + RRF | "理解了 dense/sparse 各自的优势，通过 RRF 互补" |
| **Python 后端** | FastAPI + SQLite + WAL 模式 | "生产级的 API 设计，不是 Demo 级别的 print 输出" |

---

## 基准测试对比

### 测试设计

- 20 题测试集，覆盖 5 个类别：症状咨询、紧急判断、疾病知识、预防护理、边界测试
- 分别用 qwen3:4b (本地 INT4) 和 GPT-4o-mini (云端) 运行
- 对比维度：回答质量、引用准确率、响应延迟

### 结果 (qwen3:4b 本地 INT4, RTX 4060 8GB)

| 指标 | qwen3:4b (本地) |
|------|----------------|
| 平均延迟 | 74.8s |
| 有效引用率 | 65% (13/20 有文献引用) |
| 诚实拒答率 | 35% (7/20 知识库未覆盖，正确说"不知道") |
| 急诊识别率 | 100% (3/3 急诊题正确触发就医建议) |
| 边界测试通过 | 100% (人用药拒绝 + FIV诚实 + FIP时效性) |
| 总显存占用 | ~4GB (Embedding ~1GB + LLM ~2.5GB) |

**分析**: 知识库内容覆盖了猫上呼吸道、FIP、FLUTD、疫苗、营养、皮肤病、寄生虫等领域。但对猫糖尿病、龈口炎、HCM、CKD 等覆盖不足——这正是后续需下载 CNKI 论文补充的方向。

> 运行: `python -m tests.benchmark --compare --openai-key $OPENAI_API_KEY`

---

## 项目结构

```
rag/
├── README.md                          # 本文
├── CNKI论文下载清单.md                 # 论文下载指南
├── pyproject.toml                     # 项目配置 + 依赖
│
├── data/
│   ├── papers/                        # CNKI PDF 论文 (需自行下载)
│   ├── chroma_db/                     # ChromaDB 持久化数据
│   ├── conversations.db               # SQLite 对话记录
│   ├── traces/                        # JSONL 可观测性日志
│   └── models/                        # 本地嵌入模型 (ModelScope)
│
├── src/
│   ├── config.py                      # 全局配置
│   ├── ingest.py                      # 文档导入 CLI
│   ├── cli.py                         # 命令行测试对话
│   │
│   ├── document/
│   │   ├── parser.py                  # PDF 解析 (PyMuPDF)
│   │   └── chunker.py                # 语义分块 (RecursiveCharacterTextSplitter)
│   │
│   ├── retrieval/
│   │   ├── embeddings.py             # BGE-small-zh 嵌入 (SentenceTransformer)
│   │   ├── vector_store.py           # ChromaDB 客户端
│   │   └── hybrid_search.py          # Dense + BM25 混合检索 + RRF
│   │
│   ├── agent/
│   │   ├── orchestrator.py           # ReAct Agent 核心 (tool calling + retry + verify)
│   │   ├── tools.py                  # 3 个 Tool 实现
│   │   ├── prompts.py                # System prompt + guardrails + disclaimer
│   │   └── context.py               # 四区上下文工程 + token 预算
│   │
│   ├── api/
│   │   ├── main.py                   # FastAPI app 工厂
│   │   ├── models.py                 # SQLite 数据模型 (CRUD)
│   │   └── routes.py                 # REST 端点
│   │
│   ├── observability/
│   │   └── logger.py                 # JSONL trace 日志 + 查询
│   │
│   └── ui/
│       └── app.py                    # Streamlit UI + Human-in-the-Loop
│
└── tests/
    ├── benchmark.py                   # 基准测试运行器
    └── benchmark_questions.json       # 20 题测试集
```

---

## 技术亮点与面试故事

### 1. Ollama 中文路径 Bug 调试

**问题:** 用户 Windows 用户名是中文 (`芋頭粥粥`)，Ollama 默认模型路径 `C:\Users\芋頭粥粥\.ollama\models\blobs\...` 导致 llama.cpp 无法加载模型。

**排查过程:** 检查 Ollama 日志 → 发现 `llama_model_loader: failed to load model` → 定位到 llama.cpp 对非 ASCII 路径的支持问题 → 设置 `OLLAMA_MODELS=C:/ollama_models` 环境变量解决。

**面试话术:** "通过这个 bug 我理解了推理引擎的文件 I/O 机制、Windows 编码问题，以及 GGUF 模型的加载流程。"

### 2. qwen3:4b thinking 模式 Token 消耗

**问题:** qwen3:4b 默认启用 `thinking` (内部推理链)，在 max_tokens=1024 时全部 token 用于 thinking，输出为空。

**解决:** 增加到 max_tokens=4096，确保 thinking + output 都有足够空间。

**面试话术:** "理解了推理模型的 token 分配机制和 thinking tokens 的 overhead，这对成本优化和用户体验都很关键。"

### 3. RRF 混合检索设计

选择了 Reciprocal Rank Fusion 而非线性加权，因为 dense score (余弦相似度) 和 sparse score (BM25) 的量纲不同，直接加权需要归一化，而 RRF 天然 scale-free。

---

## 已知限制与改进方向

| 限制 | 改进方向 (TODO) |
|------|----------------|
| 知识库仅 8 篇 PDF (1353 chunks) | 下载 CNKI 论文扩展到 20+ 篇 |
| 仅支持简体中文 | 添加多语言支持 |
| 单轮对话为主 | 多轮对话上下文管理优化 |
| 基于关键词的 triage 检测 | 引入结构化 intent 分类模型 |
| 无 MCP 协议支持 | 调研 MCP server 集成 |
| 无模型服务多实例管理 | 调研 vLLM 替换 Ollama |

---

## 免责声明

本助手**仅供教育参考**，不构成兽医诊断或医疗建议。宠物出现健康问题请及时就医。

---

## License

MIT
