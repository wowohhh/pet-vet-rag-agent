# 🐱 宠物兽医知识助手 — RAG Agent

> **从零实现的 ReAct Agent 系统（非 LangChain 封装）· 消费级 GPU 本地部署 · 30 篇学术论文知识库 · 95% 引用率 0% 拒答**
>
> 每句话都有出处 · 高风险建议需人工确认 · 不推荐药物 · [工程日志](ENGINEERING_LOG.md) 10 个 debug 案例

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Ollama](https://img.shields.io/badge/LLM-qwen3:4b-green)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)](https://fastapi.tiangolo.com)

---

## 目录

1. [项目简介](#项目简介)
2. [系统架构](#系统架构)
3. [快速开始](#快速开始)
4. [Agent 核心能力](#agent-核心能力)
5. [基准测试](#基准测试)
6. [项目结构](#项目结构)
7. [已知限制与改进方向](#已知限制与改进方向)

---

## 项目简介

一个面向宠物猫健康咨询的 RAG Agent 系统。知识库基于中文畜牧兽医学术文献，通过混合检索 + ReAct Agent 在消费级 GPU 上本地运行。

**技术特点：**

- **混合检索** — ChromaDB 向量检索 + BM25 稀疏检索 + RRF 融合排序
- **ReAct Agent** — 从零实现的 Plan-Execute-Observe 循环，3 个 Function Calling 工具
- **完整后端** — FastAPI + SQLite 持久化，6 个 REST 端点（含 SSE 流式）
- **安全机制** — 硬约束不推荐药物、每句话引用来源、高风险建议人机协同确认
- **可观测性** — JSONL trace 日志，记录每次对话的工具调用耗时、token 消耗
- **本地部署** — Ollama + qwen3:4b INT4 + BGE-small-zh，RTX 4060 8GB 即可运行

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
    │  ~2.5GB VRAM   │  │  1353 chunks         │
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
| 显存 | 嵌入模型 ~1GB + LLM ~2.5GB ≈ 4GB |

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

### 命令行测试

```bash
python -m src.cli
```

---

## Agent 核心能力

### 四个 Tool

| Tool | 功能 | 关键技术 |
|------|------|---------|
| `search_knowledge_base` | 检索知识库 | ChromaDB 向量检索 + BM25 稀疏检索 + RRF 融合 |
| `analyze_symptoms` | 分析症状 | 结构化提取症状 + 可能疾病方向 + 文献依据 |
| `triage_decision` | 就医决策 | 急诊信号检测 → 急诊/尽快就医/居家观察 三级 |
| `search_cnki` | 知网搜索 | 本地无结果时通过网络检索知网论文摘要作为补充 |

### 安全 Guardrails

- ❌ **不推荐任何药物**（包括人用药、处方药）
- ✅ **每句话都要有文献来源**
- ✅ **知识库没有的信息诚实说不知道**
- ✅ **高风险情况（急诊/尽快就医）触发人工确认**

### Human-in-the-Loop

当 Agent 判断为"尽快就医"或更高风险时，Streamlit UI 弹出确认框，用户手动确认采纳或拒绝后，回答才会被记录。

---

## 基准测试

20 题测试集，覆盖 5 个类别：症状咨询、紧急判断、疾病知识、预防护理、边界测试。

### qwen3:4b 本地 INT4 (RTX 4060 8GB)

| 指标 | v1 (8篇/1353 chunks) | v2 (29篇/2277 chunks + CNKI fallback) |
|------|----------------------|---------------------------------------|
| 平均延迟 | 74.8s | 46.2s |
| 有效引用率 | 65% (13/20) | 95% (19/20) |
| 拒答率 | 35% (7/20) | 15% (3/20) |
| 回答长度 | ~200 字 | 404 字 |

v2 知识库覆盖猫上呼吸道、猫瘟、FIP、FLUTD、CKD、HCM、龈口炎、猫癣、糖尿病、FCoV、杯状/支原体/衣原体/疱疹、寄生虫、疫苗等全类别。未覆盖 3 题均为知识库未收录的极端边界场景（中国 FIV 流行率具体数字、猫高空坠落后特定处理方案）。

```bash
python -m tests.benchmark
```

---

## 项目结构

```
rag/
├── README.md
├── CNKI论文下载清单.md
├── pyproject.toml
│
├── data/
│   ├── papers/          # PDF 论文 (需自行下载)
│   ├── chroma_db/       # ChromaDB 持久化数据
│   ├── conversations.db # SQLite 对话记录
│   ├── traces/          # JSONL 可观测性日志
│   └── models/          # 本地嵌入模型 (ModelScope)
│
├── src/
│   ├── config.py        # 全局配置
│   ├── ingest.py        # 文档导入 CLI
│   ├── cli.py           # 命令行对话
│   │
│   ├── document/
│   │   ├── parser.py    # PDF 解析 (PyMuPDF)
│   │   └── chunker.py   # 语义分块
│   │
│   ├── retrieval/
│   │   ├── embeddings.py    # BGE-small-zh 嵌入
│   │   ├── vector_store.py  # ChromaDB 客户端
│   │   └── hybrid_search.py # Dense + BM25 + RRF
│   │
│   ├── agent/
│   │   ├── orchestrator.py  # ReAct Agent 核心
│   │   ├── tools.py         # 4 个 Tool 实现
│   │   ├── prompts.py       # System prompt + guardrails
│   │   └── context.py       # 上下文工程 + token 预算
│   │
│   ├── api/
│   │   ├── main.py      # FastAPI app 工厂
│   │   ├── models.py    # SQLite CRUD
│   │   └── routes.py    # REST 端点
│   │
│   ├── observability/
│   │   └── logger.py    # JSONL trace 日志
│   │
│   └── ui/
│       └── app.py       # Streamlit UI + Human-in-the-Loop
│
└── tests/
    ├── benchmark.py              # 基准测试运行器
    └── benchmark_questions.json  # 20 题测试集
```

---

## 已知限制与改进方向

| 限制 | 改进方向 |
|------|---------|
| 知识库仅 8 篇 PDF (1353 chunks) | 扩展 CNKI 论文至 20+ 篇 |
| 仅支持简体中文 | 添加多语言支持 |
| 单轮对话为主 | 多轮对话上下文管理优化 |
| 基于关键词的 triage 检测 | 引入结构化 intent 分类 |
| 无 MCP 协议支持 | 调研 MCP server 集成 |
| 模型服务单实例 | 调研 vLLM 替换 Ollama |

---

## 免责声明

本助手**仅供教育参考**，不构成兽医诊断或医疗建议。宠物出现健康问题请及时就医。

---

## License

MIT
