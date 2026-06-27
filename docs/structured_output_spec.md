# 结构化输出接口规范

## JSON Schema

```json
{
  "answer": "string (必填) — 用户可读的完整回答，markdown 格式",
  "citations": [
    {
      "title": "string — 论文标题",
      "journal": "string — 期刊/来源",
      "year": "string — 出版年份",
      "relevant_text": "string — 引用的原文摘录（<100字）"
    }
  ],
  "triage": {
    "level": "string — EMERGENCY | URGENT | OBSERVE | UNKNOWN",
    "signal": "string — 触发该级别的关键词，如 '拉血'/'呼吸困难'",
    "reasoning": "string — 分级理由，一句话"
  },
  "requires_confirmation": "bool — true 当 triage.level ∈ {EMERGENCY, URGENT}",
  "source": "string — local | cnki | fallback — 回答的知识来源",
  "disclaimer": "string (必填) — 固定免责声明文本"
}
```

## 字段行为

| 字段 | 为空时 | 含义 |
|------|--------|------|
| `citations` | `[]` | 无文献支持，前端只展示 answer + disclaimer |
| `triage.signal` | `""` | UNKNOWN 级别，无明确信号 |
| `source` | `"fallback"` | 本地和网络都未找到，靠 prompt 硬约束回答 |

## Triage 级别映射

```
EMERGENCY — 检测到危险信号（呼吸困难/抽搐/大出血等），需立即就医
URGENT    — 症状持续>24h 或需要兽医诊断，建议24-48h就医
OBSERVE   — 症状轻微/自限性，可居家观察
UNKNOWN   — 信息不足，无法判断
```

## 兼容策略

旧的 `chat()` 方法保留不变，新增 `chat_structured()` 返回 dict。
Streamlit UI 逐步迁移到消费结构化输出，未迁移前回退到自由文本路径。

## 实现文件

| 文件 | 改动 |
|------|------|
| `src/agent/orchestrator.py` | 新增 `chat_structured()` 方法 |
| `src/agent/prompts.py` | System prompt 加入输出格式要求 |
| `src/ui/app.py` | `_detect_triage_level()` → 消费 `triage` 字段 |
| `src/api/routes.py` | `/api/chat` 响应增加 `structured` 字段 |
