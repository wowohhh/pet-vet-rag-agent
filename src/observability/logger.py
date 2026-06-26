"""Observability: JSONL tracing and metrics for Agent runs."""

import json
import time
from pathlib import Path
from src.config import DATA_DIR

TRACE_DIR = DATA_DIR / "traces"


def _ensure_dir():
    TRACE_DIR.mkdir(parents=True, exist_ok=True)


class Trace:
    """Context manager for tracing a single Agent interaction."""

    def __init__(self, query: str, conversation_id: str = ""):
        _ensure_dir()
        self.trace_id = f"trace_{int(time.time()*1000)}_{hash(query) & 0xffff}"
        self.start_time = time.time()
        self.query = query
        self.conversation_id = conversation_id
        self.steps: list[dict] = []
        self.total_tokens = 0
        self.total_tool_calls = 0
        self.retrieval_hits = 0

    def log_step(self, step_type: str, **kwargs):
        elapsed = time.time() - self.start_time
        step = {
            "type": step_type,
            "elapsed_ms": round(elapsed * 1000),
            **kwargs,
        }
        self.steps.append(step)

    def log_retrieval(self, query: str, hit_count: int, top_score: float):
        self.retrieval_hits = hit_count
        self.log_step("retrieval", query=query, hits=hit_count, top_score=round(top_score, 4))

    def log_tool_call(self, tool_name: str, success: bool, duration_ms: float, error: str = ""):
        self.total_tool_calls += 1
        entry = {"tool": tool_name, "success": success, "duration_ms": round(duration_ms, 1)}
        if error:
            entry["error"] = error
        self.log_step("tool_call", **entry)

    def log_generation(self, tokens: int, duration_ms: float):
        self.total_tokens += tokens
        self.log_step("generation", tokens=tokens, duration_ms=round(duration_ms, 1))

    def finish(self, answer: str = ""):
        duration = time.time() - self.start_time
        record = {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "query": self.query[:200],
            "total_duration_ms": round(duration * 1000),
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "retrieval_hits": self.retrieval_hits,
            "answer_length": len(answer),
        }

        _ensure_dir()
        log_file = TRACE_DIR / f"{time.strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record


def get_recent_traces(limit: int = 20) -> list[dict]:
    """Read most recent traces from today's log."""
    _ensure_dir()
    log_file = TRACE_DIR / f"{time.strftime('%Y%m%d')}.jsonl"
    if not log_file.exists():
        return []

    traces = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return traces[-limit:]
