"""Resource monitoring — GPU VRAM, model status, latency stats."""

import subprocess
import json
from datetime import datetime
from collections import deque


# Rolling window for latency tracking
_latency_history: deque[float] = deque(maxlen=20)


def record_latency(ms: float):
    _latency_history.append(ms)


def get_latency_stats() -> dict:
    if not _latency_history:
        return {"avg_ms": 0, "min_ms": 0, "max_ms": 0, "count": 0}
    return {
        "avg_ms": round(sum(_latency_history) / len(_latency_history)),
        "min_ms": round(min(_latency_history)),
        "max_ms": round(max(_latency_history)),
        "count": len(_latency_history),
    }


def get_gpu_info() -> dict:
    """Get GPU VRAM usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        parts = result.stdout.strip().split(",")
        if len(parts) >= 3:
            return {
                "gpu_util_pct": int(parts[0].strip()),
                "vram_used_mb": int(parts[1].strip()),
                "vram_total_mb": int(parts[2].strip()),
                "temp_c": int(parts[3].strip()) if len(parts) >= 4 else 0,
            }
    except Exception:
        pass
    return {"gpu_util_pct": 0, "vram_used_mb": 0, "vram_total_mb": 0, "temp_c": 0}


def get_ollama_status() -> dict:
    """Get loaded models from Ollama."""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/ps", timeout=3)
        data = resp.json()
        models = []
        for m in data.get("models", []):
            models.append({
                "name": m.get("name", "?"),
                "size_mb": round(m.get("size", 0) / (1024 * 1024)),
                "vram_mb": round(m.get("size_vram", 0) / (1024 * 1024)),
            })
        return {"models": models, "count": len(models)}
    except Exception:
        return {"models": [], "count": 0}


def get_full_status() -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "gpu": get_gpu_info(),
        "ollama": get_ollama_status(),
        "latency": get_latency_stats(),
    }
