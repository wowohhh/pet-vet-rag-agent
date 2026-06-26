"""Benchmark runner: qwen3:4b vs GPT-4o-mini comparison.

Usage:
    python -m tests.benchmark          # run all 20 questions
    python -m tests.benchmark --model qwen3:4b    # single model
    python -m tests.benchmark --model gpt-4o-mini --openai-key sk-xxx
"""

import json
import time
import argparse
from pathlib import Path
from src.agent.orchestrator import VetAgent
from src.config import OLLAMA_BASE_URL


def load_questions() -> list[dict]:
    path = Path(__file__).parent / "benchmark_questions.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def run_benchmark(model: str, questions: list[dict], openai_key: str = "") -> list[dict]:
    """Run benchmark for a single model."""
    results = []

    # Determine if using OpenAI
    if "gpt" in model and openai_key:
        base_url = "https://api.openai.com/v1"
        agent = VetAgent(base_url=base_url, model=model)
        agent.client.api_key = openai_key
    else:
        agent = VetAgent(model=model)

    print(f"\n{'='*60}")
    print(f"Benchmark: {model}")
    print(f"{'='*60}")

    for i, q in enumerate(questions, 1):
        agent.reset()
        print(f"\n[{i}/{len(questions)}] {q['category']}: {q['question'][:60]}...")

        start = time.time()
        try:
            answer = agent.chat(q["question"], conversation_id=f"bench_{i}")
            duration = time.time() - start
        except Exception as e:
            answer = f"[ERROR] {e}"
            duration = 0

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": answer[:500],
            "duration_s": round(duration, 1),
        })

        print(f"  Duration: {duration:.1f}s, Answer length: {len(answer)} chars")

    return results


def save_results(results: dict[str, list[dict]]):
    """Save benchmark results to JSON."""
    path = Path(__file__).parent / "benchmark_results.json"
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": list(results.keys()),
        "question_count": len(list(results.values())[0]) if results else 0,
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {path}")


def print_comparison(results: dict[str, list[dict]]):
    """Print side-by-side comparison table."""
    models = list(results.keys())
    if len(models) < 2:
        return

    print(f"\n{'='*80}")
    print(f"COMPARISON: {models[0]} vs {models[1]}")
    print(f"{'='*80}")

    for i in range(len(results[models[0]])):
        r0 = results[models[0]][i]
        r1 = results[models[1]][i]
        print(f"\n--- Q{i+1}: {r0['question'][:80]} ---")
        print(f"  {models[0]:<20} {r0['duration_s']:.1f}s  |  {len(r0['answer'])} chars")
        print(f"  {models[1]:<20} {r1['duration_s']:.1f}s  |  {len(r1['answer'])} chars")

    # Summary
    avg0 = sum(r["duration_s"] for r in results[models[0]]) / len(results[models[0]])
    avg1 = sum(r["duration_s"] for r in results[models[1]]) / len(results[models[1]])
    print(f"\n{'='*80}")
    print(f"Average duration: {models[0]}={avg0:.1f}s, {models[1]}={avg1:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="RAG Agent Benchmark")
    parser.add_argument("--model", type=str, help="Single model to test")
    parser.add_argument("--openai-key", type=str, default="", help="OpenAI API key for GPT models")
    parser.add_argument("--compare", action="store_true", help="Run both models and compare")
    args = parser.parse_args()

    questions = load_questions()
    print(f"Loaded {len(questions)} benchmark questions")

    results = {}

    if args.compare:
        results["qwen3:4b"] = run_benchmark("qwen3:4b", questions)
        if args.openai_key:
            results["gpt-4o-mini"] = run_benchmark("gpt-4o-mini", questions, args.openai_key)
        else:
            print("\n[WARNING] No OpenAI key provided. Skipping GPT-4o-mini.")
            print("Usage: python -m tests.benchmark --compare --openai-key sk-xxx")
    elif args.model:
        results[args.model] = run_benchmark(args.model, questions)
    else:
        # Default: run qwen3:4b only
        results["qwen3:4b"] = run_benchmark("qwen3:4b", questions)

    if results:
        save_results(results)
        print_comparison(results)


if __name__ == "__main__":
    main()
