#!/usr/bin/env python3
"""Write one auditable summary for a new-benchmark result JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--prompt-mode", required=True)
    ap.add_argument("--input-jsonl", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").split("\n") if line.strip()]
    parsed = [row for row in rows if row.get("parsed_answer") is not None and not str(row.get("raw_answer", "")).startswith("[API_ERROR]")]
    score_sum = sum(float(row.get("score", 0.0)) for row in rows)
    is_textvqa = any(row.get("answer_type") == "textvqa" for row in rows)
    is_exact_text = any(row.get("answer_type") == "exact_text" for row in rows)
    summary = {
        "benchmark": args.benchmark,
        "model_id": args.model_id,
        "prompt_mode": args.prompt_mode,
        "n": len(rows),
        "parsed_n": len(parsed),
        "parse_rate_pct": round(100 * len(parsed) / len(rows), 4) if rows else 0.0,
        "score_sum": round(score_sum, 6),
        "correct_n": sum(bool(row.get("correct")) for row in rows),
        "accuracy_pct": round(100 * score_sum / len(rows), 4) if rows else 0.0,
        "metric": (
            "TextVQA VQA consensus accuracy"
            if is_textvqa
            else "exact text accuracy"
            if is_exact_text
            else "exact multiple-choice accuracy"
        ),
        "input": str(args.input_jsonl),
        "output": str(args.input),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
