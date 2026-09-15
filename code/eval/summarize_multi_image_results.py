#!/usr/bin/env python3
"""Summarize strict full-only multi-image evaluation outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def pct(value: int, total: int) -> float | None:
    return round(100.0 * value / total, 4) if total else None


def summarize(path: Path) -> dict:
    # JSONL records are delimited by ASCII LF only. str.splitlines() also
    # treats Unicode line-separator characters inside JSON strings as breaks.
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]
    correct = sum(bool(x.get("correct")) for x in rows)
    parsed = sum(x.get("parsed_answer") is not None for x in rows)
    by_task = defaultdict(lambda: [0, 0])
    by_type = defaultdict(lambda: [0, 0])
    for row in rows:
        task = str(row.get("task", "unknown"))
        typ = str(row.get("answer_type", "unknown"))
        by_task[task][0] += 1
        by_task[task][1] += bool(row.get("correct"))
        by_type[typ][0] += 1
        by_type[typ][1] += bool(row.get("correct"))
    return {
        "file": str(path),
        "benchmark": (rows[0].get("benchmark") or path.stem.removesuffix("_full")) if rows else path.stem,
        "model": rows[0].get("model_id") if rows else None,
        "condition": sorted({x.get("condition") for x in rows}),
        "n": len(rows),
        "parsed_n": parsed,
        "correct_n": correct,
        "parse_rate_pct": pct(parsed, len(rows)),
        "accuracy_pct": pct(correct, len(rows)),
        "tasks": {k: {"n": v[0], "correct_n": v[1], "accuracy_pct": pct(v[1], v[0])} for k, v in sorted(by_task.items())},
        "answer_types": {k: {"n": v[0], "correct_n": v[1], "accuracy_pct": pct(v[1], v[0])} for k, v in sorted(by_type.items())},
        "raw_answer_empty_n": sum(not str(x.get("raw_answer", "")).strip() for x in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    paths = sorted(args.results.glob("*.jsonl")) if args.results.is_dir() else [args.results]
    summary = [summarize(p) for p in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in summary:
        print(json.dumps({k: item[k] for k in ["benchmark", "model", "n", "parsed_n", "correct_n", "accuracy_pct"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
