#!/usr/bin/env python3
"""Evaluate canonical benchmark JSONL through an OpenAI-compatible vLLM API."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cot_protocol import SYSTEM_PROMPT, answer_payload


def normalize_vqa(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"([.!?,;:\"'()\[\]{}])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def vqa_score(prediction: str | None, answers: list[str]) -> float:
    if prediction is None or not answers:
        return 0.0
    pred = normalize_vqa(prediction)
    normalized = [normalize_vqa(answer) for answer in answers]
    scores = []
    for index in range(len(normalized)):
        others = normalized[:index] + normalized[index + 1 :]
        matches = sum(value == pred for value in others)
        scores.append(min(1.0, matches / 3.0))
    return sum(scores) / len(scores)


def parse_mcq(raw: str, cot: bool) -> str | None:
    text = answer_payload(raw) if cot else (raw or "")
    if not text.strip():
        return None
    lines = [line.strip(" `*_") for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-6:]):
        match = re.fullmatch(
            r"(?:final\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?[.!)]?",
            line,
            flags=re.I,
        )
        if match:
            return match.group(1).upper()
        match = re.fullmatch(r"\(?([A-J])\)?(?:\s*[.:)]\s*(?:yes|no))?\s*", line, flags=re.I)
        if match:
            return match.group(1).upper()
        if not cot:
            # Base checkpoints often obey the benchmark's native format and
            # return `B. <option text>` even after being asked for a direct
            # answer. Keep the option letter and ignore the display text.
            match = re.fullmatch(r"\(?([A-J])\)?\s*[.:)]\s*.+", line, flags=re.I)
            if match:
                return match.group(1).upper()
    if not cot:
        # Direct-answer prompting is advisory for a base checkpoint. If it
        # still emits an explanation, recover only an explicit answer marker
        # or an option-formatted line; never guess from an arbitrary letter.
        for line in reversed(lines):
            match = re.search(
                r"\b(?:final\s+)?(?:answer|option|choice|letter)\s*(?:is|:)?\s*\(?([A-J])\)?\b",
                line,
                flags=re.I,
            )
            if match:
                return match.group(1).upper()
            match = re.match(r"\(?([A-J])\)?\s*[.:)]\s+\S+", line, flags=re.I)
            if match:
                return match.group(1).upper()
            match = re.fullmatch(
                r"(?:final\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?\s*[.:)]\s*.*",
                line,
                flags=re.I,
            )
            if match:
                return match.group(1).upper()
    if not cot:
        match = re.fullmatch(
            r"\s*(?:the\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?[.!)]?\s*",
            text,
            flags=re.I,
        )
        if match:
            return match.group(1).upper()
    return None


def parse_textvqa(raw: str, cot: bool) -> str | None:
    text = answer_payload(raw) if cot else (raw or "")
    if not text.strip():
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    value = lines[-1] if lines else text.strip()
    value = re.sub(r"^(?:final\s+)?answer\s*(?:is|:)\s*", "", value, flags=re.I).strip()
    value = value.strip("`*_ \"'")
    return value or None


def parse_numeric_direct(raw: str) -> str | None:
    """Parse a native numeric response such as ``2`` or ``Answer: 2``."""
    text = raw or ""
    if not text.strip():
        return None
    lines = [line.strip("`*_ \t") for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        match = re.fullmatch(
            r"(?:final\s+)?answer\s*(?:is|:)\s*(-?\d+(?:\.\d+)?)\s*[.!]?",
            line,
            flags=re.I,
        )
        if match:
            return match.group(1)
        match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*[.!]?", line)
        if match:
            return match.group(1)
    return None


def parse_answer(raw: str, item: dict, cot: bool) -> str | None:
    if item.get("answer_type") in {"textvqa", "exact_text"}:
        return parse_textvqa(raw, cot)
    # ZoomBench's numeric questions are represented as ``answer_type=mcq``
    # even though they have no A/B/C/D options (``options`` repeats the
    # question). Treat a direct numeric response as a prediction, not as an
    # unparsable multiple-choice answer.
    if (
        not cot
        and item.get("answer_type") == "mcq"
        and re.fullmatch(r"-?\d+(?:\.\d+)?", str(item.get("answer", "")).strip())
        and item.get("options") == item.get("query")
    ):
        return parse_numeric_direct(raw)
    return parse_mcq(raw, cot)


def attach_score(result: dict, item: dict, parsed: str | None) -> dict:
    result["parsed_answer"] = parsed
    if item.get("answer_type") == "textvqa":
        score = vqa_score(parsed, list(item.get("answer_items", [])))
        result["score"] = score
        result["correct"] = score >= 0.5
        result["comparable_answer"] = normalize_vqa(parsed) if parsed is not None else None
    elif item.get("answer_type") == "exact_text":
        predicted = normalize_vqa(parsed) if parsed is not None else None
        expected = normalize_vqa(item.get("answer", ""))
        correct = predicted is not None and predicted == expected
        result["score"] = 1.0 if correct else 0.0
        result["correct"] = correct
        result["comparable_answer"] = predicted
    else:
        correct = parsed is not None and parsed.upper() == str(item["answer"]).strip().upper()
        result["score"] = 1.0 if correct else 0.0
        result["correct"] = correct
        result["comparable_answer"] = parsed.lower() if parsed is not None else None
    return result


def data_uri(path: str) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--prompt-mode", choices=("none", "cot"), required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--offline-reparse", action="store_true", help="reparse saved raw outputs without calling the model")
    args = ap.parse_args()
    cot = args.prompt_mode == "cot"
    protocol_id = "new_benchmark_v2_cot" if cot else "new_benchmark_v2_base_direct"
    # JSONL records are delimited by ASCII LF only.  str.splitlines() also
    # splits on Unicode separators that may legitimately occur inside query
    # strings (MMIU contains such examples).
    items = [json.loads(line) for line in args.input.read_text(encoding="utf-8").split("\n") if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                row = json.loads(line)
                existing[row["query_uid"]] = row
    item_by_uid = {item["query_uid"]: item for item in items}
    # Re-score rows produced by an older parser when their raw model output is
    # still available. This avoids another 10k+ model calls after a parser fix.
    for uid, row in list(existing.items()):
        item = item_by_uid.get(uid)
        raw = str(row.get("raw_answer", ""))
        if (
            item is not None
            and raw
            and not raw.startswith("[API_ERROR]")
            and row.get("protocol_id") == protocol_id
            and row.get("prompt_mode") == args.prompt_mode
        ):
            reparsed = parse_answer(raw, item, cot)
            if reparsed is not None:
                row = attach_score(dict(row), item, reparsed)
                row["prompt_mode"] = args.prompt_mode
                row["protocol_id"] = protocol_id
                existing[uid] = row
    todo = [] if args.offline_reparse else [
        item
        for item in items
        if item["query_uid"] not in existing
        or existing[item["query_uid"]].get("parsed_answer") is None
        or existing[item["query_uid"]].get("protocol_id") != protocol_id
        or str(existing[item["query_uid"]].get("raw_answer", "")).startswith("[API_ERROR]")
    ]
    print(json.dumps({"input": len(items), "existing": len(existing), "todo": len(todo), "prompt_mode": args.prompt_mode}), flush=True)
    lock = threading.Lock()

    def run_one(item: dict) -> dict:
        content = [{"type": "image_url", "image_url": {"url": data_uri(path)}} for path in item["images"]]
        query = item["query"]
        if not cot:
            query += "\n\nAnswer directly. Return only the final answer, without explanation."
        content.append({"type": "text", "text": query})
        raw, error = "", ""
        started = time.time()
        messages = ([{"role": "system", "content": SYSTEM_PROMPT}] if cot else []) + [{"role": "user", "content": content}]
        for attempt in range(args.retries):
            client = OpenAI(api_key="EMPTY", base_url=args.api_base, timeout=180, max_retries=0)
            try:
                response = client.chat.completions.create(
                    model=args.model_id,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = (response.choices[0].message.content or "").strip()
                break
            except Exception as exc:
                error = repr(exc)
                if attempt + 1 < args.retries:
                    time.sleep(1.0)
            finally:
                client.close()
        parsed = parse_answer(raw, item, cot)
        result = dict(item)
        result.update({
            "model_id": args.model_id,
            "prompt_mode": args.prompt_mode,
            "protocol_id": protocol_id,
            "raw_answer": raw or f"[API_ERROR] {error}",
            "parsed_answer": parsed,
            "latency_sec": round(time.time() - started, 4),
            "input_sha1": hashlib.sha1("\n".join(item["images"]).encode()).hexdigest(),
        })
        return attach_score(result, item, parsed)

    with args.output.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, item) for item in todo]
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                print(".", end="", flush=True)
    merged = {}
    for line in args.output.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            merged[row["query_uid"]] = row
    # In-memory reparses must take precedence over stale rows already on disk.
    merged.update(existing)
    ordered = [merged[item["query_uid"]] for item in items if item["query_uid"] in merged]
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in ordered) + "\n", encoding="utf-8")
    print()
    print(json.dumps({"completed": len(ordered), "output": str(args.output)}))


if __name__ == "__main__":
    main()
