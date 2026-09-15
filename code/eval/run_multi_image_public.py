#!/usr/bin/env python3
"""Run and parse native small-benchmark multi-image evaluations."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from urllib.parse import urlparse

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cot_protocol import SYSTEM_PROMPT, answer_payload


def parse_mcq(raw: str, binary: bool = False) -> str | None:
    text = answer_payload(raw).replace("**", "").strip() if raw else ""
    # Only accept a standalone final answer (or an explicit answer label) in
    # the tail.  Falling back to the last A-E anywhere in a long explanation
    # turns a truncated response into a fabricated answer.
    lines = [line.strip(" `*_\t") for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-4:]):
        # MMMU-Pro-10c has ten choices (A-J); MuirBench can use up to H.
        labeled = re.search(r"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?\s*[.!)]?\s*$", line)
        if labeled:
            return labeled.group(1).upper()
        direct = re.fullmatch(r"\(?([A-J])\)?(?:\s*[.!:)]\s*(?:yes|no))?\s*", line, flags=re.I)
        if direct:
            return direct.group(1).upper()
        # Some native visual-choice questions end with e.g. ``C. [Image 2]``
        # rather than a bare letter. Accept only this compact final-answer
        # form; do not search arbitrary explanation text for a letter.
        visual_choice = re.fullmatch(r"\(?([A-J])\)?\s*[.:)]\s*\[?image\s*\d+\]?\s*", line, flags=re.I)
        if visual_choice:
            return visual_choice.group(1).upper()
        none_choice = re.fullmatch(r"\(?([A-J])\)?\s*[.:)]\s*none\s+of\s+the\s+(?:choices|options)(?:\s+provided)?\s*[.!]?", line, flags=re.I)
        if none_choice:
            return none_choice.group(1).upper()
    if binary:
        for line in reversed(lines[-4:]):
            yes_no = re.fullmatch(r"(?:yes|true|no|false)[.!]?", line, flags=re.I)
            if yes_no:
                return "A" if yes_no.group(0).lower().startswith(("yes", "true")) else "B"
    return None


def parse_direct(raw: str, answer_type: str, answer: str, answer_items: list[str]) -> str | None:
    text = answer_payload(raw).strip() if raw else ""
    lines = [line.strip(" `*_\t") for line in text.splitlines() if line.strip()]
    last = lines[-1].lower() if lines else text.lower()
    if answer_type == "mimic_counting":
        numbers = re.findall(r"(?<![A-Za-z])\d+(?![A-Za-z])", last)
        if not numbers:
            numbers = re.findall(r"(?<![A-Za-z])\d+(?![A-Za-z])", text)
        return numbers[-1] if numbers else None
    if answer_type == "mimic_listing":
        # Models often return one category per line. Parse the complete final
        # answer instead of only the last line, and retain extra categories so
        # over-listing is scored as wrong rather than as a parser failure.
        items = []
        for line in lines or [text]:
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
            line = re.sub(r"^(?:final\s+)?answer\s*(?:is|:)\s*", "", line, flags=re.I)
            items.extend(part.strip() for part in re.split(r"[,;|]", line) if part.strip())
        normalized = []
        for item in items:
            item = re.sub(r"[^a-z0-9 ]+", " ", item.lower().replace("-", " "))
            item = re.sub(r"\s+", " ", item).strip()
            if item.endswith("ies") and len(item) > 4:
                item = item[:-3] + "y"
            elif item.endswith("s") and not item.endswith(("ss", "us")) and len(item) > 3:
                item = item[:-1]
            if item:
                normalized.append(item)
        return ", ".join(sorted(set(normalized))) or None
    target = answer.strip().lower()
    if target and re.search(rf"\b{re.escape(target)}\b", last):
        return target
    return last or None


def parse_answer(raw: str, item: dict) -> str | None:
    if item["answer_type"] in {"mcq", "binary_mcq"}:
        return parse_mcq(raw, binary=item["answer_type"] == "binary_mcq")
    return parse_direct(raw, item["answer_type"], str(item["answer"]), item.get("answer_items", []))


def comparable_answer(item: dict, parsed: str | None) -> str | None:
    if parsed is None:
        return None
    if item["answer_type"] == "mimic_listing":
        return ", ".join(sorted(set(parsed.split(", "))))
    return parsed.strip().lower()


def is_correct(item: dict, parsed: str | None) -> bool:
    if parsed is None:
        return False
    if item["answer_type"] in {"mcq", "binary_mcq"}:
        return parsed.upper() == str(item["answer"]).upper()
    if item["answer_type"] == "mimic_listing":
        expected = {re.sub(r"[^a-z0-9 ]+", " ", value.lower().replace("-", " ")).strip() for value in item.get("answer_items", [])}
        expected = {re.sub(r"\s+", " ", value) for value in expected}
        expected = {(value[:-3] + "y") if value.endswith("ies") and len(value) > 4 else (value[:-1] if value.endswith("s") and not value.endswith(("ss", "us")) and len(value) > 3 else value) for value in expected}
        return set(parsed.split(", ")) == expected
    return parsed.strip().lower() == str(item["answer"]).strip().lower()


def good(row: dict) -> bool:
    return not str(row.get("raw_answer", "")).startswith("[API_ERROR]") and row.get("parsed_answer") is not None


def data_uri(path: str) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


def disable_proxy_for_local_api(api_base: str) -> None:
    """Prevent HTTP(S) proxies from intercepting requests to local vLLM."""
    if urlparse(api_base).hostname in {"127.0.0.1", "localhost", "::1"}:
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            os.environ.pop(name, None)
        os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
        os.environ["no_proxy"] = os.environ["NO_PROXY"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    disable_proxy_for_local_api(args.api_base)
    # JSONL records are delimited by ASCII LF only. str.splitlines() also
    # treats Unicode line-separator characters inside JSON strings as breaks.
    items = [json.loads(line) for line in args.input.read_text(encoding="utf-8").split("\n") if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                row = json.loads(line)
                existing[row["query_uid"]] = row
    todo = [item for item in items if item["query_uid"] not in existing or not good(existing[item["query_uid"]])]
    print(json.dumps({"input": len(items), "existing": len(existing), "todo": len(todo)}), flush=True)
    lock = threading.Lock()

    def run_one(item: dict) -> dict:
        content = [{"type": "image_url", "image_url": {"url": data_uri(path)}} for path in item["images"]]
        content.append({"type": "text", "text": item["query"]})
        raw = ""
        error = ""
        started = time.time()
        for attempt in range(args.retries):
            # Do not reuse a long-lived httpx connection across hundreds of
            # multimodal requests. Some vLLM/httpx combinations leave a
            # worker blocked on a stale keep-alive socket after a long run.
            # Bound an individual request so one pathological visual sample
            # cannot hold the whole full-benchmark run forever.
            request_client = OpenAI(api_key="EMPTY", base_url=args.api_base, timeout=180, max_retries=0)
            try:
                response = request_client.chat.completions.create(
                    model=args.model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
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
                request_client.close()
        parsed = parse_answer(raw, item)
        result = dict(item)
        result["model_id"] = args.model_id
        result["raw_answer"] = raw or f"[API_ERROR] {error}"
        result["parsed_answer"] = parsed
        result["comparable_answer"] = comparable_answer(item, parsed)
        result["correct"] = is_correct(item, parsed)
        result["latency_sec"] = round(time.time() - started, 4)
        result["input_sha1"] = hashlib.sha1("\n".join(item["images"]).encode()).hexdigest()
        return result

    with args.output.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, item) for item in todo]
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                print(".", end="", flush=True)
    merged = dict(existing)
    for line in args.output.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            merged[row["query_uid"]] = row
    ordered = [merged[item["query_uid"]] for item in items if item["query_uid"] in merged]
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in ordered) + "\n", encoding="utf-8")
    print()
    print(json.dumps({"completed": sum(good(row) for row in ordered), "output": str(args.output)}))


if __name__ == "__main__":
    main()
