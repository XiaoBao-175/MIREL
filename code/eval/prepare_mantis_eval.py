#!/usr/bin/env python3
"""Convert the Hugging Face Mantis-Eval parquet into the local JSONL format."""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image


def safe_stem(value: object) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return stem or "sample"


def as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def image_bytes(value: object) -> bytes:
    if isinstance(value, dict):
        blob = value.get("bytes")
        if blob is not None:
            return bytes(blob)
        path = value.get("path")
        if path:
            return Path(path).read_bytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError(f"unsupported Mantis-Eval image value: {type(value)!r}")


def image_suffix(blob: bytes) -> str:
    try:
        image_format = Image.open(io.BytesIO(blob)).format
    except Exception:
        image_format = None
    return {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
        "BMP": ".bmp",
    }.get(image_format, ".jpg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for _, row in frame.iterrows():
        item_id = str(row["id"])
        stem = safe_stem(item_id)
        paths = []
        for index, value in enumerate(as_list(row["images"])):
            blob = image_bytes(value)
            path = args.image_dir / f"{stem}_{index + 1}{image_suffix(blob)}"
            if not path.exists() or path.stat().st_size != len(blob):
                path.write_bytes(blob)
            paths.append(str(path))

        options = [str(value) for value in as_list(row.get("options")) if str(value).strip()]
        question = str(row["question"])
        if options:
            question += "\n\nOptions:\n" + "\n".join(options)
        question_type = str(row["question_type"]).lower()
        answer_type = "mcq" if "choice" in question_type else "exact_text"
        records.append(
            {
                "query_uid": f"mantis_eval:{item_id}",
                "benchmark": "Mantis-Eval",
                "images": paths,
                "query": question,
                "question": str(row["question"]),
                "options": "\n".join(options),
                "answer": str(row["answer"]),
                "answer_type": answer_type,
                "answer_items": [],
                "source_id": item_id,
                "category": str(row.get("category", "")),
                "data_source": str(row.get("data_source", "")),
                "native_image_count": len(paths),
            }
        )

    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(records), "output": str(args.output), "image_dir": str(args.image_dir)}))


if __name__ == "__main__":
    main()
