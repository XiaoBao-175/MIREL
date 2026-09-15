#!/usr/bin/env python3
"""Rewrite image roots in a release parquet without changing rows.

The released parquet uses repository-relative image paths. Use this helper
when the downloaded M4/Mantis images live under another local root:

  python rewrite_image_paths.py \
    --input ../../data/Qwen35-4B/pg_opd.parquet \
    --output /tmp/pg_opd_local.parquet \
    --old-prefix ../../data/images \
    --new-prefix /data/multi-image

Pass --check-only to verify that every recorded image path exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def rewrite(value: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(value, str):
        if value == old_prefix or value.startswith(old_prefix + "/"):
            return new_prefix + value[len(old_prefix) :]
        return value
    if isinstance(value, list):
        return [rewrite(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, tuple):
        return tuple(rewrite(item, old_prefix, new_prefix) for item in value)
    if isinstance(value, dict):
        return {key: rewrite(item, old_prefix, new_prefix) for key, item in value.items()}
    return value


def collect_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(collect_paths(item))
        return result
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            if key in {"path", "image"} and isinstance(item, str):
                result.append(item)
            elif key in {"images", "teacher_images", "teacher_negative_images"}:
                result.extend(collect_paths(item))
        return result
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--new-prefix", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    table = pq.read_table(args.input)
    rows = table.to_pylist()
    image_columns = {
        name
        for name in table.column_names
        if name in {"images", "teacher_images", "teacher_negative_images"}
    }
    missing_before: list[str] = []
    missing_after: list[str] = []
    for row in rows:
        for name in image_columns:
            for path in collect_paths(row.get(name)):
                if not Path(path).exists():
                    missing_before.append(path)
                rewritten = rewrite(path, args.old_prefix, args.new_prefix)
                if not Path(rewritten).exists():
                    missing_after.append(rewritten)

    print(f"rows={len(rows)} image_columns={sorted(image_columns)}")
    print(f"missing_recorded_paths={len(set(missing_before))}")
    if args.check_only:
        if missing_before:
            raise SystemExit("some recorded image paths do not exist")
        return
    if args.output is None:
        raise SystemExit("--output is required unless --check-only is used")

    for name in image_columns:
        values = [rewrite(row.get(name), args.old_prefix, args.new_prefix) for row in rows]
        column_index = table.column_names.index(name)
        table = table.set_column(column_index, name, pa.array(values, type=table[name].type))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output)
    print(f"wrote={args.output}")
    print(f"missing_rewritten_paths={len(set(missing_after))}")


if __name__ == "__main__":
    main()
