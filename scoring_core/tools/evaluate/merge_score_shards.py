#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_problem_ids_from_index(index_path: Path) -> List[str]:
    obj = read_json(index_path)
    problems = obj.get("problems", [])
    out: List[str] = []
    for p in problems:
        if isinstance(p, dict):
            pid = str(p.get("id", "")).strip()
            if pid:
                out.append(pid)
    return out


def find_shards(search_root: Path, pattern: str) -> List[Path]:
    return sorted(p for p in search_root.rglob(pattern) if p.is_file())


def merge_rows(shard_files: List[Path]) -> Tuple[Dict[str, dict], List[str], dict]:
    merged: Dict[str, dict] = {}
    sources: List[str] = []
    last_meta: dict = {}

    for path in shard_files:
        obj = read_json(path)
        rows = obj.get("results", [])
        if not isinstance(rows, list):
            continue
        last_meta = obj
        sources.append(str(path))
        for row in rows:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("problem_id", "")).strip()
            if not pid:
                continue
            merged[pid] = row
    return merged, sources, last_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge score shard json files into one score result.")
    parser.add_argument("--search-root", required=True, help="Root directory to scan for shard files")
    parser.add_argument("--pattern", default="score_shard*.json", help="Shard filename pattern")
    parser.add_argument("--index", required=True, help="Benchmark index json used to order problem ids")
    parser.add_argument("--pred", required=True, help="Prediction file path to record in merged output")
    parser.add_argument("--out", required=True, help="Output merged score json")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write output even if not all problem ids are present in found shards",
    )
    args = parser.parse_args()

    search_root = Path(args.search_root)
    shard_files = find_shards(search_root, args.pattern)
    if not shard_files:
        raise SystemExit(f"No shard files found under {search_root} with pattern {args.pattern}")

    merged_map, sources, last_meta = merge_rows(shard_files)
    ordered_ids = collect_problem_ids_from_index(Path(args.index))
    missing = [pid for pid in ordered_ids if pid not in merged_map]

    if missing and not args.allow_partial:
        preview = ", ".join(missing[:20])
        raise SystemExit(
            f"Merged shards are incomplete: missing {len(missing)} problems. "
            f"First missing: {preview}"
        )

    ordered_rows = [merged_map[pid] for pid in ordered_ids if pid in merged_map]

    out_obj = {
        "index": args.index,
        "pred": args.pred,
        "merged_from_shards": sources,
        "num_shards": len(shard_files),
        "num_problems": len(ordered_rows),
        "missing_problem_ids": missing,
        "text_score_mode": last_meta.get("text_score_mode"),
        "text_attention_tau": last_meta.get("text_attention_tau"),
        "text_step_source": last_meta.get("text_step_source"),
        "results": ordered_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"found_shards = {len(shard_files)}")
    print(f"merged_rows = {len(ordered_rows)}")
    print(f"missing_rows = {len(missing)}")
    print(f"wrote = {out_path}")


if __name__ == "__main__":
    main()
