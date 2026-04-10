#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))


def stringify_final_answer(fa: Dict[str, Any]) -> str:
    if not isinstance(fa, dict):
        return ""
    return str(fa.get("answer", "") or fa.get("content", "") or "").replace("\r", " ").replace("\n", " ").strip()


def collect_problem_rows(index_obj: Dict[str, Any], pred_obj: Dict[str, Any], judge_rows: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], List[str], List[str]]:
    rows: List[Dict[str, Any]] = []
    correct_ids: List[str] = []
    wrong_ids: List[str] = []
    image_ids: List[str] = []

    for problem in index_obj.get("problems", []) or []:
        pid = str(problem.get("id") or problem.get("problem_id") or "").strip()
        if not pid:
            continue
        pred_entry = pred_obj.get(pid, {}) if isinstance(pred_obj, dict) else {}
        pred_fa = pred_entry.get("final_answer", {}) or {}
        pred_modality = str(pred_fa.get("modality", "") or "").strip().lower()
        pred_answer = stringify_final_answer(pred_fa)

        gt_modalities: List[str] = []
        gt_answers: List[str] = []
        for idx, sol in enumerate(problem.get("solutions", []) or [], start=1):
            gfa = sol.get("final_answer", {}) or {}
            gt_mod = str(gfa.get("modality", "") or "").strip().lower()
            gt_modalities.append(gt_mod)
            gt_answers.append(f"{idx}:{stringify_final_answer(gfa)}")

        is_image_problem = "image" in gt_modalities
        judge_row = judge_rows.get(pid, {})
        score = float(judge_row.get("score", 0.0) or 0.0) if isinstance(judge_row, dict) else 0.0
        status = str(judge_row.get("status", "") or "").strip() if isinstance(judge_row, dict) else ""

        if is_image_problem:
            bucket = "image"
            image_ids.append(pid)
        elif score >= 1.0:
            bucket = "correct"
            correct_ids.append(pid)
        else:
            bucket = "wrong"
            wrong_ids.append(pid)

        rows.append(
            {
                "problem_id": pid,
                "bucket": bucket,
                "pred_modality": pred_modality,
                "pred_final_answer": pred_answer,
                "gt_final_answers": " || ".join(gt_answers),
                "judge_score": score,
                "judge_status": status,
                "judge_reason": str(judge_row.get("judge_reason", "") or "").strip() if isinstance(judge_row, dict) else "",
            }
        )

    return rows, correct_ids, wrong_ids, image_ids


def write_lines(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tsv(path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines), encoding="utf-8")


def build_review_dir(*, index_obj: Dict[str, Any], pred_obj: Dict[str, Any], judge_obj: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    judge_rows = {
        str(r.get("problem_id", "")).strip(): r
        for r in (judge_obj.get("results", []) or [])
        if isinstance(r, dict) and str(r.get("problem_id", "")).strip()
    }

    rows, correct_ids, wrong_ids, image_ids = collect_problem_rows(index_obj, pred_obj, judge_rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_header = [
        "problem_id",
        "bucket",
        "pred_modality",
        "pred_final_answer",
        "gt_final_answers",
        "judge_score",
        "judge_status",
        "judge_reason",
    ]
    left_right_header = [
        "problem_id",
        "gt_final_answers",
        "pred_final_answer",
        "judge_score",
        "judge_status",
        "judge_reason",
    ]

    write_tsv(out_dir / "all_final_answers_side_by_side.tsv", all_header, rows)
    write_tsv(out_dir / "correct_gt_left.tsv", left_right_header, [r for r in rows if r["bucket"] == "correct"])
    write_tsv(out_dir / "wrong_gt_left.tsv", left_right_header, [r for r in rows if r["bucket"] == "wrong"])
    write_tsv(out_dir / "image_gt_left.tsv", left_right_header, [r for r in rows if r["bucket"] == "image"])

    write_lines(out_dir / "correct_ids.txt", correct_ids)
    write_lines(out_dir / "wrong_ids.txt", wrong_ids)
    write_lines(out_dir / "image_problem_ids.txt", image_ids)

    summary = {
        "num_total": len(rows),
        "num_correct": len(correct_ids),
        "num_wrong": len(wrong_ids),
        "num_image": len(image_ids),
        "index": judge_obj.get("index", ""),
        "pred": judge_obj.get("pred", ""),
        "judge_model": judge_obj.get("judge_model", ""),
        "judge_model_path": judge_obj.get("judge_model_path", ""),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final-answer review directory from LLM judge output.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--judge-json", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    index_obj = read_json(Path(args.index))
    pred_obj = read_json(Path(args.pred))
    judge_obj = read_json(Path(args.judge_json))
    if isinstance(pred_obj, dict) and "problems" in pred_obj and isinstance(pred_obj["problems"], list):
        pred_obj = {str(p.get("id", "")): p for p in pred_obj["problems"] if isinstance(p, dict)}

    summary = build_review_dir(
        index_obj=index_obj,
        pred_obj=pred_obj,
        judge_obj=judge_obj,
        out_dir=Path(args.out_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
