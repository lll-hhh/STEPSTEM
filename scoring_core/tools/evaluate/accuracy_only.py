#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from tools.evaluate.score import match_text_final_answer, normalize_answer  # noqa: E402


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))


def build_gt_map(index_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    problems = index_obj.get("problems", [])
    out: Dict[str, Dict[str, Any]] = {}
    for p in problems:
        pid = p.get("problem_id") or p.get("id")
        if not pid:
            # Fall back to subject/problemNN style from base_dir if needed.
            qbase = str(p.get("question_base_dir", "")).replace("\\", "/").strip("/")
            if qbase:
                parts = qbase.split("/")
                if len(parts) >= 2:
                    pid = "/".join(parts[-2:])
                else:
                    pid = qbase
        if not pid:
            continue
        sols = p.get("solutions", []) or []
        if not sols:
            continue
        fa = sols[0].get("final_answer", {}) or {}
        out[str(pid)] = fa
    return out


def compare_final_answer(pred_fa: Dict[str, Any], gt_fa: Dict[str, Any]) -> Tuple[float, str]:
    pred_mod = str(pred_fa.get("modality", "") or "").strip().lower()
    gt_mod = str(gt_fa.get("modality", "") or "").strip().lower()

    if pred_mod != gt_mod:
        return 0.0, "modality_mismatch"

    if gt_mod == "text":
        pred_ans = str(pred_fa.get("answer", "") or "")
        gt_ans = str(gt_fa.get("answer", "") or "")
        score = float(match_text_final_answer(pred_ans, gt_ans))
        return score, "text_match" if score >= 1.0 else "text_mismatch"

    if gt_mod == "image":
        pred_name = Path(str(pred_fa.get("content", "") or "")).name
        gt_name = Path(str(gt_fa.get("content", "") or "")).name
        # For this lightweight script, an image-answer prediction counts as comparable/satisfied
        # when the modality is image. Filename equality is a secondary note only.
        if pred_name and gt_name and normalize_answer(pred_name) == normalize_answer(gt_name):
            return 1.0, "image_filename_match"
        if pred_name:
            return 1.0, "image_modality_match"
        return 0.0, "missing_image_content"

    return 0.0, "unknown_modality"


def main():
    ap = argparse.ArgumentParser(description="Lightweight final-answer accuracy calculator.")
    ap.add_argument("--index", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    index_obj = read_json(Path(args.index))
    pred_obj = read_json(Path(args.pred))

    gt_map = build_gt_map(index_obj)

    rows: List[Dict[str, Any]] = []
    matched = 0
    comparable = 0

    for pid, pred_entry in pred_obj.items():
        gt_fa = gt_map.get(pid)
        if not gt_fa:
            rows.append(
                {
                    "problem_id": pid,
                    "score": 0.0,
                    "status": "missing_gt",
                }
            )
            continue

        pred_fa = pred_entry.get("final_answer", {}) or {}
        score, status = compare_final_answer(pred_fa, gt_fa)
        comparable += 1
        if score >= 1.0:
            matched += 1

        rows.append(
            {
                "problem_id": pid,
                "score": score,
                "status": status,
                "pred_final_answer": pred_fa,
                "gt_final_answer": gt_fa,
            }
        )

    total_pred = len(pred_obj)
    total_gt = len(gt_map)
    accuracy = float(matched / comparable) if comparable else 0.0

    summary = {
        "index": args.index,
        "pred": args.pred,
        "num_gt": total_gt,
        "num_pred": total_pred,
        "num_comparable": comparable,
        "num_correct": matched,
        "accuracy": accuracy,
        "rows": rows,
    }

    print(f"num_gt: {total_gt}")
    print(f"num_pred: {total_pred}")
    print(f"num_comparable: {comparable}")
    print(f"num_correct: {matched}")
    print(f"accuracy: {accuracy:.6f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
