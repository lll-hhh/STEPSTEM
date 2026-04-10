#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.evaluate.accuracy_only import compare_final_answer  # type: ignore
from tools.evaluate.build_final_answer_review_dir import build_review_dir  # type: ignore
from tools.evaluate.score import read_json  # type: ignore
from tools.models.unified_model_platform import (  # type: ignore
    GenerationRequest,
    UnifiedModelPlatform,
    configure_cuda,
    resolve_device,
)


DEFAULT_SYSTEM = (
    "You are a careful judge for final-answer equivalence. "
    "Your only job is to decide whether the predicted final answer is semantically the same as "
    "one acceptable ground-truth answer, even if the formatting is different."
)

DEFAULT_PROMPT_TEMPLATE = """You are judging final-answer equivalence.

Task:
Decide whether the predicted final answer should count as correct against at least one acceptable ground-truth answer.

Important:
1. Focus on final-answer equivalence only.
2. Be lenient to formatting differences: punctuation, whitespace, LaTeX wrappers, surrounding words, ordering style, equivalent notation, and algebraically equivalent expressions.
3. Do give credit when the mathematical or semantic answer is the same but the formatting is different.
4. Do not give credit when the value, sign, unit meaning, set membership, interval boundary, or logical meaning is different.
5. Use the candidate response tail only as supporting context when the extracted predicted final answer is awkwardly formatted.
6. Output exactly one block between <judge_result> and </judge_result>.
7. Inside the block output a single JSON object with fields:
   "verdict": 0 or 1
   "matched_gt_index": integer index starting from 1, or 0 if none
   "reason": short string

Problem:
{question_text}

Predicted extracted final answer:
{pred_block}

Candidate response tail:
{raw_block}

Acceptable ground-truth final answers:
{gt_block}
"""


def extract_question_text(problem: Dict[str, Any]) -> str:
    parts: List[str] = []
    for item in problem.get("question", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("modality", "")).strip().lower() != "text":
            continue
        text = str(item.get("content", "") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def build_gt_map(index_obj: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p in index_obj.get("problems", []) or []:
        pid = str(p.get("id") or p.get("problem_id") or "").strip()
        if not pid:
            continue
        finals: List[Dict[str, Any]] = []
        for sol in p.get("solutions", []) or []:
            fa = sol.get("final_answer", {}) or {}
            if isinstance(fa, dict) and fa:
                finals.append(fa)
        if finals:
            out[pid] = finals
    return out


def one_line(s: Any) -> str:
    return str(s if s is not None else "").replace("\r", " ").replace("\n", " ").strip()


def raw_response_tail(pred_entry: Dict[str, Any], max_chars: int = 1200) -> str:
    raw = str(pred_entry.get("raw_response", "") or "").strip()
    if not raw:
        return ""
    if len(raw) <= max_chars:
        return raw
    return raw[-max_chars:]


def build_prompt(
    *,
    question_text: str,
    pred_final_answer: str,
    gt_answers: List[str],
    raw_tail: str,
    prompt_template: str,
) -> str:
    gt_block = "\n".join(f"{idx + 1}. {ans}" for idx, ans in enumerate(gt_answers))
    raw_block = raw_tail if raw_tail else "[EMPTY]"
    pred_block = pred_final_answer if str(pred_final_answer).strip() else "[EMPTY]"
    return prompt_template.format(
        question_text=question_text,
        pred_block=pred_block,
        raw_block=raw_block,
        gt_block=gt_block,
    )


def parse_judge_response(text: str) -> Tuple[float, int, str]:
    raw = str(text or "").strip()
    m = re.search(r"<judge_result>(.*?)</judge_result>", raw, flags=re.S | re.I)
    payload = m.group(1).strip() if m else raw
    try:
        obj = json.loads(payload)
        verdict = int(obj.get("verdict", 0))
        matched_gt_index = int(obj.get("matched_gt_index", 0))
        reason = str(obj.get("reason", "") or "").strip()
        return (1.0 if verdict == 1 else 0.0), matched_gt_index, reason
    except Exception:
        return 0.0, 0, "parse_error"


def row_has_final_score(row: Dict[str, Any]) -> bool:
    try:
        float(row.get("score", 0.0))
        return True
    except Exception:
        return False


def read_text_file(path_str: str, default: str) -> str:
    path = str(path_str or "").strip()
    if not path:
        return default
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge for final-answer equivalence.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--model", required=True, help="Judge model name in runtime registry")
    parser.add_argument("--model-registry", default="configs/models/runtime_model_registry.json")
    parser.add_argument("--model-path", default="", help="Optional local model path override")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", default="")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--system-file", default="", help="Optional file to override system prompt")
    parser.add_argument("--prompt-template-file", default="", help="Optional prompt template file with {question_text} {pred_block} {raw_block} {gt_block}")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--max-problems", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--review-dir", default="", help="Optional directory to export correct/wrong/image review files")
    args = parser.parse_args()

    index_obj = read_json(Path(args.index))
    pred_obj = read_json(Path(args.pred))
    if isinstance(pred_obj, dict) and "problems" in pred_obj and isinstance(pred_obj["problems"], list):
        pred_map = {str(p.get("id", "")): p for p in pred_obj["problems"] if isinstance(p, dict)}
    elif isinstance(pred_obj, dict):
        pred_map = pred_obj
    else:
        raise SystemExit("Unsupported prediction JSON format")

    gt_map = build_gt_map(index_obj)
    problems = index_obj.get("problems", []) or []
    if args.max_problems and int(args.max_problems) > 0:
        problems = problems[: int(args.max_problems)]

    out_path = Path(args.out)
    out_payload: Dict[str, Any] = {
        "index": args.index,
        "pred": args.pred,
        "judge_model": args.model,
        "judge_model_path": args.model_path or "",
        "device": args.device,
        "gpu_id": args.gpu_id,
        "dtype": args.dtype,
        "results": [],
    }
    system_prompt = read_text_file(args.system_file, args.system)
    prompt_template = read_text_file(args.prompt_template_file, DEFAULT_PROMPT_TEMPLATE)
    out_payload["system_prompt"] = system_prompt
    out_payload["prompt_template_file"] = str(args.prompt_template_file or "")
    done_map: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_path.exists():
        try:
            old = read_json(out_path)
            rows = old.get("results", []) if isinstance(old, dict) else []
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    pid = str(row.get("problem_id", "")).strip()
                    if pid and row_has_final_score(row):
                        done_map[pid] = row
                out_payload.update({k: v for k, v in old.items() if k != "results"})
                print(f"Resume: loaded {len(done_map)} existing rows")
        except Exception as e:
            print(f"Resume warning: failed to load {out_path}: {e}")

    platform = UnifiedModelPlatform(registry_path=args.model_registry)
    if args.model not in platform.registry:
        raise SystemExit(f"Unknown model: {args.model}")
    resolved_device = resolve_device(configure_cuda(args.gpu_id, args.device))
    runner, _info = platform.create_runner(
        model_name=args.model,
        model_path=(args.model_path or None),
        device=resolved_device,
        dtype=args.dtype,
    )

    started = time.time()
    newly_done = 0
    save_every = max(0, int(args.save_every))
    runner.load()
    try:
        for idx, problem in enumerate(problems, start=1):
            pid = str(problem.get("id", "")).strip()
            if not pid or pid in done_map:
                continue

            pred_entry = pred_map.get(pid, {}) if isinstance(pred_map, dict) else {}
            gt_answers = gt_map.get(pid, [])
            if not gt_answers:
                done_map[pid] = {
                    "problem_id": pid,
                    "score": 0.0,
                    "status": "missing_gt",
                }
                continue

            pred_fa = pred_entry.get("final_answer", {}) or {}
            best_strict = 0.0
            best_gt_index = 0
            strict_status = "text_mismatch"
            for gt_idx, gt_fa in enumerate(gt_answers, start=1):
                strict_score, status = compare_final_answer(pred_fa, gt_fa)
                if strict_score > best_strict:
                    best_strict = float(strict_score)
                    best_gt_index = gt_idx
                    strict_status = status
            if best_strict >= 1.0:
                done_map[pid] = {
                    "problem_id": pid,
                    "score": 1.0,
                    "status": "strict_match",
                    "strict_score": 1.0,
                    "llm_score": 1.0,
                    "matched_gt_index": best_gt_index,
                    "pred_final_answer": pred_fa,
                    "gt_final_answers": gt_answers,
                    "judge_reason": "strict_match",
                    "judge_raw_response": "",
                }
                newly_done += 1
                print(f"[{idx}/{len(problems)}] {pid}: strict_match", flush=True)
            else:
                pred_mod = str(pred_fa.get("modality", "") or "").strip().lower()
                gt_mods = {str((fa or {}).get("modality", "") or "").strip().lower() for fa in gt_answers}
                if pred_mod != "text" or gt_mods != {"text"}:
                    done_map[pid] = {
                        "problem_id": pid,
                        "score": 0.0,
                        "status": strict_status,
                        "strict_score": best_strict,
                        "llm_score": 0.0,
                        "matched_gt_index": best_gt_index,
                        "pred_final_answer": pred_fa,
                        "gt_final_answers": gt_answers,
                        "judge_reason": "non_text_or_modality_mismatch",
                        "judge_raw_response": "",
                    }
                    newly_done += 1
                    print(f"[{idx}/{len(problems)}] {pid}: {strict_status}", flush=True)
                else:
                    gt_text_answers = [str((fa or {}).get("answer", "") or "").strip() for fa in gt_answers]
                    prompt = build_prompt(
                        question_text=extract_question_text(problem),
                        pred_final_answer=str(pred_fa.get("answer", "") or "").strip(),
                        gt_answers=gt_text_answers,
                        raw_tail=raw_response_tail(pred_entry),
                        prompt_template=prompt_template,
                    )
                    req = GenerationRequest(
                        prompt=prompt,
                        system=system_prompt,
                        enable_thinking=bool(args.enable_thinking),
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        do_sample=bool(args.do_sample),
                    )
                    t0 = time.time()
                    judge_raw = runner.generate(req)
                    elapsed = float(time.time() - t0)
                    llm_score, matched_gt_index, reason = parse_judge_response(judge_raw)
                    done_map[pid] = {
                        "problem_id": pid,
                        "score": float(llm_score),
                        "status": "llm_match" if llm_score >= 1.0 else "llm_mismatch",
                        "strict_score": best_strict,
                        "llm_score": float(llm_score),
                        "matched_gt_index": int(matched_gt_index),
                        "pred_final_answer": pred_fa,
                        "gt_final_answers": gt_answers,
                        "judge_reason": reason,
                        "judge_elapsed_sec": round(elapsed, 3),
                        "judge_raw_response": judge_raw,
                    }
                    newly_done += 1
                    print(
                        f"[{idx}/{len(problems)}] {pid}: llm_score={llm_score:.1f} matched_gt={matched_gt_index}",
                        flush=True,
                    )

            if save_every > 0 and newly_done % save_every == 0:
                out_payload["results"] = [
                    done_map[str(p.get("id", "")).strip()]
                    for p in problems
                    if str(p.get("id", "")).strip() in done_map
                ]
                out_payload["num_problems"] = len(out_payload["results"])
                out_payload["total_elapsed_sec"] = round(time.time() - started, 3)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Checkpoint saved after {newly_done} newly completed problems", flush=True)
    finally:
        runner.unload()

    out_payload["results"] = [
        done_map[str(p.get("id", "")).strip()]
        for p in problems
        if str(p.get("id", "")).strip() in done_map
    ]
    out_payload["num_problems"] = len(out_payload["results"])
    out_payload["total_elapsed_sec"] = round(time.time() - started, 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    review_dir = str(args.review_dir or "").strip()
    if review_dir:
        summary = build_review_dir(
            index_obj=index_obj,
            pred_obj=pred_map,
            judge_obj=out_payload,
            out_dir=Path(review_dir),
        )
        print(f"Wrote review dir: {review_dir}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
