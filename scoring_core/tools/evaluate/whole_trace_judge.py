#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whole-trace LLM judge over GT step references.

Idea:
  - Feed the whole model response to a judge model.
  - Ask the judge to mark each GT text reference as covered (1) or not covered (0).
  - Ask the judge to mark each GT image reference as covered (1) or not covered (0)
    using the GT image plus its important bbox annotations.
  - Compute per-solution text/image/combined coverage scores and take the best
    solution per problem.

This is intentionally separate from score.py so it can be tested as an
alternative evaluation path without disturbing the existing scorer.
"""

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

from tools.evaluate.score import read_json, step_key  # type: ignore
from tools.models.unified_model_platform import (  # type: ignore
    GenerationRequest,
    UnifiedModelPlatform,
    configure_cuda,
    resolve_device,
)


DEFAULT_JUDGE_SYSTEM = "You are a generous, evidence-based judge for reasoning-process coverage."
DEFAULT_JUDGE_SOURCE = "content"


def one_line(v: Any) -> str:
    return str(v if v is not None else "").replace("\r", " ").replace("\n", " ").strip()


def synthesize_raw_response_from_steps(pred_entry: Dict[str, Any]) -> str:
    parts: List[str] = []
    steps = pred_entry.get("steps", [])
    if isinstance(steps, list):
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            mod = str(step.get("modality", "") or "").strip().lower()
            content = str(step.get("content", "") or "").strip()
            key_point = str(step.get("key_point", "") or "").strip()
            if mod == "text":
                text = content or key_point
                if text:
                    parts.append(text)
            elif mod == "image":
                label = content or f"image_step_{idx}"
                parts.append(f"[Generated image step {idx}: {label}]")

    final_answer = pred_entry.get("final_answer", {}) or {}
    if isinstance(final_answer, dict):
        fa_mod = str(final_answer.get("modality", "") or "").strip().lower()
        if fa_mod == "text":
            fa_text = str(final_answer.get("answer", "") or "").strip()
            if fa_text:
                parts.append(f"<final_answer>{fa_text}</final_answer>")
        elif fa_mod == "image":
            fa_img = str(final_answer.get("content", "") or "").strip()
            if fa_img:
                parts.append(f"<final_answer_image>{fa_img}</final_answer_image>")

    return "\n\n".join(p for p in parts if p).strip()


def candidate_raw_response(pred_entry: Dict[str, Any]) -> str:
    raw = str(pred_entry.get("raw_response", "") or "").strip()
    if raw:
        return raw
    return synthesize_raw_response_from_steps(pred_entry)


def collect_pred_image_paths(pred_entry: Dict[str, Any], pred_root: Path) -> List[str]:
    out: List[str] = []
    steps = pred_entry.get("steps", [])
    if not isinstance(steps, list):
        return out
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("modality", "")).strip().lower() != "image":
            continue
        content = str(step.get("content", "") or "").strip()
        if not content:
            continue
        resolved = _resolve_existing_path(content, base_dir=str(pred_root))
        if resolved:
            out.append(str(resolved))
    return out


def _path_candidates(path_str: str) -> List[Path]:
    s = str(path_str or "").strip()
    if not s:
        return []
    variants = [s]
    if "\\" in s:
        variants.append(s.replace("\\", "/"))
    if "/" in s:
        variants.append(s.replace("/", "\\"))
    out: List[Path] = []
    seen = set()
    for v in variants:
        if v in seen:
            continue
        seen.add(v)
        out.append(Path(v))
    return out


def _resolve_existing_path(path_str: str, base_dir: str = "") -> Optional[str]:
    rel = str(path_str or "").strip()
    if not rel:
        return None
    for p in _path_candidates(rel):
        if p.is_file():
            return str(p)
    base = str(base_dir or "").strip()
    if base:
        for b in _path_candidates(base):
            for r in _path_candidates(rel):
                cand = b / r
                if cand.is_file():
                    return str(cand)
    return None


def extract_question_text(problem: Dict[str, Any]) -> str:
    parts: List[str] = []
    for item in problem.get("question", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("modality", "")).strip().lower() != "text":
            continue
        text = str(item.get("content", "")).strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def resolve_question_image(problem: Dict[str, Any]) -> Optional[str]:
    rel_path = None
    for item in problem.get("question", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("modality", "")).strip().lower() != "image":
            continue
        rel_path = str(item.get("content", "")).strip()
        if rel_path:
            break
    if not rel_path:
        return None
    q_base = str(problem.get("question_base_dir", "") or "").strip()
    return _resolve_existing_path(rel_path, base_dir=q_base)


def step_text_for_judge(step: Dict[str, Any], judge_source: str) -> str:
    key_point = str(step.get("key_point", "") or "").strip()
    content = str(step.get("content", "") or "").strip()
    if judge_source == "content":
        return content or key_point
    if judge_source == "keypoint":
        return key_point or content
    if judge_source == "keypoint+content":
        if key_point and content:
            return f"Content: {content}\nKey point: {key_point}"
        return content or key_point
    return key_point or content


def collect_text_references(solution: Dict[str, Any], judge_source: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    items = sorted(solution.get("solution", {}).items(), key=step_key)
    for step_id, step in items:
        if not isinstance(step, dict):
            continue
        if str(step.get("modality", "")).strip().lower() != "text":
            continue
        text = step_text_for_judge(step, judge_source=judge_source)
        if text:
            rows.append((str(step_id), text))
    return rows


def resolve_solution_image(solution: Dict[str, Any], rel_path: str) -> Optional[str]:
    base_dir = str(solution.get("base_dir", "") or "").strip()
    return _resolve_existing_path(rel_path, base_dir=base_dir)


def _normalize_bbox_rows(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    boxes = step.get("important_bbox", [])
    if not isinstance(boxes, list) or not boxes:
        boxes = [{"bbox": [0.0, 0.0, 1.0, 1.0], "weight": 1.0}]
    out: List[Dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        bbox = box.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x, y, w, h = [float(v) for v in bbox]
        except Exception:
            continue
        out.append(
            {
                "bbox": [x, y, w, h],
                "weight": float(box.get("weight", 0.0)),
            }
        )
    if not out:
        out.append({"bbox": [0.0, 0.0, 1.0, 1.0], "weight": 1.0})
    return out


def collect_image_references(solution: Dict[str, Any], judge_source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    items = sorted(solution.get("solution", {}).items(), key=step_key)
    for step_id, step in items:
        if not isinstance(step, dict):
            continue
        if str(step.get("modality", "")).strip().lower() != "image":
            continue
        rel_path = str(step.get("content", "") or "").strip()
        image_path = resolve_solution_image(solution, rel_path)
        if not image_path:
            continue
        rows.append(
            {
                "step_id": str(step_id),
                "text": step_text_for_judge(step, judge_source=judge_source),
                "image_path": image_path,
                "image_rel_path": rel_path,
                "bboxes": _normalize_bbox_rows(step),
            }
        )
    return rows


def build_judge_prompt(
    *,
    question_text: str,
    raw_response: str,
    references: List[Tuple[str, str]],
) -> str:
    ref_lines = [f"{step_id}: {text}" for step_id, text in references]
    ref_block = "\n".join(ref_lines)
    return f"""You are a generous, evidence-based judge for reasoning-process coverage.

Task:
Given the original problem, a candidate model's whole reply, and a list of reference contents,
determine for each reference content whether the candidate reply semantically covers it.

Core principle:
Your goal is to judge semantic entailment and coverage, not literal wording overlap.
Prefer avoiding false negatives on actually correct reasoning.
When the candidate reply is mathematically consistent and clearly on the right solution path, be willing to count semantically implied references as covered.
Treat a clearly correct final derivation as positive evidence for intermediate coverage whenever the missing intermediate item is a standard consequence of that derivation.

Judging rules:
1. Match against the entire candidate reply, not only the final answer.
2. Mark 1 if the candidate reply explicitly states the reference content, clearly paraphrases it, or provides enough mathematical or semantic evidence that entails it.
3. Mark 1 if the candidate reply gives an algebraically equivalent formula, an equivalent constraint, an equivalent domain statement, or a stronger statement that clearly subsumes the reference.
4. Mark 1 if the evidence is distributed across multiple nearby or logically connected sentences or equations, as long as together they support the reference content.
5. Mark 1 if a later derivation, equation, or final expression clearly implies the reference content, even if the intermediate wording is omitted.
6. Mark 1 if the candidate reply reaches the same local conclusion or a directly equivalent stronger conclusion by a different but valid derivation path.
7. Do not require the same wording, the same notation, the same variable names, the same derivation order, or the same decomposition granularity as the reference.
8. Ignore harmless differences in formatting, equation rearrangement, symbol renaming, simplification, factorization, unit style, and equivalent notation.
9. If the candidate reply states a more concrete result that logically contains the reference idea, count the reference as covered.
10. Use a generous semantic matching standard for correct or near-correct reasoning traces; do not demand verbatim intermediate steps.
11. If the candidate reply provides the correct equation, invariant, constraint, ranking, or final symbolic result from which the reference naturally follows, count it as covered even when the explicit wording differs.
12. Mark 0 only if the reference content is truly missing, too vague to verify, merely topically related, or contradicted by the candidate reply.
13. Do not give 1 for generic topic overlap without concrete supporting evidence.
14. When uncertain between 0 and 1, prefer 1 if there is plausible mathematical or semantic evidence in the candidate reply.

Output format:
- Output exactly one block between <judge_result> and </judge_result>.
- Inside the block, output one line per reference in the format step_id=0 or step_id=1.
- Do not output explanations.

Problem:
{question_text}

Candidate whole reply:
{raw_response}

Reference contents:
{ref_block}
"""


def build_image_judge_prompt(
    *,
    question_text: str,
    raw_response: str,
    step_id: str,
    reference_text: str,
    bboxes: List[Dict[str, Any]],
    has_question_image: bool,
    num_pred_images: int,
) -> str:
    bbox_lines = []
    for idx, box in enumerate(bboxes, start=1):
        bbox = box.get("bbox", [0.0, 0.0, 1.0, 1.0])
        weight = float(box.get("weight", 0.0))
        bbox_lines.append(
            f"bbox{idx}=[x={bbox[0]:.4f}, y={bbox[1]:.4f}, w={bbox[2]:.4f}, h={bbox[3]:.4f}], weight={weight:.4f}"
        )
    bbox_block = "\n".join(bbox_lines) if bbox_lines else "bbox1=[x=0.0000, y=0.0000, w=1.0000, h=1.0000], weight=1.0000"
    image_inputs_desc = [
        "Image 1: optional question image for task context" if has_question_image else None,
        "Next image: the GT reference image step",
        f"Remaining {num_pred_images} image(s): candidate model generated image steps in original order",
    ]
    image_inputs_block = "\n".join(f"- {line}" for line in image_inputs_desc if line)
    return f"""You are a careful multimodal judge for reasoning-process coverage.

Task:
Given the original problem, a candidate model's whole reply, and a GT reference image step,
determine whether the candidate reply semantically covers the visual reasoning evidence
represented by the reference step.

You will receive multiple images as native model inputs in this exact order:
{image_inputs_block}

The important regions are specified by normalized bounding boxes over the GT reference image only.
You should compare the GT reference image against the candidate generated images and decide whether
the candidate generated images support the same visual reasoning evidence or conclusion associated
with this GT step.

Judging rules:
1. Consider both the reference text and the important GT image regions.
2. Mark 1 only if at least one candidate generated image semantically matches the GT visual evidence.
3. Pure text alone is not sufficient for image coverage. If the candidate has no relevant generated image
   evidence, mark 0.
4. Mark 1 if the candidate image evidence plus the candidate reply together capture the same visual idea,
   relation, geometry, highlight, or conclusion as the GT reference.
5. Mark 0 if the relevant visual evidence is missing from the candidate image panels, contradicted, or too
   vague to verify.
6. The question image, if provided, is context only. Do not count it as candidate evidence.
7. When uncertain between 0 and 1, prefer 0 unless there is visible evidence in the candidate image panels.

Output format:
- Output exactly one block between <judge_result> and </judge_result>.
- Inside the block, output exactly one line in the format {step_id}=0 or {step_id}=1.
- Do not output explanations.

Problem:
{question_text}

Candidate whole reply:
{raw_response}

Reference image step id:
{step_id}

Reference text:
{reference_text}

Important GT image regions:
{bbox_block}
"""


def build_image_judge_inputs(
    *,
    question_image: Optional[str],
    gt_image_path: str,
    pred_image_paths: List[str],
) -> List[str]:
    out: List[str] = []
    if str(question_image or "").strip():
        out.append(str(question_image).strip())
    if str(gt_image_path or "").strip():
        out.append(str(gt_image_path).strip())
    out.extend([str(p).strip() for p in pred_image_paths if str(p).strip()])
    deduped: List[str] = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


_JUDGE_RESULT_BLOCK_RE = re.compile(r"(?is)<judge_result>\s*(.*?)\s*</judge_result>")
_JUDGE_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*([01])\s*$")


def parse_judge_response(text: str, expected_step_ids: List[str]) -> Dict[str, int]:
    raw = str(text or "")
    m = _JUDGE_RESULT_BLOCK_RE.search(raw)
    body = m.group(1) if m else raw
    out: Dict[str, int] = {}
    for line in body.splitlines():
        mm = _JUDGE_LINE_RE.match(line.strip())
        if not mm:
            continue
        out[str(mm.group(1))] = int(mm.group(2))
    for step_id in expected_step_ids:
        out.setdefault(step_id, 0)
    return out


def _row_has_full_scores(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    required = (
        "whole_trace_judge_text_score",
        "whole_trace_judge_image_score",
        "whole_trace_judge_score",
    )
    return all(k in row for k in required)


def dump_outputs(
    out_path: Path,
    payload: Dict[str, Any],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Whole-trace LLM judge over GT text and image references")
    parser.add_argument("--index", default="data/benchmarks/mllm_bench/index.json", help="Benchmark index JSON")
    parser.add_argument("--pred", required=True, help="Prediction JSON file")
    parser.add_argument("--pred-root", default=".", help="Base dir for relative predicted image paths")
    parser.add_argument("--model", default="qwen35_9b", help="Judge model key in unified platform")
    parser.add_argument("--model-registry", default="configs/models/runtime_model_registry.json")
    parser.add_argument("--model-path", default="", help="Optional judge model local path override")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0...")
    parser.add_argument("--gpu-id", default="", help="GPU id(s) for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--system", default=DEFAULT_JUDGE_SYSTEM, help="System prompt for judge model")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable judge model thinking mode")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--judge-source",
        default=DEFAULT_JUDGE_SOURCE,
        choices=["content", "keypoint", "keypoint_or_content", "keypoint+content"],
        help="Which GT text field to expose to the judge",
    )
    parser.add_argument("--max-problems", type=int, default=0, help="0 means all")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint every N newly finished problems")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--pass-question-image", action="store_true", help="When judge model is VLM, also pass the question image")
    parser.add_argument("--out", default="reports/eval/whole_trace_judge.json", help="Output JSON path")
    args = parser.parse_args()

    index = read_json(Path(args.index))
    pred_payload = read_json(Path(args.pred))
    if "problems" in pred_payload and isinstance(pred_payload["problems"], list):
        pred_map = {str(p.get("id", "")): p for p in pred_payload["problems"] if isinstance(p, dict)}
    elif isinstance(pred_payload, dict):
        pred_map = pred_payload
    else:
        raise SystemExit("Unsupported prediction JSON format")

    problems = index.get("problems", [])
    if args.max_problems and int(args.max_problems) > 0:
        problems = problems[: int(args.max_problems)]

    platform = UnifiedModelPlatform(registry_path=args.model_registry)
    if args.model not in platform.registry:
        raise SystemExit(f"Unknown model: {args.model}")

    resolved_device = resolve_device(configure_cuda(args.gpu_id, args.device))
    runner, info = platform.create_runner(
        model_name=args.model,
        model_path=(args.model_path or None),
        device=resolved_device,
        dtype=args.dtype,
    )

    out_path = Path(args.out)
    pred_root = Path(args.pred_root)
    out_payload: Dict[str, Any] = {
        "index": args.index,
        "pred": args.pred,
        "pred_root": str(pred_root),
        "judge_model": args.model,
        "judge_model_path": args.model_path or platform.registry[args.model]["path"],
        "device": resolved_device,
        "gpu_id": args.gpu_id,
        "dtype": args.dtype,
        "judge_source": args.judge_source,
        "pass_question_image": bool(args.pass_question_image),
        "strict_modality_matching": True,
        "results": [],
    }
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
                    if pid and _row_has_full_scores(row):
                        done_map[pid] = row
                out_payload.update({k: v for k, v in old.items() if k != "results"})
                print(f"Resume: loaded {len(done_map)} existing problem results")
        except Exception as e:
            print(f"Resume warning: failed to load {out_path}: {e}")

    runner.load()
    started = time.time()
    save_every = max(0, int(args.save_every))
    newly_done = 0

    try:
        for idx, problem in enumerate(problems, start=1):
            pid = str(problem.get("id", "")).strip()
            if not pid:
                continue
            if pid in done_map:
                continue

            pred_entry = pred_map.get(pid, {}) if isinstance(pred_map, dict) else {}
            raw_response = candidate_raw_response(pred_entry)
            pred_image_paths = collect_pred_image_paths(pred_entry, pred_root=pred_root)
            question_text = extract_question_text(problem)
            question_image = resolve_question_image(problem) if args.pass_question_image else None
            if info.get("family") == "text":
                question_image = None

            solution_details: List[Dict[str, Any]] = []
            best_solution_id = None
            best_score = 0.0
            best_text_score = 0.0
            best_image_score = 0.0

            for sol in problem.get("solutions", []) or []:
                text_references = collect_text_references(sol, judge_source=args.judge_source)
                image_references = collect_image_references(sol, judge_source=args.judge_source)
                if not text_references and not image_references:
                    continue
                text_rows: List[Dict[str, Any]] = []
                image_rows: List[Dict[str, Any]] = []
                text_covered = 0
                image_covered = 0
                text_elapsed = 0.0
                image_elapsed = 0.0
                text_judge_raw = ""

                if text_references:
                    prompt = build_judge_prompt(
                        question_text=question_text,
                        raw_response=raw_response,
                        references=text_references,
                    )
                    req = GenerationRequest(
                        prompt=prompt,
                        system=args.system,
                        enable_thinking=bool(args.enable_thinking),
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        do_sample=bool(args.do_sample),
                    )
                    t0 = time.time()
                    text_judge_raw = runner.generate(req)
                    text_elapsed = float(time.time() - t0)
                    parsed = parse_judge_response(text_judge_raw, [step_id for step_id, _ in text_references])
                    text_covered = sum(int(parsed.get(step_id, 0)) for step_id, _ in text_references)
                    text_rows = [
                        {
                            "step_id": step_id,
                            "text": text,
                            "covered": int(parsed.get(step_id, 0)),
                        }
                        for step_id, text in text_references
                    ]

                if image_references and pred_image_paths and info.get("family") == "text":
                    raise RuntimeError(
                        f"Image judging for problem {pid} requires a multimodal judge model when predicted image steps exist."
                    )

                for ref in image_references:
                    if not pred_image_paths:
                        image_rows.append(
                            {
                                "step_id": str(ref.get("step_id", "")),
                                "text": str(ref.get("text", "")),
                                "image_rel_path": str(ref.get("image_rel_path", "")),
                                "image_path": str(ref.get("image_path", "")),
                                "bboxes": list(ref.get("bboxes", [])),
                                "covered": 0,
                                "judge_elapsed_sec": 0.0,
                                "judge_raw_response": "No predicted image steps available; image coverage forced to 0.",
                            }
                        )
                        continue

                    prompt = build_image_judge_prompt(
                        question_text=question_text,
                        raw_response=raw_response,
                        step_id=str(ref.get("step_id", "")),
                        reference_text=str(ref.get("text", "")),
                        bboxes=list(ref.get("bboxes", [])),
                        has_question_image=bool(question_image),
                        num_pred_images=len(pred_image_paths),
                    )
                    judge_images = build_image_judge_inputs(
                        question_image=question_image,
                        gt_image_path=str(ref.get("image_path", "")),
                        pred_image_paths=pred_image_paths,
                    )
                    req = GenerationRequest(
                        prompt=prompt,
                        images=judge_images if info.get("family") != "text" else None,
                        system=args.system,
                        enable_thinking=bool(args.enable_thinking),
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        do_sample=bool(args.do_sample),
                    )
                    t0 = time.time()
                    judge_raw = runner.generate(req)
                    elapsed = float(time.time() - t0)
                    image_elapsed += elapsed
                    parsed = parse_judge_response(judge_raw, [str(ref.get("step_id", ""))])
                    covered = int(parsed.get(str(ref.get("step_id", "")), 0))
                    image_covered += covered
                    image_rows.append(
                        {
                            "step_id": str(ref.get("step_id", "")),
                            "text": str(ref.get("text", "")),
                            "image_rel_path": str(ref.get("image_rel_path", "")),
                            "image_path": str(ref.get("image_path", "")),
                            "judge_image_inputs": judge_images,
                            "bboxes": list(ref.get("bboxes", [])),
                            "covered": covered,
                            "judge_elapsed_sec": round(elapsed, 3),
                            "judge_raw_response": judge_raw,
                        }
                    )

                num_text_refs = len(text_references)
                num_image_refs = len(image_references)
                num_refs = num_text_refs + num_image_refs
                text_score = float(text_covered / num_text_refs) if num_text_refs else 0.0
                image_score = float(image_covered / num_image_refs) if num_image_refs else 0.0
                score = float((text_covered + image_covered) / num_refs) if num_refs else 0.0
                row = {
                    "solution_id": str(sol.get("solution_id", "")),
                    "num_text_references": num_text_refs,
                    "matched_text_references": text_covered,
                    "text_score": text_score,
                    "num_image_references": num_image_refs,
                    "matched_image_references": image_covered,
                    "image_score": image_score,
                    "num_references": num_refs,
                    "matched_references": text_covered + image_covered,
                    "score": score,
                    "judge_elapsed_sec": round(text_elapsed + image_elapsed, 3),
                    "judge_raw_response": text_judge_raw,
                    "text_references": text_rows,
                    "image_references": image_rows,
                    "references": text_rows,
                }
                solution_details.append(row)
                if best_solution_id is None or score > best_score:
                    best_score = score
                    best_text_score = text_score
                    best_image_score = image_score
                    best_solution_id = str(sol.get("solution_id", ""))

            problem_result = {
                "problem_id": pid,
                "best_solution_id": best_solution_id,
                "whole_trace_judge_text_score": float(best_text_score),
                "whole_trace_judge_image_score": float(best_image_score),
                "whole_trace_judge_score": float(best_score),
                "raw_response_preview": one_line(raw_response)[:400],
                "solution_details": solution_details,
            }
            done_map[pid] = problem_result
            newly_done += 1
            print(f"[{idx}/{len(problems)}] {pid}: judge_score={best_score:.4f}", flush=True)

            if save_every > 0 and newly_done % save_every == 0:
                out_payload["results"] = [done_map[str(p.get("id", "")).strip()] for p in problems if str(p.get("id", "")).strip() in done_map]
                out_payload["num_problems"] = len(out_payload["results"])
                out_payload["total_elapsed_sec"] = round(time.time() - started, 3)
                dump_outputs(out_path, out_payload)
                print(f"Checkpoint saved after {newly_done} newly completed problems", flush=True)
    finally:
        runner.unload()

    out_payload["results"] = [done_map[str(p.get("id", "")).strip()] for p in problems if str(p.get("id", "")).strip() in done_map]
    out_payload["num_problems"] = len(out_payload["results"])
    out_payload["total_elapsed_sec"] = round(time.time() - started, 3)
    dump_outputs(out_path, out_payload)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
