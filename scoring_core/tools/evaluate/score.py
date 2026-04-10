import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.common.utils import max_ordered_matching


DEFAULT_IMAGE_DIVERSITY_LAMBDA = 0.2
DEFAULT_TEXT_SCORE_MODE = "conditional_diff"
TEXT_SCORE_MODES = ("conditional_diff", "attention_rollout_sum_prob")

class NullImageScorer:
    def score(self, gt_img: Path, pred_img: Path, bboxes: List[Dict[str, Any]]) -> float:
        return 0.0


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))


def _normalize_rel_path_str(value: str) -> str:
    return str(value or "").strip().replace("\\", "/")


def _resolve_repo_path(path_str: str) -> Path:
    """
    Resolve dataset/index paths robustly across Windows-generated JSON and Linux runs.
    """
    raw = str(path_str or "").strip()
    if not raw:
        return Path("")
    p = Path(raw)
    if p.is_absolute():
        return p
    norm = _normalize_rel_path_str(raw)
    return ROOT_DIR / Path(norm)


def _resolve_pred_path(pred_root: Path, rel_path: str) -> Path:
    raw = str(rel_path or "").strip()
    if not raw:
        return Path("")
    p = Path(raw)
    if p.is_absolute():
        return p
    norm = _normalize_rel_path_str(raw)
    return pred_root / Path(norm)


def _split_pred_text_step(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand a predicted text step into newline-separated substeps.
    Image steps are handled elsewhere and should bypass this helper.
    """
    content = str(step.get("content", "") or "")
    key_point = str(step.get("key_point", "") or "")
    raw = content if content.strip() else key_point
    if not raw.strip():
        return []

    parts = [seg.strip() for seg in raw.splitlines()]
    parts = [seg for seg in parts if not is_noisy_pred_step_text(seg)]
    if not parts:
        compact = raw.strip()
        return [{"modality": "text", "content": compact}] if compact else []

    return [{"modality": "text", "content": seg} for seg in parts]


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    return s


_SYMBOL_ONLY_STEP_RE = re.compile(r"^[\|\`_\-=/\\\[\]\(\)\{\}\^<>:;,\.\*\+\~\s]+$")


def is_noisy_pred_step_text(s: str) -> bool:
    """
    Filter newline-split artifacts (ASCII frame lines / punctuation-only fragments).
    """
    t = str(s or "").strip()
    if not t:
        return True
    if t == "```":
        return True
    if _SYMBOL_ONLY_STEP_RE.fullmatch(t):
        return True
    if len(t) <= 2 and not any(ch.isalnum() for ch in t):
        return True
    return False


def _strip_latex_wrappers(s: str) -> str:
    t = str(s or "").strip()
    t = t.replace("$", "")
    t = t.replace("\\left", "").replace("\\right", "")
    return t.strip()


def _latex_frac_to_python(s: str) -> str:
    out = str(s or "")
    for _ in range(20):
        nxt = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", out)
        if nxt == out:
            break
        out = nxt
    return out


def _to_sympy_expr(s: str):
    try:
        import sympy as sp
    except Exception:
        return None

    t = _strip_latex_wrappers(s)
    if not t:
        return None
    t = _latex_frac_to_python(t)
    t = t.replace("^", "**")
    # Robust normalization for common math glyphs (protect against encoding issues).
    t = t.replace("π", "pi").replace("\\pi", "pi")
    t = t.replace("×", "*").replace("·", "*")
    t = t.replace("{", "(").replace("}", ")")
    # Keep identifiers like v_0 / r_0 as symbols.
    names = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", t))
    if not names:
        names = set()

    locals_map: Dict[str, Any] = {}
    for fn in ("sin", "cos", "tan", "exp", "log", "sqrt", "asin", "acos", "atan"):
        locals_map[fn] = getattr(sp, fn)
    locals_map["pi"] = sp.pi
    locals_map["E"] = sp.E
    for name in names:
        if name not in locals_map:
            locals_map[name] = sp.symbols(name)
    try:
        return sp.sympify(t, locals=locals_map)
    except Exception:
        return None


def algebraically_equivalent(a: str, b: str) -> bool:
    """
    Return True when two text answers are algebraically equivalent.
    This is a fallback after strict normalized-string match fails.
    """
    ea = _to_sympy_expr(a)
    eb = _to_sympy_expr(b)
    if ea is None or eb is None:
        return False
    try:
        import sympy as sp

        d = sp.simplify(ea - eb)
        if d == 0:
            return True
        try:
            return abs(float(d.evalf())) < 1e-9
        except Exception:
            return False
    except Exception:
        return False


def match_text_final_answer(pred_answer: str, gt_answer: str) -> float:
    p = str(pred_answer or "").strip()
    g = str(gt_answer or "").strip()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    if normalize_answer(p) == normalize_answer(g):
        return 1.0
    if algebraically_equivalent(p, g):
        return 1.0
    return 0.0


def step_key(item: Tuple[str, Dict[str, Any]]) -> int:
    step_id = item[0]
    m = re.search(r"(\d+)", step_id)
    return int(m.group(1)) if m else 0


def summarize_stats(values: List[float], default: float = 0.0) -> Tuple[float, float, float, float]:
    if not values:
        v = float(default)
        return v, v, v, 0.0
    best = float(max(values))
    worst = float(min(values))
    mean = float(sum(values) / len(values))
    var = float(sum((x - mean) ** 2 for x in values) / len(values))
    std = float(math.sqrt(var))
    return best, worst, mean, std


def compute_gt_anchor_image_m2m(
    image_score_matrix: List[List[float]],
    diversity_lambda: float = DEFAULT_IMAGE_DIVERSITY_LAMBDA,
) -> Dict[str, Any]:
    lam = float(max(0.0, min(1.0, diversity_lambda)))
    if not image_score_matrix:
        return {
            "coverage": 0.0,
            "diversity_ratio": 0.0,
            "score": 0.0,
            "argmax_pred_local_indices": [],
            "max_scores": [],
            "unique_pred_hits": 0,
        }

    n_gt = len(image_score_matrix)
    argmax_pred_local_indices: List[int] = []
    max_scores: List[float] = []

    for row in image_score_matrix:
        if not row:
            argmax_pred_local_indices.append(-1)
            max_scores.append(0.0)
            continue
        best_j = 0
        best_s = float(row[0])
        for j, s in enumerate(row[1:], start=1):
            ss = float(s)
            if ss > best_s:
                best_s = ss
                best_j = j
        argmax_pred_local_indices.append(best_j)
        max_scores.append(float(max(0.0, min(1.0, best_s))))

    coverage = float(sum(max_scores) / n_gt) if n_gt else 0.0
    unique_hits = len({j for j in argmax_pred_local_indices if j >= 0})
    diversity_ratio = float(unique_hits / n_gt) if n_gt else 0.0
    score = float(coverage * ((1.0 - lam) + lam * diversity_ratio))
    score = float(max(0.0, min(1.0, score)))

    return {
        "coverage": coverage,
        "diversity_ratio": diversity_ratio,
        "score": score,
        "argmax_pred_local_indices": argmax_pred_local_indices,
        "max_scores": max_scores,
        "unique_pred_hits": unique_hits,
    }


class LocalQwenTextScorer:
    def __init__(
        self,
        model_dir: Path,
        device: str = "auto",
        max_prefix_tokens: int = 4096,
        max_pred_tokens: int = 1024,
        mode: str = DEFAULT_TEXT_SCORE_MODE,
        attention_tau: float = 0.3,
    ):
        if not model_dir.exists():
            raise ValueError(f"Local text model not found: {model_dir}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_dir = str(model_dir)
        self.cache: Dict[Tuple[str, str], float] = {}
        self.max_prefix_tokens = max_prefix_tokens
        self.max_pred_tokens = max_pred_tokens
        self.mode = str(mode or DEFAULT_TEXT_SCORE_MODE).strip().lower()
        if self.mode not in TEXT_SCORE_MODES:
            raise ValueError(f"Unsupported text score mode: {self.mode}")
        self.attention_tau = float(attention_tau)
        self._pure_prefix_length = None
        self._active_ref_prefix = None
        self._active_ref: Optional[str] = None
        self._active_ref_past = None
        self._active_ref_next_logits = None
        self.stats: Dict[str, int] = {
            "ensure_ref_cache_errors": 0,
            "candidate_errors": 0,
            "candidate_oom_errors": 0,
        }

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_load_kwargs = {
            "trust_remote_code": True,
        }
        # Attention rollout requires real attention tensors; force eager attention
        # because sdpa can ignore output_attentions=True in this environment.
        if self.mode == "attention_rollout_sum_prob":
            model_load_kwargs["attn_implementation"] = "eager"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                dtype=dtype,
                **model_load_kwargs,
            ).to(self.device).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=dtype,
                **model_load_kwargs,
            ).to(self.device).eval()

    def _make_prefix(self, ref: str) -> Tuple[str, str]:
        # return (
        #     "You are a strict grader.\n"
        #     "The reference key point as well as the predicted reasoning step are provided. "
        #     "You are required to evaluate if the predicted reasoning step contains the contents "
        #     "of key point. 1 for most semantically consistent, and 0 for least semantically "
        #     "consistent.\n\n"
        #     f"Reference key point:\n{ref}\n\n"
        #     "Predicted reasoning step:\n"
        # )
        prompt1 = f"""
You are a strict grader.

Your goal is to determine whether the Predicted reasoning step between <predicted> and </predicted> semantically contains the Reference key point between <reference> and </reference>.

Judging procedure:
1. Identify the essential propositions in the Reference key point.
2. Check whether each essential proposition is present in the Predicted reasoning step, allowing paraphrase.
3. If all essential propositions are covered and there is no contradiction, output 1.
4. Otherwise output 0.

Rules:
- Paraphrase counts as match.
- Missing any essential proposition => 0.
- Contradiction => 0.
- Mere topic similarity => 0.
- Extra correct information is acceptable.
- Exactly same => 1.
- Same information => 1.

Reference key point: <reference>{ref}</reference>

Predicted reasoning step: <predicted>
"""
        prompt2 = f"""
You are a strict grader.

Your goal is to determine whether the Predicted reasoning step semantically contains the Reference key point.

Judging procedure:
1. Identify the essential propositions in the Reference key point.
2. Check whether each essential proposition is present in the Predicted reasoning step, allowing paraphrase.
3. If all essential propositions are covered and there is no contradiction, output 1.
4. Otherwise output 0.

Rules:
- Paraphrase counts as match.
- Missing any essential proposition => 0.
- Contradiction => 0.
- Mere topic similarity => 0.
- Extra correct information is acceptable.
- Exactly same => 1.
- Same information => 1.

"""
        return prompt1, prompt2

    def _make_attention_instruction(self) -> str:
        _, instruction = self._make_prefix("")
        return instruction.strip()

    def _encode_piece(self, text: str) -> List[int]:
        return self.tokenizer(str(text or ""), add_special_tokens=False)["input_ids"]

    def _build_attention_input(self, ref: str, pred: str):
        pieces = [
            ("instruction_prefix", "Instruction:\n"),
            ("instruction", self._make_attention_instruction()),
            ("reference_prefix", "\n\nReference:\n"),
            ("reference", str(ref or "")),
            ("prediction_prefix", "\n\nPrediction:\n"),
            ("prediction", str(pred or "")),
        ]

        ids: List[int] = []
        spans: Dict[str, Tuple[int, int]] = {}

        if self.tokenizer.bos_token_id is not None:
            ids.append(self.tokenizer.bos_token_id)

        for name, text in pieces:
            part_ids = self._encode_piece(text)
            start = len(ids)
            ids.extend(part_ids)
            end = len(ids)
            spans[name] = (start, end)

        input_ids = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        attention_mask = self.torch.ones_like(input_ids)
        return input_ids, attention_mask, spans

    def _forward_with_attentions(self, input_ids, attention_mask):
        with self.torch.no_grad():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

    def _softmax_with_temperature(self, x):
        tau = max(float(self.attention_tau), 1e-6)
        x = x.float() / tau
        x = x - x.max()
        return self.torch.softmax(x, dim=-1)

    def _compute_attention_rollout_weights(self, outputs, spans):
        attentions = outputs.attentions
        if not attentions:
            return None, None

        seq_len = attentions[0].shape[-1]
        device = attentions[0].device
        rollout = self.torch.eye(seq_len, dtype=self.torch.float32, device=device)

        for layer_attn in attentions:
            attn = layer_attn[0].float().mean(dim=0)
            attn = attn + self.torch.eye(seq_len, dtype=attn.dtype, device=attn.device)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            rollout = attn @ rollout

        ref_start, ref_end = spans["reference"]
        pred_start, pred_end = spans["prediction"]
        if pred_end <= pred_start:
            return None, None

        ref_positions = self.torch.arange(ref_start, ref_end, device=device)
        pred_positions = self.torch.arange(pred_start, pred_end, device=device)
        raw_scores = rollout[pred_positions][:, ref_positions].sum(dim=-1)
        weights = self._softmax_with_temperature(raw_scores)
        return raw_scores, weights

    def _compute_prediction_token_logprobs(self, outputs, input_ids, spans):
        pred_start, pred_end = spans["prediction"]
        if pred_end <= pred_start:
            return None
        if pred_start == 0:
            raise ValueError("Prediction span starts at position 0, cannot score causally.")

        logits = outputs.logits[0].float()
        log_probs = self.torch.log_softmax(logits, dim=-1)
        target_ids = input_ids[0, pred_start:pred_end]
        pred_positions = self.torch.arange(pred_start, pred_end, device=logits.device)
        context_positions = pred_positions - 1
        return log_probs[context_positions, target_ids]

    def _ensure_ref_cache(self, ref: str):
        if self._active_ref == ref and self._active_ref_past is not None:
            return
        prefix, instruction = self._make_prefix(ref)
        if self._pure_prefix_length is None:
            self._pure_prefix_length = self.tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prefix_tokens
            )["input_ids"].size(-1)
        toks = self.tokenizer(
            prefix,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prefix_tokens
        ).to(self.device)

        with self.torch.no_grad():
            if self._active_ref_prefix is None:

                input_ids_instruction = toks["input_ids"][:, :self._pure_prefix_length]
                attention_mask_instruction = None
                if toks.get("attention_mask") is not None:
                    attention_mask_instruction = toks["attention_mask"][:, :self._pure_prefix_length]

                out_instruction = self.model(
                    input_ids=input_ids_instruction,
                    attention_mask=attention_mask_instruction,
                    use_cache=True,
                    return_dict=True,
                )

                self._active_ref_prefix = out_instruction

        with self.torch.no_grad():
            out = self.model(
                input_ids=toks["input_ids"],
                attention_mask=toks.get("attention_mask"),
                use_cache=True,
                return_dict=True
            )
        self._active_ref = ref
        self._active_ref_past = out.past_key_values
        self._active_ref_next_logits = out.logits[:, -1, :].detach()

    def _score_conditional_prob(self, pred: str) -> Optional[float]:
        """
        Native continuous score in [0,1]:
        score = exp(mean_t log P(d_t | k,h,d_<t)) - exp(mean_t log P(d_t | h,d_<t))
        """
        if self._active_ref_past is None or self._active_ref_next_logits is None:
            return None

        pred_ids = self.tokenizer(
            pred,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_pred_tokens,
            add_special_tokens=False
        )["input_ids"][0]
        if pred_ids.numel() == 0:
            return 0.0

        pred_ids = pred_ids.to(self.device)
        n = int(pred_ids.numel())
        # Keep accumulation in fp32 to avoid fp16/bf16 underflow (which can
        # collapse moderate mismatches to exact 0.0 in long runs).
        all_logps = self.torch.empty(n, device=self.device, dtype=self.torch.float32)
        all_logps2 = self.torch.empty(n, device=self.device, dtype=self.torch.float32)

        first_log_probs = self.torch.log_softmax(self._active_ref_next_logits[0].float(), dim=-1)
        all_logps[0] = first_log_probs[pred_ids[0]]
        if self._active_ref_prefix is not None and getattr(self._active_ref_prefix, "logits", None) is not None:
            prefix_log_probs = self.torch.log_softmax(self._active_ref_prefix.logits[0, -1, :].float(), dim=-1)
            all_logps2[0] = prefix_log_probs[pred_ids[0]]
        else:
            all_logps2[0] = all_logps[0]

        if n > 1:
            with self.torch.no_grad():
                out = self.model(
                    input_ids=pred_ids[:-1].unsqueeze(0),
                    past_key_values=self._active_ref_past,
                    use_cache=False,
                    return_dict=True
                )
            next_log_probs = self.torch.log_softmax(out.logits[0].float(), dim=-1)
            targets = pred_ids[1:]
            all_logps[1:] = next_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            with self.torch.no_grad():
                out2 = self.model(
                    input_ids=pred_ids[:-1].unsqueeze(0),
                    past_key_values=self._active_ref_prefix.past_key_values,
                    use_cache=True,
                    return_dict=True
                )
            next_log_probs2 = self.torch.log_softmax(out2.logits[0].float(), dim=-1)
            targets = pred_ids[1:]
            all_logps2[1:] = next_log_probs2.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        score = float(self.torch.exp(all_logps.mean()).item() -
                      self.torch.exp(all_logps2.mean()).item())
        return max(0.0, min(1.0, score))

    def _score_attention_rollout_sum_prob(self, ref: str, pred: str) -> Optional[float]:
        """
        Keep the new method's aggregation exactly as requested:
        score = sum_i w_i * exp(log p(x_i | d, h))
        where w_i comes from all-layer attention rollout over reference tokens.
        """
        input_ids, attention_mask, spans = self._build_attention_input(ref, pred)
        outputs = self._forward_with_attentions(input_ids, attention_mask)
        _, weights = self._compute_attention_rollout_weights(outputs, spans)
        token_logprobs = self._compute_prediction_token_logprobs(outputs, input_ids, spans)
        if weights is None or token_logprobs is None:
            return 0.0
        token_probs = self.torch.exp(token_logprobs.float())
        score = float((weights.float() * token_probs).sum().item())
        return max(0.0, min(1.0, score))

    def score(self, ref: str, pred: str) -> float:
        scores = self.score_many(ref, [pred])
        return float(scores[0]) if scores else 0.0

    def score_many(self, ref: str, preds: List[str]) -> List[float]:
        ref = str(ref or "")
        preds = [str(p or "") for p in (preds or [])]
        if not preds:
            return []

        out_scores: List[Optional[float]] = [None for _ in preds]

        ref_n = normalize_answer(ref)
        for i, pred in enumerate(preds):
            key = (ref, pred)
            if key in self.cache:
                out_scores[i] = float(self.cache[key])
                continue

            pred_n = normalize_answer(pred)
            if not ref_n and not pred_n:
                out_scores[i] = 1.0
            elif not ref_n or not pred_n:
                out_scores[i] = 0.0

            if out_scores[i] is not None:
                self.cache[key] = float(out_scores[i])

        unresolved = [i for i, s in enumerate(out_scores) if s is None]
        if not unresolved:
            return [float(s) for s in out_scores]

        if self.mode == "conditional_diff":
            try:
                self._ensure_ref_cache(ref)
            except Exception:
                self.stats["ensure_ref_cache_errors"] += 1
                for i in unresolved:
                    pred = preds[i]
                    out_scores[i] = 0.0
                    self.cache[(ref, pred)] = 0.0
                return [float(s if s is not None else 0.0) for s in out_scores]

        # Isolate failures per candidate; do not zero the whole unresolved batch.
        for i in unresolved:
            pred = preds[i]
            try:
                if self.mode == "attention_rollout_sum_prob":
                    score = self._score_attention_rollout_sum_prob(ref, pred)
                else:
                    score = self._score_conditional_prob(pred)
                if score is None:
                    score = 0.0
            except Exception as e:
                self.stats["candidate_errors"] += 1
                msg = str(e).lower()
                if "out of memory" in msg or "cuda out of memory" in msg:
                    self.stats["candidate_oom_errors"] += 1
                    try:
                        if self.device == "cuda" and self.torch.cuda.is_available():
                            self.torch.cuda.empty_cache()
                    except Exception:
                        pass
                score = 0.0
            out_scores[i] = float(score)
            self.cache[(ref, pred)] = float(score)

        return [float(s if s is not None else 0.0) for s in out_scores]


class ResNet50ImageScorer:
    def __init__(self, device: Optional[str] = None, weights_path: Optional[Path] = None):
        import torch
        from torchvision import models, transforms
        from torchvision.models import ResNet50_Weights

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet50_Weights.DEFAULT
        if weights_path and Path(weights_path).exists():
            model = models.resnet50(weights=None)
            try:
                state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(str(weights_path), map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            model = models.resnet50(weights=weights)
        # feature extractor (no avgpool/fc)
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-2]).to(self.device).eval()
        preset = weights.transforms()
        mean = getattr(preset, "mean", [0.485, 0.456, 0.406])
        std = getattr(preset, "std", [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.cache: Dict[str, Any] = {}

    def _load_feature_map(self, img_path: Path):
        key = str(img_path)
        if key in self.cache:
            return self.cache[key]
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.feature_extractor(x)  # (1, C, H, W)
        feat = feat.squeeze(0).cpu()  # (C, H, W)
        self.cache[key] = feat
        return feat

    def _cosine(self, a, b) -> float:
        # a, b: (C,)
        a = a / (a.norm() + 1e-8)
        b = b / (b.norm() + 1e-8)
        return float((a * b).sum().item())

    def score(self, gt_img: Path, pred_img: Path, bboxes: List[Dict[str, Any]]) -> float:
        if not bboxes:
            return 0.0
        gt_feat = self._load_feature_map(gt_img)
        pred_feat = self._load_feature_map(pred_img)

        _, H, W = gt_feat.shape
        _, Hp, Wp = pred_feat.shape
        if (H, W) != (Hp, Wp):
            # should not happen with fixed transform, but guard anyway
            H = min(H, Hp)
            W = min(W, Wp)
            gt_feat = gt_feat[:, :H, :W]
            pred_feat = pred_feat[:, :H, :W]

        total = 0.0
        for box in bboxes:
            bbox = box.get("bbox")
            weight = float(box.get("weight", 0.0))
            if not bbox or len(bbox) != 4:
                continue
            x, y, bw, bh = bbox
            x0 = int(max(0, min(W - 1, x * W)))
            y0 = int(max(0, min(H - 1, y * H)))
            x1 = int(max(x0 + 1, min(W, (x + bw) * W)))
            y1 = int(max(y0 + 1, min(H, (y + bh) * H)))
            dx = max(1, x1 - x0)
            dy = max(1, y1 - y0)

            gt_vec = gt_feat[:, y0:y1, x0:x1].mean(dim=(1, 2))
            best = -1.0
            for py in range(0, H - dy + 1):
                for px in range(0, W - dx + 1):
                    pred_vec = pred_feat[:, py:py + dy, px:px + dx].mean(dim=(1, 2))
                    sim = self._cosine(gt_vec, pred_vec)
                    if sim > best:
                        best = sim
            # map cosine [-1,1] -> [0,1]
            score = (best + 1.0) / 2.0
            total += weight * score

        # clamp
        if total < 0:
            return 0.0
        if total > 1:
            return 1.0
        return total


def extract_pred_steps(pred_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_steps: List[Dict[str, Any]] = []
    if "steps" in pred_entry and isinstance(pred_entry["steps"], list):
        raw_steps = [s for s in pred_entry["steps"] if isinstance(s, dict)]
    elif "solution" in pred_entry and isinstance(pred_entry["solution"], dict):
        items = sorted(pred_entry["solution"].items(), key=step_key)
        raw_steps = [v for _, v in items if isinstance(v, dict)]

    expanded_steps: List[Dict[str, Any]] = []
    for step in raw_steps:
        mod = str(step.get("modality", "")).strip().lower()
        if mod == "image":
            expanded_steps.append(step)
        elif mod == "text":
            expanded_steps.extend(_split_pred_text_step(step))
        else:
            expanded_steps.append(step)
    return expanded_steps


def is_image_path(s: str) -> bool:
    s = str(s or "").strip().lower()
    return s.endswith(".png") or s.endswith(".jpg") or s.endswith(".jpeg") or s.endswith(".bmp") or s.endswith(".webp")


def extract_pred_final(pred_entry: Dict[str, Any]) -> Dict[str, Any]:
    fa = pred_entry.get("final_answer")
    if isinstance(fa, dict):
        mod = str(fa.get("modality", "")).strip().lower()
        content = str(fa.get("content", "")).strip()
        answer = str(fa.get("answer", "")).strip()
        if mod == "image":
            return {"modality": "image", "content": content}
        if mod == "text":
            return {"modality": "text", "answer": answer}
        # infer if modality missing
        if content and is_image_path(content):
            return {"modality": "image", "content": content}
        return {"modality": "text", "answer": answer}
    if isinstance(fa, str):
        v = fa.strip()
        if is_image_path(v):
            return {"modality": "image", "content": v}
        return {"modality": "text", "answer": v}
    return {"modality": "text", "answer": ""}


def step_text_for_scoring_gt(step: Dict[str, Any], text_step_source: str) -> str:
    """
    Build GT textual content used in step scoring.
    text_step_source:
      - keypoint: key_point only (fallback to content if key_point missing)
      - content:  content only (fallback to key_point if content missing)
      - keypoint+content: content with key_point as auxiliary context
    """
    content = str(step.get("content", "") or "").strip()
    key_point = str(step.get("key_point", "") or "").strip()

    if text_step_source == "keypoint":
        return key_point or content
    if text_step_source == "content":
        return content or key_point

    # keypoint+content (default behavior)
    if content and key_point:
        if normalize_answer(content) == normalize_answer(key_point):
            return content
        return f"{content}\n\nKey point: {key_point}"
    if content:
        return content
    return key_point


def step_text_for_scoring_pred(step: Dict[str, Any]) -> str:
    """
    Build predicted-step text for scoring.
    Prediction is primarily content; key_point is only fallback.
    """
    content = str(step.get("content", "") or "").strip()
    key_point = str(step.get("key_point", "") or "").strip()
    # Do not filter predicted text steps here; keep raw textual stream.
    return content or key_point


def score_problem(
    problem: Dict[str, Any],
    pred_entry: Dict[str, Any],
    text_scorer,
    image_scorer,
    pred_root: Path,
    text_step_source: str = "keypoint+content",
):
    # final answer score (text/image aware, best over references)
    pred_final = extract_pred_final(pred_entry)

    def score_final_against_solution(sol: Dict[str, Any]) -> float:
        fa = sol.get("final_answer", {})
        if not isinstance(fa, dict):
            gt_answer = str(fa).strip()
            pred_answer = str(pred_final.get("answer", "")).strip()
            return float(match_text_final_answer(pred_answer, gt_answer))

        gt_mod = str(fa.get("modality", "text")).strip().lower()
        if gt_mod == "image":
            if str(pred_final.get("modality", "")).strip().lower() != "image":
                return 0.0
            gt_img = _resolve_repo_path(Path(_normalize_rel_path_str(sol["base_dir"])) / _normalize_rel_path_str(fa.get("content", "")))
            pred_img = _resolve_pred_path(pred_root, pred_final.get("content", ""))
            if not gt_img.exists() or not pred_img.exists():
                return 0.0
            bboxes = fa.get("important_bbox", [])
            if not bboxes:
                bboxes = [{"bbox": [0.0, 0.0, 1.0, 1.0], "weight": 1.0}]
            return float(image_scorer.score(gt_img, pred_img, bboxes))

        gt_answer = str(fa.get("answer", "")).strip()
        pred_answer = str(pred_final.get("answer", "")).strip()
        return float(match_text_final_answer(pred_answer, gt_answer))

    final_score = 0.0
    for sol in problem["solutions"]:
        fs = score_final_against_solution(sol)
        if fs > final_score:
            final_score = fs

    pred_steps = extract_pred_steps(pred_entry)

    solution_details = []

    for sol in problem["solutions"]:
        gt_steps_items = sorted(sol["solution"].items(), key=step_key)
        gt_steps = [v for _, v in gt_steps_items if isinstance(v, dict)]
        if not gt_steps:
            continue

        text_ref: List[str] = []
        image_ref: List[Dict[str, Any]] = []
        text_gt_idx: List[int] = []
        image_gt_idx: List[int] = []
        gt_step_modalities: List[str] = []
        for gi, gt_step in enumerate(gt_steps):
            mod = str(gt_step.get("modality", "")).strip().lower()
            gt_step_modalities.append(mod)
            if mod == "text":
                text_ref.append(step_text_for_scoring_gt(gt_step, text_step_source=text_step_source))
                text_gt_idx.append(gi)
            elif mod == "image":
                image_ref.append(gt_step)
                image_gt_idx.append(gi)

        text_pred = []
        for pred_step in pred_steps:
            if str(pred_step.get("modality", "")).strip().lower() != "text":
                continue
            t = step_text_for_scoring_pred(pred_step)
            if t:
                text_pred.append(t)
        image_pred = [
            pred_step
            for pred_step in pred_steps
            if str(pred_step.get("modality", "")).strip().lower() == "image"
        ]

        def compute_single_image_score(gt_step: Dict[str, Any], pred_step: Dict[str, Any]) -> float:
            gt_img = _resolve_repo_path(Path(_normalize_rel_path_str(sol["base_dir"])) / _normalize_rel_path_str(gt_step.get("content", "")))
            pred_img = _resolve_pred_path(pred_root, pred_step.get("content", ""))
            if not gt_img.exists() or not pred_img.exists():
                return 0.0
            bboxes = gt_step.get("important_bbox", [])
            if not bboxes:
                bboxes = [{"bbox": [0.0, 0.0, 1.0, 1.0], "weight": 1.0}]
            return float(image_scorer.score(gt_img, pred_img, bboxes))

        text_score_list = [[float(x) for x in text_scorer.score_many(ref, text_pred)] for ref in text_ref]
        image_score_list = [
            [compute_single_image_score(gt_step, pred_step) for pred_step in image_pred]
            for gt_step in image_ref
        ]

        gt_text_local_by_global = {gidx: lidx for lidx, gidx in enumerate(text_gt_idx)}
        gt_image_local_by_global = {gidx: lidx for lidx, gidx in enumerate(image_gt_idx)}
        pred_text_global_idx = [
            idx
            for idx in range(len(pred_steps))
            if str(pred_steps[idx].get("modality", "")).strip().lower() == "text"
        ]
        pred_image_global_idx = [
            idx
            for idx in range(len(pred_steps))
            if str(pred_steps[idx].get("modality", "")).strip().lower() == "image"
        ]
        pred_text_local_by_global = {gidx: lidx for lidx, gidx in enumerate(pred_text_global_idx)}
        pred_image_local_by_global = {gidx: lidx for lidx, gidx in enumerate(pred_image_global_idx)}

        text_score_sum, text_match_pairs = max_ordered_matching(text_score_list)
        image_score_sum, image_match_pairs = max_ordered_matching(image_score_list)

        joint_score_list: List[List[float]] = []
        for gi, gt_step in enumerate(gt_steps):
            gt_mod = str(gt_step.get("modality", "")).strip().lower()
            row: List[float] = []
            for pj, pred_step in enumerate(pred_steps):
                pred_mod = str(pred_step.get("modality", "")).strip().lower()
                score = 0.0
                if gt_mod == "text" and pred_mod == "text":
                    gt_local = gt_text_local_by_global.get(gi)
                    pred_local = pred_text_local_by_global.get(pj)
                    if gt_local is not None and pred_local is not None:
                        if gt_local < len(text_score_list) and pred_local < len(text_score_list[gt_local]):
                            score = float(text_score_list[gt_local][pred_local])
                elif gt_mod == "image" and pred_mod == "image":
                    gt_local = gt_image_local_by_global.get(gi)
                    pred_local = pred_image_local_by_global.get(pj)
                    if gt_local is not None and pred_local is not None:
                        if gt_local < len(image_score_list) and pred_local < len(image_score_list[gt_local]):
                            score = float(image_score_list[gt_local][pred_local])
                row.append(score)
            joint_score_list.append(row)

        joint_score_sum, joint_match_pairs = max_ordered_matching(joint_score_list)

        n_gt_text = len(text_ref)
        n_gt_image = len(image_ref)
        n_gt_total = n_gt_text + n_gt_image
        # Legacy score: text DP and image DP are computed independently, then merged.
        step_score_legacy = float((text_score_sum + image_score_sum) / n_gt_total) if n_gt_total else 0.0
        # Standalone text DP and image DP/m2m.
        text_dp_score = float(text_score_sum / n_gt_text) if n_gt_text else 0.0
        image_step_score_dp_legacy = float(image_score_sum / n_gt_image) if n_gt_image else 0.0
        image_m2m = compute_gt_anchor_image_m2m(image_score_list)
        image_m2m_score = float(image_m2m["score"])
        w_text = float(n_gt_text / n_gt_total) if n_gt_total else 0.0
        w_image = float(n_gt_image / n_gt_total) if n_gt_total else 0.0
        textdp_image_m2m_score = float(w_text * text_dp_score + w_image * image_m2m_score)

        joint_text_score_sum = 0.0
        joint_image_score_sum = 0.0
        for gi, pj in joint_match_pairs:
            if gi >= len(gt_steps) or pj >= len(pred_steps):
                continue
            gt_mod = str(gt_steps[gi].get("modality", "")).strip().lower()
            pred_mod = str(pred_steps[pj].get("modality", "")).strip().lower()
            if gt_mod != pred_mod:
                continue
            score = 0.0
            if gi < len(joint_score_list) and pj < len(joint_score_list[gi]):
                score = float(joint_score_list[gi][pj])
            if gt_mod == "text":
                joint_text_score_sum += score
            elif gt_mod == "image":
                joint_image_score_sum += score

        dp_text_score = float(joint_text_score_sum / n_gt_text) if n_gt_text else 0.0
        dp_image_score = float(joint_image_score_sum / n_gt_image) if n_gt_image else 0.0
        joint_dp_score = float(joint_score_sum / n_gt_total) if n_gt_total else 0.0

        # Pred-axis / GT-axis joint-DP scores.
        step_scores_dp = [0.0] * len(pred_steps)
        gt_step_scores_dp = [0.0] * len(gt_steps)
        step_scores_legacy = [0.0] * len(pred_steps)
        gt_step_scores_legacy = [0.0] * len(gt_steps)
        # GT-axis best-hit scores: max over predicted steps for each GT step.
        gt_step_scores_best = [0.0] * len(gt_steps)
        text_idx = pred_text_global_idx
        image_idx = pred_image_global_idx
        for i, j in text_match_pairs:
            if i < len(text_score_list) and j < len(text_idx):
                step_scores_legacy[text_idx[j]] = float(text_score_list[i][j])
            if i < len(text_score_list) and i < len(text_gt_idx) and j < len(text_pred):
                gt_step_scores_legacy[text_gt_idx[i]] = float(text_score_list[i][j])
        for i, j in image_match_pairs:
            if i < len(image_score_list) and j < len(image_idx):
                step_scores_legacy[image_idx[j]] = float(image_score_list[i][j])
            if i < len(image_score_list) and i < len(image_gt_idx) and j < len(image_pred):
                gt_step_scores_legacy[image_gt_idx[i]] = float(image_score_list[i][j])

        for i, j in joint_match_pairs:
            if i < len(joint_score_list) and j < len(pred_steps):
                step_scores_dp[j] = float(joint_score_list[i][j])
            if i < len(joint_score_list) and i < len(gt_steps) and j < len(pred_steps):
                gt_step_scores_dp[i] = float(joint_score_list[i][j])

        for i, row in enumerate(text_score_list):
            if i < len(text_gt_idx):
                gt_step_scores_best[text_gt_idx[i]] = float(max(row)) if row else 0.0
        for i, row in enumerate(image_score_list):
            if i < len(image_gt_idx):
                if row:
                    gt_step_scores_best[image_gt_idx[i]] = float(max(row))
                elif i < len(image_m2m.get("max_scores", [])):
                    gt_step_scores_best[image_gt_idx[i]] = float(image_m2m["max_scores"][i])

        image_argmax_local = list(image_m2m.get("argmax_pred_local_indices", []))
        image_argmax_global = [
            image_idx[pj] if isinstance(pj, int) and pj >= 0 and pj < len(image_idx) else None
            for pj in image_argmax_local
        ]
        unique_global_hits = sorted({x for x in image_argmax_global if isinstance(x, int)})

        solution_details.append(
            {
                "solution_id": sol["solution_id"],
                "step_score": joint_dp_score,
                "step_score_legacy": step_score_legacy,
                "step_score_v2": textdp_image_m2m_score,
                "joint_dp_score": joint_dp_score,
                "textdp_image_m2m_score": textdp_image_m2m_score,
                "dp_text_score": dp_text_score,
                "dp_image_score": dp_image_score,
                "text_dp_score": text_dp_score,
                "image_m2m_score": image_m2m_score,
                "text_step_score": text_dp_score,
                "image_step_score_dp": image_step_score_dp_legacy,
                "image_step_score_v2": image_m2m_score,
                "image_coverage": float(image_m2m["coverage"]),
                "image_diversity_ratio": float(image_m2m["diversity_ratio"]),
                "image_gt_argmax_pred_local_indices": image_argmax_local,
                "image_gt_argmax_pred_global_indices": image_argmax_global,
                "image_unique_pred_hits": len(unique_global_hits),
                "image_unique_pred_global_indices": unique_global_hits,
                "weights": {"text": w_text, "image": w_image},
                "gt_step_modalities": gt_step_modalities,
                "gt_step_scores": gt_step_scores_dp,
                "gt_step_scores_legacy": gt_step_scores_legacy,
                "gt_step_scores_best": gt_step_scores_best,
                "step_scores": step_scores_dp,
                "step_scores_legacy": step_scores_legacy,
            }
        )

    def collect_metric(metric: str) -> List[float]:
        return [float(x.get(metric, 0.0)) for x in solution_details]

    def best_solution_id_for(metric: str) -> Optional[str]:
        if not solution_details:
            return None
        best_row = max(solution_details, key=lambda x: float(x.get(metric, 0.0)))
        sid = best_row.get("solution_id")
        return str(sid) if sid is not None else None

    def best_solution_row_for(metric: str) -> Optional[Dict[str, Any]]:
        if not solution_details:
            return None
        return max(solution_details, key=lambda x: float(x.get(metric, 0.0)))

    per_solution_scores_joint_dp = collect_metric("joint_dp_score")
    joint_dp_score_best, joint_dp_score_min, joint_dp_score_mean, joint_dp_score_std = summarize_stats(
        per_solution_scores_joint_dp,
        default=0.0,
    )
    solution_gap = float(joint_dp_score_best - joint_dp_score_min)
    per_solution_scores_dp_legacy = collect_metric("step_score_legacy")
    step_score_legacy_best, step_score_legacy_min, step_score_legacy_mean, step_score_legacy_std = summarize_stats(
        per_solution_scores_dp_legacy,
        default=0.0,
    )

    per_solution_scores_dp_text = collect_metric("dp_text_score")
    dp_text_score_best, dp_text_score_min, dp_text_score_mean, dp_text_score_std = summarize_stats(
        per_solution_scores_dp_text,
        default=0.0,
    )

    per_solution_scores_dp_image = collect_metric("dp_image_score")
    dp_image_score_best, dp_image_score_min, dp_image_score_mean, dp_image_score_std = summarize_stats(
        per_solution_scores_dp_image,
        default=0.0,
    )

    per_solution_scores_text = collect_metric("text_dp_score")
    text_dp_score_best, text_dp_score_min, text_dp_score_mean, text_dp_score_std = summarize_stats(
        per_solution_scores_text,
        default=0.0,
    )

    per_solution_scores_image_dp = collect_metric("image_step_score_dp")
    image_step_score_dp_legacy_best, image_step_score_dp_legacy_min, image_step_score_dp_legacy_mean, image_step_score_dp_legacy_std = summarize_stats(
        per_solution_scores_image_dp,
        default=0.0,
    )

    per_solution_scores_image_v2 = collect_metric("image_m2m_score")
    image_m2m_score_best, image_m2m_score_min, image_m2m_score_mean, image_m2m_score_std = summarize_stats(
        per_solution_scores_image_v2,
        default=0.0,
    )

    per_solution_scores_v2 = collect_metric("textdp_image_m2m_score")
    textdp_image_m2m_score_best, textdp_image_m2m_score_min, textdp_image_m2m_score_mean, textdp_image_m2m_score_std = summarize_stats(
        per_solution_scores_v2,
        default=0.0,
    )
    solution_gap_v2 = float(textdp_image_m2m_score_best - textdp_image_m2m_score_min)

    best_joint_dp_detail = best_solution_row_for("joint_dp_score")
    best_solution_id = str(best_joint_dp_detail.get("solution_id", "")) if best_joint_dp_detail else None
    best_solution_id_joint_dp = best_solution_id
    best_v2_detail = best_solution_row_for("textdp_image_m2m_score")
    best_solution_id_v2 = str(best_v2_detail.get("solution_id", "")) if best_v2_detail else None

    return {
        "problem_id": problem["id"],
        "final_score": final_score,
        "step_score": joint_dp_score_best,
        "step_score_best": joint_dp_score_best,
        "step_score_min": joint_dp_score_min,
        "step_score_mean": joint_dp_score_mean,
        "step_score_std": joint_dp_score_std,
        "solution_gap": solution_gap,
        "best_solution_id": best_solution_id,
        "joint_dp_score": joint_dp_score_best,
        "joint_dp_score_best": joint_dp_score_best,
        "joint_dp_score_min": joint_dp_score_min,
        "joint_dp_score_mean": joint_dp_score_mean,
        "joint_dp_score_std": joint_dp_score_std,
        "best_solution_id_joint_dp": best_solution_id_joint_dp,
        "step_score_legacy": step_score_legacy_best,
        "step_score_legacy_best": step_score_legacy_best,
        "step_score_legacy_min": step_score_legacy_min,
        "step_score_legacy_mean": step_score_legacy_mean,
        "step_score_legacy_std": step_score_legacy_std,
        "dp_text_score": float(best_joint_dp_detail.get("dp_text_score", 0.0)) if best_joint_dp_detail else 0.0,
        "dp_text_score_best": dp_text_score_best,
        "dp_text_score_min": dp_text_score_min,
        "dp_text_score_mean": dp_text_score_mean,
        "dp_text_score_std": dp_text_score_std,
        "best_solution_id_dp_text": best_solution_id_joint_dp,
        "dp_image_score": float(best_joint_dp_detail.get("dp_image_score", 0.0)) if best_joint_dp_detail else 0.0,
        "dp_image_score_best": dp_image_score_best,
        "dp_image_score_min": dp_image_score_min,
        "dp_image_score_mean": dp_image_score_mean,
        "dp_image_score_std": dp_image_score_std,
        "best_solution_id_dp_image": best_solution_id_joint_dp,
        "text_step_score": float(best_v2_detail.get("text_dp_score", 0.0)) if best_v2_detail else 0.0,
        "text_step_score_best": text_dp_score_best,
        "text_step_score_min": text_dp_score_min,
        "text_step_score_mean": text_dp_score_mean,
        "text_step_score_std": text_dp_score_std,
        "text_dp_score": float(best_v2_detail.get("text_dp_score", 0.0)) if best_v2_detail else 0.0,
        "text_dp_score_best": text_dp_score_best,
        "text_dp_score_min": text_dp_score_min,
        "text_dp_score_mean": text_dp_score_mean,
        "text_dp_score_std": text_dp_score_std,
        "best_solution_id_text": best_solution_id_v2,
        "image_step_score_dp": image_step_score_dp_legacy_best,
        "image_step_score_dp_best": image_step_score_dp_legacy_best,
        "image_step_score_dp_min": image_step_score_dp_legacy_min,
        "image_step_score_dp_mean": image_step_score_dp_legacy_mean,
        "image_step_score_dp_std": image_step_score_dp_legacy_std,
        "best_solution_id_image_dp": best_solution_id_joint_dp,
        "image_step_score_v2": float(best_v2_detail.get("image_m2m_score", 0.0)) if best_v2_detail else 0.0,
        "image_step_score_v2_best": image_m2m_score_best,
        "image_step_score_v2_min": image_m2m_score_min,
        "image_step_score_v2_mean": image_m2m_score_mean,
        "image_step_score_v2_std": image_m2m_score_std,
        "image_m2m_score": float(best_v2_detail.get("image_m2m_score", 0.0)) if best_v2_detail else 0.0,
        "image_m2m_score_best": image_m2m_score_best,
        "image_m2m_score_min": image_m2m_score_min,
        "image_m2m_score_mean": image_m2m_score_mean,
        "image_m2m_score_std": image_m2m_score_std,
        "best_solution_id_image_v2": best_solution_id_v2,
        "image_coverage": float(best_v2_detail.get("image_coverage", 0.0)) if best_v2_detail else 0.0,
        "image_diversity_ratio": float(best_v2_detail.get("image_diversity_ratio", 0.0)) if best_v2_detail else 0.0,
        "step_score_v2": textdp_image_m2m_score_best,
        "step_score_v2_best": textdp_image_m2m_score_best,
        "step_score_v2_min": textdp_image_m2m_score_min,
        "step_score_v2_mean": textdp_image_m2m_score_mean,
        "step_score_v2_std": textdp_image_m2m_score_std,
        "textdp_image_m2m_score": textdp_image_m2m_score_best,
        "textdp_image_m2m_score_best": textdp_image_m2m_score_best,
        "textdp_image_m2m_score_min": textdp_image_m2m_score_min,
        "textdp_image_m2m_score_mean": textdp_image_m2m_score_mean,
        "textdp_image_m2m_score_std": textdp_image_m2m_score_std,
        "solution_gap_v2": solution_gap_v2,
        "best_solution_id_v2": best_solution_id_v2,
        "solution_details": solution_details,
    }


def build_trace_problem_from_score_result(
    problem_id: str,
    pred_entry: Dict[str, Any],
    score_result: Dict[str, Any],
) -> Dict[str, Any]:
    pred_steps = extract_pred_steps(pred_entry)
    solution_details = score_result.get("solution_details", [])
    if not isinstance(solution_details, list):
        solution_details = []

    solutions = []
    for sd in solution_details:
        if not isinstance(sd, dict):
            continue
        solutions.append(
            {
                "solution_id": sd.get("solution_id", ""),
                "step_score": float(sd.get("step_score", 0.0)),
                "step_score_legacy": float(sd.get("step_score_legacy", 0.0)),
                "joint_dp_score": float(sd.get("joint_dp_score", sd.get("step_score", 0.0))),
                "step_score_v2": float(sd.get("step_score_v2", 0.0)),
                "textdp_image_m2m_score": float(sd.get("textdp_image_m2m_score", sd.get("step_score_v2", 0.0))),
                "dp_text_score": float(sd.get("dp_text_score", 0.0)),
                "dp_image_score": float(sd.get("dp_image_score", 0.0)),
                "text_step_score": float(sd.get("text_step_score", 0.0)),
                "text_dp_score": float(sd.get("text_dp_score", sd.get("text_step_score", 0.0))),
                "image_step_score_dp": float(sd.get("image_step_score_dp", 0.0)),
                "image_step_score_v2": float(sd.get("image_step_score_v2", 0.0)),
                "image_m2m_score": float(sd.get("image_m2m_score", sd.get("image_step_score_v2", 0.0))),
                "image_coverage": float(sd.get("image_coverage", 0.0)),
                "image_diversity_ratio": float(sd.get("image_diversity_ratio", 0.0)),
                "weights": sd.get("weights", {"text": 0.0, "image": 0.0}),
                "gt_step_modalities": sd.get("gt_step_modalities", []),
                "gt_step_scores": sd.get("gt_step_scores", []),
                "gt_step_scores_legacy": sd.get("gt_step_scores_legacy", []),
                "gt_step_scores_best": sd.get("gt_step_scores_best", []),
                "step_scores": sd.get("step_scores", []),
                "step_scores_legacy": sd.get("step_scores_legacy", []),
                "image_gt_argmax_pred_local_indices": sd.get("image_gt_argmax_pred_local_indices", []),
                "image_gt_argmax_pred_global_indices": sd.get("image_gt_argmax_pred_global_indices", []),
                "image_unique_pred_hits": int(sd.get("image_unique_pred_hits", 0)),
                "image_unique_pred_global_indices": sd.get("image_unique_pred_global_indices", []),
            }
        )

    return {
        "problem_id": problem_id,
        "raw_response": str(pred_entry.get("raw_response", "")),
        "pred_steps": pred_steps,
        "best_solution_id": score_result.get("best_solution_id"),
        "best_step_score": float(score_result.get("step_score", 0.0)),
        "best_joint_dp_score": float(score_result.get("joint_dp_score", score_result.get("step_score", 0.0))),
        "best_dp_text_score": float(score_result.get("dp_text_score", 0.0)),
        "best_dp_image_score": float(score_result.get("dp_image_score", 0.0)),
        "best_text_dp_score": float(score_result.get("text_dp_score", score_result.get("text_step_score", 0.0))),
        "best_image_m2m_score": float(score_result.get("image_m2m_score", score_result.get("image_step_score_v2", 0.0))),
        "best_solution_id_v2": score_result.get("best_solution_id_v2"),
        "best_step_score_v2": float(score_result.get("step_score_v2", 0.0)),
        "best_textdp_image_m2m_score": float(score_result.get("textdp_image_m2m_score", score_result.get("step_score_v2", 0.0))),
        "best_final_score": float(score_result.get("final_score", 0.0)),
        "solutions": solutions,
    }


def _write_score_payload(
    out_path: Path,
    *,
    index_path: str,
    pred_path: str,
    text_score_mode: str,
    text_attention_tau: float,
    text_step_source: str,
    ordered_results: List[Dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "index": index_path,
                "pred": pred_path,
                "text_score_mode": text_score_mode,
                "text_attention_tau": text_attention_tau,
                "text_step_source": text_step_source,
                "results": ordered_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_trace_payload(
    trace_path: Path,
    *,
    index_path: str,
    pred_path: str,
    text_step_source: str,
    ordered_trace_rows: List[Dict[str, Any]],
) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "index": index_path,
                "pred": pred_path,
                "text_step_source": text_step_source,
                "num_problems": len(ordered_trace_rows),
                "problems": ordered_trace_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/benchmarks/xxbench/index.json", help="Index JSON file")
    parser.add_argument("--pred", required=True, help="Prediction JSON file")
    parser.add_argument("--pred-root", default=".", help="Base dir for prediction image paths")
    # Text scorer is fixed to local qwen KV-cache.
    # Keep this hidden legacy arg for backward-compatible CLI calls.
    parser.add_argument("--text-scorer", default="qwen", choices=["qwen"], help=argparse.SUPPRESS)
    parser.add_argument("--text-local-model", default=os.environ.get("QWEN3_LOCAL_MODEL_DIR", "models/Qwen3.5-9B"), help="Local qwen model directory")
    parser.add_argument("--text-device", default=os.environ.get("TEXT_SCORE_DEVICE", "auto"), help="Text scorer device: auto/cpu/cuda")
    parser.add_argument(
        "--text-score-mode",
        default=os.environ.get("TEXT_SCORE_MODE", DEFAULT_TEXT_SCORE_MODE),
        choices=list(TEXT_SCORE_MODES),
        help="Text scorer mode",
    )
    parser.add_argument(
        "--text-attention-tau",
        type=float,
        default=float(os.environ.get("TEXT_ATTENTION_TAU", "0.3")),
        help="Temperature used when normalizing all-layer attention weights",
    )
    parser.add_argument(
        "--text-step-source",
        default=os.environ.get("TEXT_STEP_SOURCE", "keypoint+content"),
        choices=["keypoint", "content", "keypoint+content"],
        help="GT text source for step matching: keypoint/content/keypoint+content",
    )
    parser.add_argument("--image-scorer", default="resnet50", choices=["resnet50", "none"], help="Image step scorer")
    parser.add_argument("--resnet50-weights", default=os.environ.get("RESNET50_WEIGHTS_PATH", "models/resnet50/resnet50-11ad3fa6.pth"), help="Local resnet50 .pth path; fallback to torchvision download if not found")
    parser.add_argument("--progress", action="store_true", help="Print per-problem progress")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N problems")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file if present")
    parser.add_argument("--save-every", type=int, default=0, help="Checkpoint every N newly scored problems")
    parser.add_argument("--out", default="reports/eval/score_results.json", help="Output score file")
    parser.add_argument(
        "--trace-out",
        default="",
        help="Optional: also write full_match_trace-style JSON in the same scoring pass",
    )
    args = parser.parse_args()

    index = read_json(Path(args.index))
    pred_data = read_json(Path(args.pred))
    if "problems" in pred_data and isinstance(pred_data["problems"], list):
        pred_map = {p["id"]: p for p in pred_data["problems"]}
    elif isinstance(pred_data, dict):
        pred_map = pred_data
    else:
        raise SystemExit("Unsupported prediction JSON format")

    # force KV-cache text scoring
    text_scorer = LocalQwenTextScorer(
        model_dir=Path(args.text_local_model),
        device=args.text_device,
        mode=args.text_score_mode,
        attention_tau=args.text_attention_tau,
    )

    if args.image_scorer == "resnet50":
        weights_path = Path(args.resnet50_weights) if args.resnet50_weights else None
        if weights_path and not weights_path.exists():
            weights_path = None
        image_scorer = ResNet50ImageScorer(weights_path=weights_path)
    else:
        image_scorer = NullImageScorer()

    problems = index.get("problems", [])
    total = len(problems)
    every = max(1, int(args.progress_every))
    started = time.time()
    ordered_problem_ids = [str(problem.get("id", "")) for problem in problems]
    out_path = Path(args.out)
    trace_path = Path(args.trace_out) if args.trace_out else None

    result_map: Dict[str, Dict[str, Any]] = {}
    trace_map: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_path.exists():
        existing_payload = read_json(out_path)
        existing_rows = existing_payload.get("results", [])
        if isinstance(existing_rows, list):
            result_map = {
                str(row.get("problem_id", "")).strip(): row
                for row in existing_rows
                if isinstance(row, dict) and str(row.get("problem_id", "")).strip()
            }
            print(f"Resume: loaded {len(result_map)} scored problems from {out_path}", flush=True)
    if args.resume and trace_path and trace_path.exists():
        existing_trace_payload = read_json(trace_path)
        existing_trace_rows = existing_trace_payload.get("problems", [])
        if isinstance(existing_trace_rows, list):
            trace_map = {
                str(row.get("problem_id", "")).strip(): row
                for row in existing_trace_rows
                if isinstance(row, dict) and str(row.get("problem_id", "")).strip()
            }
            print(f"Resume: loaded {len(trace_map)} trace rows from {trace_path}", flush=True)

    newly_completed = 0
    for i, problem in enumerate(problems, start=1):
        t0 = time.time()
        pid = problem["id"]
        pred_entry = pred_map.get(pid, {})
        if pid in result_map:
            if args.trace_out and pid not in trace_map:
                trace_map[pid] = build_trace_problem_from_score_result(
                    problem_id=pid,
                    pred_entry=pred_entry if isinstance(pred_entry, dict) else {},
                    score_result=result_map[pid],
                )
            continue
        score_result = score_problem(
            problem,
            pred_entry,
            text_scorer,
            image_scorer,
            Path(args.pred_root),
            text_step_source=args.text_step_source,
        )
        result_map[pid] = score_result
        newly_completed += 1
        if args.trace_out:
            trace_map[pid] = build_trace_problem_from_score_result(
                problem_id=pid,
                pred_entry=pred_entry if isinstance(pred_entry, dict) else {},
                score_result=score_result,
            )
        if args.progress and (i == 1 or i == total or i % every == 0):
            print(f"[{i}/{total}] {pid}: elapsed={time.time() - t0:.2f}s", flush=True)
        if int(args.save_every) > 0 and newly_completed % int(args.save_every) == 0:
            ordered_results = [result_map[pid] for pid in ordered_problem_ids if pid in result_map]
            _write_score_payload(
                out_path,
                index_path=args.index,
                pred_path=args.pred,
                text_score_mode=args.text_score_mode,
                text_attention_tau=args.text_attention_tau,
                text_step_source=args.text_step_source,
                ordered_results=ordered_results,
            )
            if args.trace_out and trace_path is not None:
                ordered_trace_rows = [trace_map[pid] for pid in ordered_problem_ids if pid in trace_map]
                _write_trace_payload(
                    trace_path,
                    index_path=args.index,
                    pred_path=args.pred,
                    text_step_source=args.text_step_source,
                    ordered_trace_rows=ordered_trace_rows,
                )
            if args.progress:
                print(f"Checkpoint saved after {newly_completed} newly completed problems", flush=True)

    ordered_results = [result_map[pid] for pid in ordered_problem_ids if pid in result_map]
    _write_score_payload(
        out_path,
        index_path=args.index,
        pred_path=args.pred,
        text_score_mode=args.text_score_mode,
        text_attention_tau=args.text_attention_tau,
        text_step_source=args.text_step_source,
        ordered_results=ordered_results,
    )
    if args.progress:
        print(f"Scoring done: {len(ordered_results)} problems in {time.time() - started:.2f}s", flush=True)
    print(f"Wrote {out_path}")

    if args.trace_out and trace_path is not None:
        ordered_trace_rows = [trace_map[pid] for pid in ordered_problem_ids if pid in trace_map]
        _write_trace_payload(
            trace_path,
            index_path=args.index,
            pred_path=args.pred,
            text_step_source=args.text_step_source,
            ordered_trace_rows=ordered_trace_rows,
        )
        print(f"Wrote {trace_path}")


if __name__ == "__main__":
    main()
