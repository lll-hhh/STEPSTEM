#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified local model testing platform for text, VLM, and local image generation models.
"""

import argparse
import gc
import inspect
import json
import os
import re
import sys
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_REGISTRY_PATH = "configs/models/runtime_model_registry.json"

DEFAULT_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen35_9b": {
        "family": "text",
        "runner_kind": "text_causal",
        "path": "models/Qwen3.5-9B",
        "hf_repo": "Qwen/Qwen3.5-9B",
    },
    "internvl3_8b": {
        "family": "vlm",
        "runner_kind": "internvl_chat",
        "path": "models/InternVL3-8B",
        "hf_repo": "OpenGVLab/InternVL3-8B",
    },
    "llava_ov_qwen2_7b": {
        "family": "vlm",
        "runner_kind": "vlm_auto",
        "path": "models/llava-onevision-qwen2-7b-ov-hf",
        "hf_repo": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    },
}


def _derive_family(runner_kind: str, family: str) -> str:
    fam = str(family or "").strip().lower()
    rk = str(runner_kind or "").strip().lower()
    if fam:
        return fam
    if rk in ("text_causal",):
        return "text"
    if rk in ("vlm_auto", "internvl_chat", "causal_mm_chat", "molmo2_vlm", "bagel_official"):
        return "vlm"
    if rk in ("image_gen_local",):
        return "generation"
    return "text"


def _derive_runner_kind(model_name: str, family: str, runner_kind: str) -> str:
    rk = str(runner_kind or "").strip().lower()
    if rk:
        return rk
    name = str(model_name).strip().lower()
    if "molmo2" in name:
        return "molmo2_vlm"
    if "bagel" in name or "unireason" in name:
        return "bagel_official"
    if "intern-s1" in name:
        return "causal_mm_chat"
    fam = str(family or "").strip().lower()
    if name.startswith("internvl"):
        return "internvl_chat"
    if fam == "text":
        return "text_causal"
    if fam in ("vlm", "vision", "multimodal"):
        return "vlm_auto"
    if fam in ("generation", "image"):
        return "image_gen_local"
    return "text_causal"


def load_model_registry(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return DEFAULT_MODELS
    p = Path(path)
    if not p.exists():
        return DEFAULT_MODELS
    payload = json.loads(p.read_text(encoding="utf-8"))
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError(f"Invalid model registry format: {path}")

    norm: Dict[str, Dict[str, Any]] = {}
    for model_name, meta in models.items():
        if not isinstance(meta, dict):
            continue
        mpath = str(meta.get("path", "")).strip()
        if not model_name or not mpath:
            continue
        family = str(meta.get("family", "")).strip().lower()
        runner_kind = _derive_runner_kind(str(model_name), family, str(meta.get("runner_kind", "")))
        norm[str(model_name)] = {
            "family": _derive_family(runner_kind, family),
            "runner_kind": runner_kind,
            "path": mpath,
            "hf_repo": str(meta.get("hf_repo", "")).strip(),
            "track": str(meta.get("track", "")).strip(),
            "tier": str(meta.get("tier", "")).strip(),
            "recommended_vram_gb": float(meta.get("recommended_vram_gb", 0.0) or 0.0),
            "min_vram_gb": float(meta.get("min_vram_gb", 0.0) or 0.0),
            "env_name": str(meta.get("env_name", "")).strip(),
            "runtime": meta.get("runtime", {}) if isinstance(meta.get("runtime", {}), dict) else {},
        }
    return norm or DEFAULT_MODELS


def parse_dtype(dtype: str):
    import torch

    d = str(dtype).strip().lower()
    if d == "auto":
        return "auto"
    if d == "bf16":
        return torch.bfloat16
    if d == "fp16":
        return torch.float16
    if d == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def configure_cuda(gpu_id: Optional[str], device: str) -> str:
    """
    Configure GPU visibility before model/tensor creation.
    If gpu_id is provided, map selected GPU to local cuda:0.
    """
    if gpu_id not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if device in ("auto", "cuda"):
            return "cuda:0"
    return device


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def validate_model_path(model_path: Path, runner_kind: str = ""):
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    rk = str(runner_kind or "").strip().lower()
    if rk == "image_gen_local":
        has_marker = (model_path / "model_index.json").exists() or (model_path / "config.json").exists()
        if not has_marker:
            raise FileNotFoundError(f"Neither model_index.json nor config.json found under: {model_path}")
        return
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"config.json not found under: {model_path}")


def _config_supports_images(model_path: Path) -> bool:
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(cfg, dict):
        return False
    # Common multimodal markers across Qwen/LLaVA/Phi/Intern families.
    image_markers = (
        "image_token_id",
        "vision_config",
        "mm_vision_tower",
        "vision_tower",
        "image_size",
        "image_seq_length",
    )
    return any(k in cfg for k in image_markers)


def _resolve_dynamic_module_safe_path(model_path: Path) -> Path:
    """
    HF dynamic module names can break on local directory names containing dots
    (e.g. InternVL3.5-8B -> transformers_modules.InternVL3).
    Create a sibling symlink with sanitized name and load from it when possible.
    """
    p = model_path.resolve()
    name = p.name
    if "." not in name:
        return p
    alias = p.parent / name.replace(".", "_")
    if alias.exists():
        return alias
    try:
        alias.symlink_to(p, target_is_directory=True)
        return alias
    except Exception:
        return p


def _install_backoff_shim_if_missing():
    try:
        import backoff  # noqa: F401
        return
    except Exception:
        pass

    import types

    mod = types.ModuleType("backoff")

    def _expo(base: int = 2, factor: float = 1.0):
        n = 0
        while True:
            yield factor * (base ** n)
            n += 1

    def _constant(interval: float = 1.0):
        while True:
            yield interval

    def _decorator_passthrough(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

    mod.expo = _expo
    mod.constant = _constant
    mod.full_jitter = lambda value: value
    mod.on_exception = _decorator_passthrough
    mod.on_predicate = _decorator_passthrough
    sys.modules["backoff"] = mod


def _patch_transformers_cache_utils_compat():
    try:
        from transformers import cache_utils
    except Exception:
        return
    dynamic_cache = getattr(cache_utils, "DynamicCache", None) or getattr(cache_utils, "Cache", object)
    if not hasattr(cache_utils, "SlidingWindowCache"):
        setattr(cache_utils, "SlidingWindowCache", dynamic_cache)
    if not hasattr(cache_utils, "HybridCache"):
        setattr(cache_utils, "HybridCache", dynamic_cache)


def _patch_transformers_video_utils_compat():
    try:
        from transformers import video_utils
    except Exception:
        return
    if hasattr(video_utils, "make_batched_metadata"):
        return

    def make_batched_metadata(*args, **kwargs):
        # Best-effort shim for older transformers; Molmo2 smoke path is image-only.
        return kwargs.get("metadata", None)

    video_utils.make_batched_metadata = make_batched_metadata


def load_image(image_path: str):
    from PIL import Image

    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p).convert("RGB")


def load_images(image_paths: Optional[List[str]]) -> List[Any]:
    paths = [str(p).strip() for p in (image_paths or []) if str(p).strip()]
    return [load_image(p) for p in paths]


def request_image_paths(req: "GenerationRequest") -> List[str]:
    out: List[str] = []
    if str(req.image or "").strip():
        out.append(str(req.image).strip())
    if isinstance(req.images, list):
        for item in req.images:
            s = str(item or "").strip()
            if s:
                out.append(s)
    deduped: List[str] = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


def _processor_images_arg(image_paths: List[str]) -> Any:
    images = load_images(image_paths)
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    return images


def build_messages(
    system_prompt: str,
    prompt: str,
    image_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    paths: List[str] = []
    if str(image_path or "").strip():
        paths.append(str(image_path).strip())
    for p in image_paths or []:
        s = str(p or "").strip()
        if s:
            paths.append(s)

    if paths:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": p} for p in paths]
                + [{"type": "text", "text": prompt}],
            }
        )
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def build_multimodal_user_message(
    prompt: str,
    image_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    paths: List[str] = []
    if str(image_path or "").strip():
        paths.append(str(image_path).strip())
    for p in image_paths or []:
        s = str(p or "").strip()
        if s:
            paths.append(s)
    if paths:
        return {
            "role": "user",
            "content": [{"type": "text", "text": prompt}] + [{"type": "image", "image": p} for p in paths],
        }
    return {"role": "user", "content": prompt}


def _signature_allowed_keys(fn: Any) -> Tuple[set, bool]:
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        return set(), True
    allowed = set()
    has_var_kwargs = False
    for name, p in params.items():
        if name in ("self", "kwargs"):
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kwargs = True
            continue
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(name)
    return allowed, has_var_kwargs


def _filter_inputs_for_model_generate(model: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
    methods = [
        getattr(model, "prepare_inputs_for_generation", None),
        getattr(model, "forward", None),
    ]
    allowed_union: set = set()
    for fn in methods:
        if fn is None:
            continue
        allowed, _ = _signature_allowed_keys(fn)
        allowed_union |= allowed
    if not allowed_union:
        return dict(inputs)
    filtered = {k: v for k, v in inputs.items() if k in allowed_union}
    if "input_ids" in inputs and "input_ids" not in filtered:
        filtered["input_ids"] = inputs["input_ids"]
    return filtered


def _extract_unexpected_kw_from_error(err: Exception) -> List[str]:
    msg = str(err or "")
    out: List[str] = []
    m = re.search(r"Unexpected keyword argument\s+`?([A-Za-z0-9_]+)`?", msg)
    if m:
        out.append(m.group(1))
    m2 = re.search(r"The following `model_kwargs` are not used by the model:\s*\[(.*?)\]", msg, flags=re.S)
    if m2:
        raw = m2.group(1)
        for part in raw.split(","):
            key = str(part).strip().strip("'").strip('"').strip("`")
            if key:
                out.append(key)
    # De-dup and preserve order.
    dedup: List[str] = []
    seen = set()
    for k in out:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    return dedup


def _safe_model_generate(model: Any, inputs: Dict[str, Any], gen_kwargs: Dict[str, Any]):
    model_inputs = _filter_inputs_for_model_generate(model, inputs)
    for _ in range(8):
        try:
            return model.generate(**model_inputs, **gen_kwargs)
        except (TypeError, ValueError, RuntimeError) as e:
            bad_keys = _extract_unexpected_kw_from_error(e)
            if not bad_keys:
                raise
            changed = False
            for k in bad_keys:
                if k in model_inputs:
                    model_inputs.pop(k, None)
                    changed = True
            if not changed:
                raise
    raise RuntimeError("model.generate failed after repeated unexpected keyword filtering")


@dataclass
class GenerationRequest:
    prompt: str
    image: Optional[str] = None
    images: Optional[List[str]] = None
    system: str = ""
    enable_thinking: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False
    out_image_path: Optional[str] = None


def _apply_chat_template_optional_thinking(
    renderer: Any,
    messages: Any,
    *,
    enable_thinking: Optional[bool],
    **kwargs: Any,
):
    if enable_thinking is None:
        return renderer(messages, **kwargs)
    try:
        return renderer(messages, enable_thinking=bool(enable_thinking), **kwargs)
    except TypeError:
        return renderer(messages, **kwargs)


class BaseRunner:
    def __init__(self, model_path: Path, device: str, dtype: str):
        self.model_path = model_path
        self.device = device
        self.dtype = parse_dtype(dtype)
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load(self):
        raise NotImplementedError

    def generate(self, req: GenerationRequest) -> Any:
        raise NotImplementedError

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class QwenTextRunner(BaseRunner):
    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"trust_remote_code": True}
        if self.dtype != "auto":
            kwargs["torch_dtype"] = self.dtype
        load_path = _resolve_dynamic_module_safe_path(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(load_path), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(str(load_path), **kwargs).to(self.device).eval()

    def generate(self, req: GenerationRequest) -> str:
        import torch

        messages: List[Dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        text = _apply_chat_template_optional_thinking(
            self.tokenizer.apply_chat_template,
            messages,
            enable_thinking=req.enable_thinking,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            gen_kwargs: Dict[str, Any] = {
                "do_sample": bool(req.do_sample),
            }
            # max_new_tokens<=0 means "do not enforce manual cap here";
            # let model generation config control stopping length.
            if int(req.max_new_tokens) > 0:
                gen_kwargs["max_new_tokens"] = int(req.max_new_tokens)
            if req.do_sample:
                gen_kwargs["temperature"] = req.temperature
                gen_kwargs["top_p"] = req.top_p
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


class VisionLanguageRunner(BaseRunner):
    def load(self):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

        kwargs = {"trust_remote_code": True}
        if self.dtype != "auto":
            kwargs["torch_dtype"] = self.dtype
        load_path = _resolve_dynamic_module_safe_path(self.model_path)
        self.processor = AutoProcessor.from_pretrained(str(load_path), trust_remote_code=True)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(str(load_path), **kwargs).to(self.device).eval()
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(str(load_path), **kwargs).to(self.device).eval()

    def generate(self, req: GenerationRequest) -> str:
        import torch

        image_paths = request_image_paths(req)
        messages = build_messages(req.system, req.prompt, image_paths=image_paths)
        chat_text = _apply_chat_template_optional_thinking(
            self.processor.apply_chat_template,
            messages,
            enable_thinking=req.enable_thinking,
            tokenize=False,
            add_generation_prompt=True,
        )

        images = _processor_images_arg(image_paths)
        inputs = self.processor(text=chat_text, images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": req.max_new_tokens,
                "do_sample": bool(req.do_sample),
            }
            if req.do_sample:
                gen_kwargs["temperature"] = req.temperature
                gen_kwargs["top_p"] = req.top_p
            output_ids = _safe_model_generate(self.model, inputs, gen_kwargs)

        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        gen_ids = output_ids[:, input_len:] if input_len > 0 else output_ids
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


class CausalMMRunner(BaseRunner):
    def load(self):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        from transformers import tokenization_utils

        _install_backoff_shim_if_missing()
        _patch_transformers_cache_utils_compat()

        kwargs = {"trust_remote_code": True}
        if self.dtype != "auto":
            kwargs["torch_dtype"] = self.dtype
        # Prefer SDPA to avoid hard dependency on flash-attn in smoke environments.
        kwargs["attn_implementation"] = "sdpa"

        # Compat patch for custom tokenizers (Intern-S1 family) on some transformers versions.
        if not hasattr(tokenization_utils.PreTrainedTokenizer, "_update_trie"):
            def _noop_update_trie(self, *args, **kwargs):
                return None
            setattr(tokenization_utils.PreTrainedTokenizer, "_update_trie", _noop_update_trie)

        load_path = _resolve_dynamic_module_safe_path(self.model_path)
        self.tokenizer = None
        try:
            self.processor = AutoProcessor.from_pretrained(str(load_path), trust_remote_code=True, use_fast=False)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(str(load_path), trust_remote_code=True)
        if self.processor is not None and hasattr(self.processor, "tokenizer"):
            self.tokenizer = getattr(self.processor, "tokenizer")
        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(load_path),
                    trust_remote_code=True,
                    use_fast=False,
                )
            except TypeError:
                self.tokenizer = AutoTokenizer.from_pretrained(str(load_path), trust_remote_code=True)
        if self.tokenizer is not None and not hasattr(self.tokenizer, "_update_trie"):
            def _noop_update_trie_inst(*args, **kwargs):
                return None
            setattr(self.tokenizer.__class__, "_update_trie", _noop_update_trie_inst)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(str(load_path), **kwargs).to(self.device).eval()
        except Exception as e:
            msg = str(e or "").lower()
            if "flashattention2" in msg or "flash_attn" in msg:
                kwargs["attn_implementation"] = "eager"
                cfg = AutoConfig.from_pretrained(str(load_path), trust_remote_code=True)
                for obj in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "vision_config", None)):
                    if obj is None:
                        continue
                    try:
                        setattr(obj, "_attn_implementation", "eager")
                    except Exception:
                        pass
                    try:
                        setattr(obj, "_attn_implementation_internal", "eager")
                    except Exception:
                        pass
                    try:
                        setattr(obj, "attn_implementation", "eager")
                    except Exception:
                        pass
                kwargs["config"] = cfg
                self.model = AutoModelForCausalLM.from_pretrained(str(load_path), **kwargs).to(self.device).eval()
            else:
                raise

    def _build_prompt_text(self, req: GenerationRequest) -> str:
        messages: List[Dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append(build_multimodal_user_message(req.prompt, image_paths=request_image_paths(req)))

        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            try:
                return str(
                    _apply_chat_template_optional_thinking(
                        self.processor.apply_chat_template,
                        messages,
                        enable_thinking=req.enable_thinking,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception:
                pass
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return str(
                    _apply_chat_template_optional_thinking(
                        self.tokenizer.apply_chat_template,
                        messages,
                        enable_thinking=req.enable_thinking,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception:
                pass
        return req.prompt

    def generate(self, req: GenerationRequest) -> str:
        import torch

        prompt_text = self._build_prompt_text(req)
        image_paths = request_image_paths(req)
        images = _processor_images_arg(image_paths)

        if self.processor is not None:
            inputs = self.processor(text=prompt_text, images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        else:
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": req.max_new_tokens,
            "do_sample": bool(req.do_sample),
        }
        if req.do_sample:
            gen_kwargs["temperature"] = req.temperature
            gen_kwargs["top_p"] = req.top_p

        with torch.no_grad():
            output_ids = _safe_model_generate(self.model, inputs, gen_kwargs)
        gen_ids = output_ids[:, input_len:] if input_len > 0 else output_ids

        if self.processor is not None and hasattr(self.processor, "batch_decode"):
            return str(self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]).strip()
        return str(self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]).strip()


class Molmo2Runner(BaseRunner):
    def load(self):
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

        _patch_transformers_video_utils_compat()

        kwargs = {"trust_remote_code": True}
        if self.dtype != "auto":
            kwargs["torch_dtype"] = self.dtype
        load_path = _resolve_dynamic_module_safe_path(self.model_path)
        self.processor = AutoProcessor.from_pretrained(str(load_path), trust_remote_code=True, use_fast=False)
        last_err: Optional[Exception] = None
        try:
            self.model = AutoModelForCausalLM.from_pretrained(str(load_path), **kwargs).to(self.device).eval()
            return
        except Exception as e:
            last_err = e
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(str(load_path), **kwargs).to(self.device).eval()
            return
        except Exception as e:
            last_err = e
        try:
            # Some Molmo2 checkpoints register custom classes outside AutoModel mappings.
            self.model = AutoModel.from_pretrained(str(load_path), **kwargs).to(self.device).eval()
            return
        except Exception as e:
            last_err = e

        # Last-resort: instantiate the exact model class from auto_map.
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            cfg = AutoConfig.from_pretrained(str(load_path), trust_remote_code=True)
            auto_map = getattr(cfg, "auto_map", {}) or {}
            for key in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
                ref = auto_map.get(key)
                if not ref:
                    continue
                if isinstance(ref, (list, tuple)):
                    ref = ref[0]
                model_cls = get_class_from_dynamic_module(str(ref), str(load_path), trust_remote_code=True)
                self.model = model_cls.from_pretrained(str(load_path), **kwargs).to(self.device).eval()
                return
        except Exception as e:
            last_err = e

        raise RuntimeError(f"Molmo2 load failed for {load_path}: {last_err}")

    def generate(self, req: GenerationRequest) -> str:
        import torch

        messages: List[Dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append(build_multimodal_user_message(req.prompt, image_paths=request_image_paths(req)))
        # Molmo2 processors vary across versions:
        # - some return encoded dict directly from apply_chat_template(tokenize=True),
        # - some return text and require a second processor(...) call.
        inputs: Dict[str, Any]
        templated: Any = None
        try:
            templated = _apply_chat_template_optional_thinking(
                self.processor.apply_chat_template,
                messages,
                enable_thinking=req.enable_thinking,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except Exception:
            try:
                templated = _apply_chat_template_optional_thinking(
                    self.processor.apply_chat_template,
                    messages,
                    enable_thinking=req.enable_thinking,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            except Exception:
                templated = None

        if isinstance(templated, Mapping):
            inputs = dict(templated)
        else:
            # Fallback: template to text then call processor for multimodal encode.
            try:
                prompt_text = _apply_chat_template_optional_thinking(
                    self.processor.apply_chat_template,
                    messages,
                    enable_thinking=req.enable_thinking,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_text = req.prompt
            if req.system:
                prompt_text = f"{req.system}\n\n{prompt_text}"
            images = _processor_images_arg(request_image_paths(req))
            inputs = self.processor(text=prompt_text, images=images, return_tensors="pt")

        if isinstance(inputs, Mapping):
            inputs = dict(inputs)
        elif hasattr(inputs, "items"):
            inputs = dict(inputs.items())
        else:
            raise RuntimeError(f"Molmo2 processor returned unsupported input format: {type(inputs)}")

        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        # Known incompatible key for some Molmo2 checkpoints.
        inputs.pop("image_use_col_tokens", None)
        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": req.max_new_tokens,
            "do_sample": bool(req.do_sample),
        }
        if req.do_sample:
            gen_kwargs["temperature"] = req.temperature
            gen_kwargs["top_p"] = req.top_p

        with torch.no_grad():
            output = _safe_model_generate(self.model, inputs, gen_kwargs)
        generated_tokens = output[:, input_len:] if input_len > 0 else output
        if hasattr(self.processor, "tokenizer"):
            return str(self.processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]).strip()
        return str(generated_tokens)


def _internvl_build_transform(input_size: int = 448):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _internvl_find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[tuple],
    width: int,
    height: int,
    image_size: int,
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _internvl_load_image(image_file: str, input_size=448, max_num=12):
    import torch
    from PIL import Image

    image = Image.open(image_file).convert("RGB")
    transform = _internvl_build_transform(input_size=input_size)
    images = _internvl_dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


class InternVLRunner(BaseRunner):
    def load(self):
        from transformers import AutoModel, AutoTokenizer

        kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": False, "use_flash_attn": False}
        if self.dtype != "auto":
            kwargs["dtype"] = self.dtype
        load_path = _resolve_dynamic_module_safe_path(self.model_path)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(load_path),
                trust_remote_code=True,
                use_fast=False,
                fix_mistral_regex=True,
            )
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(load_path),
                trust_remote_code=True,
                use_fast=False,
            )
        try:
            self.model = AutoModel.from_pretrained(str(load_path), **kwargs).eval()
        except Exception as e:
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            try:
                self.model = AutoModel.from_pretrained(str(load_path), **kwargs).eval()
            except TypeError:
                kwargs.pop("use_flash_attn", None)
                self.model = AutoModel.from_pretrained(str(load_path), **kwargs).eval()
        if self.device.startswith("cuda"):
            self.model = self.model.to(self.device)

    def generate(self, req: GenerationRequest) -> str:
        generation_config = {
            "max_new_tokens": int(req.max_new_tokens),
            "do_sample": bool(req.do_sample),
        }
        if req.do_sample:
            generation_config["temperature"] = float(req.temperature)
            generation_config["top_p"] = float(req.top_p)

        question = req.prompt
        if req.system:
            question = f"{req.system}\n\n{req.prompt}"

        pixel_values = None
        image_paths = request_image_paths(req)
        if len(image_paths) > 1:
            raise ValueError("InternVLRunner does not support multi-image requests yet.")
        if image_paths:
            pixel_values = _internvl_load_image(image_paths[0], max_num=12)
            if self.dtype != "auto":
                pixel_values = pixel_values.to(self.dtype)
            if self.device.startswith("cuda"):
                pixel_values = pixel_values.to(self.device)

        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()


class ImageGenLocalRunner(BaseRunner):
    def __init__(self, model_path: Path, device: str, dtype: str):
        super().__init__(model_path, device, dtype)
        self.generator = None

    def load(self):
        from tools.inference.image_backends import LocalImageGenerator

        dtype_name = "fp16"
        if self.dtype != "auto":
            if "bfloat16" in str(self.dtype):
                dtype_name = "bf16"
            elif "float32" in str(self.dtype):
                dtype_name = "fp32"
        cfg = {
            "backend": "local",
            "model_path": str(self.model_path),
            "device": self.device,
            "dtype": dtype_name,
        }
        self.generator = LocalImageGenerator(cfg)

    def generate(self, req: GenerationRequest) -> str:
        if self.generator is None:
            raise RuntimeError("ImageGenLocalRunner not loaded.")
        target = str(req.out_image_path or "").strip()
        if not target:
            ts = int(time.time() * 1000)
            target = str((Path("demo") / "generated_images" / f"gen_{ts}.png").resolve())
        out_path = Path(target)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = self.generator.generate(str(req.prompt or ""), out_path, source_image=req.image)
        return str(result.get("image_path", "")).strip()

    def unload(self):
        self.generator = None
        super().unload()


class BagelOfficialRunner(BaseRunner):
    def __init__(self, model_path: Path, device: str, dtype: str, runtime_cfg: Dict[str, Any]):
        super().__init__(model_path, device, dtype)
        self.runtime_cfg = runtime_cfg or {}

    def load(self):
        entry = str(self.runtime_cfg.get("entrypoint", "")).strip()
        if not entry:
            raise RuntimeError(
                "bagel_official runner requires runtime.entrypoint in model registry "
                "(e.g. tools/runtimes/bagel_official_runtime.py)."
            )

    def generate(self, req: GenerationRequest) -> str:
        entry = str(self.runtime_cfg.get("entrypoint", "")).strip()
        python_bin = str(self.runtime_cfg.get("python_bin", "python")).strip() or "python"
        env_name = str(self.runtime_cfg.get("env_name", "")).strip()
        use_conda_run = bool(self.runtime_cfg.get("use_conda_run", False))
        timeout_sec = int(self.runtime_cfg.get("timeout_sec", 900))
        extra_args = self.runtime_cfg.get("extra_args", [])
        if not isinstance(extra_args, list):
            extra_args = []

        entry_path = Path(entry)
        if not entry_path.is_absolute():
            entry_path = (Path(__file__).resolve().parents[2] / entry).resolve()
        if not entry_path.exists():
            raise FileNotFoundError(f"bagel_official entrypoint not found: {entry_path}")

        payload = {
            "prompt": str(req.prompt or ""),
            "image": str(req.image or ""),
            "images": request_image_paths(req),
            "system": str(req.system or ""),
            "enable_thinking": bool(req.enable_thinking),
            "max_new_tokens": int(req.max_new_tokens),
            "temperature": float(req.temperature),
            "top_p": float(req.top_p),
            "do_sample": bool(req.do_sample),
            "device": self.device,
            "dtype": str(self.dtype),
        }

        with tempfile.TemporaryDirectory(prefix="bagel_official_") as td:
            td_path = Path(td)
            in_json = td_path / "input.json"
            out_json = td_path / "output.json"
            in_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            cmd: List[str] = [python_bin, str(entry_path), "--model-path", str(self.model_path), "--input-json", str(in_json), "--output-json", str(out_json)]
            cmd.extend([str(x) for x in extra_args])
            if use_conda_run and env_name:
                cmd = ["conda", "run", "-n", env_name] + cmd

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            if proc.returncode != 0:
                err = (proc.stderr or "").strip()
                out = (proc.stdout or "").strip()
                raise RuntimeError(
                    f"bagel_official runtime failed: rc={proc.returncode}; "
                    f"stderr={err[:4000]}; stdout={out[:1000]}"
                )
            if not out_json.exists():
                raise RuntimeError("bagel_official runtime produced no output json")

            result = json.loads(out_json.read_text(encoding="utf-8"))
            text = str(result.get("text", "") or result.get("response", "")).strip()
            if not text:
                raise RuntimeError("bagel_official runtime output missing 'text'/'response'")
            return text


class UnifiedModelPlatform:
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = registry_path or DEFAULT_REGISTRY_PATH
        self.registry = load_model_registry(self.registry_path)

    def _resolve_runner_kind(self, model_name: str, info: Dict[str, Any]) -> str:
        rk = str(info.get("runner_kind", "")).strip().lower()
        if rk:
            return rk
        family = str(info.get("family", "")).strip().lower()
        return _derive_runner_kind(model_name, family, "")

    def create_runner(self, model_name: str, model_path: Optional[str], device: str, dtype: str):
        if model_name not in self.registry:
            raise ValueError(f"Unknown model '{model_name}'. Use --list-models.")
        info = self.registry[model_name]
        runner_kind = self._resolve_runner_kind(model_name, info)
        final_path = Path(model_path or str(info.get("path", "")))
        validate_model_path(final_path, runner_kind=runner_kind)
        effective_info = dict(info)
        # Auto-upgrade text runner when config clearly indicates image capability.
        # This avoids silently dropping question images due to stale registry labels.
        if runner_kind == "text_causal" and _config_supports_images(final_path):
            runner_kind = "vlm_auto"
            effective_info["family"] = "vlm"
            effective_info["runner_kind"] = runner_kind
        if runner_kind == "text_causal":
            return QwenTextRunner(final_path, device, dtype), effective_info
        if runner_kind == "internvl_chat":
            return InternVLRunner(final_path, device, dtype), effective_info
        if runner_kind == "vlm_auto":
            return VisionLanguageRunner(final_path, device, dtype), effective_info
        if runner_kind == "causal_mm_chat":
            return CausalMMRunner(final_path, device, dtype), effective_info
        if runner_kind == "molmo2_vlm":
            return Molmo2Runner(final_path, device, dtype), effective_info
        if runner_kind == "bagel_official":
            return BagelOfficialRunner(final_path, device, dtype, runtime_cfg=info.get("runtime", {})), effective_info
        if runner_kind == "image_gen_local":
            return ImageGenLocalRunner(final_path, device, dtype), effective_info
        raise ValueError(f"Unsupported runner_kind '{runner_kind}' for model '{model_name}'")

    def run_for_model(
        self,
        model_name: str,
        model_path: Optional[str],
        device: str,
        dtype: str,
        requests: List[GenerationRequest],
    ) -> Dict[str, Any]:
        runner, info = self.create_runner(model_name, model_path, device, dtype)
        started = time.time()
        runner.load()
        load_sec = time.time() - started

        outputs: List[Dict[str, Any]] = []
        for idx, req in enumerate(requests, start=1):
            t0 = time.time()
            response_value = runner.generate(req)
            outputs.append(
                {
                    "index": idx,
                    "prompt": req.prompt,
                    "image": req.image,
                    "images": request_image_paths(req),
                    "system": req.system,
                    "response": response_value,
                    "elapsed_sec": round(time.time() - t0, 3),
                }
            )

        runner.unload()
        return {
            "model": model_name,
            "family": info.get("family", ""),
            "runner_kind": info.get("runner_kind", self._resolve_runner_kind(model_name, info)),
            "hf_repo": info.get("hf_repo", ""),
            "model_path": str(Path(model_path or str(info.get("path", ""))).resolve()),
            "device": device,
            "dtype": dtype,
            "load_sec": round(load_sec, 3),
            "outputs": outputs,
        }


def read_requests(args) -> List[GenerationRequest]:
    if args.input_jsonl:
        reqs: List[GenerationRequest] = []
        p = Path(args.input_jsonl)
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            reqs.append(
                GenerationRequest(
                    prompt=str(obj.get("prompt", "")),
                    image=obj.get("image"),
                    images=(obj.get("images") if isinstance(obj.get("images"), list) else None),
                    system=str(obj.get("system", "")),
                    enable_thinking=bool(obj.get("enable_thinking", False)),
                    max_new_tokens=int(obj.get("max_new_tokens", args.max_new_tokens)),
                    temperature=float(obj.get("temperature", args.temperature)),
                    top_p=float(obj.get("top_p", args.top_p)),
                    do_sample=bool(obj.get("do_sample", args.do_sample)),
                    out_image_path=(obj.get("out_image_path") or obj.get("output_image_path")),
                )
            )
        return reqs

    return [
        GenerationRequest(
            prompt=args.prompt,
            image=args.image,
            system=args.system,
            enable_thinking=bool(args.enable_thinking),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            out_image_path=args.out_image,
        )
    ]


def _check_model_entry(model_name: str, model_meta: Dict[str, Any], model_path_override: str = "") -> Tuple[bool, str]:
    path_val = model_path_override or str(model_meta.get("path", ""))
    if not path_val:
        return False, f"MISS {model_name}: missing path"
    p = Path(path_val)
    runner_kind = str(model_meta.get("runner_kind", "")).strip().lower()
    try:
        validate_model_path(p, runner_kind=runner_kind)
    except Exception as e:
        return False, f"FAIL {model_name}: {e}"
    return True, f"OK {model_name} -> {p}"


def main():
    parser = argparse.ArgumentParser(description="Unified local model testing platform")
    parser.add_argument(
        "--model-registry",
        default=DEFAULT_REGISTRY_PATH,
        help="Model registry JSON path (default: configs/models/runtime_model_registry.json)",
    )
    parser.add_argument(
        "--models",
        default="qwen35_9b",
        help="Comma-separated model keys. e.g. qwen35_9b,internvl3_8b,llava_ov_qwen2_7b",
    )
    parser.add_argument("--model-path", default="", help="Optional override path when running one model only")
    parser.add_argument("--list-models", action="store_true", help="List supported model keys and exit")
    parser.add_argument("--check-models", action="store_true", help="Check local model paths and exit")

    parser.add_argument("--prompt", default="Please describe your reasoning briefly.", help="Single-run prompt")
    parser.add_argument("--image", default="", help="Optional image path for VLM")
    parser.add_argument("--system", default="", help="Optional system prompt")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable model thinking/reasoning mode when backend supports it (default: disabled).",
    )
    parser.add_argument("--input-jsonl", default="", help="Batch input jsonl path")
    parser.add_argument("--out", default="demo/unified_model_outputs.json", help="Output JSON path")
    parser.add_argument("--out-image", default="", help="Optional output image path for image generation models")

    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0...")
    parser.add_argument("--gpu-id", default="", help="GPU id(s) for CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"], help="Model dtype")

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true")
    args = parser.parse_args()

    platform = UnifiedModelPlatform(registry_path=args.model_registry)

    if args.list_models:
        for k, v in platform.registry.items():
            print(json.dumps({"model": k, **v}, ensure_ascii=False))
        return

    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    if not model_names:
        raise SystemExit("No model selected.")
    if args.model_path and len(model_names) != 1:
        raise SystemExit("--model-path can only be used when running one model.")

    if args.check_models:
        for m in model_names:
            if m not in platform.registry:
                raise SystemExit(f"Unknown model: {m}")
            ok, msg = _check_model_entry(m, platform.registry[m], args.model_path)
            print(msg)
            if not ok:
                raise SystemExit(2)
        return

    resolved_device = configure_cuda(args.gpu_id, args.device)
    resolved_device = resolve_device(resolved_device)
    requests = read_requests(args)

    all_results: List[Dict[str, Any]] = []
    for m in model_names:
        model_path = args.model_path if args.model_path else None
        res = platform.run_for_model(
            model_name=m,
            model_path=model_path,
            device=resolved_device,
            dtype=args.dtype,
            requests=requests,
        )
        all_results.append(res)
        print(f"Done {m}: {len(res['outputs'])} request(s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "models": model_names,
                "device": resolved_device,
                "gpu_id": args.gpu_id,
                "dtype": args.dtype,
                "num_requests": len(requests),
                "results": all_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
