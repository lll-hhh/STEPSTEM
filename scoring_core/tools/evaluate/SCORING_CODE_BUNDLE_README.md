# Scoring Code Bundle

This bundle is intended for running score/judge code on another server.

Included pieces:

- `tools/evaluate/score.py`
- `tools/evaluate/whole_trace_judge.py`
- `tools/evaluate/example_interleaved_scoring.py`
- `tools/evaluate/fuse_score_and_judge_rms.py`
- `tools/models/unified_model_platform.py`
- `configs/models/runtime_model_registry.json`

The example runner demonstrates interleaved text+image scoring:

```bash
python tools/evaluate/example_interleaved_scoring.py \
  --index data/benchmarks/mllm_bench/index.json \
  --pred experiments/gpt54_openai_tool/pred_split_nl_slim.json \
  --pred-root experiments/gpt54_openai_tool \
  --score-out experiments/gpt54_openai_tool/score_slim.json \
  --run-judge \
  --judge-out experiments/gpt54_openai_tool/whole_trace_judge_vlm_strict.json
```

Notes:

- `score.py` uses `--text-step-source keypoint+content` together with `--image-scorer resnet50`.
- `whole_trace_judge.py` should be run with `--pass-question-image` for multimodal judging.
- The caller still needs to provide benchmark data, predictions, model weights, and image assets on the target machine.
