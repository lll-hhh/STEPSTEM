# Beyond Final Answers: Evaluating MultiModal Interleaved Reasoning Chains in Multimodal STEM Tasks

This repository contains the **StepSTEM** benchmark and the **core evaluation code** used in our paper.

StepSTEM is a multimodal STEM benchmark for **process-level evaluation**. It includes:

- `283` problems across Engineering, Physics, Chemistry, Biology, Mathematics, and Other
- `269` text-final problems
- `14` drawing problems with image final answers
- interleaved reference solutions with text steps, image steps, and bounding-box annotations

## Contents

- `benchmark/`
  - StepSTEM benchmark
- `scoring_core/tools/evaluate/score.py`
  - local process score (`m2m + dp`)
- `scoring_core/tools/evaluate/whole_trace_judge.py`
  - whole-trace judge
- `scoring_core/tools/evaluate/final_answer_llm_judge.py`
  - final-answer judge

## How To Use

### 1. Local process score

  ```bash
  python scoring_core/tools/evaluate/score.py \
  --index benchmark/index.json \
  --pred path/to/pred.json \
  --pred-root path/to/pred_root \
  --text-local-model path/to/Qwen3.5-9B \
  --text-device auto \
  --text-score-mode attention_rollout_sum_prob \
  --text-attention-tau 0.3 \
  --text-step-source keypoint+content \
  --image-scorer resnet50 \
  --resnet50-weights path/to/resnet50.pth \
  --out path/to/score.json
```

### 2. Whole-trace judge

  ```bash
  python scoring_core/tools/evaluate/whole_trace_judge.py \
  --index benchmark/index.json \
  --pred path/to/pred.json \
  --pred-root path/to/pred_root \
  --model qwen35_9b \
  --model-registry scoring_core/configs/models/runtime_model_registry.json \
  --device auto \
  --dtype bf16 \
  --judge-source content \
  --pass-question-image \
  --out path/to/whole_trace_judge.json
```

### 3. Final-answer judge

  ```bash
  python scoring_core/tools/evaluate/final_answer_llm_judge.py \
  --index benchmark/index.json \
  --pred path/to/pred.json \
  --model qwen35_9b \
  --model-registry scoring_core/configs/models/runtime_model_registry.json \
  --device auto \
  --dtype bf16 \
  --out path/to/final_answer_judge.json
```

## Notes

- This is a local release preview and has not been pushed yet.
- The dataset copy here is the pure benchmark data only.
