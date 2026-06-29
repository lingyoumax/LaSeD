# LaSeD: Label-Semantic Self-Distillation for Visual-Only Surgical Phase Recognition

Official implementation of **LaSeD** — *Label-Semantic Self-Distillation for Visual-Only Surgical Phase Recognition*.

This repository provides training code for **frame-level surgical phase recognition** on laparoscopic cholecystectomy video (Cholec80-style), built on **Qwen3-VL**. LaSeD injects ground-truth phase semantics into a teacher branch at training time, then distills a **visual-only student** that needs no text input at inference.

## Overview

During training, a single vision-language model (VLM) plays two roles:

| Branch | Input | Role |
|--------|-------|------|
| **Teacher** | Image + ground-truth phase name text | Privileged semantic context (training only) |
| **Student** | Image only | Deployable visual-only predictor |

**Training pipeline:**

1. The teacher runs in eval mode and precomputes final interaction features for every training sample.
2. The teacher is released from memory; the student is trained.
3. Student loss = **λ_ce × L_ce** (cross-entropy on ground-truth phase labels) + **λ_feat × L_mse** (MSE on teacher hidden state).
4. At **inference**, only the student branch is used: image in, digit token **1-7** out (seven Cholec80 phases).

Phase names: Preparation, CalotTriangleDissection, ClippingCutting, GallbladderDissection, GallbladderPackaging, CleaningCoagulation, GallbladderRetraction.

## Data layout

Default root: `./data/cholec80`

```
data/cholec80/
  train/
    video01/
      0.jpg
      1.jpg
      ...
  test/
    video41/
      ...
  csvs/
    video01.csv
    video41.csv
    ...
```

**CSV format:** skip the header row; **column 2** is the integer phase label (**1-7**). Row index `i` must match frame file `{i}.jpg`.

**Common Cholec80 protocol:** videos 1-40 for training, 41-80 for testing, sampled at **1 fps**.

## Quick start

### Train (LaSeD)

From the repository root:

```bash
python train.py
```

Equivalent:

```bash
python -m vl_kd.main
```

Multi-GPU with LoRA / QLoRA:

```bash
accelerate launch train.py \
  --output_dir ./checkpoints/run1 \
  --num_train_epochs 2 \
  --use_lora --use_qlora \
  --teacher_fp16
```

List all options:

```bash
python train.py --help
```

Useful flags: `--lambda_ce`, `--lambda_feat`, `--use_label_weight`, `--train_visual_only`, `--teacher_model_name_or_path`, `--student_model_name_or_path`.

### Evaluate a trained student

```bash
python evaluate.py \
  --model_name_or_path ./checkpoints/run1/best_model \
  --save_confusion_matrix
```

For **LoRA adapters**, also pass the base model:

```bash
python evaluate.py \
  --model_name_or_path ./checkpoints/run1/best_model \
  --base_model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
  --save_confusion_matrix
```

Multi-GPU evaluation:

```bash
accelerate launch evaluate.py \
  --model_name_or_path ./checkpoints/run1/final_model \
  --save_confusion_matrix
```

Metrics are written to `{model_path}/eval_results/metrics.json`. With `--save_confusion_matrix`, count and heatmap files are saved under `eval_results/eval/`.

## Customizing prompts

Default prompts and phase names live in `vl_kd/prompts.py`. You can override them in `settings.py` at the project root without editing the package.
