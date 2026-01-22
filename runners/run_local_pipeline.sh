#!/usr/bin/env bash
set -euo pipefail

STAGE1_CFG="configs/local/sharegpt4v_coco_stage1_qwen0.5b_siglip224.yaml"
STAGE2_CFG="configs/local/llava_textvqa_train_qwen0.5b_siglip224.yaml"

MAX_STEPS_STAGE1="${MAX_STEPS_STAGE1:-50}"
MAX_STEPS_STAGE2="${MAX_STEPS_STAGE2:-50}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
STAGE2_STAGE="${STAGE2_STAGE:-peft_lora}"

echo "stage1: train"
uv run python -m scripts.train \
  --config "$STAGE1_CFG" \
  --stage projector \
  --max-steps "$MAX_STEPS_STAGE1"

RUN1=$(ls -dt artifacts/local_sharegpt4v_coco_stage1_qwen0.5b_siglip224_* | head -1)
CKPT1=$(ls -dt "$RUN1"/step_* | head -1)
echo "stage1: latest run=$RUN1 ckpt=$CKPT1"

echo "stage1: eval (sharegpt4v_coco validation)"
uv run python -m scripts.eval \
  --ckpt "$CKPT1" \
  --override data.split=validation \
  --override logging.backend=wandb \
  --max-samples "$EVAL_MAX_SAMPLES"

echo "stage1: eval (textvqa validation)"
uv run python -m scripts.eval \
  --ckpt "$CKPT1" \
  --dataset textvqa \
  --split validation \
  --override logging.backend=wandb \
  --max-samples "$EVAL_MAX_SAMPLES"

echo "stage2: train"
uv run python -m scripts.train \
  --config "$STAGE2_CFG" \
  --stage "$STAGE2_STAGE" \
  --max-steps "$MAX_STEPS_STAGE2" \
  --resume "$CKPT1"

RUN2=$(ls -dt artifacts/local_llava_textvqa_train_qwen0.5b_siglip224_* | head -1)
CKPT2=$(ls -dt "$RUN2"/step_* | head -1)
echo "stage2: latest run=$RUN2 ckpt=$CKPT2"

echo "stage2: eval (textvqa validation)"
uv run python -m scripts.eval \
  --ckpt "$CKPT2" \
  --dataset textvqa \
  --split validation \
  --override logging.backend=wandb \
  --max-samples "$EVAL_MAX_SAMPLES"
