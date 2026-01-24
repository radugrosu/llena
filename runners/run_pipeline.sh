#!/usr/bin/env bash
set -euo pipefail

STAGE1_CFG="${STAGE1_CFG:-configs/local/sharegpt4v_train_qwen2.5-0.5b_siglip224.yaml}"
STAGE2_CFG="${STAGE2_CFG:-configs/local/llava_textvqa_train_qwen2.5-0.5b_siglip224.yaml}"

STAGE2_STAGE="${STAGE2_STAGE:-peft_lora}"

echo "stage1: train"
stage1_args=()
if [[ -n "${MAX_STEPS_STAGE1:-}" ]]; then
  stage1_args+=(--max-steps "$MAX_STEPS_STAGE1")
fi
uv run python -m scripts.train \
  --config "$STAGE1_CFG" \
  --stage projector \
  "${stage1_args[@]}"

RUN1=$(cat artifacts/latest_run.txt)
printf '%s' "$RUN1" > artifacts/latest_stage1.txt
CKPT1=$(ls -dt "$RUN1"/step_* | head -1)
echo "stage1: latest run=$RUN1 ckpt=$CKPT1"

echo "stage1: eval (sharegpt4v_coco validation)"
eval_args=()
if [[ -n "$EVAL_MAX_SAMPLES" ]]; then
  eval_args+=(--max-samples "$EVAL_MAX_SAMPLES")
fi
uv run python -m scripts.eval \
  --ckpt "$CKPT1" \
  --override data.split=validation \
  --override logging.backend=wandb \
  "${eval_args[@]}"

echo "stage1: eval (textvqa validation)"
uv run python -m scripts.eval \
  --ckpt "$CKPT1" \
  --dataset textvqa \
  --split validation \
  --override logging.backend=wandb \
  "${eval_args[@]}"

echo "stage2: train"
stage2_args=()
if [[ -n "${MAX_STEPS_STAGE2:-}" ]]; then
  stage2_args+=(--max-steps "$MAX_STEPS_STAGE2")
fi
uv run python -m scripts.train \
  --config "$STAGE2_CFG" \
  --stage "$STAGE2_STAGE" \
  "${stage2_args[@]}" \
  --resume "$CKPT1"

RUN2=$(cat artifacts/latest_run.txt)
printf '%s' "$RUN2" > artifacts/latest_stage2.txt
CKPT2=$(ls -dt "$RUN2"/step_* | head -1)
echo "stage2: latest run=$RUN2 ckpt=$CKPT2"

echo "stage2: eval (textvqa validation)"
uv run python -m scripts.eval \
  --ckpt "$CKPT2" \
  --dataset textvqa \
  --split validation \
  --override logging.backend=wandb \
  "${eval_args[@]}"
