#!/usr/bin/env bash
set -euo pipefail

CFG_PROFILE="${1:-L4}"
STAGE1_CFG="${STAGE1_CFG:-configs/${CFG_PROFILE}/sharegpt4v_train_qwen2.5-0.5b_siglip224.yaml}"

echo "stage1: train (cfg=$STAGE1_CFG)"
uv run python -m scripts.train --config "$STAGE1_CFG"

RUN1=$(cat artifacts/latest_run.txt)
printf '%s' "$RUN1" >artifacts/latest_stage1.txt

CKPT1=$(ls -dt "$RUN1"/step_* | head -1)
echo "stage1: latest run=$RUN1 ckpt=$CKPT1"

echo "stage1: eval (sharegpt4v_coco validation)"
uv run python -m scripts.eval \
  --ckpt "$CKPT1" \
  --split validation

echo "stage1: eval (textvqa validation)"
uv run python -m scripts.eval \
  --ckpt "$CKPT1" \
  --dataset textvqa \
  --split validation

echo "stage1: final eval (textvqa validation, generate)"
uv run python -m scripts.eval \
  --ckpt "$CKPT1" \
  --dataset textvqa \
  --split validation \
  --eval-mode generate
