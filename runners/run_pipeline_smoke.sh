#!/usr/bin/env bash
set -euo pipefail

CFG_PROFILE="${1:-cpu}"
STAGE1_CFG="${STAGE1_CFG:-configs/${CFG_PROFILE}/sharegpt4v_train_qwen2.5-0.5b_siglip224.yaml}"
STAGE2_CFG="${STAGE2_CFG:-configs/${CFG_PROFILE}/llava_textvqa_train_qwen2.5-0.5b_siglip224.yaml}"

## ===========================================================================
if [[ -f artifacts/latest_stage1.txt ]]; then
  RUN1=$(cat artifacts/latest_stage1.txt)
  echo "stage1: using existing run=$RUN1"
else
  echo "stage1: train"
  uv run python -m scripts.train --config "$STAGE1_CFG"
  RUN1=$(cat artifacts/latest_run.txt)
  printf '%s' "$RUN1" >artifacts/latest_stage1.txt
fi

CKPT1=$(ls -dt "$RUN1"/step_* | head -1)
echo "stage1: latest run=$RUN1 ckpt=$CKPT1"

## ===========================================================================
echo "stage1: eval (sharegpt4v_coco validation)"
uv run python -m scripts.eval \
  --ckpt "$CKPT1" \
  --override data.split=validation

## ===========================================================================
echo "stage1: eval (textvqa validation)"
uv run python -m scripts.eval \
  --ckpt "$CKPT1" \
  --dataset textvqa \
  --split validation \
  --eval-samples-path datasets/samples.json

## ===========================================================================
if [[ -f artifacts/latest_stage2.txt ]]; then
  RUN2=$(cat artifacts/latest_stage2.txt)
  echo "stage2: using existing run=$RUN2"
else
  echo "stage2: train"
  uv run python -m scripts.train \
    --config "$STAGE2_CFG" \
    --resume "$RUN1"
  RUN2=$(cat artifacts/latest_run.txt)
  printf '%s' "$RUN2" >artifacts/latest_stage2.txt
fi

CKPT2=$(ls -dt "$RUN2"/step_* | head -1)
echo "stage2: latest run=$RUN2 ckpt=$CKPT2"

echo "stage2: eval (textvqa validation)"
uv run python -m scripts.eval \
  --ckpt "$CKPT2" \
  --dataset textvqa \
  --split validation \
  --eval-samples-path datasets/samples.json
