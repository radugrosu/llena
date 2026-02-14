#!/usr/bin/env bash
set -euo pipefail

RUN1="${1:-}"
CFG_PROFILE="${2:-L4}"
STAGE2_CFG="${STAGE2_CFG:-configs/${CFG_PROFILE}/llava_textvqa_train_qwen2.5-0.5b_siglip224.yaml}"

if [[ -z "$RUN1" ]]; then
  if [[ -f artifacts/latest_stage1.txt ]]; then
    RUN1=$(cat artifacts/latest_stage1.txt)
  else
    echo "stage2: missing stage1 run path. Usage: ./runners/run_stage2.sh <stage1_run_dir> [CFG_PROFILE]" >&2
    exit 1
  fi
fi

if [[ ! -d "$RUN1" ]]; then
  echo "stage2: stage1 run directory does not exist: $RUN1" >&2
  exit 1
fi

echo "stage2: train (cfg=$STAGE2_CFG, resume=$RUN1)"
uv run python -m scripts.train \
  --config "$STAGE2_CFG" \
  --resume "$RUN1"

RUN2=$(cat artifacts/latest_run.txt)
printf '%s' "$RUN2" >artifacts/latest_stage2.txt

CKPT2=$(ls -dt "$RUN2"/step_* | head -1)
echo "stage2: latest run=$RUN2 ckpt=$CKPT2"

echo "stage2: eval (textvqa validation)"
uv run python -m scripts.eval \
  --ckpt "$CKPT2" \
  --dataset textvqa \
  --split validation

echo "stage2: final eval (textvqa validation, generate)"
uv run python -m scripts.eval \
  --ckpt "$CKPT2" \
  --dataset textvqa \
  --split validation \
  --eval-mode generate
