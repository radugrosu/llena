#!/usr/bin/env bash
set -euo pipefail

MAX_WORKERS="${MAX_WORKERS:-8}"
TEXTVQA_SPLITS="${TEXTVQA_SPLITS:-train,validation}"

echo "download: sharegpt4v_coco + COCO images"
uv run python -m scripts.download_sharegpt4v_coco \
  --out-dir datasets/raw/sharegpt4v_coco \
  --coco-dir datasets/raw/coco2017 \
  --max-workers "$MAX_WORKERS"

echo "download: llava_instruct metadata"
uv run python -m scripts.download_llava_instruct \
  --out-dir datasets/raw/llava_instruct

echo "download: textvqa parquet"
uv run python -m scripts.download_dataset \
  --dataset textvqa \
  --out-dir datasets/raw

echo "process: sharegpt4v_coco"
uv run python -m scripts.process_sharegpt4v_coco \
  --out-dir datasets/processed/sharegpt4v_coco \
  --coco-images-dir datasets/raw/coco2017/images

echo "process: llava_instruct"
uv run python -m scripts.process_llava_instruct \
  --out-dir datasets/processed/llava_instruct \
  --coco-dir datasets/raw/coco2017

IFS=',' read -r -a splits <<< "$TEXTVQA_SPLITS"
for split in "${splits[@]}"; do
  split_trimmed="$(echo "$split" | xargs)"
  if [[ -z "$split_trimmed" ]]; then
    continue
  fi
  echo "process: textvqa ($split_trimmed)"
  uv run python -m scripts.process_dataset \
    --dataset textvqa \
    --split "$split_trimmed" \
    --raw-dir datasets/raw \
    --out-dir datasets/processed
done
