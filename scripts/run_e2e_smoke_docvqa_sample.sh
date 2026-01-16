#!/usr/bin/env bash
set -euo pipefail

uv run python -m scripts.e2e_smoke \
  --config configs/qwen2_5_0_5b_docvqa_sample.yaml \
  --stage projector \
  --override train.device=cpu
