#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ckpt_path>"
  exit 1
fi

uv run python -m scripts.eval --ckpt "$1" --log-every 100
