#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml>"
  exit 1
fi

uv run python -m scripts.eval --config "$1" --log-every 50
