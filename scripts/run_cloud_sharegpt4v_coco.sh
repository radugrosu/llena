#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 download|process [args...]"
  exit 1
fi

cmd="$1"
shift

case "$cmd" in
  download)
    uv run python -m scripts.download_sharegpt4v_coco --out-dir datasets/raw/sharegpt4v_coco --max-workers 8 "$@"
    ;;
  process)
    uv run python -m scripts.process_sharegpt4v_coco --out-dir datasets/processed/sharegpt4v_coco "$@"
    ;;
  *)
    echo "Unknown command: $cmd"
    exit 1
    ;;
esac
