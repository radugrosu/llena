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
  uv run python -m scripts.download_llava_instruct --out-dir datasets/raw/llava_instruct --max-workers 16 "$@"
  ;;
process)
  uv run python -m scripts.process_llava_instruct --out-dir datasets/processed/llava_instruct "$@"
  ;;
*)
  echo "Unknown command: $cmd"
  exit 1
  ;;
esac
