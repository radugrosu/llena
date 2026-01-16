#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [extra args...]"
  exit 1
fi

cfg="$1"
shift
uv run python -m scripts.train --config "$cfg" "$@"
