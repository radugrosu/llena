#!/usr/bin/env bash

cd /lambda/nfs/rg-data

git clone git@github.com:radugrosu/llena.git
cd llena

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Configure uv to keep the cache and venv on the persistent disk
export UV_CACHE_DIR="/lambda/nfs/my-data/.uv_cache"
export UV_PROJECT_ENVIRONMENT="/lambda/nfs/my-data/llena/.venv"

uv sync

uv run bash runners/run_data_prep.sh
