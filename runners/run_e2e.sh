#!/usr/bin/env bash
set -euo pipefail

echo "data prep: download + process"
./runners/run_data_prep.sh

echo "pipeline: train + eval"
./runners/run_pipeline.sh
