#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export CPM_LLM_MODEL_DIR="${CPM_LLM_MODEL_DIR:-$PWD/models/instruct}"
export CPM_OUTPUT_DIR="${CPM_OUTPUT_DIR:-$PWD/results/pilot}"
export CPM_SCOPE="${CPM_SCOPE:-pilot}"
export CPM_SHARDS=1 CPM_SHARD_ID=0
mkdir -p "$CPM_OUTPUT_DIR"
python src/check_inputs.py
python src/smoke_model.py 2>&1 | tee "$CPM_OUTPUT_DIR/smoke.log"
python src/run.py 2>&1 | tee "$CPM_OUTPUT_DIR/run.log"
python src/summarize.py "$CPM_OUTPUT_DIR"
