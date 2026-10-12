#!/usr/bin/env bash
# Run only inside an allocation that already grants all requested GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export CPM_LLM_MODEL_DIR="${CPM_LLM_MODEL_DIR:-$PWD/models/instruct}"
export CPM_SCOPE=all
export CPM_SHARDS="${CPM_SHARDS:-8}"
OUT="${CPM_OUTPUT_DIR:-$PWD/results/all}"
mkdir -p "$OUT"
python src/check_inputs.py
# Preserve scheduler-assigned GPU identifiers, including UUIDs.
IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES:-$(python -c 'import torch; print(",".join(map(str,range(torch.cuda.device_count()))))')}"
if (( ${#DEVICES[@]} < CPM_SHARDS )) || [[ -z "${DEVICES[0]}" ]]; then
  echo 'Not enough allocated GPUs. Set CPM_SHARDS to allocated GPU count.' >&2; exit 2
fi
if (( CPM_SHARDS < 1 || CPM_SHARDS > 20 )); then echo 'CPM_SHARDS must be 1..20' >&2; exit 2; fi
CUDA_VISIBLE_DEVICES="${DEVICES[0]}" CPM_OUTPUT_DIR="$OUT/preflight" python src/smoke_model.py 2>&1 | tee "$OUT/preflight.log"
pids=()
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
for ((i=0;i<CPM_SHARDS;i++)); do
  mkdir -p "$OUT/shard_$i"
  CUDA_VISIBLE_DEVICES="${DEVICES[$i]}" CPM_SHARD_ID="$i" CPM_OUTPUT_DIR="$OUT/shard_$i" python src/run.py > "$OUT/shard_$i/run.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then echo 'A shard failed; inspect shard logs. No complete-run success claimed.' >&2; exit 1; fi
python src/summarize.py "$OUT"
