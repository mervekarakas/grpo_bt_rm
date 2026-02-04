#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --save_jsonl PATH [options]

Multi-GPU launcher for variance.py. Splits work across GPUs via sharding,
then merges shard results into a single output.

Required:
  --save_jsonl PATH        Base path for output JSONL (shards get .shard<N>.jsonl)

GPU:
  --gpus "0 1 2 3"         Space-separated CUDA device IDs (default: "0")

Variance params (passed through to variance.py):
  --dataset NAME           Dataset name from registry
  --split NAME             Dataset split
  --prompt NAME            Prompt name from registry
  --parser NAME            Parser name (default: prompt default)
  --n_pairs N              Number of pairs
  --n_samples N            Number of samples per summary
  --seed N                 Random seed
  --batch_pairs N          Batch size in pairs
  --max_new_tokens N       Max new tokens
  --temperature X          Sampling temperature
  --top_p X                Top-p sampling
  --top_k N                Top-k sampling
  --use_chat_template      Use chat template
  --range_thresholds "..." Comma-separated range thresholds
  --unc_low X              Uncertainty low threshold
  --unc_high X             Uncertainty high threshold
EOF
}

# ---------------- repo root + env ----------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"

if [[ -f "$REPO_ROOT/scripts/setup_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$REPO_ROOT/scripts/setup_env.sh"
else
  export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
fi

# ---------------- defaults ----------------
GPUS=(0)
SAVE_JSONL=""
PASSTHROUGH_ARGS=()

# ---------------- parse args ----------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)      read -r -a GPUS <<< "$2"; shift 2 ;;
    --save_jsonl) SAVE_JSONL="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    # Everything else is passed through to variance.py
    *)           PASSTHROUGH_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$SAVE_JSONL" ]]; then
  echo "ERROR: --save_jsonl is required"
  usage
  exit 1
fi

NUM_SHARDS=${#GPUS[@]}
BASE="${SAVE_JSONL%.jsonl}"

echo "=== run_variance_multigpu.sh ==="
echo "GPUS:       ${GPUS[*]}"
echo "NUM_SHARDS: $NUM_SHARDS"
echo "SAVE_JSONL: $SAVE_JSONL"
echo "PASSTHROUGH: ${PASSTHROUGH_ARGS[*]:-<none>}"
echo ""

# Create output directory
OUT_DIR="$(dirname "$SAVE_JSONL")"
if [[ -n "$OUT_DIR" ]]; then
  mkdir -p "$OUT_DIR"
fi

# ---------------- launch shards ----------------
SHARD_FILES=()
PIDS=()

for (( i=0; i<NUM_SHARDS; i++ )); do
  GPU=${GPUS[$i]}
  SHARD_JSONL="${BASE}.shard${i}.jsonl"
  SHARD_LOG="${BASE}.shard${i}.log"
  SHARD_FILES+=("$SHARD_JSONL")

  echo "Launching shard $i on GPU $GPU -> $SHARD_JSONL (log: $SHARD_LOG)"

  CUDA_VISIBLE_DEVICES="$GPU" python -m grpo_bt_rm.eval.variance \
    --shard_id "$i" \
    --num_shards "$NUM_SHARDS" \
    --save_jsonl "$SHARD_JSONL" \
    "${PASSTHROUGH_ARGS[@]}" \
    2>&1 | tee "$SHARD_LOG" &

  PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} shards to finish..."

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAILED=$((FAILED + 1))
  fi
done

if [[ $FAILED -gt 0 ]]; then
  echo "ERROR: $FAILED shard(s) failed. Check logs: ${BASE}.shard*.log"
  exit 1
fi

echo "All shards complete."
echo ""

# ---------------- merge ----------------
echo "Merging shards -> $SAVE_JSONL"
python "$REPO_ROOT/tools/merge_variance_shards.py" \
  "${SHARD_FILES[@]}" \
  --out "$SAVE_JSONL"

echo ""
echo "DONE. Merged output: $SAVE_JSONL"
echo "Shard logs: ${BASE}.shard*.log"
