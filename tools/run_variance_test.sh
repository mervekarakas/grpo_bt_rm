#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [options]

Core:
  --split SPLIT                 (default: validation)
  --n_pairs N                   (default: 500)
  --n_samples N                 (default: 8)
  --seed N                      (default: 0)

Gen:
  --max_new_tokens N            (default: 128)
  --temperature X               (default: 0.7)
  --top_p X                     (default: 0.9)
  --top_k N                     (default: 50)
  --batch_pairs N               (default: 16)
  --use_chat_template           (default: on)

Prompt/parsing:
  --prompt NAME                 (default: score100_v1)
  --parser NAME                 (default: use prompt default)

Diagnostics:
  --range_thresholds "a,b,c"    (default: 20,30,40)
  --unc_low X                   (default: auto: score100->20, score5->2)
  --unc_high X                  (default: auto: score100->80, score5->4)

Outputs:
  --out_prefix NAME             (default: variance_run)
  --out_dir PATH                (default: outputs/variance)
  --no_jsonl                    (don’t write jsonl)

Examples:
  $0 --prompt score100_v1 --range_thresholds "10,20,40"
  $0 --prompt score5_v1 --parser score5_last --n_pairs 200
EOF
}

# -------- defaults --------
SPLIT="validation"
N_PAIRS=500
N_SAMPLES=8
SEED=0

MAX_NEW_TOKENS=128
TEMP=0.7
TOP_P=0.9
TOP_K=50
BATCH_PAIRS=16
USE_CHAT_TEMPLATE=1

PROMPT="score100_v1"
PARSER=""

RANGE_THRESHOLDS="20,30,40"
UNC_LOW=""
UNC_HIGH=""

OUT_PREFIX="variance_run"
OUT_DIR="outputs/variance"
WRITE_JSONL=1

# -------- parse args --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --n_pairs) N_PAIRS="$2"; shift 2 ;;
    --n_samples) N_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;

    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMP="$2"; shift 2 ;;
    --top_p) TOP_P="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --batch_pairs) BATCH_PAIRS="$2"; shift 2 ;;
    --use_chat_template) USE_CHAT_TEMPLATE=1; shift 1 ;;
    --no_chat_template) USE_CHAT_TEMPLATE=0; shift 1 ;;

    --prompt) PROMPT="$2"; shift 2 ;;
    --parser) PARSER="$2"; shift 2 ;;

    --range_thresholds) RANGE_THRESHOLDS="$2"; shift 2 ;;
    --unc_low) UNC_LOW="$2"; shift 2 ;;
    --unc_high) UNC_HIGH="$2"; shift 2 ;;

    --out_prefix) OUT_PREFIX="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --no_jsonl) WRITE_JSONL=0; shift 1 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR"

# -------- repo root + PYTHONPATH --------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH-}"

# -------- auto uncertainty thresholds if not provided --------
if [[ -z "$UNC_LOW" || -z "$UNC_HIGH" ]]; then
  if [[ "$PROMPT" == score100_* ]]; then
    : "${UNC_LOW:=20}"
    : "${UNC_HIGH:=80}"
  else
    : "${UNC_LOW:=2}"
    : "${UNC_HIGH:=4}"
  fi
fi

STAMP="${OUT_PREFIX}_${PROMPT}_${SPLIT}_${N_PAIRS}x${N_SAMPLES}_t${TEMP}_seed${SEED}"
LOG_PATH="$OUT_DIR/${STAMP}.log"
JSONL_PATH="$OUT_DIR/${STAMP}.jsonl"

echo "Running variance:"
echo "  prompt=$PROMPT parser=${PARSER:-<default>}"
echo "  split=$SPLIT n_pairs=$N_PAIRS n_samples=$N_SAMPLES seed=$SEED"
echo "  temp=$TEMP top_p=$TOP_P top_k=$TOP_K max_new_tokens=$MAX_NEW_TOKENS batch_pairs=$BATCH_PAIRS"
echo "  range_thresholds=$RANGE_THRESHOLDS unc_low=$UNC_LOW unc_high=$UNC_HIGH"
echo "  out=$LOG_PATH"
echo ""

cmd=(python -m grpo_bt_rm.eval.variance
  --split "$SPLIT"
  --n_pairs "$N_PAIRS"
  --n_samples "$N_SAMPLES"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMP"
  --top_p "$TOP_P"
  --top_k "$TOP_K"
  --seed "$SEED"
  --batch_pairs "$BATCH_PAIRS"
  --prompt "$PROMPT"
  --range_thresholds "$RANGE_THRESHOLDS"
  --unc_low "$UNC_LOW"
  --unc_high "$UNC_HIGH"
)

if [[ -n "$PARSER" ]]; then
  cmd+=(--parser "$PARSER")
fi
if [[ $USE_CHAT_TEMPLATE -eq 1 ]]; then
  cmd+=(--use_chat_template)
fi
if [[ $WRITE_JSONL -eq 1 ]]; then
  cmd+=(--save_jsonl "$JSONL_PATH")
fi

"${cmd[@]}" 2>&1 | tee "$LOG_PATH"
echo "Done."
[[ $WRITE_JSONL -eq 1 ]] && echo "JSONL: $JSONL_PATH"
