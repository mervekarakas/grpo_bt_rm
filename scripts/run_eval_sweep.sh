if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
fi

usage() {
  cat <<EOF
Usage: $0 --run_dir RUN_DIR [options]

Core:
  --run_dir PATH         Run dir containing checkpoints (may contain nested v0-*/)
  --eval_subdir NAME     Eval folder name under resolved run dir (default: from \$EVAL_SUBDIR or "eval_score100_2k")

Sweep:
  --ckpts "..."          Space-separated list of checkpoints
  --seeds "..."          Space-separated list of seeds
  --gpus "..."           Space-separated CUDA_VISIBLE_DEVICES IDs to use

Eval params:
  --n_pairs N            (default: from \$EVAL_N_PAIRS or 2000)
  --dtype bf16|fp16      (default: from \$EVAL_DTYPE or bf16)
  --max_new_tokens N     (default: from \$EVAL_MAX_NEW_TOKENS or 128)
  --batch_pairs N        (default: from \$EVAL_BATCH_PAIRS or 16)
  --do_sample            (default: from \$EVAL_DO_SAMPLE or on)
  --no_sample            disable sampling
  --n_samples N          (default: from \$EVAL_N_SAMPLES or 2)
  --temperature X        (default: from \$EVAL_TEMPERATURE or 0.7)
  --top_p X              (default: from \$EVAL_TOP_P or 0.9)
  --top_k N              (default: from \$EVAL_TOP_K or 50)

Prompt/parsing:
  --prompt NAME          (default: from \$EVAL_PROMPT or score100_v1)
  --parser NAME          (default: from \$EVAL_PARSER or score100_first)

BT + uncertainty:
  --bt_temp X            (default: from \$EVAL_BT_TEMP or 10)
  --unc_low X            (default: from \$EVAL_UNC_LOW or 20)
  --unc_high X           (default: from \$EVAL_UNC_HIGH or 80)
  --no_report_margins    Disable --report_margins

Output:
  --no_summary           Skip tools/summarize_eval_logs.py
EOF
}

# ---------------- repo root + env ----------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"

# Prefer sourcing setup_env (loads .env + sets PYTHONPATH + HF cache)
if [[ -f "$REPO_ROOT/scripts/setup_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$REPO_ROOT/scripts/setup_env.sh"
else
  # fallback: minimal PYTHONPATH so -m works
  export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
fi

# ---------------- defaults (env -> fallback) ----------------
RUN_DIR=""
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_score100_2k}"

N_PAIRS="${EVAL_N_PAIRS:-2000}"
DTYPE="${EVAL_DTYPE:-bf16}"
MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
BATCH_PAIRS="${EVAL_BATCH_PAIRS:-16}"

DO_SAMPLE="${EVAL_DO_SAMPLE:-1}"   # 1=on, 0=off
N_SAMPLES="${EVAL_N_SAMPLES:-2}"
TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
TOP_P="${EVAL_TOP_P:-0.9}"
TOP_K="${EVAL_TOP_K:-50}"

PROMPT="${EVAL_PROMPT:-score100_v1}"
PARSER="${EVAL_PARSER:-score100_first}"

BT_TEMP="${EVAL_BT_TEMP:-10}"
UNC_LOW="${EVAL_UNC_LOW:-20}"
UNC_HIGH="${EVAL_UNC_HIGH:-80}"
REPORT_MARGINS=1
DO_SUMMARY=1

USE_TRAIN_ARGS=0

# defaults similar to your old sweeps
CKPTS=(checkpoint-0 checkpoint-1000 checkpoint-2000 checkpoint-3000 checkpoint-4000 checkpoint-6000 checkpoint-8000)
SEEDS=(0 1)
GPUS=(0 1 2 3 4 5 6)

# ---------------- parse args ----------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir) RUN_DIR="$2"; shift 2 ;;
    --eval_subdir) EVAL_SUBDIR="$2"; shift 2 ;;

    --ckpts) read -r -a CKPTS <<< "$2"; shift 2 ;;
    --seeds) read -r -a SEEDS <<< "$2"; shift 2 ;;
    --gpus)  read -r -a GPUS  <<< "$2"; shift 2 ;;

    --n_pairs) N_PAIRS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --batch_pairs) BATCH_PAIRS="$2"; shift 2 ;;

    --do_sample) DO_SAMPLE=1; shift 1 ;;
    --no_sample) DO_SAMPLE=0; shift 1 ;;
    --n_samples) N_SAMPLES="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top_p) TOP_P="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;

    --prompt) PROMPT="$2"; shift 2 ;;
    --parser) PARSER="$2"; shift 2 ;;

    --bt_temp) BT_TEMP="$2"; shift 2 ;;
    --unc_low) UNC_LOW="$2"; shift 2 ;;
    --unc_high) UNC_HIGH="$2"; shift 2 ;;
    --no_report_margins) REPORT_MARGINS=0; shift 1 ;;

    --no_summary) DO_SUMMARY=0; shift 1 ;;
    --use_train_args) USE_TRAIN_ARGS=1; shift 1 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "ERROR: --run_dir is required"
  usage
  exit 1
fi

# ---------------- resolve nested v0-* ----------------
RESOLVED_RUN_DIR="$RUN_DIR"
if [[ -d "$RUN_DIR" ]]; then
  mapfile -t V0S < <(find "$RUN_DIR" -maxdepth 1 -type d -name "v0-*" | sort)
  if [[ ${#V0S[@]} -ge 1 ]]; then
    RESOLVED_RUN_DIR="${V0S[-1]}"   # pick last/newest
  fi
fi

OUT_DIR="$RESOLVED_RUN_DIR/$EVAL_SUBDIR"
mkdir -p "$OUT_DIR"

ARGS_JSON="$RESOLVED_RUN_DIR/args.json"
if [[ $USE_TRAIN_ARGS -eq 1 && -f "$ARGS_JSON" ]]; then
  echo "Loading defaults from $ARGS_JSON"

  jget () {
    python - <<PY
import json,sys
p=sys.argv[1]; k=sys.argv[2]
try:
  d=json.load(open(p))
  v=d.get(k,"")
  print(v if v is not None else "")
except Exception:
  print("")
PY
  "$ARGS_JSON" "$1"
  }

  v=$(jget temperature); [[ -n "$v" && "$TEMPERATURE" == "${EVAL_TEMPERATURE:-0.7}" ]] && TEMPERATURE="$v"
  v=$(jget top_p);       [[ -n "$v" && "$TOP_P" == "${EVAL_TOP_P:-0.9}" ]]            && TOP_P="$v"
  v=$(jget top_k);       [[ -n "$v" && "$TOP_K" == "${EVAL_TOP_K:-50}" ]]             && TOP_K="$v"

  v=$(jget max_new_tokens)
  if [[ -z "$v" ]]; then v=$(jget max_completion_length); fi
  [[ -n "$v" && "$MAX_NEW_TOKENS" == "${EVAL_MAX_NEW_TOKENS:-128}" ]] && MAX_NEW_TOKENS="$v"

  v=$(jget n_samples)
  if [[ -z "$v" ]]; then v=$(jget num_generations); fi
  [[ -n "$v" && "$N_SAMPLES" == "${EVAL_N_SAMPLES:-2}" ]] && N_SAMPLES="$v"
fi

echo "RUN_DIR (input):    $RUN_DIR"
echo "RUN_DIR (resolved): $RESOLVED_RUN_DIR"
echo "OUT_DIR:            $OUT_DIR"
echo "CKPTS:              ${CKPTS[*]}"
echo "SEEDS:              ${SEEDS[*]}"
echo "GPUS:               ${GPUS[*]}"
echo "n_pairs=$N_PAIRS dtype=$DTYPE max_new_tokens=$MAX_NEW_TOKENS batch_pairs=$BATCH_PAIRS"
echo "sample=$DO_SAMPLE n_samples=$N_SAMPLES temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K"
echo "prompt=$PROMPT parser=$PARSER bt_temp=$BT_TEMP unc_low=$UNC_LOW unc_high=$UNC_HIGH report_margins=$REPORT_MARGINS"
echo ""

MAXJOBS=${#GPUS[@]}
i=0

for SEED in "${SEEDS[@]}"; do
  for CKPT in "${CKPTS[@]}"; do
    GPU=${GPUS[$((i % MAXJOBS))]}
    OUT="$OUT_DIR/eval_${CKPT}_seed${SEED}.log"
    echo "EVAL seed=$SEED ckpt=$CKPT cuda=$GPU -> $OUT"

    cmd=(python -m grpo_bt_rm.eval.eval_bt
      --run_dir "$RESOLVED_RUN_DIR"
      --checkpoint "$RESOLVED_RUN_DIR/$CKPT"
      --n_pairs "$N_PAIRS"
      --seed "$SEED"
      --dtype "$DTYPE"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --batch_pairs "$BATCH_PAIRS"
      --n_samples "$N_SAMPLES"
      --temperature "$TEMPERATURE"
      --top_p "$TOP_P"
      --top_k "$TOP_K"
      --prompt "$PROMPT"
      --parser "$PARSER"
      --bt_temp "$BT_TEMP"
      --unc_low "$UNC_LOW"
      --unc_high "$UNC_HIGH"
    )

    if [[ $DO_SAMPLE -eq 1 ]]; then
      cmd+=(--do_sample)
    fi
    if [[ $REPORT_MARGINS -eq 1 ]]; then
      cmd+=(--report_margins)
    fi

    CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" > "$OUT" 2>&1 &

    i=$((i+1))
    if (( i % MAXJOBS == 0 )); then
      wait
    fi
  done
done

wait
echo "DONE. Logs in $OUT_DIR"

if [[ $DO_SUMMARY -eq 1 ]]; then
  python "$REPO_ROOT/tools/summarize_eval_logs.py" \
    --logdir "$OUT_DIR" \
    --pattern 'eval_checkpoint-*_seed*.log' \
    --out_md "$OUT_DIR/summary.md" \
    --out_csv "$OUT_DIR/summary.csv"
  echo "Summary written:"
  echo "  $OUT_DIR/summary.md"
  echo "  $OUT_DIR/summary.csv"

    # ---- also push to W&B (as artifact + table) ----
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    python "$REPO_ROOT/tools/wandb_log_eval_summary.py" \
      --logdir "$OUT_DIR" \
      --summary_csv "$OUT_DIR/summary.csv" \
      --summary_md "$OUT_DIR/summary.md" \
      --run_name "eval_${EVAL_SUBDIR}_$(basename "$RESOLVED_RUN_DIR")" \
      ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"}
  else
    echo "WANDB_PROJECT not set; skipping W&B summary upload."
  fi
fi
