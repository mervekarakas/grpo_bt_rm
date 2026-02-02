if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
fi

# Always run from repo root, but source env relative to this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/setup_env.sh"

GPU_LIST="${1:-0,1}"  # e.g. "0,1" or "2,3,4,5"

# Required (ideally set by configs/runs/*.env)
DATASET="${DATASET:-}"
OUT_DIR="${OUT_DIR:-}"

# Plugin wiring (set by configs/runs/*.env)
PLUGIN="${PLUGIN:-src/grpo_bt_rm/training/reward_plugins/bt_baseline.py}"
REWARD_NAME="${REWARD_NAME:-bt_pointwise_baseline}"

# Model + core training knobs (overridable via env)
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-128}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"

PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-2}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"

LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_STEPS="${MAX_STEPS:-8000}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"

DATASET_SHUFFLE="${DATASET_SHUFFLE:-true}"
DYNAMIC_SAMPLE="${DYNAMIC_SAMPLE:-true}"
MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-3}"

REPORT_TO="${REPORT_TO:-wandb}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-true}"

# W&B run name to avoid warning; override per run if you want
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$(basename "$OUT_DIR")}"

# Optional: try to silence DDP warning (only if swift supports this arg)
# Set DDP_FIND_UNUSED=false/true; default empty => don't pass flag.
DDP_FIND_UNUSED="${DDP_FIND_UNUSED:-}"

# Optional passthrough args (e.g. EXTRA_ARGS="--seed 42")
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$DATASET" || -z "$OUT_DIR" ]]; then
  echo "ERROR: DATASET and OUT_DIR must be set (source a configs/runs/*.env)."
  echo "Example:"
  echo "  source scripts/setup_env.sh"
  echo "  source configs/runs/score100_T20_C10_posclip.env"
  echo "  bash scripts/train_grpo.sh 0,1"
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NPROC="${#GPUS[@]}"

echo "Launching GRPO:"
echo "  CUDA_VISIBLE_DEVICES=$GPU_LIST (nproc=$NPROC)"
echo "  MODEL=$MODEL"
echo "  DATASET=$DATASET"
echo "  OUT_DIR=$OUT_DIR"
echo "  PLUGIN=$PLUGIN  REWARD_NAME=$REWARD_NAME"
echo "  BT_SCORE_PARSER=${BT_SCORE_PARSER:-<unset>}  BT_DELTA_TEMP=${BT_DELTA_TEMP:-<unset>}  BT_DELTA_CLIP=${BT_DELTA_CLIP:-<unset>}  BT_DELTA_NEG_CLIP=${BT_DELTA_NEG_CLIP:-<unset>}  BT_REWARD_SCALE=${BT_REWARD_SCALE:-<unset>}"
echo "  WANDB_PROJECT=${WANDB_PROJECT:-<unset>}  WANDB_RUN_NAME=$WANDB_RUN_NAME"
echo ""

cmd=(
  "$(which swift)" rlhf
  --rlhf_type grpo
  --use_hf true
  --model "$MODEL"
  --dataset "$DATASET"
  --external_plugins "$PLUGIN"
  --reward_funcs "$REWARD_NAME"
  --dataset_shuffle "$DATASET_SHUFFLE"
  --num_generations "$NUM_GENERATIONS"
  --generation_batch_size "$GENERATION_BATCH_SIZE"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE" --top_p "$TOP_P" --top_k "$TOP_K"
  --dynamic_sample "$DYNAMIC_SAMPLE" --max_resample_times "$MAX_RESAMPLE_TIMES"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BS"
  --gradient_accumulation_steps "$GRAD_ACC_STEPS"
  --learning_rate "$LEARNING_RATE"
  --max_steps "$MAX_STEPS"
  --save_steps "$SAVE_STEPS"
  --logging_steps "$LOGGING_STEPS"
  --report_to "$REPORT_TO"
  --run_name "$WANDB_RUN_NAME"
  --output_dir "$OUT_DIR"
  --log_completions "$LOG_COMPLETIONS"
)

if [[ -n "$DDP_FIND_UNUSED" ]]; then
  # If swift/transformers honors this flag, it will remove the warning.
  # If not recognized, you can just leave DDP_FIND_UNUSED unset.
  cmd+=(--ddp_find_unused_parameters "$DDP_FIND_UNUSED")
fi

# shellcheck disable=SC2206
if [[ -n "$EXTRA_ARGS" ]]; then
  cmd+=($EXTRA_ARGS)
fi

CUDA_VISIBLE_DEVICES="$GPU_LIST" torchrun --standalone --nproc_per_node="$NPROC" "${cmd[@]}"
