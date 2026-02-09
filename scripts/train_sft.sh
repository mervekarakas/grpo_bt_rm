#!/usr/bin/env bash
# SFT warm-up training for GRPO-BT reward modeling.
#
# Usage:
#   source scripts/setup_env.sh
#   bash scripts/train_sft.sh <gpu_list> <dataset_jsonl> <output_dir>
#
# Example:
#   bash scripts/train_sft.sh 2,3 /data/.../hh_sft_correct_only.jsonl /data/.../sft_v1

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/setup_env.sh"

GPU_LIST="${1:-0,1}"
DATASET="${2:-}"
OUT_DIR="${3:-}"

if [[ -z "$DATASET" || -z "$OUT_DIR" ]]; then
  echo "Usage: bash scripts/train_sft.sh <gpu_list> <dataset_jsonl> <output_dir>"
  echo "Example:"
  echo "  bash scripts/train_sft.sh 2,3 /data/.../hh_sft_correct_only.jsonl /data/.../sft_v1"
  exit 1
fi

# Model (overridable via env)
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# Training hyperparams
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-2}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$(basename "$OUT_DIR")}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NPROC="${#GPUS[@]}"

echo "Launching SFT:"
echo "  CUDA_VISIBLE_DEVICES=$GPU_LIST (nproc=$NPROC)"
echo "  MODEL=$MODEL"
echo "  DATASET=$DATASET"
echo "  OUT_DIR=$OUT_DIR"
echo "  LR=$LEARNING_RATE  EPOCHS=$NUM_EPOCHS  BS=$PER_DEVICE_TRAIN_BS  GRAD_ACC=$GRAD_ACC_STEPS"
echo ""

cmd=(
  "$(which swift)" sft
  --model "$MODEL"
  --use_hf true
  --dataset "$DATASET"
  --output_dir "$OUT_DIR"
  --num_train_epochs "$NUM_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BS"
  --gradient_accumulation_steps "$GRAD_ACC_STEPS"
  --max_length "$MAX_LENGTH"
  --save_strategy "$SAVE_STRATEGY"
  --logging_steps "$LOGGING_STEPS"
  --report_to "$REPORT_TO"
  --run_name "$WANDB_RUN_NAME"
  --gradient_checkpointing true
  --bf16 true
)

CUDA_VISIBLE_DEVICES="$GPU_LIST" torchrun --standalone --nproc_per_node="$NPROC" "${cmd[@]}"
