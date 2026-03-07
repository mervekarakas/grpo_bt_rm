#!/bin/bash
# Master launch script: runs experiments sequentially with auto-eval.
# Usage: bash scripts/launch_lambda_experiments.sh [GPU_LIST] [exp4 exp1 exp3 ...]
# Default: GPU_LIST=0,1, runs exp4 (main) + exp1 (ablation) + exp3 (ablation)
#
# Experiments:
#   exp4 = Score-Only Hard-BT GRPO (MAIN METHOD)
#   exp1 = Full-token graded reward (ablation: shows credit assignment matters)
#   exp3 = Score-only graded reward (ablation: shows reward formula matters)
#   exp2 = Full-token graded + hard-pair (ablation)
#
# On Lambda with 8xA100-40GB, run in parallel:
#   tmux new -s main  'bash scripts/launch_lambda_experiments.sh 0,1 exp4'
#   tmux new -s abl   'bash scripts/launch_lambda_experiments.sh 2,3 exp1 exp3'
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_LIST="${1:-0,1}"
# Which experiments to run (default: all three)
shift || true
EXPERIMENTS=("${@:-exp4 exp1 exp3}")
if [[ ${#EXPERIMENTS[@]} -eq 0 ]] || [[ "${EXPERIMENTS[0]}" == "exp4 exp1 exp3" ]]; then
    EXPERIMENTS=(exp4 exp1 exp3)
fi

# Activate venv (Lambda uses /data/venv, local uses ~/venvs/swift)
if [[ -d /data/venv ]]; then
    source /data/venv/bin/activate
else
    source /home/mkarakas/venvs/swift/bin/activate
fi
export WANDB_PROJECT="grpo_bt_rm"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# A100 40GB safe defaults (reduce from 128 default)
export GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-64}"
export PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-2}"

RESULTS_DIR="/data/$USER/experiments/grpo_bt_rm"
mkdir -p "$RESULTS_DIR"
SUMMARY_FILE="$RESULTS_DIR/lambda_results_$(date +%Y%m%d_%H%M%S).txt"
echo "=== Lambda Experiment Results ===" > "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "GPU_LIST: $GPU_LIST" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

run_eval() {
    local run_dir="$1"
    local checkpoint="$2"
    local prompt="$3"
    local parser="$4"
    local bt_temp="$5"
    local max_tokens="${6:-128}"
    local eval_gpu="${7:-$GPU_LIST}"

    local ckpt_name
    ckpt_name=$(basename "$checkpoint")
    echo "[EVAL] $ckpt_name with prompt=$prompt parser=$parser bt_temp=$bt_temp"

    # Use first GPU only for eval (single-GPU)
    local first_gpu
    first_gpu=$(echo "$eval_gpu" | cut -d',' -f1)

    CUDA_VISIBLE_DEVICES="$first_gpu" python -u -m grpo_bt_rm.eval.eval_bt \
        --run_dir "$run_dir" \
        --checkpoint "$checkpoint" \
        --dataset anthropic_hh \
        --prompt "$prompt" \
        --parser "$parser" \
        --bt_temp "$bt_temp" \
        --n_pairs 500 \
        --batch_pairs 4 \
        --max_new_tokens "$max_tokens" \
        --report_margins \
        2>&1 | tee "$run_dir/eval_${ckpt_name}.log"

    # Extract accuracy from log
    local acc
    acc=$(grep "Accuracy (ties=0.5):" "$run_dir/eval_${ckpt_name}.log" | tail -1 | awk '{print $NF}')
    local tie_rate
    tie_rate=$(grep "Tie rate:" "$run_dir/eval_${ckpt_name}.log" | tail -1 | awk '{print $NF}')
    echo "  -> ACC=$acc TIE=$tie_rate"
    echo "$ckpt_name: ACC=$acc TIE=$tie_rate" >> "$SUMMARY_FILE"
}

run_experiment() {
    local exp_name="$1"
    local env_file="$2"
    local eval_prompt="$3"
    local eval_parser="$4"
    local eval_bt_temp="$5"
    local eval_max_tokens="${6:-128}"

    echo ""
    echo "=============================================="
    echo "  EXPERIMENT: $exp_name"
    echo "  $(date)"
    echo "=============================================="
    echo ""

    echo "--- $exp_name ---" >> "$SUMMARY_FILE"

    source "$env_file"

    echo "Config: $env_file"
    echo "Plugin: $PLUGIN"
    echo "Dataset: $DATASET"
    echo "Output: $OUT_DIR"

    # Ensure output dir exists
    mkdir -p "$OUT_DIR"

    # Check dataset exists
    if [[ ! -f "$DATASET" ]]; then
        echo "ERROR: Dataset not found: $DATASET"
        echo "$exp_name: SKIPPED (dataset missing)" >> "$SUMMARY_FILE"
        return 1
    fi

    # Train
    echo "[TRAIN] Starting at $(date)"
    bash scripts/train_grpo.sh "$GPU_LIST" 2>&1 | tee "$OUT_DIR/train.log" || {
        echo "$exp_name: TRAINING FAILED" >> "$SUMMARY_FILE"
        return 1
    }
    echo "[TRAIN] Finished at $(date)"

    # Find checkpoints to eval
    local run_dir
    run_dir=$(ls -dt "$OUT_DIR"/v*/ 2>/dev/null | head -1)
    if [[ -z "$run_dir" ]]; then
        echo "ERROR: No run directory found in $OUT_DIR"
        echo "$exp_name: NO RUN DIR" >> "$SUMMARY_FILE"
        return 1
    fi
    run_dir="${run_dir%/}"
    echo "Run dir: $run_dir"

    # Eval checkpoints (sorted numerically by step number)
    local checkpoints=()
    while IFS= read -r ckpt; do
        checkpoints+=("$ckpt")
    done < <(find "$run_dir" -maxdepth 1 -name "checkpoint-*" -type d | sort -t'-' -k2 -n)

    if [[ ${#checkpoints[@]} -eq 0 ]]; then
        echo "WARNING: No checkpoints found"
        echo "$exp_name: NO CHECKPOINTS" >> "$SUMMARY_FILE"
        return 1
    fi

    echo "[EVAL] Found ${#checkpoints[@]} checkpoints"
    # Eval every other checkpoint to save time, plus always eval last
    local n=${#checkpoints[@]}
    local eval_indices=()
    for ((i=0; i<n; i+=2)); do
        eval_indices+=("$i")
    done
    # Always include last
    if [[ $((n - 1)) -ne ${eval_indices[-1]} ]]; then
        eval_indices+=("$((n - 1))")
    fi

    for idx in "${eval_indices[@]}"; do
        local ckpt="${checkpoints[$idx]}"
        run_eval "$run_dir" "$ckpt" "$eval_prompt" "$eval_parser" "$eval_bt_temp" "$eval_max_tokens"
    done

    echo "" >> "$SUMMARY_FILE"
    echo "[DONE] $exp_name finished at $(date)"
}

# ===== EXPERIMENTS =====

for exp in "${EXPERIMENTS[@]}"; do
    case "$exp" in
        exp1)
            run_experiment \
                "EXP1: PaTaRM Graded Reward" \
                "configs/runs/hh_exp1_graded.env" \
                "hh_score100_v3" "score100_last" "5" "128"
            ;;
        exp2)
            run_experiment \
                "EXP2: Graded + Hard-Pair" \
                "configs/runs/hh_exp2_graded_hardpair.env" \
                "hh_score100_v3" "score100_last" "5" "128"
            ;;
        exp3)
            run_experiment \
                "EXP3: Score-Only Graded" \
                "configs/runs/hh_exp3_scoreonly_graded.env" \
                "hh_score100_scoreonly" "score100_last" "5" "16"
            ;;
        exp4)
            run_experiment \
                "EXP4: Score-Only Hard-BT (main)" \
                "configs/runs/hh_exp4_scoreonly_hardmargin.env" \
                "hh_score100_scoreonly" "score100_last" "5" "16"
            ;;
        *)
            echo "Unknown experiment: $exp (expected exp1, exp2, exp3, or exp4)"
            ;;
    esac
done

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "=============================================="
echo ""
echo "Results summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
