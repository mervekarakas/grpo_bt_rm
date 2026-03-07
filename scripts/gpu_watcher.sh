#!/bin/bash
# GPU watcher: monitors specified nvidia-smi GPUs and launches experiments
# when they've been idle for a sustained period.
#
# Usage:
#   bash scripts/gpu_watcher.sh
#
# Watches nvidia-smi GPUs 4,5 (= CUDA 0,1 = A100 80GB).
# When both GPUs have <1GB used memory for IDLE_MINUTES straight,
# launches the next queued experiment.
#
# Run in tmux:
#   tmux new -s watcher 'bash scripts/gpu_watcher.sh'
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- Configuration ----
# nvidia-smi GPU indices to watch
WATCH_GPUS=(4 5)
# CUDA_VISIBLE_DEVICES to use when launching (maps to the A100s)
CUDA_DEVICES="0,1"
# Memory threshold (MiB): GPU is "idle" if used memory below this
MEM_THRESHOLD_MIB=2000
# How many consecutive idle checks before launching (check_interval * idle_count = wait time)
CHECK_INTERVAL_SEC=60
IDLE_CHECKS_NEEDED=5  # 5 minutes of sustained idleness
# ---- End Configuration ----

# Experiment queue: list of experiments to run in order
# Edit this array to change what gets launched
EXPERIMENT_QUEUE=(exp4 exp1 exp3)

source /home/mkarakas/venvs/swift/bin/activate
export WANDB_PROJECT="grpo_bt_rm"

LOGDIR="/data/mkarakas/experiments/grpo_bt_rm"
mkdir -p "$LOGDIR"
WATCHER_LOG="$LOGDIR/gpu_watcher_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHER_LOG"
}

check_gpus_idle() {
    # Returns 0 (true) if ALL watched GPUs are below memory threshold
    for gpu_idx in "${WATCH_GPUS[@]}"; do
        local mem_used
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null | tr -d ' ')
        if [[ -z "$mem_used" ]]; then
            return 1  # Can't read GPU -> not idle
        fi
        if (( mem_used > MEM_THRESHOLD_MIB )); then
            return 1  # This GPU is busy
        fi
    done
    return 0  # All GPUs idle
}

get_gpu_status() {
    # Print current status of watched GPUs
    for gpu_idx in "${WATCH_GPUS[@]}"; do
        local info
        info=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$gpu_idx" 2>/dev/null)
        echo "  GPU $gpu_idx: $info"
    done
}

run_experiment() {
    local exp_name="$1"
    log "LAUNCHING experiment: $exp_name on CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"

    # Source the right env file and run training + eval
    bash scripts/launch_lambda_experiments.sh "$CUDA_DEVICES" "$exp_name" \
        2>&1 | tee -a "$LOGDIR/watcher_${exp_name}_$(date +%Y%m%d_%H%M%S).log"

    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        log "COMPLETED experiment: $exp_name (exit code 0)"
    else
        log "FAILED experiment: $exp_name (exit code $exit_code)"
    fi
    return $exit_code
}

# ---- Main Loop ----
log "GPU Watcher started"
log "Watching nvidia-smi GPUs: ${WATCH_GPUS[*]} (CUDA: $CUDA_DEVICES)"
log "Idle threshold: ${MEM_THRESHOLD_MIB} MiB, need ${IDLE_CHECKS_NEEDED} consecutive idle checks (${CHECK_INTERVAL_SEC}s each)"
log "Experiment queue: ${EXPERIMENT_QUEUE[*]}"
log ""

exp_index=0
idle_count=0
check_count=0

while (( exp_index < ${#EXPERIMENT_QUEUE[@]} )); do
    current_exp="${EXPERIMENT_QUEUE[$exp_index]}"
    check_count=$((check_count + 1))

    # Heartbeat every 10 checks (~10 min)
    if (( check_count % 10 == 0 )); then
        log "Heartbeat: waiting for GPUs. Next=$current_exp. Status:"
        get_gpu_status | while read -r line; do log "$line"; done
    fi

    if check_gpus_idle; then
        idle_count=$((idle_count + 1))
        if (( idle_count >= IDLE_CHECKS_NEEDED )); then
            log "GPUs idle for $((idle_count * CHECK_INTERVAL_SEC))s — launching $current_exp"
            get_gpu_status | while read -r line; do log "$line"; done

            run_experiment "$current_exp" || true

            exp_index=$((exp_index + 1))
            idle_count=0

            if (( exp_index < ${#EXPERIMENT_QUEUE[@]} )); then
                log "Next in queue: ${EXPERIMENT_QUEUE[$exp_index]}"
                log "Waiting for GPUs to become idle again..."
            fi
        else
            remaining=$(( (IDLE_CHECKS_NEEDED - idle_count) * CHECK_INTERVAL_SEC ))
            log "GPUs idle (check $idle_count/$IDLE_CHECKS_NEEDED). Launching $current_exp in ~${remaining}s if still idle."
        fi
    else
        if (( idle_count > 0 )); then
            log "GPUs busy again (was idle for $idle_count checks). Resetting counter."
        fi
        idle_count=0
    fi

    sleep "$CHECK_INTERVAL_SEC"
done

log ""
log "ALL EXPERIMENTS COMPLETE. Queue exhausted."
log "Results in: $LOGDIR"
