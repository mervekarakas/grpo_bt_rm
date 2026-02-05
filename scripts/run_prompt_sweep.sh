#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_prompt_sweep.sh — Parallel prompt/scale sweep for HH base model
#
# Runs multiple variance.py experiments in parallel, one per GPU,
# then parses logs and prints a comparison table.
# ==============================================================================

# ---------------- repo root + env ----------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"

if [[ -f "$REPO_ROOT/scripts/setup_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$REPO_ROOT/scripts/setup_env.sh"
else
  export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
fi

# Activate venv
source /home/mkarakas/venvs/swift/bin/activate

# ---------------- config ----------------
OUT_BASE="/data/mkarakas/experiments/grpo_bt_rm/hh_score100_T20_C10_posclip/v0-20260204-002615/diagnostics/prompt_sweep"
mkdir -p "$OUT_BASE"

N_PAIRS=200
N_SAMPLES=8
BATCH_PAIRS=4
DATASET="anthropic_hh"
SPLIT="test"
SEED=0
TEMPERATURE=0.7
RANGE_THRESHOLDS="20,30"

# Each line: GPU  PROMPT  PARSER  CHAT(chat|nochat)  LABEL
EXPERIMENTS=(
  "2 hh_score100_v1 score100_first chat  hh_score100_v1_chat"
  "3 hh_score5_v1   score5_first   nochat hh_score5_v1_nochat"
  "4 hh_score5_v1   score5_first   chat  hh_score5_v1_chat"
  "5 hh_score100_v2 score100_first chat  hh_score100_v2_chat"
  "6 hh_score5_v2   score5_first   chat  hh_score5_v2_chat"
)

echo "=== Prompt Sweep ==="
echo "OUT_BASE: $OUT_BASE"
echo "N_PAIRS=$N_PAIRS  N_SAMPLES=$N_SAMPLES  BATCH_PAIRS=$BATCH_PAIRS"
echo "DATASET=$DATASET  SPLIT=$SPLIT  SEED=$SEED"
echo "Experiments: ${#EXPERIMENTS[@]}"
echo ""

# ---------------- launch experiments ----------------
PIDS=()
LABELS=()
LOG_FILES=()

for exp in "${EXPERIMENTS[@]}"; do
  read -r GPU PROMPT PARSER CHAT LABEL <<< "$exp"
  LABELS+=("$LABEL")

  EXP_DIR="$OUT_BASE/$LABEL"
  mkdir -p "$EXP_DIR"
  LOG_FILE="$EXP_DIR/variance.log"
  JSONL_FILE="$EXP_DIR/variance.jsonl"
  LOG_FILES+=("$LOG_FILE")

  CHAT_FLAG=""
  if [[ "$CHAT" == "chat" ]]; then
    CHAT_FLAG="--use_chat_template"
  fi

  echo "Launching: $LABEL  (GPU=$GPU, prompt=$PROMPT, parser=$PARSER, chat=$CHAT)"

  CUDA_VISIBLE_DEVICES="$GPU" python -m grpo_bt_rm.eval.variance \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --prompt "$PROMPT" \
    --parser "$PARSER" \
    --n_pairs "$N_PAIRS" \
    --n_samples "$N_SAMPLES" \
    --batch_pairs "$BATCH_PAIRS" \
    --seed "$SEED" \
    --temperature "$TEMPERATURE" \
    --range_thresholds "$RANGE_THRESHOLDS" \
    --save_jsonl "$JSONL_FILE" \
    $CHAT_FLAG \
    > "$LOG_FILE" 2>&1 &

  PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} experiments launched. Waiting..."
echo ""

# ---------------- wait for completion ----------------
FAILED=0
for i in "${!PIDS[@]}"; do
  PID=${PIDS[$i]}
  LABEL=${LABELS[$i]}
  if wait "$PID"; then
    echo "  DONE: $LABEL (pid=$PID)"
  else
    echo "  FAILED: $LABEL (pid=$PID) — check ${LOG_FILES[$i]}"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
if [[ $FAILED -gt 0 ]]; then
  echo "WARNING: $FAILED experiment(s) failed. Printing table for successful ones."
fi

# ---------------- parse logs + print table ----------------
echo ""
echo "=== Comparison Table ==="
echo ""

# Header
printf "%-24s %-5s %6s %6s %6s %6s %5s %6s %8s %8s %7s %7s\n" \
  "PROMPT" "CHAT" "VALID%" "ACC" "ACC_S" "W_STD" "UNIQ" "TIES" "UNC_ANY" "UNC_BTH" "RNG≥20" "RNG≥30"
printf "%s\n" "------------------------------------------------------------------------------------------------------------------------------"

# Baseline row (hardcoded from existing results)
printf "%-24s %-5s %6s %6s %6s %6s %5s %6s %8s %8s %7s %7s\n" \
  "hh_score100_v1" "no" "90.2" "56.9" "47.1" "7.08" "3.65" "19.5" "0.0" "0.0" "63.7" "37.5"

# Parse each log
for i in "${!LABELS[@]}"; do
  LABEL="${LABELS[$i]}"
  LOG="${LOG_FILES[$i]}"

  if [[ ! -f "$LOG" ]]; then
    printf "%-24s %-5s %s\n" "$LABEL" "?" "LOG NOT FOUND"
    continue
  fi

  # Check if log has final summary
  if ! grep -q "=== Final summary across pairs ===" "$LOG"; then
    printf "%-24s %-5s %s\n" "$LABEL" "?" "INCOMPLETE/FAILED"
    continue
  fi

  # Extract prompt name and chat status from label
  # Label format: <prompt>_chat or <prompt>_nochat
  if [[ "$LABEL" == *"_chat" ]]; then
    CHAT_COL="yes"
  else
    CHAT_COL="no"
  fi
  # Prompt name: remove _chat or _nochat suffix
  PROMPT_COL="${LABEL%_chat}"
  PROMPT_COL="${PROMPT_COL%_nochat}"

  # Parse metrics from log (use head -1 to avoid matching Notes section)
  VALID=$(grep -m1 "avg valid score rate:" "$LOG" | awk '{print $NF}')
  W_STD=$(grep -m1 "avg within-summary std:" "$LOG" | awk '{print $NF}')
  UNIQ=$(grep -m1 "avg unique scores/summary:" "$LOG" | awk '{print $NF}')
  TIES=$(grep -m1 "avg tie rate (delta==0):" "$LOG" | awk '{print $NF}')
  ACC=$(grep -m1 "^avg accuracy (ties=0.5):" "$LOG" | awk '{print $NF}')
  ACC_S=$(grep -m1 "^avg accuracy (strict ties=0):" "$LOG" | awk '{print $NF}')

  # Uncertainty metrics
  UNC_ANY=$(grep -m1 "uncertain_any among wrong:" "$LOG" | awk '{print $NF}' || echo "N/A")
  UNC_BOTH=$(grep -m1 "uncertain_both among wrong:" "$LOG" | awk '{print $NF}' || echo "N/A")

  # Range metrics
  RNG20=$(grep -m1 "range_any>=20 among wrong:" "$LOG" | awk '{print $NF}' || echo "N/A")
  RNG30=$(grep -m1 "range_any>=30 among wrong:" "$LOG" | awk '{print $NF}' || echo "N/A")

  # Convert fractions to percentages
  to_pct() {
    local val="$1"
    if [[ "$val" == "N/A" || -z "$val" || "$val" == "nan" ]]; then
      echo "N/A"
    else
      awk -v v="$val" 'BEGIN {printf "%.1f", v * 100}'
    fi
  }

  VALID_PCT=$(to_pct "$VALID")
  ACC_PCT=$(to_pct "$ACC")
  ACC_S_PCT=$(to_pct "$ACC_S")
  TIES_PCT=$(to_pct "$TIES")
  UNC_ANY_PCT=$(to_pct "$UNC_ANY")
  UNC_BOTH_PCT=$(to_pct "$UNC_BOTH")
  RNG20_PCT=$(to_pct "$RNG20")
  RNG30_PCT=$(to_pct "$RNG30")

  # W_STD and UNIQ are already in absolute form
  if [[ "$W_STD" == "nan" || -z "$W_STD" ]]; then
    W_STD_FMT="N/A"
  else
    W_STD_FMT=$(awk -v v="$W_STD" 'BEGIN {printf "%.2f", v}')
  fi
  if [[ "$UNIQ" == "nan" || -z "$UNIQ" ]]; then
    UNIQ_FMT="N/A"
  else
    UNIQ_FMT=$(awk -v v="$UNIQ" 'BEGIN {printf "%.2f", v}')
  fi

  printf "%-24s %-5s %6s %6s %6s %6s %5s %6s %8s %8s %7s %7s\n" \
    "$PROMPT_COL" "$CHAT_COL" "$VALID_PCT" "$ACC_PCT" "$ACC_S_PCT" "$W_STD_FMT" "$UNIQ_FMT" "$TIES_PCT" "$UNC_ANY_PCT" "$UNC_BOTH_PCT" "$RNG20_PCT" "$RNG30_PCT"
done

echo ""
echo "Legend:"
echo "  VALID%    = format compliance (parse success rate)"
echo "  ACC       = accuracy with ties=0.5"
echo "  ACC_S     = strict accuracy (ties=wrong)"
echo "  W_STD     = avg within-summary std"
echo "  UNIQ      = avg unique scores per summary"
echo "  TIES      = tie rate"
echo "  UNC_ANY   = % of wrong pairs with at least one uncertain summary"
echo "  UNC_BTH   = % of wrong pairs where both summaries uncertain"
echo "  RNG≥20/30 = % of wrong pairs with score range ≥ threshold (any side)"
echo ""
echo "Logs and JSONL files in: $OUT_BASE"
