#!/bin/bash
# Poll Gemini API every 3 hours until rate limit resets, then collect teacher data.
# Run inside tmux: tmux new-session -d -s gemini_poll "bash scripts/gemini_poll_and_run.sh"

set -euo pipefail

source /home/mkarakas/venvs/swift/bin/activate
source /home/mkarakas/projects/grpo_bt_rm/scripts/setup_env.sh

API_KEY="${GEMINI_API_KEY:?Set GEMINI_API_KEY before running}"
POLL_INTERVAL=10800  # 3 hours in seconds
TEST_DIR="/tmp/gemini_rate_test"
OUTPUT_DIR="/data/mkarakas/experiments/grpo_bt_rm/teacher_data_gemini_batch2"

# How many pairs to collect once rate limit resets
N_PAIRS=5000
BATCH_SIZE=1
# Skip pairs already scored by all previous batches (B1=400, B2=500, B3=1000, Gemini_B1=25)
SKIP_PAIRS=5400

echo "$(date): Starting Gemini API polling (every ${POLL_INTERVAL}s / 3h)"
echo "  Will collect $N_PAIRS pairs once API is available"
echo "  Output: $OUTPUT_DIR"
echo ""

while true; do
    echo "$(date): Testing Gemini API..."
    rm -rf "$TEST_DIR"

    # Try to score 2 pairs as a test
    timeout 120 python -u scripts/teacher_generate.py \
        --backend gemini --model gemini-2.5-flash \
        --api_key "$API_KEY" \
        --n_pairs 2 --batch_size 1 --scale score100 --split train \
        --output_dir "$TEST_DIR" \
        --max_retries 1 \
        2>&1 | tail -5

    TEST_FILE="$TEST_DIR/score100_gemini_gemini_2.5_flash/results.jsonl"
    if [[ -f "$TEST_FILE" ]] && [[ $(wc -l < "$TEST_FILE") -ge 1 ]]; then
        echo ""
        echo "$(date): ✓ Gemini API is working! Starting full data collection..."
        echo ""

        python -u scripts/teacher_generate.py \
            --backend gemini --model gemini-2.5-flash \
            --api_key "$API_KEY" \
            --n_pairs "$N_PAIRS" --batch_size 1 --scale score100 --split train \
            --skip_pairs "$SKIP_PAIRS" \
            --resume \
            --output_dir "$OUTPUT_DIR" \
            2>&1 | tee "$OUTPUT_DIR/gemini_batch2.log"

        echo ""
        echo "$(date): Data collection finished!"
        RESULT_FILE="$OUTPUT_DIR/score100_gemini_gemini_2.5_flash/results.jsonl"
        if [[ -f "$RESULT_FILE" ]]; then
            echo "Total pairs: $(wc -l < "$RESULT_FILE")"
        fi
        break
    else
        echo "$(date): ✗ Still rate-limited. Sleeping ${POLL_INTERVAL}s (3h)..."
        echo ""
        sleep "$POLL_INTERVAL"
    fi
done

echo "$(date): Done. Exiting."
exec bash
