#!/bin/bash
# Run ON the Lambda instance after rsync to set up the environment.
# Usage: bash /data/grpo_bt_rm/scripts/setup_lambda_remote.sh
set -euo pipefail

echo "=== Setting up Lambda environment ==="

# 1. Create venv if not exists
if [[ ! -d /data/venv ]]; then
    echo "[1/5] Creating Python venv..."
    python3 -m venv /data/venv
else
    echo "[1/5] Venv already exists at /data/venv"
fi
source /data/venv/bin/activate

# 2. Install ms-swift + dependencies
echo "[2/5] Installing ms-swift..."
if [[ ! -d /data/ms-swift ]]; then
    pip install ms-swift[rlhf] 2>&1 | tail -5
else
    echo "ms-swift already installed"
fi

# 3. Apply ms-swift patches
echo "[3/5] Applying ms-swift patches..."
SWIFT_DIR=$(python3 -c "import swift; import os; print(os.path.dirname(swift.__file__))")
echo "Swift location: $SWIFT_DIR"
cp /data/ms-swift-patches/args_mixin.py "$SWIFT_DIR/rlhf_trainers/args_mixin.py"
cp /data/ms-swift-patches/grpo_trainer.py "$SWIFT_DIR/rlhf_trainers/grpo_trainer.py"
cp /data/ms-swift-patches/reward_trainer.py "$SWIFT_DIR/rlhf_trainers/reward_trainer.py"
echo "Patches applied."

# 4. Install project dependencies
echo "[4/5] Installing project dependencies..."
pip install datasets transformers accelerate wandb 2>&1 | tail -3

# 5. Setup symlinks for dataset paths (scripts expect /data/$USER/datasets/)
echo "[5/5] Setting up data paths..."
mkdir -p /data/$USER/datasets /data/$USER/experiments/grpo_bt_rm
if [[ ! -L /data/$USER/datasets/grpo_bt_hh_train_hh_score100_v3.jsonl ]] && [[ ! -f /data/$USER/datasets/grpo_bt_hh_train_hh_score100_v3.jsonl ]]; then
    ln -sf /data/datasets/grpo_bt_hh_train_hh_score100_v3.jsonl /data/$USER/datasets/
fi
if [[ ! -L /data/$USER/datasets/grpo_bt_hh_train_hh_score100_scoreonly.jsonl ]] && [[ ! -f /data/$USER/datasets/grpo_bt_hh_train_hh_score100_scoreonly.jsonl ]]; then
    ln -sf /data/datasets/grpo_bt_hh_train_hh_score100_scoreonly.jsonl /data/$USER/datasets/
fi

# 6. Download model (if not cached)
echo "[6] Downloading Qwen2.5-7B-Instruct (if needed)..."
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); print('Tokenizer OK')" 2>&1 | tail -2
python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype='auto'); print('Model OK')" 2>&1 | tail -2

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run experiments:"
echo "  source /data/venv/bin/activate"
echo "  cd /data/grpo_bt_rm"
echo "  export WANDB_API_KEY=<your-key>"
echo "  wandb login \$WANDB_API_KEY"
echo ""
echo "  # Option A: All 3 sequentially on GPUs 0,1"
echo "  tmux new -s exps 'bash scripts/launch_lambda_experiments.sh 0,1'"
echo ""
echo "  # Option B: Parallel (2 GPUs each)"
echo "  tmux new -s exp12 'bash scripts/launch_lambda_experiments.sh 0,1 exp1 exp2'"
echo "  tmux new -s exp3  'bash scripts/launch_lambda_experiments.sh 2,3 exp3'"
