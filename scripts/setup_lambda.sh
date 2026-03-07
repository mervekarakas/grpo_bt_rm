#!/bin/bash
# Setup script for Lambda AI instance.
# Run on the LOCAL machine to sync code + data to Lambda.
#
# Usage:
#   bash scripts/setup_lambda.sh <LAMBDA_IP> [SSH_KEY]
#
# Prerequisites:
#   - Lambda instance running with persistent storage at /data
#   - SSH access configured
#
# After rsync, SSH in and run:
#   bash /data/grpo_bt_rm/scripts/setup_lambda_remote.sh
set -euo pipefail

LAMBDA_IP="${1:?Usage: setup_lambda.sh <LAMBDA_IP> [SSH_KEY]}"
SSH_KEY="${2:-}"

SSH_OPTS=""
if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS="-i $SSH_KEY"
fi

echo "=== Syncing code to Lambda: $LAMBDA_IP ==="

# 1. Sync project code (exclude wandb, outputs, __pycache__)
echo "[1/4] Syncing project code..."
rsync -avz --progress \
    --exclude='wandb/' \
    --exclude='outputs/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    -e "ssh $SSH_OPTS" \
    /home/mkarakas/projects/grpo_bt_rm/ \
    ubuntu@${LAMBDA_IP}:/data/grpo_bt_rm/

# 2. Sync ms-swift patches
echo "[2/4] Syncing ms-swift patches..."
rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    /data/mkarakas/src/ms-swift/swift/rlhf_trainers/args_mixin.py \
    /data/mkarakas/src/ms-swift/swift/rlhf_trainers/grpo_trainer.py \
    /data/mkarakas/src/ms-swift/swift/rlhf_trainers/reward_trainer.py \
    ubuntu@${LAMBDA_IP}:/data/ms-swift-patches/

# 3. Sync training datasets
echo "[3/4] Syncing training datasets..."
rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    /data/mkarakas/datasets/grpo_bt_hh_train_hh_score100_v3.jsonl \
    /data/mkarakas/datasets/grpo_bt_hh_train_hh_score100_scoreonly.jsonl \
    ubuntu@${LAMBDA_IP}:/data/datasets/

# 4. Print next steps
echo ""
echo "=== Sync complete ==="
echo ""
echo "Next steps:"
echo "  ssh $SSH_OPTS ubuntu@${LAMBDA_IP}"
echo "  bash /data/grpo_bt_rm/scripts/setup_lambda_remote.sh"
