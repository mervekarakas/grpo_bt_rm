#!/bin/bash
# Binary reward GRPO on Qwen2.5-7B-Instruct (A100 GPUs)
# Compares against BT RM baseline (68.6%) and continuous advfilt (63.9%)
cd /home/mkarakas/projects/grpo_bt_rm
source /home/mkarakas/venvs/swift/bin/activate
export WANDB_PROJECT="grpo_bt_rm"
source configs/runs/hh_v3_binary_advfilt_base.env
source scripts/setup_env.sh
bash scripts/train_grpo.sh 0,1 2>&1 | tee /data/mkarakas/experiments/grpo_bt_rm/hh_v3_binary_advfilt_base/train.log
