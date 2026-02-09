#!/bin/bash
cd /home/mkarakas/projects/grpo_bt_rm
source /home/mkarakas/venvs/swift/bin/activate
# WANDB_API_KEY must be set in environment before running
export WANDB_PROJECT="grpo_bt_rm"
source configs/runs/hh_s5v3_T1_NC2_offset_base.env
export GENERATION_BATCH_SIZE=96  # must be divisible by global_batch=6 AND 2*num_gen=16
source scripts/setup_env.sh
bash scripts/train_grpo.sh 2,3,4 2>&1 | tee /data/mkarakas/experiments/grpo_bt_rm/hh_s5v3_T1_NC2_offset_base/train.log
