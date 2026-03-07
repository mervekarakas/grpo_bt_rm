#!/bin/bash
# Standard Bradley-Terry reward model training on Anthropic/hh-rlhf
# DWRL-inspired hyperparams: LR=5e-7, batch=32, 2 epochs, LoRA
# Uses Qwen2.5-7B-Instruct on 4x A6000 48GB

set -e

source /home/mkarakas/venvs/swift/bin/activate
source /home/mkarakas/projects/grpo_bt_rm/scripts/setup_env.sh

export CUDA_VISIBLE_DEVICES=4,5,6,7
NPROC_PER_NODE=4
OUTPUT_DIR=/data/mkarakas/experiments/grpo_bt_rm/bt_rm_baseline

NPROC_PER_NODE=$NPROC_PER_NODE \
swift rlhf \
    --rlhf_type rm \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use_hf true \
    --dataset /data/mkarakas/experiments/grpo_bt_rm/bt_rm_data/hh_train.jsonl \
    --val_dataset /data/mkarakas/experiments/grpo_bt_rm/bt_rm_data/hh_test.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 4 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --load_from_cache_file true \
    --report_to wandb
