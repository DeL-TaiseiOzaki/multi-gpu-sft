#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --config_file configs/accelerate_config_zero3.yaml \
    train_full.py \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 128 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.1 \
    --bf16 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --dataloader_pin_memory True \
    --max_seq_length 1024 \
    --data_files data/sft_reasoning_missinfo.jsonl \
    --model_name_or_path elyza/Llama-3-ELYZA-JP-8B \
    --output_dir output_elyza_ep3_lowlr \