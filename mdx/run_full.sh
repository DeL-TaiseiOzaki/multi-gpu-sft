#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --config_file configs/accelerate_config_zero3.yaml \
    train_gemma.py \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 128 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --fp16 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --dataloader_pin_memory True \
    --max_seq_length 2048 \
    --data_files data/sft_filtered_processed.jsonl \
    --model_name_or_path DeL-TaiseiOzaki/Tengentoppa-gemma-2-9b-base \
    --output_dir output_gemma \
    --new_model_id DeL-TaiseiOzaki/Tengentoppa-gemma-2-9b-reasoning-it \