#!/bin/bash

accelerate launch --config_file configs/default_config.yaml \
    train_full.py \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --max_seq_length 2048 \
    --data_files data/sft_filtered.jsonl \
    --bf16 \
    --model_name_or_path DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-3.7B \
    --output_dir output3 \
    --hub_model_id DeL-TaiseiOzaki/Tengentoppa-llm-jp-3.7B-reasoning-instruct