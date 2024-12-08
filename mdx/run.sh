#!/bin/bash

accelerate launch --config_file configs/accelerate_config_notzero.yaml \
    train_4bit.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --load_in_8bit \
    --max_seq_length 2048 \
    --data_files data/sft_filtered.jsonl \
    --use_peft \
    --bf16 \
    --peft_lora_r 8 \
    --peft_lora_alpha 16 \
    --peft_lora_dropout 0.05 \
    --model_name_or_path DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B \
    --output_dir output2 \
    --hub_model_id DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B-reasoning-instruct \
