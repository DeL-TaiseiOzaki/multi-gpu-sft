#!/bin/bash

accelerate launch --config_file configs/accelerate_config_zero3.yaml \
    train_original.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --load_in_8bit \
    --max_seq_length 2048 \
    --data_files data/sft_filtered.jsonl \
    --use_peft \
    --bf16 \
    --peft_target_model llama-all \
    --peft_lora_r 8 \
    --peft_lora_alpha 32 \
    --peft_lora_dropout 0.05 \
    --model_name_or_path DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B \
    --output_dir output2 \
    --push_to_hub \
    --hub_model_id DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B-reasoning-instruct \
