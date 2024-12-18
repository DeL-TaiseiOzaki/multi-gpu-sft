#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --config_file configs/accelerate_config_notzero.yaml \
    train_unsloth.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --max_seq_length 1536 \
    --bf16 \
    --data_files data/sft_reasoning_filterd.jsonl \
    --use_peft \
    --peft_lora_r 8 \
    --peft_lora_alpha 16 \
    --peft_lora_dropout 0.05 \
    --model_name_or_path DeL-TaiseiOzaki/Tengentoppa-llm-jp-13B-base \
    --output_dir output_final \
