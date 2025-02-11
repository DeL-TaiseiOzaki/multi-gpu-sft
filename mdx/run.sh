#!/bin/bash

accelerate launch --config_file configs/accelerate_config_notzero.yaml \
    train_4bit.py \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --max_seq_length 2048 \
    --bf16 \
    --data_files data/sft_IN3_reasoning.jsonl \
    --use_peft \
    --peft_lora_r 8 \
    --peft_lora_alpha 32 \
    --peft_lora_dropout 0.05 \
    --peft_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj embed_tokens lm_head \
    --model_name_or_path MKJ-TOE/Qwen2.5-7B-Instruct-addsptoken-v1.0 \
    --output_dir output3 \
