#!/bin/bash
accelerate launch --config_file configs/accelerate_config_zero3.yaml \
    train.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --max_seq_length 2048 \
    --data_files data/sft_base.jsonl \
    --use_flash_attention_2 \
    --use_peft \
    --peft_target_model llama-all \
    --peft_lora_r 16 \
    --peft_lora_alpha 32 \
    --peft_lora_dropout 0.05 \
    --use_flash_attention_2 True \
    --model_name_or_path DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B \
    --output_dir output