#!/bin/bash
# train_stage1.sh
accelerate launch --config_file configs/accelerate_config_zero1.yaml \
    train.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --max_seq_length 2048 \
    --data_files data/sft_dataset.jsonl \
    --use_peft \
    --peft_target_model llama-all \
    --peft_lora_r 8 \
    --peft_lora_alpha 32 \
    --peft_lora_dropout 0.05 \
    --use_flash_attention_2 True \
    --model_name_or_path DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B \
    --output_dir output/stage1_output

#!/bin/bash
# train_stage2.sh
accelerate launch --config_file configs/accelerate_config_zero1.yaml \
    train.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --max_seq_length 2048 \
    --data_files data/sft_reasoning1.jsonl \
    --use_peft \
    --peft_target_model llama-all \
    --peft_lora_r 8 \
    --peft_lora_alpha 32 \
    --peft_lora_dropout 0.05 \
    --use_flash_attention_2 True \
    --model_name_or_path output/stage1_output \
    --output_dir output/stage2_output

#!/bin/bash
# train_stage3.sh
accelerate launch --config_file configs/accelerate_config_zero1.yaml \
    train.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --max_seq_length 2048 \
    --data_files data/sft_reasoning2.jsonl \
    --use_peft \
    --peft_target_model llama-all \
    --peft_lora_r 8 \
    --peft_lora_alpha 32 \
    --peft_lora_dropout 0.05 \
    --use_flash_attention_2 True \
    --model_name_or_path output/stage2_output \
    --output_dir output/stage3_output

#!/bin/bash
# train_stage3.sh
accelerate launch --config_file configs/accelerate_config_zero1.yaml \
    train.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --max_seq_length 2048 \
    --data_files data/sft_reasoning3.jsonl \
    --use_peft \
    --peft_target_model llama-all \
    --peft_lora_r 8 \
    --peft_lora_alpha 32 \
    --peft_lora_dropout 0.05 \
    --use_flash_attention_2 True \
    --model_name_or_path output/stage2_output \
    --output_dir output/stage3_output

#!/bin/bash
# train_stage3.sh
accelerate launch --config_file configs/accelerate_config_zero1.yaml \
    train.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --max_seq_length 2048 \
    --data_files data/sft_final.jsonl \
    --use_peft \
    --peft_target_model llama-all \
    --peft_lora_r 8 \
    --peft_lora_alpha 32 \
    --peft_lora_dropout 0.05 \
    --use_flash_attention_2 True \
    --model_name_or_path output/stage2_output \
    --output_dir output/stage3_output