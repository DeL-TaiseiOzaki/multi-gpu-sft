#!/bin/bash
config_file=multi-gpu-sft/configs/accelerate_config_ddp.yaml
model_name_or_path=DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B
tokenizer_name_or_path=$3
dataset_path=$4
dataset_sh=$5
num_train_epochs=1
output_dir=multi-gpu-sft/output
per_device_train_batch_size=2
gradient_accumulation_steps=1
accelerate launch --config_file $config_file \
    train.py \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --max_seq_length 2048 \
    --gradient_checkpointing \
    --logging_steps 1 \
    --report_to wandb \
    --data_files `$dataset_sh $dataset_path` \
    --output_dir $output_dir \
    ${@:10}
