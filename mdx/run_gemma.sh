#!/bin/bash

accelerate launch --config_file configs/accelerate_config_notzero.yaml \
    train_gemma_LoRA.py \
    --model_name_or_path DeL-TaiseiOzaki/Tengentoppa-gemma-2-9b-base \
    --new_model_id DeL-TaiseiOzaki/Tengentoppa-gemma-2-9b-LoRA-it-Reasoning \
    --data_files data/sft_filtered_processed.jsonl \