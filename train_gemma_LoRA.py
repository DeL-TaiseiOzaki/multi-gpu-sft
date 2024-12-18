import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
from datasets import disable_caching, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

disable_caching()
logger = logging.getLogger(__name__)

@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    new_model_id: str
    data_files: List[str]
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    max_seq_length: int = 2048
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

def load_datasets(data_files, tokenizer, max_length: int):
    """JSONLファイルからデータセットを読み込み、フォーマットを適用"""
    dataset = load_dataset("json", data_files=data_files)
    
    def format_prompt(example):
        formatted_text = f"{tokenizer.bos_token}{example['text']}{tokenizer.eos_token}"
        return {"formatted_text": formatted_text}
    
    formatted_dataset = dataset["train"].map(
        format_prompt,
        num_proc=4,
        remove_columns=dataset["train"].column_names
    )
    
    def tokenize(examples):
        return tokenizer(examples["formatted_text"])
        
    tokenized_dataset = formatted_dataset.map(
        tokenize,
        num_proc=4,  
        remove_columns=formatted_dataset.column_names
    )
    
    return tokenized_dataset

def main() -> None:
    # シェルスクリプトのパラメータを反映したTrainingArguments
    training_args = TrainingArguments(
        output_dir="output_gemma_news",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        gradient_checkpointing=True,
        bf16=True
    )

    parser = HfArgumentParser(SFTTrainingArguments)
    sft_training_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path,
        use_fast=sft_training_args.use_fast,
        trust_remote_code=True,
    )

    logger.info("Loading and formatting data")
    train_dataset = load_datasets(
        sft_training_args.data_files,
        tokenizer,
        sft_training_args.max_seq_length
    )

    logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        sft_training_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    lora_config = LoraConfig(
        r=sft_training_args.lora_r,
        lora_alpha=sft_training_args.lora_alpha,
        lora_dropout=sft_training_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj"]  # Gemmaモデル用の設定
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        max_seq_length=sft_training_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        dataset_text_field="formatted_text",
        packing=False,
    )

    logger.info("Disabling model cache")
    model.config.use_cache = False

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()
    new_model_id = sft_training_args.new_model_id
    HF_TOKEN="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"
    model.save_pretrained(new_model_id)
    tokenizer.push_to_hub(new_model_id, token=HF_TOKEN)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()