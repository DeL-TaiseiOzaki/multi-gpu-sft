import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
from datasets import disable_caching, load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel
import os

disable_caching()
logger = logging.getLogger(__name__)

@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    data_files: List[str]
    eval_data_files: Optional[List[str]] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    max_seq_length: int = 2048
    use_peft: bool = True
    peft_target_model: Optional[str] = "llama-all"
    peft_target_modules: Optional[List[str]] = None
    peft_lora_r: int = 16
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05

    def __post_init__(self):
        if self.peft_target_model and self.peft_target_modules is None:
            if self.peft_target_model == "llama-all":
                self.peft_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

def load_datasets(data_files, tokenizer):
    """JSONLファイルからデータセットを読み込み、フォーマットを適用"""
    dataset = load_dataset("json", data_files=data_files)
    
    def format_prompt(example):
        # BOSとEOSトークンを追加
        formatted_text = f"{tokenizer.bos_token}{example['text']}{tokenizer.eos_token}"
        return {"formatted_text": formatted_text}
    
    # データセットにフォーマットを適用
    formatted_dataset = dataset["train"].map(
        format_prompt,
        num_proc=4,
        remove_columns=dataset["train"].column_names
    )
    return formatted_dataset

def main() -> None:
    parser = HfArgumentParser((TrainingArguments, SFTTrainingArguments))
    training_args, sft_training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path,
        use_fast=sft_training_args.use_fast,
        trust_remote_code=True,
    )

    logger.info("Loading and formatting data")
    train_dataset = load_datasets(sft_training_args.data_files, tokenizer)
    if sft_training_args.eval_data_files:
        eval_dataset = load_datasets(sft_training_args.eval_data_files, tokenizer)
        training_args.do_eval = True
    else:
        eval_dataset = None

    logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
    # Unslothを使用してモデルをロード
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_training_args.model_name_or_path,
        max_seq_length=sft_training_args.max_seq_length,
        dtype=torch.bfloat16
    )

    if sft_training_args.use_peft:
        logger.info("Setting up LoRA using Unsloth")
        model = FastLanguageModel.get_peft_model(
            model,
            r=sft_training_args.peft_lora_r,
            target_modules=sft_training_args.peft_target_modules,
            lora_alpha=sft_training_args.peft_lora_alpha,
            lora_dropout=sft_training_args.peft_lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,  # 必要に応じて設定
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=sft_training_args.max_seq_length,
        dataset_text_field="formatted_text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    logger.info("Disabling model cache")
    model.config.use_cache = False

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
