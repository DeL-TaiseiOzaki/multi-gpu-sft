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
    BitsAndBytesConfig,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
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
    max_seq_length: int = 1024
    use_flash_attention_2: str = "flash_attention_2"

    def from_pretrained_kwargs(self, training_args):
        kwargs = {"torch_dtype": "auto"}
        kwargs = {"torch_dtype": "auto"}
        kwargs["attn_implementation"] = self.use_flash_attention_2
        return kwargs

def load_datasets(data_files, tokenizer, max_seq_length):
    """JSONLファイルからデータセットを読み込み、フォーマットとトークナイズを適用"""
    dataset = load_dataset("json", data_files=data_files)

    def format_prompt(example):
        return {"formatted_text": f"{tokenizer.bos_token}{example['text']}{tokenizer.eos_token}"}

    dataset = dataset["train"].map(
        format_prompt,
        num_proc=4,
        remove_columns=dataset["train"].column_names
    )

    def tokenize(example):
        # トークナイズ時に truncation と max_length を指定する
        return tokenizer(
            example["formatted_text"], 
            truncation=True, 
            max_length=max_seq_length,
        )

    dataset = dataset.map(
        tokenize,
        num_proc=4,
        batched=False,  
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return dataset

def main() -> None:
    parser = HfArgumentParser((TrainingArguments, SFTTrainingArguments))
    training_args, sft_training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path,
        use_fast=sft_training_args.use_fast,
        trust_remote_code=True,
    )

    logger.info("Loading and formatting data")
    train_dataset = load_datasets(
        sft_training_args.data_files,
        tokenizer,
        max_seq_length=sft_training_args.max_seq_length)
    if sft_training_args.eval_data_files:
        eval_dataset = load_datasets(sft_training_args.eval_data_files, tokenizer)
        training_args.do_eval = True
    else:
        eval_dataset = None

    logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
    kwargs = sft_training_args.from_pretrained_kwargs(training_args)
    model = AutoModelForCausalLM.from_pretrained(
        sft_training_args.model_name_or_path,
        **kwargs,
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
    )

    logger.info("Disabling model cache")
    model.config.use_cache = False

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()

    if training_args.push_to_hub and training_args.hub_model_id:
        logger.info("Uploading to HuggingFace Hub")
        upload_to_hub(
            model=model,
            tokenizer=tokenizer,
            model_id=training_args.hub_model_id,
            token="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"
        )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()