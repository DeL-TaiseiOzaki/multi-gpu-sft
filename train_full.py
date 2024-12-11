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
from trl import SFTTrainer
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
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: str = "flash_attention_2"

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")

    def from_pretrained_kwargs(self, training_args):
        kwargs = {"torch_dtype": "auto"}
        if self.load_in_8bit:
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=["lm_head"],
                    llm_int8_enable_fp32_cpu_offload=True
                )
            }
        elif self.load_in_4bit:
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
        elif training_args.bf16:
            kwargs = {"torch_dtype": torch.bfloat16}

        kwargs["attn_implementation"] = self.use_flash_attention_2
        return kwargs

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

def upload_to_hub(model, tokenizer, model_id, token):
    """モデルとトークナイザーをHuggingFace Hubにアップロード"""
    logger.info(f"Uploading model to HuggingFace Hub as {model_id}")
    
    try:
        # モデルのアップロード
        model.push_to_hub(model_id, token=token, private=True)
        logger.info("Successfully uploaded model to HuggingFace Hub")
        
        # トークナイザーのアップロード
        tokenizer.push_to_hub(model_id, token=token, private=True)
        logger.info("Successfully uploaded tokenizer to HuggingFace Hub")
        
    except Exception as e:
        logger.error(f"Failed to upload to HuggingFace Hub: {str(e)}")
        raise

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