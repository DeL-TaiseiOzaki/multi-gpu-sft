import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
from peft import LoraConfig
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
    use_flash_attention_2: str = "eager"
    use_peft: bool = True
    peft_target_model: Optional[str] = "llama"
    peft_target_modules: Optional[List[str]] = None
    peft_lora_r: int = 16
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            if self.peft_target_model == "llama-all":
                self.peft_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", "embed_tokens",
                ]

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

    peft_config = None
    if sft_training_args.use_peft:
        logger.info("Setting up LoRA")
        peft_config = LoraConfig(
            r=sft_training_args.peft_lora_r,
            target_modules=sft_training_args.peft_target_modules,
            lora_alpha=sft_training_args.peft_lora_alpha,
            lora_dropout=sft_training_args.peft_lora_dropout,
            fan_in_fan_out=False,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if training_args.gradient_checkpointing:
            for param in model.parameters():
                param.requires_grad = False
                if param.ndim == 1:
                    param.data = param.data.to(torch.float32)
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
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
        level=logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()