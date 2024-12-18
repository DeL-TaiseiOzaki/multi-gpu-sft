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

def load_datasets(data_files, tokenizer, max_length: int):
    """JSONLファイルからデータセットを読み込み、フォーマットを適用"""
    dataset = load_dataset("json", data_files=data_files)
    
    def format_prompt(example):
        # 必要に応じてプロンプトフォーマットを調整
        formatted_text = f"{tokenizer.bos_token}{example['text']}{tokenizer.eos_token}"
        return {"formatted_text": formatted_text}
    
    # データセットの前処理
    formatted_dataset = dataset["train"].map(
        format_prompt,
        num_proc=4,
        remove_columns=dataset["train"].column_names
    )
    
    # トークナイズとフィルタリング
    def tokenize(examples):
        return tokenizer(examples["formatted_text"])
        
    tokenized_dataset = formatted_dataset.map(
        tokenize,
        num_proc=4,  
        remove_columns=formatted_dataset.column_names
    )
    
    return tokenized_dataset

def main() -> None:
    parser = HfArgumentParser((TrainingArguments, SFTTrainingArguments))
    training_args, sft_training_args = parser.parse_args_into_dataclasses()

    # トークナイザーの設定
    tokenizer = AutoTokenizer.from_pretrained(
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path,
        use_fast=sft_training_args.use_fast,
        trust_remote_code=True,
    )

    # データセットの準備
    logger.info("Loading and formatting data")
    train_dataset = load_datasets(
        sft_training_args.data_files,
        tokenizer,
        sft_training_args.max_seq_length
    )

    # モデルの設定
    logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        sft_training_args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # 勾配チェックポイントの有効化
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    # データコレーターの設定
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # トレーナーの設定
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        max_seq_length=sft_training_args.max_seq_length,
        dataset_text_field="formatted_text",
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        packing=False,
    )

    logger.info("Disabling model cache")
    model.config.use_cache = False

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()

    new_model_id=sft_training_args.new_model_id
    HF_TOKEN="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"
    model.push_to_hub(new_model_id, token=HF_TOKEN)
    tokenizer.push_to_hub(new_model_id, token=HF_TOKEN)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()