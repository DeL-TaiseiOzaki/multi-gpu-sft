import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

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

disable_caching()

logger = logging.getLogger(__name__)

@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    data_files: List[str]
    eval_data_files: Optional[List[str]] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    additional_special_tokens: List[str] = None
    max_seq_length: int = 2048
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: str = "flash_attention_2"
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
            if self.peft_target_model == "llm-jp":
                self.peft_target_modules = ["c_attn", "c_proj", "c_fc"]
            elif self.peft_target_model == "llama":
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            elif self.peft_target_model == "llama-all":
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                    "embed_tokens",
                ]
            else:
                logger.warning(
                    f"peft_target_model '{self.peft_target_model}' is not supported, "
                    f"so peft_target_modules is set to None."
                )

    def from_pretrained_kwargs(self, training_args):
        # 量子化はオフ, bf16を使用
        if self.load_in_8bit:
            kwargs = {"quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=["lm_head"],
                    llm_int8_enable_fp32_cpu_offload=True)
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
        else:
            kwargs = {"torch_dtype": "auto"}
        kwargs["attn_implementation"] = self.use_flash_attention_2
        return kwargs

def load_datasets(data_files):
    dataset1 = load_dataset("json", data_files=data_files)
    dataset2 = dataset1["train"]
    dataset3 = dataset2.select_columns("text")
    return dataset3

class ReasoningDataCollator:
    """
    このコラトレータは、与えられた"text"列のプロンプトから、
    <|REASONING|>以降を出力ラベルとして学習するように処理します。

    前提:
    <|SYSTEM|>system_prompt</|SYSTEM|>\n
    <|USER|>user_prompt</|USER|>\n
    <|HINT|>hint_prompt</|HINT|>\n (optional)
    <|REASONING|>reasoning_prompt</|REASONING|>\n
    <|ASSISTANT|>assistant_prompt</|ASSISTANT|>
    """

    def __init__(self, tokenizer, max_seq_length=2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.reasoning_token = "<|REASONING|>"
        self.reasoning_token_id = self.tokenizer.convert_tokens_to_ids(self.reasoning_token)

    def __call__(self, examples: List[Dict[str, Any]]):
        # ここでexamplesはすでに {"input_ids":..., "attention_mask":...} の形式
        # もしexamplesがまだlist of dictでバッチ化されていない場合、スタック処理が必要
        
        # input_idsを取得
        input_ids = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
        attention_masks = [torch.tensor(e["attention_mask"], dtype=torch.long) for e in examples]

        # バッチ内で最大長にパディング
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

        # max_seq_lengthでトランケート(念のため)
        if input_ids.size(1) > self.max_seq_length:
            input_ids = input_ids[:, :self.max_seq_length]
            attention_masks = attention_masks[:, :self.max_seq_length]

        labels = input_ids.clone()

        # reasoning_token_idの位置を特定し、それより前を-100
        for i in range(input_ids.size(0)):
            reasoning_pos = (input_ids[i] == self.reasoning_token_id).nonzero(as_tuple=True)[0]
            if len(reasoning_pos) == 0:
                # reasoning_tokenがない場合は全て-100
                labels[i] = -100
            else:
                start_pos = reasoning_pos[0]
                labels[i, :start_pos] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

def upload_to_hub(trainer, training_args):
    """Upload the model, tokenizer, and config to the HuggingFace Hub"""
    if not training_args.push_to_hub:
        return
    
    if not training_args.hub_model_id:
        raise ValueError("hub_model_id must be specified when push_to_hub is True")
    
    logger.info(f"Uploading model to HuggingFace Hub as {training_args.hub_model_id}")
    
    try:
        trainer.push_to_hub()
        logger.info("Successfully uploaded model to HuggingFace Hub")
    except Exception as e:
        logger.error(f"Failed to upload model to HuggingFace Hub: {str(e)}")
        raise

def main() -> None:
    parser = HfArgumentParser((TrainingArguments, SFTTrainingArguments))
    training_args, sft_training_args = parser.parse_args_into_dataclasses()

    tokenizer_name_or_path: str = (
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=sft_training_args.use_fast,
        additional_special_tokens=sft_training_args.additional_special_tokens,
        trust_remote_code=True,
    )

    logger.info("Loading data")
    train_dataset = load_datasets(sft_training_args.data_files)
    if sft_training_args.eval_data_files:
        eval_dataset = load_datasets(sft_training_args.eval_data_files)
        training_args.do_eval = True
    else:
        eval_dataset = None

    logger.info("Preparing data collator")
    collator = ReasoningDataCollator(tokenizer=tokenizer, max_seq_length=sft_training_args.max_seq_length)

    logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
    kwargs = sft_training_args.from_pretrained_kwargs(training_args)
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({sft_training_args.model_name_or_path}, trust_remote_code=True, **kwargs={kwargs})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        sft_training_args.model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )

    peft_config: Optional[LoraConfig] = None
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

    logger.info("Setting up trainer")
    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        peft_config=peft_config,
        max_seq_length=sft_training_args.max_seq_length,
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()

    # Add HuggingFace Hub upload
    upload_to_hub(trainer, training_args)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()