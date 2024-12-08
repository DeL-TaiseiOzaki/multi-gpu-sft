from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Hugging Face Token設定
HF_TOKEN = "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" # あなたのトークンを入れてください

# モデルのパラメータ設定
max_seq_length = 2048
dtype = None
load_in_4bit = False  # 8bit量子化を使用するためFalse
load_in_8bit = True   # 8bit量子化を使用

# モデルの設定
model_id = "DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B"  # 使用したいモデルIDを指定
new_model_id = "Tengentoppa-llm-jp-13B-reasoning-instruct"    # Fine-Tuning後のモデル名

# モデルとトークナイザーのロード
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    trust_remote_code=True,
)

# SFT用のモデルを準備
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj","lm_head",
                    "embed_tokens"],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
    max_seq_length = max_seq_length,
)

# カスタムデータセットの読み込み
dataset = load_dataset("json", data_files="data/sft_filtered.jsonl")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=dataset["train"],
    max_seq_length = max_seq_length,
    dataset_text_field="text",
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 1,
        num_train_epochs = 2,
        logging_steps = 10,
        warmup_steps = 10,
        save_steps = 100,
        save_total_limit = 2,
        max_steps = -1,
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        group_by_length = True,
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        gradient_checkpointing = True,
        optim = "adamw_torch_fused"
    ),
)

# GPU情報の表示
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# 学習の実行
trainer_stats = trainer.train()

# モデルの保存（LoRAアダプタのみ）
model.push_to_hub_merged(
    new_model_id+"_lora",
    tokenizer=tokenizer,
    save_method="lora",
    token=HF_TOKEN,
    private=True
)