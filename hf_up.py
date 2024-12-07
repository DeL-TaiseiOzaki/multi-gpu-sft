from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login

base_model_name_or_path = "DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B"  # 例: "gpt2"
lora_adapter_path = "output/checkpoint-322"  # 例: "./lora_adapter"

# ベースモデルとトークナイザーのロード
model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

# LoRAアダプターをロード
model = PeftModel.from_pretrained(model, lora_adapter_path)
# Hugging Face Hubにログイン
login()  # トークンが必要です。https://huggingface.co/settings/tokens で取得

# モデルをHugging Face Hubにアップロード
repository_name = "DeL-TaiseiOzaki/Tengenetoppa-llm-jp-3-13b-elyza-news-specialized2"  # 例: "username/lora-finetuned-model"

model.push_to_hub(repository_name)
tokenizer.push_to_hub(repository_name)
