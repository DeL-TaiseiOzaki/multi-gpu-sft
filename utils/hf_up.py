from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from huggingface_hub import login

login(token="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")

# 1. ベースモデルの読み込み
base_model = AutoModelForCausalLM.from_pretrained("DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B")
tokenizer = AutoTokenizer.from_pretrained("DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B")

# 2. LoRAモデルの読み込み
model = PeftModel.from_pretrained(base_model, "output_last")

# 3. HuggingFaceにプッシュ
model.push_to_hub("DeL-TaiseiOzaki/Tengentoppa-llm-jp-13B-reasoning-it")
tokenizer.push_to_hub("DeL-TaiseiOzaki/Tengentoppa-llm-jp-13B-reasoning-it")