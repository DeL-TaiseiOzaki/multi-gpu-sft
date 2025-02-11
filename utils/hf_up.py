from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from huggingface_hub import login

login(token="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")

repo="MKJ-TOE/llm-jp-3-13b-instruct-addsptoken-v1.0"

# 1. ベースモデルの読み込み
base_model = AutoModelForCausalLM.from_pretrained(repo)
tokenizer = AutoTokenizer.from_pretrained(repo)

# # 2. LoRAモデルの読み込み
model = PeftModel.from_pretrained(base_model, "output_thought_missinfo")

# 3. HuggingFaceにプッシュ
model.push_to_hub("MKJ-TOE/llm-jp-3-13b-instruct-addsptoken-v1.0-thought-missinfo")
tokenizer.push_to_hub("MKJ-TOE/llm-jp-3-13b-instruct-addsptoken-v1.0-thought-missinfo")