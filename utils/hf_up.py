from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel, PeftConfig
from huggingface_hub import login

login(token="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")

# 1. ベースモデルの読み込み
model = AutoModelForCausalLM.from_pretrained("output_elyza_ep3_lowlr")
tokenizer = AutoTokenizer.from_pretrained("output_elyza_ep3_lowlr")

# # 2. LoRAモデルの読み込み
# model = PeftModel.from_pretrained(base_model, "output_last")

# 3. HuggingFaceにプッシュ
model.push_to_hub("MKJ-TOE/elyza-llama_missinfo-detection_lr1e-6_ep3")
tokenizer.push_to_hub("MKJ-TOE/elyza-llama_missinfo-detection_lr1e-6_ep3")