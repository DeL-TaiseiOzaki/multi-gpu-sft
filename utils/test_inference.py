from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

def load_model_with_lora(base_model_name, lora_path):
    # Load the base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load and apply the LoRA adapter
    model = PeftModel.from_pretrained(model, lora_path)
    
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=2048):
    # token_type_idsを除外
    inputs = tokenizer(prompt, return_tensors="pt")
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.5
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

# 使用例
base_model_name = "MKJ-TOE/llm-jp-3-13b-instruct-addsptoken-v1.0"  # ベースモデルのパスまたは名前
lora_path = "MKJ-TOE/llm-jp-3-13b-instruct-addsptoken-v1.0-reasoning"  # HuggingFaceのLoRAアダプターのパス

# モデルとトークナイザーの読み込み
model, tokenizer = load_model_with_lora(base_model_name, lora_path)

# テキスト生成
prompt = "You are a helpful and competent assistant. You provide correct answers to questions from users.\n<|start_user|>Find the latest research on diabetes treatment.<|start_user|>\n\n<|start_reasoning|>"
generated_text = generate_text(prompt, model, tokenizer)
print(generated_text)