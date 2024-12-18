from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

def setup_model_and_tokenizer(model_name):
    """
    モデルとトークナイザーの初期設定を行う関数
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def add_special_tokens(model, tokenizer, new_tokens):
    """
    特殊トークンを追加する関数
    """
    special_tokens_dict = {'additional_special_tokens': new_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    print(f"追加された特殊トークン数: {num_added_tokens}")
    
    original_dtype = model.dtype
    model.resize_token_embeddings(len(tokenizer))
    
    for param in model.parameters():
        param.data = param.data.to(original_dtype)
    
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"追加されたトークン: {token}, ID: {token_id}")
    
    print(f"モデルのdtype: {model.dtype}")
    
    return model, tokenizer

def main():
    # 設定
    MODEL_NAME = "google/gemma-2-9b"
    NEW_SPECIAL_TOKENS = ["<|SYSTEM|>","</|SYSTEM|>", "<|USER|>","</|USER|>","<|HINT|>","</|HINT|>",
                         "<|REASONING|>","</|REASONING|>","<|ASSISTANT|>","</|ASSISTANT|>"]
    HF_TOKEN = "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"
    REPO_NAME = "DeL-TaiseiOzaki/Tengentoppa-gemma-2-9b-base"
    
    # HuggingFaceにログイン
    login(HF_TOKEN)

    # モデルとトークナイザーの初期設定
    print("モデルとトークナイザーをロード中...")
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    
    # 特殊トークンの追加
    print("\n特殊トークンを追加中...")
    model, tokenizer = add_special_tokens(model, tokenizer, NEW_SPECIAL_TOKENS)
    
    # push_to_hubを使用してモデルをアップロード
    print("\nモデルをHugging Faceにアップロード中...")
    model.push_to_hub(REPO_NAME, commit_message="Add special tokens to model (preserving bf16 dtype)")
    tokenizer.push_to_hub(REPO_NAME, commit_message="Add special tokens to tokenizer")
    
    print(f"モデルを正常にアップロードしました: {REPO_NAME}")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = main()

