from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, login
import os
from pathlib import Path
import shutil

def setup_model_and_tokenizer(model_name):
    """
    モデルとトークナイザーの初期設定を行う関数
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # pad_tokenの設定
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
    
    # モデルの埋め込み層とLM headのサイズを調整
    model.resize_token_embeddings(len(tokenizer))
    
    # 追加されたトークンの情報を表示
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"追加されたトークン: {token}, ID: {token_id}")
    
    return model, tokenizer

def upload_to_hub(model, tokenizer, upload_repo, hf_token):
    """
    モデルをHugging Faceにアップロードする関数
    """
    if not hf_token:
        raise ValueError("HF_TOKENが必要です")
    
    login(hf_token)
    api = HfApi()
    
    # 一時ディレクトリの作成
    temp_dir = Path("./temp_model")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    
    try:
        # リポジトリの作成
        api.create_repo(repo_id=upload_repo, exist_ok=True)
        
        # モデルとトークナイザーの保存
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        
        # アップロード
        api.upload_folder(
            folder_path=str(temp_dir),
            repo_id=upload_repo,
            commit_message="Add special tokens to model"
        )
        print(f"モデルを正常にアップロードしました: {upload_repo}")
        
    except Exception as e:
        print(f"アップロード中にエラーが発生しました: {e}")
        
    finally:
        # 一時ディレクトリの削除
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main():
    # 設定
    MODEL_NAME = "llm-jp/llm-jp-3-3.7b"  # 使用するモデル
    NEW_SPECIAL_TOKENS = ["<|SYSTEM|>","</|SYSTEM|>", "<|USER|>","</|USER|>","<|HINT|>","</|HINT|>"
                        "<|REASONING|>","</|REASONING|>","<|ASSISTANT|>","</|ASSISTANT|>"]
    HF_TOKEN = "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"  # 環境変数から取得
    REPO_NAME = "DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-3.7B"  # アップロード先のリポジトリ名
    
    # モデルとトークナイザーの初期設定
    print("モデルとトークナイザーをロード中...")
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    
    # 特殊トークンの追加
    print("\n特殊トークンを追加中...")
    model, tokenizer = add_special_tokens(model, tokenizer, NEW_SPECIAL_TOKENS)
    
    # Hugging Faceへのアップロード
    if HF_TOKEN and REPO_NAME:
        print("\nモデルをHugging Faceにアップロード中...")
        upload_to_hub(model, tokenizer, REPO_NAME, HF_TOKEN)
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = main()