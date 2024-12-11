from huggingface_hub import login, HfApi

login(token="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")

# APIクライアントの初期化
api = HfApi()

# モデルのアップロード
api.upload_folder(
    folder_path="model_to_upload",  # ローカルのモデルディレクトリ
    repo_id="DeL-TaiseiOzaki/Tengentoppa-llm-jp-3.7B-reasoning-instruct",  # 'ユーザー名/リポジトリ名'の形式
    repo_type="model"
)