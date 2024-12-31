import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import HfApi
from dotenv import load_dotenv

class ModelQuantizer:
    def __init__(self, original_model_name, quantized_model_name, save_directory):
        self.original_model_name = original_model_name
        self.quantized_model_name = quantized_model_name
        self.save_directory = save_directory
        self.tokenizer = None
        self.model = None

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.original_model_name)

    def load_and_quantize_model(self):
        """Load the model and apply 4-bit quantization."""

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.original_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded and quantized to 4-bit.")

    def save_and_upload(self):
        try:
            # .envからAPIキーを読み込む
            load_dotenv()
            api_key = "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"
            api = HfApi()
            username = api.whoami()["name"]
            repo_id = f"{username}/{self.quantized_model_name}"

            print(f"Uploading model to {repo_id}...")

            # モデルをHugging Face Hubにアップロード
            self.model.push_to_hub(
                repo_id,
                save_directory=self.save_directory,
                use_auth_token=True
            )
            
            # トークナイザーもアップロード
            self.tokenizer.push_to_hub(
                repo_id,
                save_directory=self.save_directory,
                use_auth_token=True
            )

            print(f"Successfully uploaded model and tokenizer to: {repo_id}")
            return repo_id

        except Exception as e:
            print(f"Error during save and upload: {e}")
            return None

def main():
    config = {
        "original_model": "llm-jp/llm-jp-3-172b-instruct3",
        "quantized_name": "llm-jp-3-172b-instruct3-4bit",
        "save_dir": "llm-jp-3-172b-instruct3-4bit"
    }
    
    quantizer = ModelQuantizer(
        config["original_model"],
        config["quantized_name"],
        config["save_dir"]
    )

    print("Loading tokenizer...")
    quantizer.load_tokenizer()

    print("Loading and quantizing model (4-bit)...")
    quantizer.load_and_quantize_model()

    print("Saving and uploading to Hugging Face Hub...")
    repo_id = quantizer.save_and_upload()

    if repo_id:
        print(f"Process completed successfully. Model available at: https://huggingface.co/{repo_id}")
    else:
        print("Process completed with errors during upload.")

if __name__ == "__main__":
    main()
