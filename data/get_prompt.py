from datasets import load_dataset
import json
from typing import Dict, Any

DEFAULT_SYSTEM_PROMPT = """あなたは親切で有能なアシスタントです。
ユーザーからの質問に対して、正しい回答を提供します。
与えられた情報を正確に整理し，論理的に説明し，簡潔に回答します．"""

def format_prompt(
    instruction: str,
    hint: str = "",
    reasoning: str = "",
    response: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    """
    指定されたフォーマットでプロンプトを生成する

    Args:
        instruction: タスクの指示
        output: 期待される出力
        input_text: 入力テキスト（オプション）
        system_prompt: システムプロンプト

    Returns:
        フォーマット済みのプロンプト
    """
        
    return f"<|SYSTEM|>{system_prompt}</|SYSTEM|>\n<|USER|>{instruction}</|USER|>\n<|HINT|>{hint}</|HINT|>\n<|REASONING|>{reasoning}</|REASONING|>\n<|ASSISTANT|>{response}</|ASSISTANT|>"

def convert_dataset(
    dataset_name: str,
    output_path: str = "sft_dataset.jsonl",
    split: str = "train",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> None:
    """
    Hugging Face データセットを指定のフォーマットに変換してJSONLファイルとして保存

    Args:
        dataset_name: Hugging Face のデータセット名
        output_path: 出力ファイルのパス
        split: データセットのスプリット（train/validation/test）
        system_prompt: カスタムシステムプロンプト（オプション）
    """
    # データセットのロード
    dataset = load_dataset(dataset_name, split=split)
    
    # JSONL ファイルとして保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            # プロンプトフォーマットの作成
            formatted_text = format_prompt(
                instruction=item['instruction'],
                hint="",
                response=item["output"],
                reasoning=item["reasoning"],
                system_prompt=system_prompt
            )
            
            # JSON 形式で書き込み
            json_line = json.dumps({'text': formatted_text}, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Converted dataset saved to {output_path}")

# 使用例
if __name__ == "__main__":
    # データセット名とファイルパスを指定して実行
    dataset_name = "DeL-TaiseiOzaki/Tengentoppa-sft-reasoning-v2.0"  # 実際のデータセット名に置き換えてください
    output_path = "sft_reasoning.jsonl"
    
    convert_dataset(
        dataset_name=dataset_name,
        output_path=output_path
    )