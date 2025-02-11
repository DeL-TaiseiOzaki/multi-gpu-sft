import json
from typing import Dict, Any, List
from pathlib import Path

def transform_data(input_json: Dict[str, Any]) -> Dict[str, str]:
    """
    単一のJSONオブジェクトを新しい形式に変換します。

    Args:
        input_json: 入力JSONオブジェクト（task, thought, missing_detailsを含む）

    Returns:
        変換後のJSONオブジェクト
    """
    try:
        transformed = {
            "text": f"""You are a helpful and competent assistant. You provide correct answers to questions from users.
Analyzes ambiguous tasks from the user and enumerates missing information. If it is not ambiguous, claim that the task is clear.

<|start_user|>{input_json['task']}<|end_user|>

<|start_reasoning|>{input_json['thought']}<|end_reasonig|>

<|start_assistant|>{json.dumps(input_json['missing_details'], ensure_ascii=False)}<|end_assistant|>"""
        }
        return transformed
    except KeyError as e:
        raise KeyError(f"Required key missing in input JSON: {e}")
    except Exception as e:
        raise Exception(f"Error transforming data: {e}")

def process_jsonl_file(input_path: str, output_path: str) -> None:
    """
    JSONLファイルを読み込み、変換して新しいJSONLファイルに保存します。

    Args:
        input_path: 入力JSONLファイルのパス
        output_path: 出力JSONLファイルのパス
    """
    try:
        # 入力ファイルの存在確認
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        transformed_lines = []
        
        # ファイルを読み込んで各行を処理
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():  # 空行をスキップ
                        json_data = json.loads(line.strip())
                        transformed = transform_data(json_data)
                        transformed_lines.append(transformed)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue

        # 変換したデータを新しいファイルに書き込み
        with open(output_path, 'w', encoding='utf-8') as f:
            for data in transformed_lines:
                json_str = json.dumps(data, ensure_ascii=False)
                f.write(json_str + '\n')

        print(f"Successfully processed {len(transformed_lines)} lines")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"Error processing file: {e}")
        raise

def main():
    """
    メイン実行関数
    使用例を含みます
    """
    process_jsonl_file('gen_dataset/regen_sets.jsonl', 'output.jsonl')

if __name__ == "__main__":
    main()