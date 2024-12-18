import json
from typing import List, Dict

def remove_lines_with_short_values(input_file: str, output_file: str, column_name: str, max_length: int = 10) -> None:
    """
    JSONLファイルを読み込み、指定された列の値が一定の長さ以下の行を削除する
    
    Args:
        input_file (str): 入力JSONLファイルのパス
        output_file (str): 出力JSONLファイルのパス
        column_name (str): チェックする列の名前
        max_length (int): この長さ以下の値を持つ行を削除する（デフォルト: 10）
    """
    # 条件を満たす行のみを保持するリスト
    valid_lines: List[Dict] = []
    
    # ファイルを読み込み、条件をチェック
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 指定された列の値をチェック
            if column_name in data and isinstance(data[column_name], str):
                # 値の長さが指定された長さより大きい場合のみ保持
                if len(data[column_name]) > max_length:
                    valid_lines.append(data)
    
    # 結果を新しいJSONLファイルに書き込み
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in valid_lines:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

# 使用例
if __name__ == "__main__":
    input_file = "generated_sets.jsonl"
    output_file = "generated_sets.jsonl"
    target_column = "response"
    remove_lines_with_short_values(input_file, output_file, target_column)