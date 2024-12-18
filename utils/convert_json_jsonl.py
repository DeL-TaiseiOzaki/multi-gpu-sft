import json
from pathlib import Path
from typing import List, Union, Dict

def convert_json_to_jsonl(input_path: str, output_path: str) -> None:
    """
    JSONファイルをJSONL形式に変換する
    
    Args:
        input_path: 入力JSONファイルのパス
        output_path: 出力JSONLファイルのパス
    """
    try:
        # JSONファイルを読み込む
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # データが配列でない場合は配列に変換
        if not isinstance(data, list):
            data = [data]
        
        # JSONLとして書き出し
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')
                
        print(f"変換完了: {len(data)}行のデータを{output_path}に書き出しました")
        
    except json.JSONDecodeError as e:
        print(f"エラー: JSONの解析に失敗しました - {e}")
    except FileNotFoundError:
        print(f"エラー: 入力ファイル {input_path} が見つかりません")
    except Exception as e:
        print(f"エラー: 予期しない問題が発生しました - {e}")

def convert_multiple_files(input_dir: str, output_dir: str) -> None:
    """
    指定ディレクトリ内の全JSONファイルをJSONL形式に変換する
    
    Args:
        input_dir: 入力JSONファイルのディレクトリ
        output_dir: 出力JSONLファイルのディレクトリ
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 出力ディレクトリが存在しない場合は作成
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSONファイルを検索して変換
    json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        print(f"警告: {input_dir} にJSONファイルが見つかりません")
        return
    
    for json_file in json_files:
        output_file = output_path / f"{json_file.stem}.jsonl"
        convert_json_to_jsonl(str(json_file), str(output_file))

def main():
    # 単一ファイルの変換
    input_file = "data/sft_base.json"
    output_file = "data/sft_base.jsonl"
    convert_json_to_jsonl(input_file, output_file)
    
    # または、ディレクトリ内の全ファイルを変換
    # input_dir = "json_files"
    # output_dir = "jsonl_files"
    # convert_multiple_files(input_dir, output_dir)

if __name__ == "__main__":
    main()