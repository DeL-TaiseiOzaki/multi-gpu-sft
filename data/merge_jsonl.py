import json
import argparse
from pathlib import Path

def merge_jsonl_files(input_files, output_file):
    """
    複数のJSONLファイルを指定された順序で1つのファイルに結合します。

    Args:
        input_files (list): 入力JSONLファイルのパスのリスト（順序維持）
        output_file (str): 出力ファイルのパス
    """
    # 入力ファイルの合計行数を数える（進捗表示用）
    total_lines = 0
    processed_lines = 0
    
    print("ファイルを結合中...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 指定された順序で処理
        for i, input_file in enumerate(input_files, 1):
            try:
                with open(input_file, 'r', encoding='utf-8') as infile:
                    print(f"処理中: {Path(input_file).name} ({i}/{len(input_files)})")
                    
                    for line in infile:
                        # 空行をスキップ
                        if line.strip():
                            # JSONとして解析可能か確認
                            try:
                                json.loads(line.strip())
                                outfile.write(line)
                                processed_lines += 1
                            except json.JSONDecodeError:
                                print(f"警告: {input_file} で無効なJSON行をスキップしました: {line.strip()[:50]}...")
            
            except FileNotFoundError:
                print(f"エラー: ファイル '{input_file}' が見つかりません")
                continue
    
    print(f"\n結合完了！")
    print(f"処理された行数: {processed_lines}")
    print(f"出力ファイル: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='複数のJSONLファイルを指定された順序で1つのファイルに結合します')
    parser.add_argument('input_files', nargs='+', help='入力JSONLファイル（順序維持）')
    parser.add_argument('-o', '--output', required=True, help='出力ファイル')
    
    args = parser.parse_args()
    merge_jsonl_files(args.input_files, args.output)

if __name__ == '__main__':
    main()

# プログラムからの使用
files_in_order = ['data/sft_reasoning.jsonl', 'data/sft_final.jsonl']
merge_jsonl_files(files_in_order, 'sft.jsonl')