import json
from pathlib import Path
from typing import List, Dict
import argparse

def concatenate_jsonl_files(input_files: List[str], output_file: str, validate: bool = True) -> Dict:
    """
    複数のJSONLファイルを指定された順序で結合する

    Args:
        input_files: 入力JSONLファイルパスのリスト（結合順）
        output_file: 出力JSONLファイルパス
        validate: JSONの妥当性チェックを行うかどうか

    Returns:
        処理結果の統計情報
    """
    stats = {
        'total_lines': 0,
        'files_processed': 0,
        'error_lines': 0,
        'file_stats': {}
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as outf:
            for input_file in input_files:
                file_stats = {'lines': 0, 'errors': 0}
                
                try:
                    with open(input_file, 'r', encoding='utf-8') as inf:
                        for line_num, line in enumerate(inf, 1):
                            line = line.strip()
                            if not line:  # 空行をスキップ
                                continue
                                
                            if validate:
                                try:
                                    # JSONの妥当性チェック
                                    json.loads(line)
                                except json.JSONDecodeError:
                                    print(f"警告: {input_file}の{line_num}行目 - 不正なJSON形式")
                                    file_stats['errors'] += 1
                                    stats['error_lines'] += 1
                                    continue
                            
                            outf.write(line + '\n')
                            file_stats['lines'] += 1
                            stats['total_lines'] += 1
                            
                    stats['files_processed'] += 1
                    stats['file_stats'][input_file] = file_stats
                    print(f"処理完了: {input_file} - {file_stats['lines']}行")
                    
                except FileNotFoundError:
                    print(f"エラー: ファイル {input_file} が見つかりません")
                except Exception as e:
                    print(f"エラー: ファイル {input_file} の処理中に問題が発生しました - {e}")

    except Exception as e:
        print(f"エラー: 出力ファイルの作成中に問題が発生しました - {e}")
        raise

    return stats

def main():
    parser = argparse.ArgumentParser(description='JSONLファイルを結合します')
    parser.add_argument('input_files', nargs='+', help='入力JSONLファイル（結合順）')
    parser.add_argument('--output', '-o', required=True, help='出力JSONLファイル')
    parser.add_argument('--no-validate', action='store_false', dest='validate',
                      help='JSON妥当性チェックを無効化（処理が高速化）')

    args = parser.parse_args()
    
    try:
        stats = concatenate_jsonl_files(args.input_files, args.output, args.validate)
        
        # 結果サマリーの表示
        print("\n処理サマリー:")
        print(f"処理完了ファイル数: {stats['files_processed']}")
        print(f"総行数: {stats['total_lines']}")
        if stats['error_lines'] > 0:
            print(f"エラー行数: {stats['error_lines']}")
            
    except Exception as e:
        print(f"エラー: {e}")
        exit(1)

if __name__ == "__main__":
    main()