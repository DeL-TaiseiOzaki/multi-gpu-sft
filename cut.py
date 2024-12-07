import json
from transformers import AutoTokenizer
from pathlib import Path

def filter_long_sequences(input_path, output_path, model_name, max_tokens=2048):
    """
    JSONLファイルから指定したトークン数を超えるデータを除外して新しいファイルを作成する
    
    Parameters:
    input_path (str): 入力JSONLファイルのパス
    output_path (str): 出力JSONLファイルのパス
    model_name (str): Hugging FaceのモデルまたはトークナイザーのID
    max_tokens (int): 最大トークン数（デフォルト: 2048）
    
    Returns:
    dict: 処理の統計情報
    """
    # トークナイザーを初期化
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"トークナイザーの読み込みに失敗しました: {e}")
        return None
    
    # 統計情報を記録する変数
    stats = {
        'total_processed': 0,
        'kept': 0,
        'removed': 0,
        'errors': 0
    }
    
    # 出力ディレクトリが存在しない場合は作成
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # ファイルを処理
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_number, line in enumerate(infile, 1):
            stats['total_processed'] += 1
            
            try:
                # JSON行をパース
                data = json.loads(line)
                
                # textカラムがあることを確認
                if 'text' not in data:
                    print(f"警告: 行 {line_number} にtextカラムがありません")
                    stats['errors'] += 1
                    continue
                
                # テキストのトークン数をカウント
                text = data['text']
                tokens = tokenizer.encode(text)
                token_count = len(tokens)
                
                # トークン数が閾値以下の場合のみ保存
                if token_count <= max_tokens:
                    outfile.write(line)
                    stats['kept'] += 1
                else:
                    stats['removed'] += 1
                    
            except json.JSONDecodeError:
                print(f"エラー: 行 {line_number} のJSONパースに失敗しました")
                stats['errors'] += 1
                continue
            except Exception as e:
                print(f"エラー: 行 {line_number} の処理中にエラーが発生しました: {e}")
                stats['errors'] += 1
                continue
            
            # 進捗表示（1000行ごと）
            if line_number % 1000 == 0:
                print(f"処理済み: {line_number}行")
    
    return stats

# 使用例
if __name__ == "__main__":
    input_file = "data/sft.jsonl"
    output_file = "data/sft_filstered.jsonl"
    model_name = "DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B"
    
    stats = filter_long_sequences(input_file, output_file, model_name)
    
    if stats:
        print("\n処理完了:")
        print(f"処理した総行数: {stats['total_processed']}")
        print(f"保持したデータ数: {stats['kept']}")
        print(f"除外したデータ数: {stats['removed']}")
        print(f"エラー数: {stats['errors']}")
        print(f"除外率: {(stats['removed'] / stats['total_processed'] * 100):.2f}%")