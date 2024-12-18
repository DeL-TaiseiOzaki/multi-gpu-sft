import json
from transformers import AutoTokenizer

def count_tokens_in_jsonl(file_path, model_name):
    """
    JSONLファイルの各行のtextカラムのトークン数をHugging Faceのトークナイザーでカウントする関数
    
    Parameters:
    file_path (str): JSONLファイルのパス
    model_name (str): Hugging FaceのモデルまたはトークナイザーのID（例：'rinna/japanese-gpt-neox-3.6b'）
    
    Returns:
    list: 各行のトークン数とテキストの情報を含む辞書のリスト
    """
    # トークナイザーを初期化
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"トークナイザーの読み込みに失敗しました: {e}")
        return None, None
    
    results = []
    total_tokens = 0
    
    # JSONLファイルを1行ずつ読み込む
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            try:
                # JSON行をパース
                data = json.loads(line)
                
                # textカラムがあることを確認
                if 'text' not in data:
                    print(f"警告: 行 {line_number} にtextカラムがありません")
                    continue
                
                # テキストのトークン数をカウント
                text = data['text']
                tokens = tokenizer.encode(text)
                token_count = len(tokens)
                
                # 結果を保存
                result = {
                    'line_number': line_number,
                    'token_count': token_count,
                    'text_preview': text[:100] + '...' if len(text) > 100 else text
                }
                results.append(result)
                total_tokens += token_count
                
            except json.JSONDecodeError:
                print(f"エラー: 行 {line_number} のJSONパースに失敗しました")
                continue
            except Exception as e:
                print(f"エラー: 行 {line_number} の処理中にエラーが発生しました: {e}")
                continue
    
    # 合計トークン数を追加
    summary = {
        'total_tokens': total_tokens,
        'total_lines': len(results),
        'average_tokens': total_tokens / len(results) if results else 0
    }
    
    return results, summary

# 使用例
if __name__ == "__main__":
    file_path = "data/sft.jsonl"  # JSONLファイルのパス
    model_name = "DeL-TaiseiOzaki/Tengentoppa-llm-jp-base-13B"  # 使用するモデルのID
    results, summary = count_tokens_in_jsonl(file_path, model_name)
    
    if results and summary:
        print(f"\n集計結果:")
        print(f"総トークン数: {summary['total_tokens']}")
        print(f"総行数: {summary['total_lines']}")
        print(f"平均トークン数: {summary['average_tokens']:.2f}")
        
        print("\n各行の詳細:")
        for result in results[:5]:  # 最初の5行だけ表示
            print(f"行 {result['line_number']}: {result['token_count']} トークン")
            print(f"テキストプレビュー: {result['text_preview']}\n")