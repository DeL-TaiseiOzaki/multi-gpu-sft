import json
import argparse

def process_jsonl(input_file, output_file):
    """
    JSONLファイルを読み込み、システムメッセージを置き換え、
    HINTが存在する場合はUSERメッセージを修正して保存する
    
    Args:
        input_file (str): 入力JSONLファイルのパス
        output_file (str): 出力JSONLファイルのパス
    """
    # 置き換える前のシステムメッセージ
    old_system_msg = "あなたは親切で有能なアシスタントです。\nユーザーからの質問に対して、正しい回答を提供します。\n与えられた情報を正確に整理し，論理的に説明し，簡潔に回答します．"
    
    # 置き換える後のシステムメッセージ
    new_system_msg = "あなたは親切で有能なアシスタントです。\nユーザーからの質問に対して、正しい回答を提供します。\n与えられた情報を正確に整理し，論理的に説明し，簡潔に回答します．<|REASONING|>，</|REASONING|>の間で思考の過程を抜けがないように記載します．"
    
    processed_lines = []
    
    # ファイルを1行ずつ読み込む
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():  # 空行をスキップ
                continue
                
            # JSONデータをパース
            data = json.loads(line)
            content = data['text']
            
            # システムメッセージを置き換え
            start_token = "<|SYSTEM|>"
            end_token = "</|SYSTEM|>"
            
            start_idx = content.find(start_token)
            end_idx = content.find(end_token)
            
            if start_idx != -1 and end_idx != -1:
                # 現在のシステムメッセージを確認
                current_system_msg = content[start_idx + len(start_token):end_idx]
                
                if current_system_msg.strip() == old_system_msg.strip():
                    # システムメッセージを置き換え
                    content = (
                        content[:start_idx] + 
                        start_token + 
                        new_system_msg + 
                        end_token + 
                        content[end_idx + len(end_token):]
                    )
            
            # HINTトークンの処理
            hint_start = "<|HINT|>"
            hint_end = "</|HINT|>"
            hint_start_idx = content.find(hint_start)
            hint_end_idx = content.find(hint_end)
            
            if hint_start_idx != -1 and hint_end_idx != -1:
                hint_content = content[hint_start_idx:hint_end_idx + len(hint_end)]
                if hint_content.strip() != hint_start + hint_end:  # HINTの中身が空でない場合
                    # USERメッセージの最後に文を追加
                    user_start = "<|USER|>"
                    user_end = "</|USER|>"
                    user_end_idx = content.find(user_end)
                    
                    if user_end_idx != -1:
                        hint_message = "\n\n以下の<|HINT|></|HINT|>の間のヒントを活用してください．"
                        content = (
                            content[:user_end_idx] + 
                            hint_message +
                            content[user_end_idx:]
                        )
            
            processed_data = {"text": content}
            processed_lines.append(json.dumps(processed_data, ensure_ascii=False))
    
    # 処理結果を新しいファイルに書き込む
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')

def main():
    parser = argparse.ArgumentParser(description='JSONLファイルのシステムメッセージとヒントを処理する')
    parser.add_argument('input_file', help='入力JSONLファイルのパス')
    parser.add_argument('output_file', help='出力JSONLファイルのパス')
    
    args = parser.parse_args()
    
    process_jsonl(args.input_file, args.output_file)

if __name__ == '__main__':
    main()