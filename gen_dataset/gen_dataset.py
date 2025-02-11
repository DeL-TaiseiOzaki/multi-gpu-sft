import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import json
import gc
import random
from vllm import LLM, SamplingParams
from huggingface_hub import login
import re

login(token="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")

# モデルとトークナイザーの初期化
model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(
    model=model_name,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.95,  # GPUメモリ使用率を調整
    max_model_len=2048,           # モデルの最大長を調整
    dtype=torch.bfloat16
)

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=1024
)

BATCH_SIZE = 500   # バッチサイズを小さくする（大きなモデルのため）
SAVE_INTERVAL = 1000  # 保存間隔を調整

SYSTEM_PROMPTS = [
    "You are a highly curious and knowledge-seeking assistant. Strive to answer any question in detail and provide new perspectives.",
    "You are a logical and analytical assistant. You excel at breaking down problems and presenting step-by-step solutions.",
    "You are a highly creative assistant who generates innovative ideas. Approach problem-solving with unconventional thinking.",
    "You are a warm and empathetic assistant. Understand and relate to people's feelings, offering compassionate solutions and responses.",
    "You are an assistant who values efficiency and productivity. Aim to provide concise and practical solutions by eliminating unnecessary elements.",
    "You are an assistant well-versed in history and culture. Utilize your knowledge of various eras and regions to offer diverse perspectives.",
    "You are a calm and impartial assistant. Provide responses based on objective facts without being influenced by emotions.",
    "You are an optimistic assistant with a sense of humor. Address difficult problems positively and incorporate humor when appropriate.",
    "You are a detail-oriented and perfectionist assistant. Strive for accuracy and meticulous information, ensuring that even the smallest details are not overlooked.",
    "You are a highly adaptable and flexible assistant. Tailor your approach based on the situation and offer a variety of solutions."
]

def load_progress(filename="gen_dataset/progress.json"):
    """進捗をロード"""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {"processed_count": 0}

def save_progress(progress, filename="gen_dataset/progress.json"):
    """進捗を保存（トランザクション的）"""
    temp_filename = filename + ".tmp"
    with open(temp_filename, "w") as f:
        json.dump(progress, f)
    os.rename(temp_filename, filename)

def append_outputs(data, filename="gen_dataset/generated_sets.jsonl"):
    """生成されたデータを保存（アペンドモード）"""
    with open(filename, "a", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_instruction_samples(filename="gen_dataset/train_data.json"):
    with open(filename, "r", encoding="utf-8") as f:
        IN3 = json.load(f)
    return IN3

# サンプル指示文の読み込み
INSTRUCTION_SAMPLES = load_instruction_samples()
SAMPLE_TASK = INSTRUCTION_SAMPLES["task"]
SAMPLE_THOUGHT = INSTRUCTION_SAMPLES["thought"]
SAMPLE_MISSING_DETAILS = INSTRUCTION_SAMPLES["missing_details"]

def generate_texts(prompts_with_system):
    """バッチ処理でテキストを生成（システムプロンプトごとに個別に）"""
    messages_batch = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        for system_prompt, prompt in prompts_with_system
    ]
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    outputs = llm.generate(texts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def sanitize_json_output(text: str) -> str:
    """
    モデルの出力からコードブロックや先頭/末尾の余計な文言を取り除き、
    純粋なJSON部分を抽出して返す簡易関数。
    """

    # 1) コードブロックのトリプルバッククォートを除去
    #    ```json ... ``` のような表記を消す
    text = re.sub(r'```(\w+)?', '', text)

    # 2) JSON以外の先頭メッセージが混ざっている場合があるので、
    #    最初に '{' か '[' が登場する位置を探す
    start_bracket_idx = None
    for i, ch in enumerate(text):
        if ch in ['{', '[']:
            start_bracket_idx = i
            break

    if start_bracket_idx is None:
        # JSONの開始カッコが見つからない場合、とりあえず全体をstripして返す
        return text.strip()

    text = text[start_bracket_idx:]  # JSONカッコ前を切り捨て

    # 3) 後方に余計なメッセージがある場合もあるので、
    #    ざっくり最後に '}' か ']' が出現する位置を探し、それ以降を切る
    #    （あまり厳密にやりすぎると遅くなるので簡易的に）
    end_bracket_idx = None
    reversed_text = text[::-1]  # 文字列を逆に
    for j, ch in enumerate(reversed_text):
        if ch in ['}', ']']:
            end_bracket_idx = len(text) - j - 1
            break

    if end_bracket_idx is not None:
        text = text[:end_bracket_idx+1]

    # 最終的に余分な空白を除去
    return text.strip()


def generate_pipeline(batch_size):
    """IN3の拡張"""

    # システムプロンプトの選択
    system_prompts = [random.choice(SYSTEM_PROMPTS) for _ in range(batch_size)]

    # === TASK Generation ===
    selected_indices = random.sample(range(len(SAMPLE_TASK)), min(2, len(SAMPLE_TASK)))  # 例として2つ選択
    reference_tasks = [SAMPLE_TASK[i] for i in selected_indices]
    task_prompts = [
        f"The following are examples of tasks that a user has ordered from their personal assistant in the past:\n\n" +
        "\n\n".join(reference_tasks) +
        "\n\nUsing the examples above, generate only one additional task that a user might request. Do not write anything other than the new task itself.\n\nNew Task:"
        for _ in range(batch_size)
    ]
    tasks = generate_texts(list(zip(system_prompts, task_prompts)))

    # === THOUGHT Generation ===
    reference_thoughts = [SAMPLE_THOUGHT[i] for i in selected_indices]  # 対応する THOUGHT
    thought_prompts = [
        f"The following are examples of tasks along with their respective thought processes:\n\n" +
        "\n\n".join([f"Task: {t}\nThought Process: {th}" for t, th in zip(reference_tasks, reference_thoughts)]) +
        f"\n\nNow, given the following new task, provide a logical thought process to accomplish it, referring to the examples.\n\nTask: {task}\n\nThought Process:"
        for task in tasks
    ]
    thoughts = generate_texts(list(zip(system_prompts, thought_prompts)))

    # === MISSING_DETAILS Generation (with JSON format) ===
    reference_missing_details = [SAMPLE_MISSING_DETAILS[i] for i in selected_indices]
    missing_detail_prompts = [
        f"The following are examples of tasks along with their thought processes and missing details:\n\n" +
        "\n\n".join([
            f"Task: {t}\nThought Process: {th}\nMissing Details: {json.dumps(md, indent=4)}"
            for t, th, md in zip(reference_tasks, reference_thoughts, reference_missing_details)
        ]) +
        f"\n\nNow, given the following new task and thought process, identify what essential details are missing for successfully completing this task.\n"
        f"Ensure the output is a structured JSON list with each entry containing 'description', 'importance', 'inquiry', and 'options'.\n\n"
        f"Task: {task}\n\nThought Process: {thought}\n\nMissing Details (JSON format):"
        for task, thought in zip(tasks, thoughts)
    ]
    missing_details_raw = generate_texts(list(zip(system_prompts, missing_detail_prompts)))

    # === JSONパース確認（フィルタリングを追加）===
    generated_sets = []
    for system_prompt, task, thought, missing_detail in zip(system_prompts, tasks, thoughts, missing_details_raw):
        # 生成後テキストの整形・フィルタリング
        sanitized = sanitize_json_output(missing_detail)
        try:
            parsed_missing_details = json.loads(sanitized)  # JSONに変換

            # 配列形式かどうかチェック
            if isinstance(parsed_missing_details, list):
                generated_sets.append({
                    "task": task,
                    "thought": thought,
                    "missing_details": parsed_missing_details
                })
            else:
                print(f"[WARNING] Invalid format (not a list), skipping sample: {missing_detail}")
        except json.JSONDecodeError:
            print(f"[WARNING] JSON parsing failed, skipping sample: {missing_detail}")

    return generated_sets

def main():
    progress = load_progress()
    processed_count = progress["processed_count"]

    total_generations = 10000
    buffer = []

    try:
        while processed_count < total_generations:
            remaining = total_generations - processed_count
            batch_size = min(BATCH_SIZE, remaining)

            generated_sets = generate_pipeline(batch_size=batch_size)
            buffer.extend(generated_sets)
            processed_count += len(generated_sets)

            print(f"処理済み: {processed_count}/{total_generations}", end='\r')

            # バッファが十分に溜まったら保存
            if len(buffer) >= SAVE_INTERVAL:
                append_outputs(buffer)
                buffer = []
                print(f"\n{processed_count}件処理しました。")
                save_progress({"processed_count": processed_count})
                clear_memory()

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        if buffer:
            append_outputs(buffer)
            buffer = []
        save_progress({"processed_count": processed_count})
        clear_memory()
        raise

    finally:
        if buffer:
            append_outputs(buffer)
            buffer = []
        print(f"\nすべての{total_generations}件の生成が完了しました。")
        save_progress({"processed_count": processed_count})
        clear_memory()

if __name__ == "__main__":
    main()
