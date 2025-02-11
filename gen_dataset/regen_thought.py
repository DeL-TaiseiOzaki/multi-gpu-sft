import json
import re
import torch
import gc
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import login

# Hugging Face にログイン
login(token="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")

# モデルの設定
model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(
    model=model_name,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.95,
    max_model_len=2048,
    dtype=torch.bfloat16
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=1024
)

BATCH_SIZE = 500  # 一度に処理するデータ数
SAVE_INTERVAL = 1000  # 何件ごとに保存するか

def refine_thoughts(tasks, original_thoughts, missing_details_list):
    """
    `task`, `original_thought`, `missing_details` を考慮し、新しい `thought` を生成する関数
    """
    system_prompt = (
        "You are an AI assistant that refines the thought process behind a task.\n"
        "You will receive a task, an initial thought process, and identified missing details.\n"
        "Your job is to refine the thought process so that it logically connects the task with the missing details.\n"
        "Make sure that the refined thought process clearly explains how the missing details were identified."
    )

    prompts = [
        f"{system_prompt}\n\nTask: {task}\n\nOriginal Thought Process: {original_thought}\n\n"
        f"Identified Missing Details: {json.dumps(missing_details, indent=4)}\n\n"
        "Refined thought process. The thought process must meet the following requirements:\n"
        "(1) Clearly determine whether the task is vague or not.\n"
        "(2) Analyze the contents of the task in depth.\n"
        "(3) The task analysis foresees what kind of missing information there is.\n"
        "Output only the contents of the refined thought process.\n\nRefined thought process:"
        for task, original_thought, missing_details in zip(tasks, original_thoughts, missing_details_list)
    ]

    messages_batch = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompts
    ]
    
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    outputs = llm.generate(texts, sampling_params)

    return [output.outputs[0].text.strip() for output in outputs]

def process_jsonl_file(input_filename, output_filename):
    """
    JSONLファイルをバッチ処理で読み込み、`thought` を再生成し、新しい JSONL ファイルに保存する。
    """
    updated_data = []
    
    with open(input_filename, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    tasks = []
    original_thoughts = []
    missing_details_list = []
    json_objects = []

    # タスクと不足情報を抽出
    for line in lines:
        data = json.loads(line.strip())
        tasks.append(data["task"])
        original_thoughts.append(data["thought"])
        missing_details_list.append(data["missing_details"])
        json_objects.append(data)

    total_records = len(tasks)
    processed_count = 0

    with open(output_filename, "w", encoding="utf-8") as outfile:
        for i in range(0, total_records, BATCH_SIZE):
            batch_tasks = tasks[i:i+BATCH_SIZE]
            batch_original_thoughts = original_thoughts[i:i+BATCH_SIZE]
            batch_missing_details = missing_details_list[i:i+BATCH_SIZE]
            batch_json_objects = json_objects[i:i+BATCH_SIZE]

            # `thought` を再生成
            new_thoughts = refine_thoughts(batch_tasks, batch_original_thoughts, batch_missing_details)

            # 更新データの構築
            for j, data in enumerate(batch_json_objects):
                data["thought"] = new_thoughts[j]
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

            processed_count += len(batch_tasks)
            print(f"Processed: {processed_count}/{total_records}", end="\r")

            # メモリ開放
            if processed_count % SAVE_INTERVAL == 0:
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\nProcessing complete. Updated JSONL file saved as: {output_filename}")

# 実行
input_filename = "gen_dataset/generated_sets.jsonl"  # 入力ファイル名
output_filename = "gen_dataset/regen_sets.jsonl"  # 出力ファイル名
process_jsonl_file(input_filename, output_filename)
