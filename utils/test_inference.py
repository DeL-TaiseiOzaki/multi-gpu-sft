import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# モデルとサンプリングパラメーターを構成する
model_id = "MKJ-TOE/elyza-llama_missinfo-detection_lr1e-6_ep3" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = LLM(model=model_id, trust_remote_code=True) # 必要な場合のみtrust_remote_code=Trueを使用する
sampling_params = SamplingParams(temperature=0.5, 
                                max_tokens=1024,
                                repetition_penalty=1.2)

# プロンプトを定義する（テスト用のプロンプト）
prompts = [
    "面白いニュースを教えてください"
]

# 推論の時間を測定する
start_time = time.time()
results = llm.generate(prompts, sampling_params)
end_time = time.time()

# 生成時間を計算して出力する
generation_time = end_time - start_time
print(f"生成時間: {generation_time:.4f}秒")

# 出力を検証して印刷する
for i, result in enumerate(results):
    generated_text = result.outputs[0].text
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"Generated text: {generated_text}")

    # 簡単な検証（必要に応じてカスタマイズ）
    assert len(generated_text) > 0, f"Prompt {i+1}の出力が空です"

print("すべてのテストが完了しました。")