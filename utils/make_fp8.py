from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from huggingface_hub import login

login(token="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")
MODEL_ID = "DeL-TaiseiOzaki/Tengentoppa-llm-jp-13B-base"

# Load model.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
adapter_model = PeftModel.from_pretrained(model, "output_final")
merged_model = adapter_model.merge_and_unload()

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# Apply quantization.
oneshot(model=merged_model, recipe=recipe)

print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = merged_model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic-merged"
merged_model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# リポジトリ名を設定します
REPO_NAME = "DeL-TaiseiOzaki/Tengentoppa-llm-jp-13B-reasoning-it-fp8"

tokenizer.push_to_hub(REPO_NAME)
merged_model.push_to_hub(REPO_NAME)