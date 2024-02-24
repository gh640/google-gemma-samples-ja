"""A sample to use Gemma 2B with GPU.

See: https://huggingface.co/google/gemma-2b
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# GPU を使うために `device_map="auto"`
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

input_text = "昔々あるところに"
# GPU を使うために `to("mps")`
input_ids = tokenizer(input_text, return_tensors="pt").to("mps")

# 推奨される `max_length` を明示的にセットする
outputs = model.generate(max_length=30, **input_ids)
print(tokenizer.decode(outputs[0]))
