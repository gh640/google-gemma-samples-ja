"""A sample to use Gemma 2B it with GPU.

See: https://huggingface.co/google/gemma-2b-it
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# GPU を使うために `device_map="auto"`
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

input_text = "古典的なダジャレを言ってください"
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

# 推奨されている `max_length` を明示的にセットする
# GPU を使うために `to(model.device)`
outputs = model.generate(max_length=30, **input_ids).to(model.device)
print(tokenizer.decode(outputs[0]))
