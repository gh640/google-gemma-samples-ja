"""A sample to use Gemma 2B it with GPU.

See: https://huggingface.co/google/gemma-2b-it
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
# GPU を使うために `device_map="auto"`
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", map_device="auto")

input_text = "古典的なダジャレを言ってください"
input_ids = tokenizer(input_text, return_tensors="pt").to("mps")

# 推奨されている `max_length` を明示的にセットする
# GPU を使うために `to("mps")`
outputs = model.generate(max_length=30, **input_ids).to("mps")
print(tokenizer.decode(outputs[0]))
