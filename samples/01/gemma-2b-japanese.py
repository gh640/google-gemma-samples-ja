"""A sample to use Gemma 2B.

See: https://huggingface.co/google/gemma-2b
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

input_text = "昔々あるところに"
input_ids = tokenizer(input_text, return_tensors="pt")

# 推奨される `max_length` を明示的にセットする
outputs = model.generate(**input_ids, max_length=20)
print(tokenizer.decode(outputs[0]))
