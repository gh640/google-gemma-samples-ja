"""A sample to use Gemma 2B it

See: https://huggingface.co/google/gemma-2b-it
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

input_text = "古典的なダジャレを言ってください"
input_ids = tokenizer(input_text, return_tensors="pt")

# 推奨されている `max_length` を明示的にセットする:
outputs = model.generate(**input_ids, max_length=40)
print(tokenizer.decode(outputs[0]))

