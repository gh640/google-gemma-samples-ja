"""Run Gemma 7B it using GPUs on Apple Silicon macOS

See: https://huggingface.co/docs/accelerate/en/usage_guides/mps
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto")

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(**input_ids, max_length=200)
print(tokenizer.decode(outputs[0]))
