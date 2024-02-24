"""A sample to use Gemma 2B.

See: https://huggingface.co/google/gemma-2b
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(max_length=30, **input_ids)
print(tokenizer.decode(outputs[0]))
