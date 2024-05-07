"""Run Gemma 7B it using GPUs on Apple Silicon macOS.

See: https://huggingface.co/docs/accelerate/en/usage_guides/mps
"""

from transformers import pipeline


MODEL = "google/gemma-7b-it"
MESSAGE = "Write me a poem about Machine Learning."

DEBUG = False


chatbot = pipeline(model=MODEL, device_map="auto")
response = chatbot(MESSAGE, max_new_tokens=200)
print(response[0]["generated_text"])

if DEBUG:
    breakpoint()
