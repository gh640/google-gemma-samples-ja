"""A sample to use Gemma 2B it and a chat template with GPU.

See: https://huggingface.co/google/gemma-2b-it
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)

chat = [
    {"role": "user", "content": "Write a hello world program"},
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print(f"{prompt=}")
# =>
# prompt='<bos><start_of_turn>user\nWrite a hello world program<end_of_turn>\n<start_of_turn>model\n'

inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(
    "mps"
)
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)

decoded = tokenizer.decode(outputs[0])
print("decoded:\n", decoded)

# => 出力サンプル:
#  <bos><start_of_turn>user
# Write a hello world program<end_of_turn>
# <start_of_turn>model
# ```python
# print("Hello, world!")
# ```
#
# **Explanation:**
#
# * `print()` is a built-in Python function that prints the given argument to the console.
# * `"Hello, world!"` is the string that we want to print.
# * `` is the string delimiter, which tells `print()` to print the string on a single line.
#
# **Output:**
#
# ```
# Hello, world!
# ```
#
# **Note:**
#
# * The `print()` function can take multiple arguments, which will be separated by commas.
# * You can also use `print()` to print multiple lines of text by passing a list of strings as arguments.
# * `print()` is a versatile function that can be used
