"""A sample to use Gemma 2B it.

See: https://huggingface.co/google/gemma-2b-it
"""
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
# 次のエラーを避けるために `device_map` は使わない:
# ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: 
# `pip install accelerate`
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
)

chat = [
    { "role": "user", "content": "Python で Hello World のプログラムを書いてください" },
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print(f"{prompt=}")
# =>
# prompt='<bos><start_of_turn>user\nWrite a hello world program<end_of_turn>\n<start_of_turn>model\n'

inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)

decoded = tokenizer.decode(outputs[0])
print("decoded:\n", decoded)

# => 出力サンプル:
#  <bos><start_of_turn>user
# Python で Hello World のプログラムを書いてください<end_of_turn>
# <start_of_turn>model
# ```python
# print("Hello World!")
# ```

# このコードは、Python で Hello World のプログラムを表しています。

# このプログラムは、Python のコンソレーションで実行されます。コンソレーションは、Python の実行環境を自動的に構築するための機能です。

# このプログラムを実行すると、次のメッセージが表示されます。

# ```
# Hello World!
# ```<eos>

