from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_sentence = "In the absence of Bangladesh's opening bowler, Mortaza, Australia opened the innings with Andrew Symonds and Michael"

print("1) Eager Attention + Full precision")
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", device_map="auto", torch_dtype=torch.float32)
tokenized_input = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
output = model(**tokenized_input)
print("Nan: ", output.logits.isnan().sum())
print(output.logits[-1])

print("2) Eager Attention + Half precision")
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", device_map="auto", torch_dtype=torch.float16)
tokenized_input = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
output = model(**tokenized_input)
print("Nan: ", output.logits.isnan().sum())

print("3) Default Attention + Full precision")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float32)
tokenized_input = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
output = model(**tokenized_input)
print("Nan: ", output.logits.isnan().sum())

print("4) Default Attention + Half precision")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
tokenized_input = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
output = model(**tokenized_input)
print("Nan: ", output.logits.isnan().sum())

print("2) Eager Attention + bfloat16")
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", device_map="auto", torch_dtype=torch.bfloat16)
tokenized_input = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
output = model(**tokenized_input)
print("Nan: ", output.logits.isnan().sum())
print(output.logits[-1])