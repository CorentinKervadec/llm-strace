import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Initialize Model and Tokenizer
olmo = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-2-1124-7B",
    device_map="auto", 
    attn_implementation="eager", 
    torch_dtype=torch.float16, 
)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")

# 2. Define the stopping tokens (sentence terminators)
stop_chars = [".", "!", "?", "\n"]

# Get the token IDs for these punctuation marks. 
# We take the last ID in case the tokenizer splits them or adds a prefix.
stop_token_ids = [tokenizer.encode(char, add_special_tokens=False)[-1] for char in stop_chars]

# It is good practice to also include the model's default End-Of-Sequence token
if tokenizer.eos_token_id is not None:
    stop_token_ids.append(tokenizer.eos_token_id)

# 3. Prepare Inputs
message = ["Language modeling is"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)

# Move inputs to the same device as the model (required for device_map="auto")
inputs = {k: v.to(olmo.device) for k, v in inputs.items()}

# 4. Generate with the new stopping criteria
for i in range(5): # repeat the generation N time, to capture the diversity
    with torch.no_grad(): 
        response = olmo.generate(
            **inputs, 
            max_new_tokens=40, 
            do_sample=True, 
            top_p=0.6, # <-- we use nucleus sampling, so the generation could be different when you try multiple time
            eos_token_id=stop_token_ids # <--- Instructs the model to stop at any of these IDs
        )
    print(f"[#{i}]", tokenizer.batch_decode(response, skip_special_tokens=True)[0])