import argparse
import os
import sys
import torch
import numpy as np
from torch.nn.functional import softmax
from pathlib import Path
import csv

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.llm_hooked.hook_constructors import get_hooked_constructor
from src.llm_trace.llm_trace import LLM_STRACE

EPS=1e-7

def sample_next_token(logits, temperature=1.0, top_p=0.9):
    """
    Runs a forward pass and samples the next token using Nucleus (Top-P) sampling.
    Returns: (token_string, token_id)
    """
    
    next_token_logits = logits[:, -1, :] / temperature
    probs = softmax(next_token_logits, dim=-1)
    
    # Sort for Nucleus Sampling
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

    # ensure the mask is boolean (scatter sometimes yields integer type) and use masked_fill
    indices_to_remove = indices_to_remove.bool()
    probs = probs.masked_fill(indices_to_remove, 0.0)
    # avoid division by zero when all probabilities are removed
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    
    next_token_id = torch.multinomial(probs, num_samples=1)
    next_token_prob = probs.gather(1, next_token_id)
    nucleus_token_id = np.nonzero(probs.cpu().numpy())[1]

    return next_token_id, next_token_prob, nucleus_token_id

def get_entropy(logits):
    # Compute probabilities for the last token
    probabilities = softmax(logits[:, -1], dim=-1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + EPS)).item()
    return entropy


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Generation & Extraction")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--prompt_index', type=int, required=True, help="Line number in prompts file")
    parser.add_argument('--prompts_file', type=str, required=True)
    parser.add_argument('--intermediate_dir', type=str, required=True, help='Directory to save intermediate graphs.')
    parser.add_argument('--max_new_tokens', type=int, default=20)
    parser.add_argument('--max_context_length', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32, help="For graph population")
    args = parser.parse_args()

    TOP_P = 0.6

    # --- 1. Load Prompt ---
    try:
        with open(args.prompts_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',', quotechar='"')
            rows = list(reader)  # DictReader is an iterator; convert to list to get length and index access
            if args.prompt_index < 0 or args.prompt_index >= len(rows):
                print(f"Error: Prompt index {args.prompt_index} out of range.")
                raise ValueError
            prompt = rows[args.prompt_index]['Paragraph Text'].strip()
    except Exception as e:
        print(f"Error loading prompt file: {e}")
        exit(0)

    # --- 2. Setup Directories ---
    # Create a specific folder for this prompt index
    prompt_dir = os.path.join(args.intermediate_dir, f"prompt_{args.prompt_index}")
    os.makedirs(prompt_dir, exist_ok=True)

    print(f"[Gen-Stage1] Prompt {args.prompt_index}: '{prompt}'")
    print(f"[Gen-Stage1] Output dir: {prompt_dir}")

    tsv_file_path = os.path.join(prompt_dir, 'generation.tsv')
    with open(tsv_file_path, 'w') as tsv_file:
        tsv_file.write("next_token_str\tnext_token_id\tnext_token_prob\tnucleus_token_id\tentropy\n")

    # --- 3. Init Model ---
    HOOKED_CONSTRUCTOR = get_hooked_constructor(args.model_name)
    llm_hooked = HOOKED_CONSTRUCTOR(args.model_name, half_precision=False, untrained=False)

    # --- 4. Generation Loop ---
    current_text_tokenized = llm_hooked.tokenizer([prompt], padding=True, return_tensors="pt")
    full_text_tokenized = current_text_tokenized.input_ids
    
    for step in range(args.max_new_tokens):
        print(f"--- Step {step} ---")

        # Trim sentence to max_context_length
        current_text_tokenized.input_ids = current_text_tokenized.input_ids[:, -args.max_context_length:]
        current_text_tokenized.attention_mask = current_text_tokenized.attention_mask[:, -args.max_context_length:]
        
        trimmed_input_str = llm_hooked.tokenizer.decode(current_text_tokenized.input_ids[0])
        print("Input:", trimmed_input_str)

        # A. Generate
        if not llm_hooked.extraction_hook_registred():
            llm_hooked.register_extraction_hooks()
        
        # Initialize trace for the transition (current_text -> next_token_str)
        strace = LLM_STRACE((current_text_tokenized, 'xxx'), llm_hooked, track_time=False)
        strace.initialize_graph(importance_mode='norm')
        strace.populate_graph(batch_size=args.batch_size, print_stats=False)
        
        logits = strace.original_logits.cpu()
        next_token_id, next_token_prob, nucleus_token_id = sample_next_token(logits, top_p=TOP_P)
        entropy = get_entropy(logits) # get entropy

        next_token_str = llm_hooked.tokenizer.decode(next_token_id[0])
        print(f"Generated: '{next_token_str}'")

        # C. Save Graph
        output_file = os.path.join(prompt_dir, f'graph_{step}.npz')
        strace.save_light(output_file)

        # D. Update Context
        current_text_tokenized.input_ids = torch.concat([current_text_tokenized.input_ids, next_token_id], dim=-1)
        current_text_tokenized.attention_mask = torch.concat([current_text_tokenized.attention_mask, torch.ones_like(next_token_id)], dim=-1)

        full_text_tokenized = torch.concat([full_text_tokenized, next_token_id], dim=-1)

        # E. Save Results to TSV ---
        with open(tsv_file_path, 'a') as tsv_file:
            tsv_file.write(f"{next_token_str}\t{next_token_id[0].item()}\t{next_token_prob.item()}\t{'_'.join([str(i) for i in nucleus_token_id])}\t{entropy}\n")

    full_text_str = llm_hooked.tokenizer.decode(full_text_tokenized[0])
    print('\nFull text:', full_text_str)

if __name__ == "__main__":
    main()