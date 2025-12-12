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
from src.llm_trace.llm_trace import LLM_STRACE, THRESHOLD_STRACE

EPS=1e-7

def sample_next_token(logits, temperature=1.0, top_p=0.9):
    """
    Runs a forward pass and samples the next token using Nucleus (Top-P) sampling.
    Returns: (token_id, token_prob, nucleus_token_ids)
    """
    logits = logits.float() # cast to float
    next_token_logits = logits[:, -1, :] / temperature
    probs = softmax(next_token_logits, dim=-1)
    
    # Sort for Nucleus Sampling
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above the threshold (Nucleus)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

    # ensure the mask is boolean and use masked_fill
    indices_to_remove = indices_to_remove.bool()
    probs = probs.masked_fill(indices_to_remove, 0.0)
    
    # Re-normalize
    # avoid division by zero
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    
    # Sample from the filtered distribution
    next_token_id = torch.multinomial(probs, num_samples=1)
    
    # Get the probability of the sampled token
    next_token_prob = probs.gather(1, next_token_id)
    
    # Get all tokens that were in the nucleus (non-zero probability after filter)
    # Warning: potentially large array
    nucleus_token_id = np.nonzero(probs.cpu().numpy())[1]

    return next_token_id, next_token_prob, nucleus_token_id

def get_entropy(logits):
    # Compute probabilities for the last token (full distribution, before filtering)
    probabilities = softmax(logits[:, -1], dim=-1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + EPS)).item()
    return entropy


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Generation & Extraction")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--importance_mode', type=str, required=True, help="Method to compute edge importance")
    parser.add_argument('--prompt_index', type=int, required=True, help="Line number in prompts file")
    parser.add_argument('--prompts_file', type=str, required=True)
    parser.add_argument('--intermediate_dir', type=str, required=True, help='Directory to save intermediate graphs.')
    parser.add_argument('--max_new_tokens', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32, help="For graph population")
    parser.add_argument('--p_sample', type=float, default=0.6, help="Parameter for nucleus sampling (0.0 to 1.0)")
    args = parser.parse_args()

    TOP_K_SAVE=100

    # --- 1. Load Prompt ---
    try:
        with open(args.prompts_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',', quotechar='"')
            rows = list(reader)
            if args.prompt_index < 0 or args.prompt_index >= len(rows):
                print(f"Error: Prompt index {args.prompt_index} out of range.")
                raise ValueError
            prompt = rows[args.prompt_index]['Paragraph Text'].strip()
    except Exception as e:
        print(f"Error loading prompt file: {e}")
        exit(1)

    # --- 2. Setup Directories ---
    prompt_dir = os.path.join(args.intermediate_dir, f"prompt_{args.prompt_index}")
    os.makedirs(prompt_dir, exist_ok=True)

    print(f"[Gen-Stage1] Prompt {args.prompt_index}: '{prompt}'")
    print(f"[Gen-Stage1] Output dir: {prompt_dir}")

    tsv_file_path = os.path.join(prompt_dir, 'generation.tsv')
    # Write header
    with open(tsv_file_path, 'w') as tsv_file:
        tsv_file.write("next_token_str\tnext_token_id\tnext_token_prob\tnucleus_token_id\tentropy\n")

    # --- 3. Init Model ---
    HOOKED_CONSTRUCTOR = get_hooked_constructor(args.model_name)
    # Using half_precision=True as in your provided code
    llm_hooked = HOOKED_CONSTRUCTOR(args.model_name, half_precision=True, untrained=False)

    
    # --- 4. Generation Loop ---
    # Tokenize initial prompt
    initial_text_tokenized = llm_hooked.tokenizer([prompt], padding=True, return_tensors="pt")
    
    # Store full history (on CPU initially to save VRAM, or GPU if needed for concatenation)
    full_text_tokenized = initial_text_tokenized.input_ids.clone()
    
    token_start = full_text_tokenized.size(-1) # token position at which the generation starts

    # --- Initialize STRACE once ---
    # Note: We pass 'xxx' as target string because we are only populating the graph 
    # based on the *current* context. The target doesn't matter for the forward pass hooks.
    strace = LLM_STRACE((initial_text_tokenized, 'xxx'), llm_hooked, track_time=False)
    strace.initialize_graph(importance_mode=args.importance_mode)

    new_last_token_id = None # will be updated after the first step

    for step in range(args.max_new_tokens):
        print(f"--- Step {step} ---")
    
        # 2. Register Hooks (if missing)
        if not llm_hooked.extraction_hook_registred():
            llm_hooked.register_extraction_hooks()
        
        # 3. Trace (Forward Pass + Graph Extraction)
        if step==0: # first step: full pass, build the graph from scratch
            strace.populate_graph(batch_size=args.batch_size, print_stats=False)
        else: # incrementally
            # todo: update the populate_graph function to add the start index
            strace.update_trace(new_last_token_id, batch_size=args.batch_size, print_stats=False)
        
        # 4. Sampling
        # We reuse the logits computed during the trace to avoid a second forward pass
        logits = strace.original_logits.cpu()
        next_token_id, next_token_prob, nucleus_token_id = sample_next_token(logits, top_p=args.p_sample)
        new_last_token_id = next_token_id
        entropy = get_entropy(logits)

        next_token_str = llm_hooked.tokenizer.decode(next_token_id[0])
        print(f"Generated: '{next_token_str}'")

        # 5. Save Graph
        # Using save_light (saves .npz)
        output_file = os.path.join(prompt_dir, f'graph_{step}.npz')
        strace.save_light(output_file)
        
        # 6. Update Context        
        # Update full history (on same device as full_text_tokenized)
        full_text_tokenized = torch.concat([full_text_tokenized, next_token_id.to('cpu')], dim=-1)

        # 7. Save Stats to TSV
        # Warning: 'nucleus_token_id' can be very large, trim to TOP_K_SAVE.
        nucleus_str = '_'.join([str(i) for i in nucleus_token_id][:TOP_K_SAVE])
        
        with open(tsv_file_path, 'a') as tsv_file:
            tsv_file.write(f"{next_token_str}\t{next_token_id[0].item()}\t{next_token_prob.item()}\t{nucleus_str}\t{entropy}\n")

    # --- End of Generation ---
    full_text_str = llm_hooked.tokenizer.decode(full_text_tokenized[0])
    print('\nFull text:', full_text_str)

    token_stop = full_text_tokenized.size(-1)

    # FIX: Move tensor to CPU and convert to list before joining for efficiency/correctness
    full_ids_list = full_text_tokenized[0].cpu().tolist()
    full_ids_str = '_'.join([str(t) for t in full_ids_list])

    # FIX: Correct indentation for final write
    with open(tsv_file_path, 'a') as tsv_file:
        tsv_file.write(f"BREAK T{token_start}-T{token_stop}\t{full_text_str}\t{full_ids_str}\n")

if __name__ == "__main__":
    main()