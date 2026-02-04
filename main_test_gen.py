import sys
import os
import torch
import numpy as np
from torch.nn.functional import softmax
from pathlib import Path
import random
import time

# --- 1. Path Setup ---
# Add project root to path so we can import src
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from src.llm_hooked.hook_constructors import get_hooked_constructor
    from src.llm_trace.llm_trace import LLM_STRACE, THRESHOLD_STRACE
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure this script is run from the project root directory containing 'src'.")
    sys.exit(1)

# --- 2. Configuration ---
class TestConfig:
    """Mock arguments configuration"""
    model_name = "Qwen/Qwen3-0.6B-Base"  # Change to your target model
    importance_mode = "norm"
    strace_mode = "threshold"                   # 'nucleus' or 'threshold'
    max_new_tokens = 5                        # Keep small for testing
    batch_size = 1
    p_sample = 0.6
    seed = 42
    prompts = [
        "At the word, color flooded into Pesh's tan cheeks, and he broke eye",
        "All colors coordinated beautifully, enhancing the feeling of the agency being a strong wedding group"
    ]
    half_precision = True

# --- 3. Helper Functions (From Stage 1) ---
EPS = 1e-7

def sample_next_token(logits, temperature=1.0, top_p=0.9):
    """Nucleus sampling helper."""
    logits = logits.float()
    next_token_logits = logits[:, -1, :] / temperature
    probs = softmax(next_token_logits, dim=-1)
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    indices_to_remove = indices_to_remove.bool()
    probs = probs.masked_fill(indices_to_remove, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    
    next_token_id = torch.multinomial(probs, num_samples=1)
    next_token_prob = probs.gather(1, next_token_id)
    nucleus_token_id = np.nonzero(probs.cpu().numpy())[1]

    return next_token_id, next_token_prob, nucleus_token_id

def get_entropy(logits):
    probabilities = softmax(logits[:, -1].float(), dim=-1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + EPS)).item()
    return entropy

# --- 4. Main Pipeline ---
def main():
    args = TestConfig()
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"--- [TEST PIPELINE] Starting ---")
    print(f"Model: {args.model_name}")
    print(f"Prompts: {len(args.prompts)}")

    # Initialize Model ONCE (Replaces the load in Stage 1 and Stage 3)
    print("\n[Init] Loading Model...")
    HOOKED_CONSTRUCTOR = get_hooked_constructor(args.model_name)
    llm_hooked = HOOKED_CONSTRUCTOR(args.model_name, half_precision=args.half_precision, untrained=False)

    # Tracking results
    pipeline_log = []

    for p_idx, prompt in enumerate(args.prompts):
        print(f"\n=== Processing Prompt {p_idx}: '{prompt}' ===")
        
        # Tokenize initial prompt
        initial_text_tokenized = llm_hooked.tokenizer([prompt], padding=True, return_tensors="pt")
        full_text_tokenized = initial_text_tokenized.input_ids.clone()
        
        # Initialize STRACE for this prompt
        # We pass 'xxx' as target as per original script logic
        strace = LLM_STRACE((initial_text_tokenized, 'xxx'), llm_hooked, track_time=False)
        strace.initialize_graph(importance_mode=args.importance_mode)

        new_last_token_id = None

        for step in range(args.max_new_tokens):
            t0 = time.time()
            print(f"  > Step {step}...")

            # -------------------------------------------------
            # STAGE 1: Generation & Extraction (GPU)
            # -------------------------------------------------
            # Ensure hooks are registered (Stage 3 removes them, so we re-add them here)
            if not llm_hooked.extraction_hook_registred():
                llm_hooked.register_extraction_hooks()

            # Trace (Forward Pass)
            # if step == 0:
            #     strace.populate_graph(batch_size=args.batch_size, print_stats=False)
            # else:
            #     strace.update_trace(new_last_token_id, batch_size=args.batch_size, print_stats=False)
            
            initial_text_tokenized.input_ids = full_text_tokenized
            initial_text_tokenized.attention_mask = torch.ones_like(full_text_tokenized)
            strace = LLM_STRACE((initial_text_tokenized, 'xxx'), llm_hooked, track_time=False)
            strace.initialize_graph(importance_mode=args.importance_mode)
            strace.populate_graph(batch_size=args.batch_size, print_stats=False)

            # Sampling (Reuse logits from trace)
            logits = strace.original_logits.cpu()
            next_token_id, next_token_prob, nucleus_token_id = sample_next_token(logits, top_p=args.p_sample)
            new_last_token_id = next_token_id
            entropy = get_entropy(logits)
            
            token_str = llm_hooked.tokenizer.decode(next_token_id[0])
            print(f"    [Gen] Token: '{token_str}' | Entropy: {entropy:.4f}")

            # -------------------------------------------------
            # STAGE 2: Stratification (CPU)
            # -------------------------------------------------
            # Retrieve thresholds
            threshold_key = '_'.join([args.importance_mode, args.strace_mode])
            threshold_values = THRESHOLD_STRACE.get(threshold_key)
            if threshold_values is None:
                print(f"    [Warn] Key {threshold_key} not found in THRESHOLD_STRACE. Using default.")
                exit()
            
            # Run extraction in-memory
            try:
                strace.extract_strace(threshold_values, mode=args.strace_mode)
                raw_size = getattr(strace, 'strata_raw_size', 'N/A')
                rel_size = getattr(strace, 'strata_rel_size', 'N/A')
                print(f"    [CPU] Stratification complete. Raw Size: {raw_size}")
                print(f"    [CPU] Rel Size: {rel_size}")
            except Exception as e:
                print(f"    [Error] Stage 2 (Stratification) failed: {e}")

            # -------------------------------------------------
            # STAGE 3: Evaluation / Reconstruction (GPU)
            # -------------------------------------------------
            # Important: Remove extraction hooks before reconstruction pass
            llm_hooked.remove_extraction_hooks()

            try:
                strace.compute_stratum_reconstruction_error()
                # Check for error metrics usually stored in the object
                reco_error = getattr(strace, 'reconstruction_error', 'N/A') # Adjust attribute name if different in src
                print(f"    [GPU] Reconstruction complete.")
            except Exception as e:
                print(f"    [Error] Stage 3 (Evaluation) failed: {e}")
                reco_error = "Error"

            # check the entropy
            strata_entropy = strace.strata_entropy['trace']['only']
            print(f"    [EVAL] Strata-entropy: {strata_entropy}")
            eval_last_entropy = strata_entropy[-1]

            if entropy != eval_last_entropy:
                reco_error = "Error"

            # check reconstruction
            strata_nu = strace.strata_reco_nu['trace']['only']
            print(f"    [EVAL] Strata-entropy: {strata_nu}")
            # -------------------------------------------------
            # Update Context & Log
            # -------------------------------------------------
            full_text_tokenized = torch.concat([full_text_tokenized, next_token_id.to('cpu')], dim=-1)
            
            pipeline_log.append({
                "prompt_idx": p_idx,
                "step": step,
                "token": token_str,
                "entropy": entropy,
                "eval_entropy": eval_last_entropy,
                "reco_status": "OK" if reco_error != "Error" else "FAIL"
            })
            
            print(f"    Step processed in {time.time() - t0:.2f}s")

    # --- Final Report ---
    print("\n\n=== Final Testing Report ===")
    print(f"{'Prompt':<10} | {'Step':<5} | {'Token':<10} | {'Entropy':<8} | {'Eval entropy':<8} | {'Status':<10}")
    print("-" * 63)
    for log in pipeline_log:
        token_display = log['token'].replace('\n', '\\n')
        print(f"{log['prompt_idx']:<10} | {log['step']:<5} | {token_display:<10} | {log['entropy']:<8.4f} | {log['eval_entropy']:<8.4f} | {log['reco_status']:<10}")

if __name__ == "__main__":
    main()