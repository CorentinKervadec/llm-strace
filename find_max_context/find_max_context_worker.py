import argparse
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.llm_hooked.hook_constructors import get_hooked_constructor
from src.llm_trace.llm_trace import LLM_STRACE

def run_trace_step(llm_hooked, context_length, batch_size):
    """
    Runs the critical path of the user's script to test memory usage.
    Returns True if successful, False if OOM.
    """
    try:
        # 1. Create Dummy Input
        # We generate random token IDs up to the vocab size.
        vocab_size = llm_hooked.tokenizer.vocab_size
        dummy_input_ids = torch.randint(0, vocab_size, (1, context_length))
        
        # Mock attention mask (all ones)
        dummy_attention_mask = torch.ones_like(dummy_input_ids)
        
        # Bundle into a dictionary/object similar to tokenizer output if needed, 
        # or pass directly if LLM_STRACE handles tensors. 
        # Based on your snippet, LLM_STRACE accepts the tokenized object.
        class MockTokenized:
            def __init__(self, ids, mask):
                self.input_ids = ids
                self.attention_mask = mask
        
        current_text_tokenized = MockTokenized(dummy_input_ids, dummy_attention_mask)

        # 2. Register Hooks
        if not llm_hooked.extraction_hook_registred():
            llm_hooked.register_extraction_hooks()

        # 3. Initialize STRACE (The wrapper)
        # We pass a dummy 'gt_next' string as it doesn't impact memory of the graph population
        strace = LLM_STRACE((current_text_tokenized, 'dummy_target'), llm_hooked, track_time=False)
        
        # 4. Initialize Graph Structure
        strace.initialize_graph(importance_mode='norm')
        
        # 5. Populate Graph (The Memory Bottleneck)
        # This is where the heavy forward pass and gradient/activation storage happens
        strace.populate_graph(batch_size=batch_size, print_stats=False)
        
        # 6. Extract strace
        strace.extract_strace([0.99], mode='threshold')

        # 7. Evaluate trace
        llm_hooked.remove_extraction_hooks()
        strace.compute_stratum_reconstruction_error(do_random=False, do_inverse=False)

        # 8. Cleanup immediately to free memory for next test
        # (Simulate the graph saving/destruction)
        strace.graph.clear()
        del strace       
        del current_text_tokenized
        
        return True

    except torch.cuda.OutOfMemoryError:
        return False
    except Exception as e:
        print(f"\n[Warning] Unexpected error at length {context_length}: {e}")
        # If it's not memory related, we might want to raise, but for this test assume failure
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument("--context_len", type=int, required=True)
    args = parser.parse_args()

    print(f"--- WORKER STARTING: Testing context {args.context_len} ---")

    # Run exactly ONE test
    try:
        # Setup model...
        HOOKED_CONSTRUCTOR = get_hooked_constructor(args.model_name)
        llm_hooked = HOOKED_CONSTRUCTOR(args.model_name, half_precision=True, untrained=False)

        # Run trace...
        success = run_trace_step(llm_hooked, args.context_len, 1)
        
        if success:
            print("SUCCESS")
            sys.exit(0) # Exit code 0 means "It worked"
        else:
            print("FAILURE_LOGIC")
            sys.exit(1) # Application error

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cudamalloc" in str(e).lower():
            print("OOM_ERROR")
            sys.exit(137) # Standard OOM exit code convention
        else:
            raise e

if __name__ == "__main__":
    main()