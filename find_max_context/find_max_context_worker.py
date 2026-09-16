import argparse
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.llm_trace.llm_trace import LLM_STRACE
from src.modified_transformers.utils import get_model_class, identify_model_type
from transformers import AutoTokenizer
from accelerate import cpu_offload

CPU_OFFLOAD = True

# adapted to new code
def run_trace_step(llm, tokenizer, context_length, batch_size):
    """
    Runs the critical path of the user's script to test memory usage.
    Returns True if successful, False if OOM.
    """
    try:
        # 1. Create Dummy Input
        # We generate random token IDs up to the vocab size.
        vocab_size = tokenizer.vocab_size
        dummy_input_ids = torch.randint(0, vocab_size, (1, context_length))
        dummy_sentence = tokenizer.decode(dummy_input_ids[0])        
        # 3. Initialize STRACE (The wrapper)
        strace = LLM_STRACE(
            sentence=dummy_sentence,
            next_word='x', 
            llm=llm, 
            tokenizer=tokenizer,
            track_time=False)
                # 5. Populate Graph (The Memory Bottleneck)
        # This is where the heavy forward pass and gradient/activation storage happens
        strace.populate_graph(importance_mode='L1-norm')
        # 6. Extract strace
        strace.extract_strace([0.99])
        # 7. Evaluate trace
        strace.compute_stratum_reconstruction_error(do_random=False, do_inverse=False)

        # 8. Cleanup immediately to free memory for next test
        # (Simulate the graph saving/destruction)
        strace.graph.clear()
        del strace       
        
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
        model_type = identify_model_type(args.model_name)
        hf_constructor = get_model_class(model_type)
        
        if CPU_OFFLOAD:
            llm = hf_constructor.from_pretrained(
                args.model_name,
                attn_implementation="eager",
                torch_dtype=torch.float16,
            )
            llm = cpu_offload(llm, execution_device="cuda:0")
        else:
            llm = hf_constructor.from_pretrained(
                args.model_name,
                device_map="auto",
                # max_memory={0: "22GB", "cpu": "60GB"}, # Leave 2GB GPU RAM free for your hidden_states/activations
                attn_implementation="eager",
                torch_dtype=torch.float16,
            )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # Run trace...
        success = run_trace_step(llm, tokenizer, args.context_len, 1)
        
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