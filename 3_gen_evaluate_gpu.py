import argparse
import os
import time
from src.llm_trace.llm_trace import load_from_file_light
from src.llm_hooked.hook_constructors import get_hooked_constructor
import re

def main():
    parser = argparse.ArgumentParser(description="Stage 3: GPU Stratum Reconstruction Error")
    parser.add_argument('--model_name', type=str, required=True, help='HF name of the llm.')
    parser.add_argument('--prompt_index', type=int, required=True, help="Line number in prompts file")
    parser.add_argument('--strace_dir', type=str, required=True, help='Directory to load strace results from.')
    parser.add_argument('--final_dir', type=str, required=True, help='Directory to save final strace results.')
    args = parser.parse_args()

    # --- Initialise Model (on GPU) - ONCE per job ---
    print(f"[GPU-JOB-2 CHUNK {args.prompt_index}] Loading model...")
    start_time = time.time()
    HOOKED_CONSTRUCT = get_hooked_constructor(args.model_name)
    llm_hooked = HOOKED_CONSTRUCT(args.model_name, half_precision=True, untrained=False)
    
    print(f"[GPU-JOB-2 PROMPT {args.prompt_index}] Time to initialise GPU Mistral_Hooked: {time.time() - start_time:.2f} s")

    # --- Calculate sentence range for this job ---

    # List all 'strace_{step}.npz' files in the strace directory
    strace_prompt_dir = os.path.join(args.strace_dir, f"prompt_{args.prompt_index}")
    strace_files = {re.search(r'strace_(\d+)\.npz', f).group(1): os.path.join(strace_prompt_dir, f) for f in os.listdir(strace_prompt_dir) if re.match(r'strace_\d+\.npz', f)}
    
    final_prompt_dir = os.path.join(args.final_dir, f"prompt_{args.prompt_index}")
    os.makedirs(final_prompt_dir, exist_ok=True)

    print(f"[GPU-JOB-2 PROMPT {args.prompt_index}] Processing sentences from folder {strace_prompt_dir}")

    for step, strace_file in strace_files.items():
        # --- Load Intermediate State (from CPU stage) ---
        # strace_file = os.path.join(args.strace_dir, f'strace_{sentence_index}.pickle')
        if not os.path.exists(strace_file):
            print(f"[GPU-JOB-2 {step}] ERROR: No strace file found at {strace_file}. Skipping.")
            continue

        print(f"[GPU-JOB-2 {step}] Loading strace file: {strace_file}")
        # strace = load_from_file(strace_file, llm_hooked, False)
        strace = load_from_file_light(strace_file, llm_hooked)
        # don't forget to remove the extraction hooks before doing the masking
        llm_hooked.remove_extraction_hooks()

        # --- Run GPU-bound Reconstruction Error ---
        print(f"[GPU-JOB-2 {step}] Starting stratum reconstruction error calculation...")
        start_time = time.time()
        strace.compute_stratum_reconstruction_error()
        print(f"[GPU-JOB-2 {step}] Time to compute stratum reco error: {time.time() - start_time:.2f} s")

        strace.print_graph_sizes_and_thresholds()

        # --- Save Final Result ---
        # final_file_path = os.path.join(args.final_dir, f'strace_final_{sentence_index}.pickle')
        final_file_path = os.path.join(final_prompt_dir, f'strace_final_{step}.npz')
        # strace.save(final_file_path)
        strace.save_light(final_file_path)
        os.remove(strace_file) # remove the intermediate file once it is consumed
        
    print(f"[GPU-JOB-2 {args.chunk_id}] Finished processing sentences {start_index} to {end_index - 1}")


if __name__ == "__main__":
    main()

