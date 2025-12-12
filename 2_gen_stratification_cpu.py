import argparse
import os
import time
from src.llm_trace.llm_trace import load_from_file_light, THRESHOLD_STRACE
import re

def main():
    parser = argparse.ArgumentParser(description="Stage 2: CPU Stratum Analysis")
    parser.add_argument('--intermediate_dir', type=str, required=True, help='Directory to load intermediate graphs from.')
    parser.add_argument('--prompt_index', type=int, required=True, help="Line number in prompts file")
    parser.add_argument('--strace_dir', type=str, required=True, help='Directory to save strace results (for stage 3).')
    parser.add_argument('--strace', type=str, required=True, help='Strace extraction mode')
    parser.add_argument('--importance', type=str, required=True, help='Importance mode')
    args = parser.parse_args()

    # --- Parameters ---
    strace_mode = args.strace # 'nucleus' or 'threshold'
    importance_mode = args.importance

    threshold_values = THRESHOLD_STRACE['_'.join([importance_mode, strace_mode])]
    
    print(f"[CPU-JOB] Processing generation from folder {args.intermediate_dir}")

    # List all 'graph_{step}.npz' files in the intermediate directory
    intermediate_prompt_dir = os.path.join(args.intermediate_dir, f"prompt_{args.prompt_index}")
    graph_files = {re.search(r'graph_(\d+)\.npz', f).group(1): os.path.join(intermediate_prompt_dir, f) for f in os.listdir(intermediate_prompt_dir) if re.match(r'graph_\d+\.npz', f)}
    print(f"[CPU-JOB] Found {len(graph_files)} graph files in {intermediate_prompt_dir}")

    strace_prompt_dir = os.path.join(args.strace_dir, f"prompt_{args.prompt_index}")
    os.makedirs(strace_prompt_dir, exist_ok=True)
    
    for step, intermediate_file in graph_files.items():
        # --- Load Intermediate State ---
        if not os.path.exists(intermediate_file):
            print(f"[CPU-JOB STEP {step}] ERROR: No intermediate file found at {intermediate_file}. Skipping.")
            continue

        print(f"[CPU-JOB STEP {step}] Loading intermediate file: {intermediate_file}")
        strace = load_from_file_light(intermediate_file, None)

        # --- Run CPU-bound Analysis ---
        
        print(f"[CPU-JOB STEP {step}] Starting strace extraction...")
        start_time = time.time()
        strace.extract_strace(threshold_values, mode=strace_mode)
        print("strata_raw_size", strace.strata_raw_size)
        print(f"[CPU-JOB STEP {step}] Time to extract strace: {time.time() - start_time:.2f} s")
        
        # --- Save Intermediate Strace Result ---
        # This file will be picked up by the 3rd stage (GPU)
        strace_file_path = os.path.join(strace_prompt_dir, f'strace_{step}.npz')
        # strace.save(strace_file_path)
        strace.save_light(strace_file_path)
        os.remove(intermediate_file) # remove the intermediate file once it is consumed       

    print(f"[CPU-JOB STEP {step}] Finished processing sentences from folder {intermediate_prompt_dir}")

if __name__ == "__main__":
    main()

