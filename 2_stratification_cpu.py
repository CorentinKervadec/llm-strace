import argparse
import os
import time
from src.llm_trace.llm_trace import load_from_file_light

def main():
    parser = argparse.ArgumentParser(description="Stage 2: CPU Stratum Analysis")
    parser.add_argument('--chunk_id', type=int, required=True, help='Slurm array task ID, used as chunk index.')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of sentences to process per job.')
    parser.add_argument('--total_sentences', type=int, required=True, help='Total number of sentences in the dataset.')
    parser.add_argument('--intermediate_dir', type=str, required=True, help='Directory to load intermediate graphs from.')
    parser.add_argument('--strace_dir', type=str, required=True, help='Directory to save strace results (for stage 3).')
    args = parser.parse_args()

    # --- Parameters ---
    strace_mode = 'nucleus' # 'nucleus' or 'threshold'
    threshold_values = [1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95, .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    # --- Calculate sentence range for this job ---
    start_index = args.chunk_id * args.chunk_size
    end_index = min((args.chunk_id + 1) * args.chunk_size, args.total_sentences)

    print(f"[CPU-JOB CHUNK {args.chunk_id}] Processing sentences from {start_index} to {end_index - 1}")

    for sentence_index in range(start_index, end_index):
        # --- Load Intermediate State ---
        intermediate_file = os.path.join(args.intermediate_dir, f'graph_{sentence_index}.npz')
        if not os.path.exists(intermediate_file):
            print(f"[CPU-JOB {sentence_index}] ERROR: No intermediate file found at {intermediate_file}. Skipping.")
            continue

        print(f"[CPU-JOB {sentence_index}] Loading intermediate file: {intermediate_file}")
        # strace = load_from_file(intermediate_file, None, False)
        strace = load_from_file_light(intermediate_file, None)

        # --- Run CPU-bound Analysis ---
        
        print(f"[CPU-JOB {sentence_index}] Starting strace extraction...")
        start_time = time.time()
        strace.extract_strace(threshold_values, mode=strace_mode)
        print(f"[CPU-JOB {sentence_index}] Time to extract strace: {time.time() - start_time:.2f} s")
        
        # --- Save Intermediate Strace Result ---
        # This file will be picked up by the 3rd stage (GPU)
        strace_file_path = os.path.join(args.strace_dir, f'strace_{sentence_index}.npz')
        # strace.save(strace_file_path)
        strace.save_light(strace_file_path)
        os.remove(intermediate_file) # remove the intermediate file once it is consumed       

    print(f"[CPU-JOB CHUNK {args.chunk_id}] Finished processing sentences {start_index} to {end_index - 1}")

if __name__ == "__main__":
    main()

