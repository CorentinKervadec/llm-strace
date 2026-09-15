"""
Stage 2: Strata Extraction (CPU)
======================================
This script represents the second stage of the s-Trace pipeline. 
Because filtering edges and extracting computational subgraphs (s-traces) 
is memory-light and requires no neural network forward passes, this stage 
runs efficiently on CPU.

For a given chunk of pre-populated graphs, this script:
1. Loads the lightweight computation graph produced in Stage 1.
2. Sweeps over a grid of sizes to isolate s-traces.
4. Saves the extracted straces to disk for predictive evaluation in Stage 3.
5. Deletes intermediate Stage 1 files.
"""

import argparse
import os
import time
from src.llm_trace.llm_trace_2 import load_from_file_light

def main():
    parser = argparse.ArgumentParser(description="s-Trace Stage 2: CPU Strata Extraction")
    parser.add_argument('--chunk_id', type=int, required=True, help='Slurm array task ID (used as chunk index).')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of sentences to process per job.')
    parser.add_argument('--total_sentences', type=int, required=True, help='Total size of the dataset.')
    parser.add_argument('--intermediate_dir', type=str, required=True, help='Directory containing Stage 1 intermediate graphs.')
    parser.add_argument('--strace_dir', type=str, required=True, help='Directory to save extracted straces for Stage 3.')
    parser.add_argument('--importance', type=str, required=True, help='Edge importance metric used during graph population.')
    parser.add_argument('--freeze', type=str, default=None, help='Optional component freezing ("attention" or "mlp").')
    args = parser.parse_args()

    # --- 1. Threshold Density Grid (tau) ---
    # Defines the target size used to extract subgraphs (1.0 == 100% of the model)
    threshold_values = [
        1e-5, 2e-5, 4e-5, 8e-5,
        1e-4, 2e-4, 4e-4, 8e-4,
        1e-3, 1.2e-3, 1.4e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3,
        1e-2, 2e-2, 4e-2, 6e-2, 8e-2,
        1e-1, 2e-1, 4e-1, 6e-1, 8e-1
    ]

    # --- 2. Calculate Data Chunk Bounds ---
    start_index = args.chunk_id * args.chunk_size
    end_index = min((args.chunk_id + 1) * args.chunk_size, args.total_sentences)

    print(f"[STAGE 2 | CHUNK {args.chunk_id}] Processing dataset subset: Sentences {start_index} to {end_index - 1}")

    # --- 3. Process Chunk ---
    for sentence_index in range(start_index, end_index):
        intermediate_file = os.path.join(args.intermediate_dir, f'graph_{sentence_index}.npz')
        strace_file_path = os.path.join(args.strace_dir, f'strace_{sentence_index}.npz')

        # Check if already completed by a previous job
        if os.path.exists(strace_file_path):
            print(f"[STAGE 2 | IDX {sentence_index}] Strace already extracted. Skipping.")
            continue

        # Check if input file exists
        if not os.path.exists(intermediate_file):
            print(f"[STAGE 2 | IDX {sentence_index}] WARNING: Intermediate graph missing at {intermediate_file}. Skipping.")
            continue

        print(f"\n[STAGE 2 | IDX {sentence_index}] Extracting subgraphs from intermediate graph...")
        
        # Load lightweight graph state from Stage 1
        strace = load_from_file_light(intermediate_file)
        strace.reset_strace()
        
        if args.freeze: # not used in the paper
            print(f"  > Applying component freeze: {args.freeze}")
            strace.freeze_strace(args.freeze)

        # Run CPU-bound subgraph extraction across all thresholds
        t_start = time.time()
        strace.extract_strace(threshold_values)
        print(f"  > Stratification done ({len(threshold_values)} strace) in {time.time() - t_start:.2f} s")

        # Save extracted strata for Stage 3 (GPU evaluation)
        strace.save_light(strace_file_path)
        
        # Clean up intermediate Stage 1 graph to free up disk space
        if os.path.exists(intermediate_file):
            os.remove(intermediate_file)

    print(f"\n[STAGE 2 | CHUNK {args.chunk_id}] Complete.")

if __name__ == "__main__":
    main()