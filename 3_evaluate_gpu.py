"""
Stage 3: Reconstruction Evaluation (GPU-Bound)
==============================================
This script represents the final stage of the s-Trace pipeline.

For each extracted computational subgraph (s-trace) produced in Stage 2, this script:
1. Reloads the language model and tokenizer onto the GPU.
2. Re-attaches the extracted subgraphs alongside control baselines (randomized and inverse subgraphs).
3. Evaluates predictive capability by computing the reconstruction error (Total Variation distance)
   between full-model output logits and pruned subgraph logits.
4. Saves the final evaluated trace objects for analysis and visualization.
5. Deletes intermediate Stage 2 files to conserve cluster storage space.
"""

import argparse
import os
import time
import torch
from transformers import AutoTokenizer
from accelerate import cpu_offload

from src.llm_trace.llm_trace import load_from_file_light
from src.modified_transformers.utils import get_model_class, identify_model_type


def main():
    parser = argparse.ArgumentParser(description="s-Trace Stage 3: GPU Reconstruction Evaluation")
    parser.add_argument('--model_name', type=str, required=True, help='Hugging Face model identifier.')
    parser.add_argument('--chunk_id', type=int, required=True, help='Slurm array task ID (used as chunk index).')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of sentences to process per job.')
    parser.add_argument('--total_sentences', type=int, required=True, help='Total size of the dataset.')
    parser.add_argument('--strace_dir', type=str, required=True, help='Directory containing Stage 2 extracted s-traces.')
    parser.add_argument('--final_dir', type=str, required=True, help='Directory to save final evaluation results.')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offload for large models.')
    parser.add_argument('--precision', type=str, choices=['float16', 'float32'], default='float16', help='Model precision.')
    parser.add_argument('--cleanup', action='store_true', help='Delete intermediate s-trace files to save disk space.')
    parser.add_argument('--checkpoint', type=str, default='main')
    args = parser.parse_args()

    # --- 1. Model Initialization ---
    print(f"[STAGE 3 | CHUNK {args.chunk_id}] Initializing model...")
    start_time = time.time()

    target_dtype = torch.float16 if args.precision == 'float16' else torch.float32
    model_type = identify_model_type(args.model_name)
    hf_constructor = get_model_class(model_type)

    if args.cpu_offload:
        llm = hf_constructor.from_pretrained(
            args.model_name,
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=target_dtype,
            revision=args.checkpoint
        )
        llm = cpu_offload(llm, execution_device="cuda:0")
    else:
        llm = hf_constructor.from_pretrained(
            args.model_name,
            device_map="auto",
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=target_dtype,
            revision=args.checkpoint
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"[STAGE 3] {args.model_name} loaded in {time.time() - start_time:.2f} s")

    # --- 2. Calculate Data Chunk Bounds ---
    start_index = args.chunk_id * args.chunk_size
    end_index = min((args.chunk_id + 1) * args.chunk_size, args.total_sentences)

    print(f"[STAGE 3 | CHUNK {args.chunk_id}] Processing dataset subset: Sentences {start_index} to {end_index - 1}")

    # --- 3. Process Chunk ---
    for sentence_index in range(start_index, end_index):
        strace_file = os.path.join(args.strace_dir, f'strace_{sentence_index}.npz')
        final_file_path = os.path.join(args.final_dir, f'strace_final_{sentence_index}.npz')

        # Skip if already evaluated by a previous run
        if os.path.exists(final_file_path):
            print(f"[STAGE 3 | IDX {sentence_index}] Final evaluation already exists. Skipping.")
            continue

        # Check if input Stage 2 file exists
        if not os.path.exists(strace_file):
            print(f"[STAGE 3 | IDX {sentence_index}] WARNING: No extracted strace file found at {strace_file}. Skipping.")
            continue

        print(f"\n[STAGE 3 | IDX {sentence_index}] Loading s-traces from {strace_file}...")
        strace = load_from_file_light(strace_file, llm, tokenizer)
        strace.reset_evaluation()

        # Run GPU-bound intervention passes to calculate reconstruction error
        print("  > Computing s-trace reconstruction error (TV distance & control baselines)...")
        t_start = time.time()
        strace.compute_stratum_reconstruction_error(
            do_random=True,
            do_inverse=True,
            save_logit=False
        )
        print(f"  > Reconstruction error computed in {time.time() - t_start:.2f} s")

        # Display summary table formatted for paper metrics
        strace.print_graph_sizes_and_thresholds()

        # Save final evaluated results
        strace.save_light(final_file_path)

        # Clean up intermediate Stage 2 graph to free up disk space
        if args.cleanup and os.path.exists(strace_file):
            os.remove(strace_file)

    print(f"\n[STAGE 3 | CHUNK {args.chunk_id}] Complete.")


if __name__ == "__main__":
    main()