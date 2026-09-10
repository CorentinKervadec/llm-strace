import argparse
import os
import time
from src.llm_trace.llm_trace_2 import load_from_file_light
from src.modified_transformers.utils import get_model_class, identify_model_type
from transformers import AutoTokenizer
import torch
from accelerate import cpu_offload

def main():
    parser = argparse.ArgumentParser(description="Stage 3: GPU Stratum Reconstruction Error")
    parser.add_argument('--model_name', type=str, required=True, help='HF name of the llm.')
    parser.add_argument('--chunk_id', type=int, required=True, help='Slurm array task ID, used as chunk index.')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of sentences to process per job.')
    parser.add_argument('--total_sentences', type=int, required=True, help='Total number of sentences in the dataset.')
    parser.add_argument('--strace_dir', type=str, required=True, help='Directory to load strace results from.')
    parser.add_argument('--final_dir', type=str, required=True, help='Directory to save final strace results.')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offload')
    parser.add_argument('--checkpoint', type=str, default='main')
    args = parser.parse_args()

    half_precision = True

    # --- Initialise Model (on GPU) - ONCE per job ---
    print(f"[GPU-JOB-2 CHUNK {args.chunk_id}] Loading model...")
    start_time = time.time()
    
    model_type = identify_model_type(args.model_name)
    hf_constructor = get_model_class(model_type)
    if args.cpu_offload:
        llm = hf_constructor.from_pretrained(
            args.model_name,
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=torch.float16 if half_precision else torch.float32,
            revision=args.checkpoint
        )
        llm = cpu_offload(llm, execution_device="cuda:0")
    else:
        llm = hf_constructor.from_pretrained(
            args.model_name,
            device_map="auto",
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=torch.float16 if half_precision else torch.float32,
            revision=args.checkpoint
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = llm.device
    
    print(f"[GPU-JOB-2 CHUNK {args.chunk_id}] Time to initialise GPU Mistral_Hooked: {time.time() - start_time:.2f} s")

    # --- Calculate sentence range for this job ---
    start_index = args.chunk_id * args.chunk_size
    end_index = min((args.chunk_id + 1) * args.chunk_size, args.total_sentences)
    
    print(f"[GPU-JOB-2 CHUNK {args.chunk_id}] Processing sentences from {start_index} to {end_index - 1}")

    for sentence_index in range(start_index, end_index):
        # --- Load Intermediate State (from CPU stage) ---
        # strace_file = os.path.join(args.strace_dir, f'strace_{sentence_index}.pickle')
        strace_file = os.path.join(args.strace_dir, f'strace_{sentence_index}.npz')
        # strace_file = os.path.join(args.strace_dir, f'strace_final_{sentence_index}.npz')
        if not os.path.exists(strace_file):
            print(f"[GPU-JOB-2 {sentence_index}] ERROR: No strace file found at {strace_file}. Skipping.")
            continue

        print(f"[GPU-JOB-2 {sentence_index}] Loading strace file: {strace_file}")
        strace = load_from_file_light(strace_file, llm, tokenizer)
        strace.reset_evaluation() # reset the evaluation, in case we are re-doing it
        
        # --- Run GPU-bound Reconstruction Error ---
        print(f"[GPU-JOB-2 {sentence_index}] Starting stratum reconstruction error calculation...")
        start_time = time.time()
        # MODIFY RANDOM INVERSE AND SAVE LOGIT IF YOU NEED IT
        strace.compute_stratum_reconstruction_error(do_random=True, do_inverse=True, save_logit=False)
        print(f"[GPU-JOB-2 {sentence_index}] Time to compute stratum reco error: {time.time() - start_time:.2f} s")

        strace.print_graph_sizes_and_thresholds()

        # --- Save Final Result ---
        # final_file_path = os.path.join(args.final_dir, f'strace_final_{sentence_index}.pickle')
        final_file_path = os.path.join(args.final_dir, f'strace_final_{sentence_index}.npz')
        strace.save_light(final_file_path)
        os.remove(strace_file) # remove the intermediate file once it is consumed
        
    print(f"[GPU-JOB-2 CHUNK {args.chunk_id}] Finished processing sentences {start_index} to {end_index - 1}")


if __name__ == "__main__":
    main()

