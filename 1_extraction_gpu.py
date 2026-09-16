"""
Stage 1: Graph Population (GPU-Bound)
=====================================
This script represents the first stage of the s-Trace extraction pipeline. 
It is designed to run in parallel across a compute cluster (e.g., via Slurm arrays).

For a given chunk of sentences, this script:
1. Loads the specified Large Language Model onto the GPU.
2. Computes the full forward pass for each sentence.
3. Populates the complete computation graph with edge importance scores.
4. Saves a lightweight version of the graph to disk for subsequent CPU processing.
"""

import argparse
import os
import time
import torch
from transformers import AutoTokenizer
from accelerate import cpu_offload

from src.llm_trace.llm_trace import LLM_STRACE
from src.modified_transformers.utils import get_model_class, identify_model_type

def load_sentence(data_file: str, index: int) -> dict:
    """
    Retrieves a specific sentence and its ground truth next word from the dataset.
    The ground truth next word is not used for faithfulness evaluation (which compares against the full model distribution)
    but is sometimes used to estimate perplexity of the LLM on the dataset.
    Handles both structured .tsv (wikitext) and plain .txt files.
    """
    if 'wikitext' in data_file or 'df' in data_file:
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == index:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        return {"input": parts[0], "gt_next": parts[3]}
                    print(f"[DATA] Error: Line {index} lacks required columns.")
                    return None
    else:
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == index:
                    return {"input": line.strip('\n'), "gt_next": 'x'}
    return None

def main():
    parser = argparse.ArgumentParser(description="s-Trace Stage 1: GPU Graph Population")
    parser.add_argument('--model_name', type=str, required=True, help='Hugging Face model identifier.')
    parser.add_argument('--chunk_id', type=int, required=True, help='Slurm array task ID (used as chunk index).')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of sentences to process in this job.')
    parser.add_argument('--total_sentences', type=int, required=True, help='Total size of the dataset.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--intermediate_dir', type=str, required=True, help='Output directory for the extracted graphs.')
    parser.add_argument('--importance', type=str, default='L1-norm', help='Edge importance metric to use.')
    parser.add_argument('--precision', type=str, choices=['float16', 'float32'], default='float16', help='Model precision.')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offload for large models.')
    parser.add_argument('--checkpoint', type=str, default='main')
    args = parser.parse_args()

    # --- 1. Model Initialization ---
    print(f"[STAGE 1 | CHUNK {args.chunk_id}] Initializing model...")
    start_time = time.time()
    
    # Set dtype based on argument
    target_dtype = torch.float16 if args.precision == 'float16' else torch.float32
    
    model_type = identify_model_type(args.model_name)
    hf_constructor = get_model_class(model_type)
    
    # Load model weights
    llm = hf_constructor.from_pretrained(
        args.model_name,
        device_map="auto" if not args.cpu_offload else None,
        attn_implementation="eager",
        use_safetensors=True,
        torch_dtype=target_dtype,
        revision=args.checkpoint
    )

    if args.cpu_offload:
        llm = cpu_offload(llm, execution_device="cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"[STAGE 1] {args.model_name} loaded in {time.time() - start_time:.2f} s")

    # --- 2. Calculate Data Chunk Bounds ---
    start_index = args.chunk_id * args.chunk_size
    end_index = min((args.chunk_id + 1) * args.chunk_size, args.total_sentences)
    print(f"[STAGE 1] Processing dataset subset: Sentences {start_index} to {end_index - 1}")

    # --- 3. Process Chunk ---
    for sentence_index in range(start_index, end_index):
        
        # Skip if already processed by a previous run
        output_file = os.path.join(args.intermediate_dir, f'graph_{sentence_index}.npz')
        if os.path.isfile(output_file):
            print(f"[STAGE 1 | IDX {sentence_index}] Graph already exists. Skipping.")
            continue

        # Fetch text
        sentence_data = load_sentence(args.data_file, sentence_index)
        if not sentence_data:
            print(f"[STAGE 1 | IDX {sentence_index}] Failed to load data. Skipping.")
            continue
            
        print(f"\n[STAGE 1 | IDX {sentence_index}] Analyzing: '{sentence_data['input']}'")

        # Initialize the strace instance
        t_init = time.time()
        strace = LLM_STRACE(
            sentence=sentence_data["input"],
            next_word=sentence_data["gt_next"], 
            llm=llm, 
            tokenizer=tokenizer,
            track_time=False
        )
        print(f"  > Strace initialized in {time.time() - t_init:.2f} s")
        
        # Populate the computation graph
        t_pop = time.time()
        strace.populate_graph(args.importance)
        print(f"  > Graph populated in {time.time() - t_pop:.2f} s")

        # Save the lightweight topological representation for Stage 2
        strace.save_light(output_file)
        
    print(f"\n[STAGE 1 | CHUNK {args.chunk_id}] Complete.")

if __name__ == "__main__":
    main()