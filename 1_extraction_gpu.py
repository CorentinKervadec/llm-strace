import argparse
import os
import time
from src.llm_trace.llm_trace import LLM_STRACE
from src.llm_hooked.hook_constructors import get_hooked_constructor
import re

def load_sentence(data_file, index):
    """Loads a specific line (sentence) from the .tsv data file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == index:
                try:
                    parts = line.strip().split('\t')
                    # Expects format: {sentence}\t{last word}\t{...}\t{next word}\t{...}
                    if len(parts) >= 4:
                        input_sentence = parts[0]
                        gt_next = parts[3]
                        return {"input": input_sentence, "gt_next": gt_next}
                    else:
                        print(f"[load_sentence] Error: Line {index} has fewer than 4 columns. Line: '{line.strip()}'")
                        return None
                except Exception as e:
                    print(f"[load_sentence] Error processing line {index}: {e}")
                    return None
    return None

def main():
    parser = argparse.ArgumentParser(description="Stage 1: GPU Graph Population")
    parser.add_argument('--model_name', type=str, required=True, help='HF name of the llm.')
    parser.add_argument('--chunk_id', type=int, required=True, help='Slurm array task ID, used as chunk index.')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of sentences to process per job.')
    parser.add_argument('--total_sentences', type=int, required=True, help='Total number of sentences in the dataset.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the sentences.jsonl file.')
    parser.add_argument('--intermediate_dir', type=str, required=True, help='Directory to save intermediate graphs.')
    parser.add_argument('--importance', type=str, required=True, help='Importance mode')
    args = parser.parse_args()

    # --- Parameters ---
    half_precision = True
    untrained = False
    importance_mode = args.importance

    match = re.search(r'(\d+)B', args.model_name)
    model_size = int(match.group(1)) if match else None
    if model_size > 10:
        batch_size = 1
    else:
        batch_size = 8

    # --- Initialise Model (on GPU) - ONCE per job ---
    print(f"[GPU-JOB CHUNK {args.chunk_id}] Loading model...")
    start_time = time.time()
    HOOKED_CONSTRUCT = get_hooked_constructor(args.model_name)
    llm_hooked = HOOKED_CONSTRUCT(args.model_name, half_precision, untrained)
    print(f"[GPU-JOB CHUNK {args.chunk_id}] Time to initialise Mistral_Hooked: {time.time() - start_time:.2f} s")

    # --- Calculate sentence range for this job ---
    start_index = args.chunk_id * args.chunk_size
    end_index = min((args.chunk_id + 1) * args.chunk_size, args.total_sentences)

    print(f"[GPU-JOB CHUNK {args.chunk_id}] Processing sentences from {start_index} to {end_index - 1}")

    for sentence_index in range(start_index, end_index):

        # --- Check If File Already Exist ---
        output_file = os.path.join(args.intermediate_dir, f'graph_{sentence_index}.npz')
        if os.path.isfile(output_file):
            print(f"[GPU-JOB {sentence_index}] Graph already extracted. Next.'")
            continue

        # --- Load Data ---
        sentence_data = load_sentence(args.data_file, sentence_index)
        if sentence_data is None:
            print(f"[GPU-JOB {sentence_index}] No sentence found at index {sentence_index}. Skipping.")
            continue
            
        input_sentence = sentence_data["input"]
        gt_next = sentence_data["gt_next"]
        
        print(f"[GPU-JOB {sentence_index}] Processing sentence: '{input_sentence}'")

        # --- Register Hooks (if needed) ---
        if not llm_hooked.extraction_hook_registred():
            llm_hooked.register_extraction_hooks()

        # --- Run GPU-bound Trace ---
        
        # 1. Initialise the strace
        start_time = time.time()
        strace = LLM_STRACE((input_sentence, gt_next), llm_hooked, track_time=True)
        print(f"[GPU-JOB {sentence_index}] Time to initialise LLM_STRACE: {time.time() - start_time:.2f} s")

        # 2. Initialise the llm graph
        start_time = time.time()
        strace.initialize_graph(importance_mode)
        print(f"[GPU-JOB {sentence_index}] Time to initialise Mistral_Graph_NX: {time.time() - start_time:.2f} s")
        
        # 3. Populate the graph
        start_time = time.time()
        strace.populate_graph(batch_size)
        print(f"[GPU-JOB {sentence_index}] Time to populate graph: {time.time() - start_time:.2f} s")

        # --- Save Intermediate State ---
        strace.save_light(output_file)
        # strace.save(output_file)
    print(f"[GPU-JOB CHUNK {args.chunk_id}] Finished processing sentences {start_index} to {end_index - 1}")

if __name__ == "__main__":
    main()

