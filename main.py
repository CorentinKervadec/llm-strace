"""
Main testing script for the LLM_STRACE pipeline.

This script is designed for local testing and debugging. It:
1. Loads a specified language model with hugging face.
2. Processes a small, hard-coded set of test sentences.
3. (GPU) Populates a full computation graph with edge importance scores.
4. (CPU) Extracts "s-traces" (subgraphs) from the full graph based on a set of target sizes.
5. (GPU) Evaluates the predictive power of each s-trace by measuring reconstruction error.
"""

# Import the main orchestration class
from src.llm_trace.llm_trace_2 import LLM_STRACE
from src.modified_transformers.utils import get_model_class, identify_model_type
from transformers import AutoTokenizer
from accelerate import cpu_offload

import time
import argparse
import torch

AVAILABLE_MODELS = [
    "mistralai/Mistral-7B-v0.1",
    'allenai/OLMo-2-0425-1B',
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-13B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen2-7B",
    "meta-llama/Llama-3.1-8B",
    "deepseek-ai/deepseek-llm-7b-base",
    "microsoft/phi-4",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf"
]

def main(model_name: str):
    """
    Main function to run the tracing and evaluation pipeline on a few test sentences.
    
    Args:
        model_name (str): The Hugging Face name of the model to load.
    """
    
    # --- 1. Define Test Data ---
    # Small, hard-coded datasets for quick local testing.
    sentences_test = [
        ("How many legs does a dog have? Answer:", "4"),
        ("How many legs does a chicken have? Answer:", "2")
    ]
    sentences_50 = [
        ("In the absence of Bangladesh's opening bowler, Mortaza, Australia opened the innings with Andrew Symonds and Michael Bevan — giving the usual middle order batsmen time at the crease. The Symonds experiment did not last long, as he was dismissed by Hasibul Hossain for seven", "."),
        ("Its southern terminus is at an intersection with NY 383 in the village of Scottsville. The northern end of the highway is located at a junction with NY 104 in the town of Greece. NY 386 meets Interstate 490 (I 490) in Chili and NY 531", "in")
    ]
    sentences_20 = [
        ("In the absence of Bangladesh's opening bowler, Mortaza, Australia opened the innings with Andrew Symonds and Michael", "Bevan"),
        ("Its southern terminus is at an intersection with NY 383 in the village of Scottsville. The northern end of",  "the")
    ]
    
    # Select the dataset to run this test with
    sentences = sentences_20
    
    # --- 2. Define Hyperparameters & Configuration ---
    
    # Set to False for full precision (float32).
    # Note: half precision causes approximation errors.
    half_precision = False 
        
    # importance_mode: 'norm' (L1-norm) is the default as per the s-Trace method.
    importance_mode = 'norm'
    
    # Target grid sizes matching the paper's Appendix D.2
    threshold_values = [
        1e-5, 1e-4, 2e-4, 4e-4, 8e-4,
        1e-3, 1.2e-3, 1.4e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3,
        1e-2, 2e-2, 4e-2, 6e-2, 8e-2,
        1e-1, 2e-1, 4e-1, 6e-1, 8e-1
    ]

    # For larger models, you might want to save GPU memory using CPU offload
    CPU_OFFLOAD = False

    # --- 3. Initialize the Model ---
    print(f"[MAIN] Initializing model: {model_name}")
    start_time = time.time()
    
    model_type = identify_model_type(model_name)
    hf_constructor = get_model_class(model_type)
    
    # Select the appropriate dtype
    target_dtype = torch.float16 if half_precision else torch.float32
    
    if CPU_OFFLOAD:
        llm = hf_constructor.from_pretrained(
            model_name,
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=target_dtype
        )
        llm = cpu_offload(llm, execution_device="cuda:0")
    else:
        llm = hf_constructor.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=target_dtype
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[MAIN] Time to initialize {model_name}: {time.time() - start_time:.2f} seconds")
    
    # --- 4. Main Processing Loop ---
    for i, (input_sentence, gt_next) in enumerate(sentences):
        
        print(f"\n[MAIN] SENTENCE {i}: '{input_sentence}'")
        
        # --- STAGE 1: GRAPH POPULATION (GPU) ---
        print("[MAIN] Stage 1: Populating Graph (GPU)...")
        start_time = time.time()
        
        strace = LLM_STRACE(
            sentence=input_sentence,
            next_word=gt_next, # only used for computing perplexity, can be set to dummy value
            llm=llm, 
            tokenizer=tokenizer,
            track_time=True # for debugging, track the time spent in each step
        )
        print(f"[MAIN]   Time to initialize LLM_STRACE: {time.time() - start_time:.2f} s")

        start_time = time.time()
        strace.populate_graph(
            importance_mode, # how the edge importance is computed
            unit_test=True # perform test to control that the LLM integrity is preserved
            )
        print(f"[MAIN]   Time to populate graph: {time.time() - start_time:.2f} s")
        
        # --- STAGE 2: STRATA EXTRACTION (CPU) ---
        print("[MAIN] Stage 2: Extracting Strata (CPU)...")
        start_time = time.time()
        strace.extract_strace(
            threshold_values # list of target sizes that are used to extract the trace
            )
        print(f"[MAIN]   Time to extract strace: {time.time() - start_time:.2f} s")
        
        # --- STAGE 3: RECONSTRUCTION EVALUATION (GPU) ---
        print("[MAIN] Stage 3: Computing Reconstruction Error (GPU)...")
        start_time = time.time()
        strace.compute_stratum_reconstruction_error(
            do_random=True, # If true, also evaluate random graph (the random baseline from the paper)
            do_inverse=True) # If true, also evaluate when ablating ONLY the trace (fig 7, Appendix)
        print(f"[MAIN]   Time to compute stratum reco error: {time.time() - start_time:.2f} s")

        # --- Display Results ---
        strace.print_graph_sizes_and_thresholds()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main script for testing the LLM_STRACE pipeline."
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Hugging Face name of the model (e.g., "mistralai/Mistral-7B-v0.1").'
    )
    
    args = parser.parse_args()

    model_is_valid = any(args.model_name.startswith(key_model) for key_model in AVAILABLE_MODELS)
    if not model_is_valid:
        raise NotImplementedError(f"LLM_STRACE not implemented for model {args.model_name}.")

    main(args.model_name)