"""
Main testing script for the LLM_STRACE (Language Model Stratified/Soft Tracing)
pipeline.

This script is designed for local testing and debugging. It:
1. Loads a specified language model with hooks.
2. Processes a small, hard-coded set of test sentences.
3. (GPU) Populates a full computation graph with edge importance scores.
4. (CPU) Extracts "strata" (subgraphs) from the full graph based on a set of thresholds.
5. (GPU) Evaluates the predictive power of each stratum by measuring reconstruction error.
"""

# Import the main orchestration class
from src.llm_trace.llm_trace_2 import LLM_STRACE
from src.modified_transformers.utils import get_model_class, identify_model_type, load_gemma3
from transformers import AutoTokenizer
from src.llm_trace.llm_trace_2 import THRESHOLD_STRACE
from accelerate import cpu_offload

import time
import argparse # For handling command-line arguments
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
    "google/gemma-3-270m",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen2-7B",
    "google/gemma-3-4b-pt",
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.1",
    "deepseek-ai/deepseek-llm-7b-base",
    "microsoft/phi-4",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf"
]

def main(model_name):
    """
    Main function to run the tracing and evaluation pipeline on a
    few test sentences.
    
    Args:
        model_name (str): The Hugging Face name of the model to load
                          (e.g., 'mistralai/Mistral-7B-v0.1').
    """
    
    # --- 1. Define Test Data ---
    # These are small, hard-coded datasets for quick local testing.
    # Each item is a tuple: (input_prompt, ground_truth_next_token)
    
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
        # ("Its southern terminus is at an intersection with NY 383 in the village of Scottsville. The northern end of",  "the")
    ]
    
    # --- Select the dataset to run this test with ---
    sentences = sentences_20
    
    # --- 2. Define Hyperparameters & Configuration ---
    
    # half_precision: If True (recommended), load the model in float16 to save VRAM.
    # Set to False for higher precision (float32).
    # For some reason, half_precision might cause approximation error that you don't have with full precision
    half_precision = False 
    
    # untrained: If True, load a "blank" model with randomized weights
    # for debugging or analysis. /!\ Not tested
    untrained = False
    
    # importance_mode: How to calculate edge weights in the graph.
    # 'norm': Use the L1-norm of the contribution vector.
    # todo: implement more importance modes.
    importance_mode = 'norm'
    
    # batch_size: How many operations to batch together during the
    # graph population step (e.g., attention decomposition).
    # Reduce the batch size if you get memory issues.
    batch_size = 1
    
    # strace_mode: The algorithm used to extract subgraphs (strata).
    # 'nucleus': (Recommended) Keeps the most important incoming edges
    #            that sum up to a cumulative mass 'tau' (like Top-P).
    # 'threshold': Keeps all edges with a weight strictly greater than 'tau'.
    strace_mode = 'size' #'threshold'
    
    # threshold_values: A manually defined list of 'tau' values.
    # The script will extract one stratum for each value in this list.
    # Values represent the cumulative probability mass for 'nucleus' mode,
    # or the raw weight cutoff for 'threshold' mode.
    # threshold_values = [
    #     1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95,
    #     .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
    # ]
    # threshold_values = [
    #     -1.0, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    # ]

    if strace_mode == 'size':
        threshold_values = [
            1e-5,
            1e-4, 2e-4, 4e-4, 8e-4,
            1e-3, 1.2e-3, 1.4e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3,
            1e-2, 2e-2, 4e-2, 6e-2, 8e-2,
            1e-1, 2e-1, 4e-1, 6e-1, 8e-1
        ]
    else:
        threshold_values = THRESHOLD_STRACE['_'.join([importance_mode, strace_mode])]
    FREEZE = None #'mlp'

    CPU_OFFLOAD = False

    # --- 3. Initialize the Hooked Model ---
    
    print(f"[MAIN] Initializing hooked model: {model_name}")
    start_time = time.time()
    
    # Use the factory to get the correct class constructor
    model_type = identify_model_type(args.model_name)
    hf_constructor = get_model_class(model_type)
    if CPU_OFFLOAD:
        if 'gemma-3' in args.model_name:
            llm = load_gemma3(model_name, half_precision, cuda=False)
        else:
            llm = hf_constructor.from_pretrained(
                args.model_name,
                # device_map="cpu",
                attn_implementation="eager",
                use_safetensors=True,
                torch_dtype=torch.float16 if half_precision else torch.float32
            )
        llm = cpu_offload(llm, execution_device="cuda:0")
    else:
        if 'gemma-3' in args.model_name:
            llm = load_gemma3(model_name, half_precision, cuda=True)
        else:
            llm = hf_constructor.from_pretrained(
                args.model_name,
                device_map="auto",
                # max_memory={0: "22GB", "cpu": "60GB"}, # Leave 2GB GPU RAM free for your hidden_states/activations
                attn_implementation="eager",
                use_safetensors=True,
                torch_dtype=torch.float16 if half_precision else torch.float32
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"[MAIN] Time to initialise {model_name}: {time.time() - start_time:.2f} seconds")
    
    # --- 4. Main Processing Loop ---
    # Iterate over each sentence in the selected test dataset.
    
    for i, (input_sentence, gt_next) in enumerate(sentences):
        
        print(f"\n[MAIN] SENTENCE {i}: '{input_sentence}'")
        
        # -----------------------------------------------------------------
        # STAGE 1: GRAPH POPULATION (GPU-HEAVY)
        # -----------------------------------------------------------------
        print("[MAIN] Stage 1: Populating Graph (GPU)...")
        
        # Initialise the main LLM_STRACE object for this sentence.
        # It holds the model, the input, and will manage the graph.
        start_time = time.time()
        strace = LLM_STRACE(
            sentence=input_sentence,
            next_word=gt_next, 
            llm=llm, 
            tokenizer=tokenizer,
            track_time=True)
        # strace.freeze_strace(FREEZE)
        print(f"[MAIN]   Time to initialise LLM_STRACE: {time.time() - start_time:.2f} s")

        # This is the most compute-intensive part of Stage 1.
        # It runs a full forward pass, uses the hooks to capture
        # all activations, and computes the importance weight for
        # every edge in the graph.
        start_time = time.time()
        strace.populate_graph(batch_size, importance_mode, unit_test=True)
        print(f"[MAIN]   Time to populate graph with importance: {time.time() - start_time:.2f} s")
        
        # -----------------------------------------------------------------
        # STAGE 2: STRATA EXTRACTION (CPU-HEAVY)
        # -----------------------------------------------------------------
        print("[MAIN] Stage 2: Extracting Strata (CPU)...")
        
        # This is a CPU-bound operation. It runs graph traversal
        # algorithms (e.g., backward BFS) to find the 'nucleus'
        # subgraphs for each threshold value.
        start_time = time.time()
        strace.extract_strace(threshold_values, mode=strace_mode)
        print(f"[MAIN]   Time to extract strace: {time.time() - start_time:.2f} s")
        
        # -----------------------------------------------------------------
        # STAGE 3: RECONSTRUCTION EVALUATION (GPU-HEAVY)
        # -----------------------------------------------------------------
        print("[MAIN] Stage 3: Computing Reconstruction Error (GPU)...")
        
        # This is the second GPU-heavy part.
        # It loops through all the strata (subgraphs) extracted in Stage 2.
        # For each stratum, it "masks" the model to *only* use the
        # components in that subgraph and runs a new forward pass.
        # It then compares the resulting logits to the original
        # full-model logits to measure reconstruction error/loss.
        # It also reproduce the ablation with random graph extraction
        # and *inverse* ablation (masking the components that belongs to the subgraph).
        start_time = time.time()
        strace.compute_stratum_reconstruction_error(do_random=True, do_inverse=True)
        print(f"[MAIN]   Time to compute stratum reco error: {time.time() - start_time:.2f} s")

        # --- 4d. Display Results ---
        strace.print_graph_sizes_and_thresholds()

        print("\n----------- Entropy -----------")
        print(strace.strata_entropy)
    
if __name__ == "__main__":
    """
    Standard Python entry point.
    This code runs when the script is executed directly.
    """
    
    # --- 5. Command-Line Argument Parsing ---
    
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Main script for testing the LLM_STRACE pipeline."
    )
    
    # Define the arguments this script accepts
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Hugging Face name of the model (e.g., "mistralai/Mistral-7B-v0.1").'
    )
    
    # Parse the arguments provided by the user
    args = parser.parse_args()

    model_is_valid = False
    for key_model in AVAILABLE_MODELS:
        if args.model_name.startswith(key_model):
            model_is_valid = True
            break
    if not model_is_valid:
        raise NotImplementedError(f"LLM_STRACE not implemented for model {args.model_name}.")

    # Call the main function, passing in the parsed model name
    main(args.model_name)
