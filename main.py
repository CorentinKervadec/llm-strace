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

# Import the factory function to get the correct model-specific class
from src.llm_hooked.hook_constructors import get_hooked_constructor

# Import the main orchestration class
from src.llm_trace.llm_trace import LLM_STRACE

import time
import argparse # For handling command-line arguments

AVAILABLE_MODELS = [
    "mistralai/Mistral-7B-v0.1",
    'allenai/OLMo-2-0425-1B',
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-13B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B-Base",
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
        ("Its southern terminus is at an intersection with NY 383 in the village of Scottsville. The northern end of",  "the")
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
    importance_mode = 'lev_32' #'norm', 'norm_l2', 'sim', 'cosim', "ifr" "lev_k"
    
    # batch_size: How many operations to batch together during the
    # graph population step (e.g., attention decomposition).
    # Reduce the batch size if you get memory issues.
    batch_size = 32
    
    # strace_mode: The algorithm used to extract subgraphs (strata).
    # 'nucleus': (Recommended) Keeps the most important incoming edges
    #            that sum up to a cumulative mass 'tau' (like Top-P).
    # 'threshold': Keeps all edges with a weight strictly greater than 'tau'.
    strace_mode = 'threshold' #'threshold'#'nucleus'
    
    # threshold_values: A manually defined list of 'tau' values.
    # The script will extract one stratum for each value in this list.
    # Values represent the cumulative probability mass for 'nucleus' mode,
    # or the raw weight cutoff for 'threshold' mode.
    empirical_thresholds = {
        'norm_nucleus': [
            1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95,
            .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
        ],
        'norm_l2_nucleus': [
            1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95,
            .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
        ],
        'ifr_nucleus': [
            1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95,
            .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
        ],
        'ifr_threshold': [
            1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
        'sim_nucleus': [
            1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95,
            .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
        ],
        'sim_threshold': [
            1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 1e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
        'norm_threshold': [
            1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
        'norm_l2_threshold': [
            1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
        'cosim_threshold': [
            0.1, 0.2, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.49, 0.5, 0.51, 0.525, 0.55, 0.575, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0
        ],
        'lev_256_threshold':
        [
            1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
        'lev_128_threshold':
        [
            1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
        'lev_64_threshold':
        [
            1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
        'lev_32_threshold':
        [
            1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
        'lev_16_threshold':
        [
            1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
        'lev_8_threshold':
        [
            1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
        'lev_4_threshold':
        [
            1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
        'lev_2_threshold':
        [
            1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
        ],
    }
    threshold_values = empirical_thresholds['_'.join([importance_mode, strace_mode])]
    
    # --- 3. Initialize the Hooked Model ---
    
    print(f"[MAIN] Initializing hooked model: {model_name}")
    start_time = time.time()
    
    # Use the factory to get the correct class constructor
    # (e.g., Mistral_Hooked, Olmo_Hooked, etc.)
    HOOKED_CONSTRUCTOR = get_hooked_constructor(model_name)
    
    # Instantiate the hooked model. This is where the model is
    # actually downloaded (if needed) and loaded into GPU VRAM.
    llm_hooked = HOOKED_CONSTRUCTOR(
        model_name,
        half_precision=half_precision,
        untrained=untrained
    )
    print(f"[MAIN] Time to initialise LLM_Hooked: {time.time() - start_time:.2f} seconds")
    
    # Enable internal performance timers within the llm_hooked object.
    llm_hooked.turn_time_tracking_on()
    
    # Enable internal sanity checks (e.g., asserting that
    # decomposed vectors sum back to the original vector).
    llm_hooked.turn_test_mode_on()

    # --- 4. Main Processing Loop ---
    # Iterate over each sentence in the selected test dataset.
    
    for i, (input_sentence, gt_next) in enumerate(sentences):
        
        # --- 4a. Register Hooks ---
        # Ensure the PyTorch hooks are attached to the model.
        # This is inside the loop because they are removed
        # later to compute reconstruction error.
        if not llm_hooked.extraction_hook_registred():
            llm_hooked.register_extraction_hooks()

        print(f"\n[MAIN] SENTENCE {i}: '{input_sentence}'")
        
        # -----------------------------------------------------------------
        # STAGE 1: GRAPH POPULATION (GPU-HEAVY)
        # -----------------------------------------------------------------
        print("[MAIN] Stage 1: Populating Graph (GPU)...")
        
        # Initialise the main LLM_STRACE object for this sentence.
        # It holds the model, the input, and will manage the graph.
        start_time = time.time()
        strace = LLM_STRACE((input_sentence, gt_next), llm_hooked, track_time=True)
        print(f"[MAIN]   Time to initialise LLM_STRACE: {time.time() - start_time:.2f} s")

        # Create the graph structure (nodes and edges, but no weights yet).
        start_time = time.time()
        strace.initialize_graph(importance_mode)
        print(f"[MAIN]   Time to initialise LLM_Graph_NX: {time.time() - start_time:.2f} s")
        
        # This is the most compute-intensive part of Stage 1.
        # It runs a full forward pass, uses the hooks to capture
        # all activations, and computes the importance weight for
        # every edge in the graph.
        start_time = time.time()
        strace.populate_graph(batch_size)
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
        
        # CRITICAL STEP: The extraction hooks (which capture activations)
        # MUST be removed before we run the model again for evaluation.
        # Otherwise, the model's forward pass will be intercepted
        # and we can't get a clean "reconstruction" run.
        llm_hooked.remove_extraction_hooks()

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