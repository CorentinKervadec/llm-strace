import random
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import the factory function to get the correct model-specific class
from src.llm_hooked.hook_constructors import get_hooked_constructor

# Import the main orchestration class
from src.llm_trace.llm_trace import LLM_STRACE

import argparse # For handling command-line arguments

def html_colored_print(pre_text, token_color_tuples, output_file="colored_output.html"):
    """
    Generates an HTML file with tokens colored based on their associated values, using hard-coded bins and colors.
    On mouse hover, the alternatives (third element of the tuple, a list of tokens) for each token are shown as a tooltip.

    Args:
        pre_text (str): Initial text to print before the tokens.
        token_color_tuples (list): List of tuples [(token, value, alternatives), ...].
        output_file (str): Path to the output HTML file.
    """
    # Define bins and corresponding colors (hex or RGB)
    bins = [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_colors = [
        "#440154",  # for <= 0.1
        "#3b528b",  # for <= 0.2
        "#21918c",  # for <= 0.4
        "#5ec962",  # for <= 0.6
        "#fde725",  # for <= 0.7
        "#fee08b",  # for <= 0.8
        "#fdae61",  # for <= 0.9
        "#d90429",  # for <= 1.0
    ]

    # Bin the values: assign each value to the smallest bin that is >= value
    values = [max(0, value) for _, value, *_ in token_color_tuples]
    binned_indices = []
    for value in values:
        idx = 0
        while idx < len(bins) and value > bins[idx]:
            idx += 1
        if idx >= len(bins):
            idx = len(bins) - 1
        binned_indices.append(idx)

    # Build HTML content
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Colored Output</title></head><body style='font-family:monospace;'>")

    # Color scale legend
    html.append("<div><b>Color scale:</b> ")
    for bin_val, color in zip(bins, bin_colors):
        html.append(f"<span style='color:{color};'>{bin_val:.0e}</span> ")
    html.append("</div><br>")

    # Pre-text
    html.append(f"<span>{pre_text}</span>")

    # Colored tokens with tooltip (title attribute)
    for i, item in enumerate(token_color_tuples):
        # Support both (token, value) and (token, value, alternatives)
        if len(item) == 3:
            token, value, alternatives = item
        else:
            token, value = item
            alternatives = []
        bin_idx = binned_indices[i]
        if value == -1:
            html.append(token)
        else:
            if alternatives:
                # Replace actual newlines with the string '\n' for display in tooltip
                alt_str = "&#10;".join(str(alt).replace('\n', '\\n') for alt in alternatives)
                # Use data-tooltip for custom tooltip styling if needed
                html.append(
                    f"<span style='color:{bin_colors[bin_idx]};' title='{alt_str}'>{token}</span>"
                )
            else:
                html.append(
                    f"<span style='color:{bin_colors[bin_idx]};'>{token}</span>"
                )

    html.append("</body></html>")

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("".join(html))
    print(f"HTML colored output written to {output_file}")


def main(model_name, init_text, generation_length, context_length):
    """
    Main function to run the tracing and evaluation pipeline on a
    few test sentences.
    
    Args:
        model_name (str): The Hugging Face name of the model to load
                          (e.g., 'mistralai/Mistral-7B-v0.1').
    """
        
    # --- 2. Define Hyperparameters & Configuration ---
    
    # half_precision: If True (recommended), load the model in float16 to save VRAM.
    # Set to False for higher precision (float32).
    # For some reason, half_precision might cause approximation error that you don't have with full precision
    half_precision = True 
    
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
    batch_size = 4
    
    # strace_mode: The algorithm used to extract subgraphs (strata).
    # 'nucleus': (Recommended) Keeps the most important incoming edges
    #            that sum up to a cumulative mass 'tau' (like Top-P).
    # 'threshold': Keeps all edges with a weight strictly greater than 'tau'.
    strace_mode = 'nucleus'
    
    # threshold_values: A manually defined list of 'tau' values.
    # The script will extract one stratum for each value in this list.
    # Values represent the cumulative probability mass for 'nucleus' mode,
    # or the raw weight cutoff for 'threshold' mode.
    threshold_values = [
        1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95,
        .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
    ]
    
    # --- 3. Initialize the Hooked Model ---
    
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
    

    # --- 4. Main Processing Loop ---
    # Iterate over each sentence in the selected test dataset.

    stop_generation = False
    input_sentence = init_text

    generated_token_w_auc = []

    while(not stop_generation):
        # --- 4a. Register Hooks ---
        # Ensure the PyTorch hooks are attached to the model.
        # This is inside the loop because they are removed
        # later to compute reconstruction error.
        if not llm_hooked.extraction_hook_registred():
            llm_hooked.register_extraction_hooks()

        trimmed_tokenized = llm_hooked.tokenizer.encode(input_sentence)
        # print("trimmed_tokenized", trimmed_tokenized)
        input_sentence = llm_hooked.tokenizer.decode(trimmed_tokenized[-context_length:])

        print(f"\n[MAIN] SENTENCE: '{input_sentence}'")
        
        # Initialise the main LLM_STRACE object for this sentence.
        # It holds the model, the input, and will manage the graph.
        strace = LLM_STRACE((input_sentence, 'xxx'), llm_hooked, track_time=False)

        # Create the graph structure (nodes and edges, but no weights yet).
        strace.initialize_graph(importance_mode)
        
        # This is the most compute-intensive part of Stage 1.
        # It runs a full forward pass, uses the hooks to capture
        # all activations, and computes the importance weight for
        # every edge in the graph.
        strace.populate_graph(batch_size)
        
        # -----------------------------------------------------------------
        # STAGE 2: STRATA EXTRACTION (CPU-HEAVY)
        # -----------------------------------------------------------------
        
        # This is a CPU-bound operation. It runs graph traversal
        # algorithms (e.g., backward BFS) to find the 'nucleus'
        # subgraphs for each threshold value.
        strace.extract_strace(threshold_values, mode=strace_mode)
        
        # -----------------------------------------------------------------
        # STAGE 3: RECONSTRUCTION EVALUATION (GPU-HEAVY)
        # -----------------------------------------------------------------
        
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
        strace.compute_stratum_reconstruction_error(do_random=False, do_inverse=False)

        nucleus_60 = strace.strata_nucleus_60['trace']['only'][-1]
        next_token = random.sample(nucleus_60, 1)[0] # sample from the nucleus

        input_sentence += next_token

        # compute tv auc
        rel_stratum_sizes = strace.strata_rel_size
        sort_size_indices = np.argsort(rel_stratum_sizes)
        size_sorted = np.array(rel_stratum_sizes)[sort_size_indices]
        tv_sorted = np.array(strace.strata_reco_tv['trace']['only'])[sort_size_indices]
        metric_auc = np.trapz(tv_sorted, size_sorted)

        # add to the generation tuple list
        generated_token_w_auc.append((next_token, metric_auc, nucleus_60))
        # print(generated_token_w_auc)

        if len(generated_token_w_auc) >= generation_length:
            stop_generation = True

    return generated_token_w_auc

if __name__ == "__main__":

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
    
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Prefix text used for the generation'
    )

    parser.add_argument(
        '--length', 
        type=int, 
        required=True, 
        help='Length of the text to generate.'
    )

    parser.add_argument(
        '--context_length', 
        type=int, 
        required=True, 
        help='Length of the sequence fed to the LLM.'
    )

    parser.add_argument(
        '--output_prefix',
        type=str,
        required=True,
        help='Output predix'
    )

    # Parse the arguments provided by the user
    args = parser.parse_args()

    generation = main(args.model_name, args.prompt, args.length, args.context_length)

    html_colored_print(args.prompt, generation, output_file=f"{args.output_prefix}.html")

