# Import the factory function to get the correct model-specific class
from src.llm_hooked.hook_constructors import get_hooked_constructor

# Import the main orchestration class
from src.llm_trace.llm_trace import LLM_STRACE

import time
import argparse # For handling command-line arguments

def main(model_name):

    # --- 1. Define Test Data ---
    sentence_test = "Let's test the graph extraction code."

    # --- 2. Define Hyperparameters & Configuration ---
    
    # half_precision: If True (recommended), load the model in float16 to save VRAM.
    # Set to False for higher precision (float32).
    # For some reason, half_precision might cause approximation error that you don't have with full precision
    half_precision = False 
    
    # untrained: If True, load a "blank" model with randomized weights
    # for debugging or analysis. /!\ Not tested
    untrained = False
    
    # batch_size: How many operations to batch together during the
    # graph population step (e.g., attention decomposition).
    # Reduce the batch size if you get memory issues.
    batch_size = 32

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

    # --- 4. Register Hooks ---
    # Ensure the PyTorch hooks are attached to the model.
    # This is inside the loop because they are removed
    # later to compute reconstruction error.
    if not llm_hooked.extraction_hook_registred():
        llm_hooked.register_extraction_hooks()

    model_input = llm_hooked.tokenizer([sentence_test], padding=True, return_tensors="pt")
    llm_hooked.forward_pass(model_input, batch_size, output_pred=True)

    print("\n\n\tᐠ( ᐛ )ᐟ\n\nYay, hooked extraction run with success! :-)")

if __name__ == "__main__":

        # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Main script for testing the LLM_HOOK extraction."
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

    main(args.model_name)