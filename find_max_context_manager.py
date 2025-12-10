import subprocess
import argparse
import os
from datetime import datetime

# Configuration
SLURM_SCRIPT = "find_max_context_sbatch.sh" 
MIN_CTX = 40
MAX_CTX = 500

def get_log_filename(model_name):
    """Creates a safe filename from the model name."""
    # Replace slashes (common in HF model names) with underscores
    safe_name = model_name.replace("/", "_")
    return f"context_logs/{safe_name}_context_search.txt"

def log_message(filename, message):
    """Prints to console and appends to the log file."""
    print(message)  # Print to console (stdout)
    
    # Append to file
    with open(filename, "a") as f:
        # Add timestamp to file entries for better debugging
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")

def run_trial(context_len, model_name, log_file):
    """
    Submits a Slurm job and waits for it to finish.
    Returns: True if success, False if OOM/Failed.
    """
    log_message(log_file, f"\n[Manager] Submitting job for context length: {context_len}")
    
    cmd = [
        "sbatch", 
        "--wait",              
        "--kill-on-invalid-dep=yes",
        SLURM_SCRIPT,          
        model_name,             # Argument 1 for the wrapper
        str(context_len),       # Argument 2 for the wrapper
    ]

    try:
        # Run and wait
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        # Check Exit Code
        if result.returncode == 0:
            log_message(log_file, f"[Manager] Job Success! Context {context_len} fits.")
            return True
        else:
            log_message(log_file, f"[Manager] Job Failed (Exit Code {result.returncode}).")
            
            # If sbatch itself failed (not the python code, but the scheduler), log that error
            if result.stderr:
                log_message(log_file, f"[Manager] Stderr debug: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        log_message(log_file, f"[Manager] System Error: {e}")
        return False

def binary_search_max_context(low, high, model_name, log_file):
    """Finds the largest working context length."""
    best_valid = low
    
    while low <= high:
        mid = (low + high) // 2
        
        log_message(log_file, f"[Binary Search] Checking range [{low}, {high}] -> Testing: {mid}")
        
        if run_trial(mid, model_name, log_file):
            best_valid = mid
            low = mid + 1  # Try larger
        else:
            high = mid - 1 # Try smaller
            
    return best_valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    # Setup log file
    log_file = get_log_filename(args.model_name)
    
    # Initialize file with a header
    with open(log_file, "w") as f:
        f.write(f"=== Optimization Run for {args.model_name} ===\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write("============================================\n")

    log_message(log_file, "Starting optimization run...")
    
    max_len = binary_search_max_context(MIN_CTX, MAX_CTX, args.model_name, log_file)
    
    log_message(log_file, "--------------------------------------------")
    log_message(log_file, f"FINAL RESULT: Maximum stable context length is {max_len}")
    log_message(log_file, "--------------------------------------------")