#!/bin/bash

# 1. Define the list of models
#    (I formatted your list into a Bash array)
MODELS=(
    "mistralai/Mistral-7B-v0.1"
    "allenai/OLMo-2-0425-1B"
    "allenai/OLMo-2-1124-7B"
    "allenai/OLMo-2-1124-13B"
    "Qwen/Qwen3-0.6B-Base"
    "Qwen/Qwen3-1.7B-Base"
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-8B-Base"
    "google/gemma-3-270m"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-32B"
    "Qwen/Qwen2-0.5B"
    "Qwen/Qwen2-1.5B"
    "Qwen/Qwen2-7B"
)

# 2. Create a directory for the manager logs
#    (These are the logs of the optimization logic itself, distinct from the Slurm logs)
mkdir -p context_logs

echo "Starting optimization managers for ${#MODELS[@]} models..."

# 3. Loop through each model and launch its manager in the background
for model in "${MODELS[@]}"; do
    
    # Create a safe filename for the log (replace / with _)
    safe_name=$(echo "$model" | tr '/' '_')
    log_file="context_logs/${safe_name}_manager.log"

    echo "Launching: $model (Log: $log_file)"

    # EXPLANATION OF COMMAND:
    # nohup ....... &  -> Keeps running even if you logout.
    # > $log_file      -> Saves the Python "print" output to a file.
    # 2>&1             -> Captures errors (stderr) to the same file.
    
    nohup python3 find_max_context_manager.py --model_name "$model" > "$log_file" 2>&1 &

    # Tiny sleep to ensure they don't all hit the scheduler at the exact same millisecond
    sleep 1

done

echo "---------------------------------------------------"
echo "All managers launched in background!"
echo "Use 'tail -f manager_logs/*.log' to monitor progress."
echo "Use 'ps -ef | grep python' to see running managers."