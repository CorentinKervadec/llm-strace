#!/bin/bash
#SBATCH --job-name=eval-gpu-gen  # Job name
#SBATCH --output=slurm_logs_gen/3_gpu_%A_%a.out
#SBATCH --nodes=1                # Run all processes on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=32G                # Job memory request
#SBATCH --time=24:00:00          # Time limit (2 hours, adjust for 100 sentences)
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --partition=alien
#SBATCH --qos=alien

# --- Environment Variables (Provided by Master Script) ---
# MODEL_NAME
# STRACE_DIR
# FINAL_DIR

echo "Starting Gen Stage 3 for Prompt Index $SLURM_ARRAY_TASK_ID"

# Loading CUDA module
module --ignore-cache load CUDA/11.8

# Activate the Python environment
source activate unnatural_prompt

python 3_gen_evaluate_gpu.py \
    --model_name "$MODEL_NAME" \
    --prompt_index $SLURM_ARRAY_TASK_ID \
    --strace_dir "$STRACE_DIR" \
    --final_dir "$FINAL_DIR"

echo "Finished Gen Stage 3"