#!/bin/bash
#SBATCH --job-name=strace-cpu-gen    # Job name
#SBATCH --output=slurm_logs_gen/2_cpu_%A_%a.out
#SBATCH --nodes=1                # Run all processes on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=64G                # Job memory request
#SBATCH --time=24:00:00          # Time limit (4 hours, adjust for 100 sentences)
#SBATCH --gpus=0                 # Explicitly request 0 GPUs
#SBATCH --partition=high          # Run on the 'cpu' partition (adjust as needed)



# --- Environment Variables (Provided by Master Script) ---
# INTERMEDIATE_DIR
# STRACE_DIR
# STRACE_MODE
# IMPORTANCE_MODE

echo "Starting Gen Stage 2 for Prompt Index $SLURM_ARRAY_TASK_ID"

# Activate the Python environment
source activate unnatural_prompt

python 2_gen_stratification_cpu.py \
    --intermediate_dir "$INTERMEDIATE_DIR" \
    --prompt_index $SLURM_ARRAY_TASK_ID \
    --strace_dir "$STRACE_DIR" \
    --strace "$STRACE_MODE" \
    --importance "$IMPORTANCE_MODE"

echo "Finished Gen Stage 2"