#!/bin/bash
#SBATCH --job-name=strace-cpu    # Job name
#SBATCH --output=slurm_logs/strata_%A_%a.out # Standard output and error log
#SBATCH --nodes=1                # Run all processes on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=64G                # Job memory request
#SBATCH --time=05:00:00          # Time limit (4 hours, adjust for 100 sentences)
#SBATCH --gpus=0                 # Explicitly request 0 GPUs
#SBATCH --partition=medium          # Run on the 'cpu' partition (adjust as needed)


# --- This script is submitted by submit_jobs.sh ---
# --- It expects $SLURM_ARRAY_TASK_ID to be set (as chunk_id) ---

echo "--- Starting CPU Job Chunk $SLURM_ARRAY_TASK_ID ---"
date

# Load necessary modules (e.g., Python)
# module load python/3.10

# Activate the Python environment
source activate unnatural_prompt

# Get arguments from the environment variables set in submit_jobs.sh
: "${INTERMEDIATE_DIR:?INTERMEDIATE_DIR not set}"
: "${STRACE_DIR:?STRACE_DIR not set}"
: "${CHUNK_SIZE:?CHUNK_SIZE not set}"
: "${TOTAL_SENTENCES:?TOTAL_SENTENCES not set}"

# Run the CPU-bound Python script
python 2_stratification_cpu.py \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --chunk_size $CHUNK_SIZE \
    --total_sentences $TOTAL_SENTENCES \
    --intermediate_dir $INTERMEDIATE_DIR \
    --strace_dir $STRACE_DIR

echo "--- Finished CPU Job Chunk $SLURM_ARRAY_TASK_ID ---"
date

