#!/bin/bash
#SBATCH --job-name=strace-gpu    # Job name
#SBATCH --output=slurm_logs/extract_%A_%a.out # Standard output and error log
#SBATCH --nodes=1                # Run all processes on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=64G                # Job memory request
#SBATCH --time=05:00:00          # Time limit (2 hours, adjust for 100 sentences)
#SBATCH --gpus=1                 # Request 1 GPU



# --- This script is submitted by submit_jobs.sh ---
# --- It expects $SLURM_ARRAY_TASK_ID to be set (as chunk_id) ---

echo "--- Starting GPU Job Chunk $SLURM_ARRAY_TASK_ID ---"
date

# Load necessary modules (e.g., Python, CUDA)
# module load python/3.10
# module load cuda/11.8

# Loading CUDA module
module --ignore-cache load CUDA/11.8

# Activate the Python environment
source activate unnatural_prompt

# Get arguments from the environment variables set in submit_jobs.sh
: "${DATA_FILE:?DATA_FILE not set}"
: "${INTERMEDIATE_DIR:?INTERMEDIATE_DIR not set}"
: "${CHUNK_SIZE:?CHUNK_SIZE not set}"
: "${TOTAL_SENTENCES:?TOTAL_SENTENCES not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
: "${IMPORTANCE:?IMPORTANCE not set}"

# Run the GPU-bound Python script
python 1_extraction_gpu.py \
    --model_name $MODEL_NAME \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --chunk_size $CHUNK_SIZE \
    --total_sentences $TOTAL_SENTENCES \
    --data_file $DATA_FILE \
    --intermediate_dir $INTERMEDIATE_DIR \
    --importance $IMPORTANCE

echo "--- Finished GPU Job Chunk $SLURM_ARRAY_TASK_ID ---"
date

