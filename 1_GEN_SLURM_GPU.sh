#!/bin/bash
#SBATCH --job-name=strace-gpu-gen    # Job name
#SBATCH --output=slurm_logs_gen/1_gpu_%A_%a.out
#SBATCH --nodes=1                # Run all processes on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=64G                # Job memory request
#SBATCH --time=24:00:00          # Time limit (2 hours, adjust for 100 sentences)
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --partition=alien
#SBATCH --qos=alien


# --- Environment Variables (Provided by Master Script) ---
# MODEL_NAME
# IMPORTANCE_MODE
# PROMPTS_FILE
# INTERMEDIATE_DIR
# MAX_NEW_TOKENS
# BATCH_SIZE
# P_SAMPLE
# SEED

echo "Starting Gen Stage 1 for Prompt Index $SLURM_ARRAY_TASK_ID"

# Loading CUDA module
module --ignore-cache load CUDA/11.8

# Activate the Python environment
source activate unnatural_prompt

python 1_gen_extraction_gpu.py \
    --model_name "$MODEL_NAME" \
    --importance_mode "$IMPORTANCE_MODE" \
    --prompt_index $SLURM_ARRAY_TASK_ID \
    --prompts_file "$PROMPTS_FILE" \
    --intermediate_dir "$INTERMEDIATE_DIR" \
    --max_new_tokens ${MAX_NEW_TOKENS:-20} \
    --batch_size ${BATCH_SIZE:-1} \
    --p_sample ${P_SAMPLE:-0.6} \
    --seed $SEED

echo "Finished Gen Stage 1"