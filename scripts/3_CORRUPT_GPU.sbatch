#!/bin/bash
#SBATCH --job-name=eval-corrupt-gpu  # Job name
#SBATCH --output=slurm_logs/eval_corrupt_%A_%a.out # Standard output and error log
#SBATCH --nodes=1                # Run all processes on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=64G                # Job memory request
#SBATCH --time=36:00:00          # Time limit 
#SBATCH --gres=gpu:1

# Explicitly set Hugging Face cache variables
export HF_HOME="/scratch/lab_mbaroni/shared_data/hf_cache"
export TRANSFORMERS_CACHE="/scratch/lab_mbaroni/shared_data/hf_cache"
export HF_HUB_CACHE="/scratch/lab_mbaroni/shared_data/hf_cache"

# Verify environment variable in job logs
echo "Using HF_HOME: $HF_HOME"

ENV_PATH='/data/samanthafs/projects/lab_mbaroni/ckervadec/conda_envs/llm-trace'
LLM_STRACE_PATH='/homes/users/ckervadec/llm-strace'

# # --- Arguments passed via command line ---
# MODEL_NAME="${1:?Error: Arg 1 (MODEL_NAME) is required}"
# CSV_PATH="${2:?Error: Arg 2 (CSV_PATH) is required}"
# STRACE_DIR="${3:?Error: Arg 3 (STRACE_DIR) is required}"
# OUTPUT_DIR="${4:?Error: Arg 4 (OUTPUT_DIR) is required}"
# CHUNK_SIZE="${5:-100}" # Default chunk_size to 100 if omitted

# echo "--- Starting GPU Job Corrupted Evaluation Chunk $SLURM_ARRAY_TASK_ID ---"
# date

# Activate the Python environment
ml modulepath/UPF/apps
ml Miniconda3
source activate $ENV_PATH

cd $LLM_STRACE_PATH

# Run the Corrupted Context GPU Python script
# python 3_evaluate_corrupted_gpu.py \
#     --model_name "$MODEL_NAME" \
#     --csv_path "$CSV_PATH" \
#     --strace_dir "$STRACE_DIR" \
#     --output_dir "$OUTPUT_DIR" \
#     --chunk_id "$SLURM_ARRAY_TASK_ID" \
#     --chunk_size "$CHUNK_SIZE"

python 3_evaluate_corrupted_gpu.py \
    --model_name "allenai/OLMo-2-1124-7B" \
    --csv_path "data/corrupted/rand_pos_succ/wikitext_rand_pos_successive_0.csv" \
    --strace_dir "/homes/users/ckervadec/projects/llm-strace/OLMo-2-1124-7B/main/final_straces_40/" \
    --output_dir "/homes/users/ckervadec/scratch/llm-strace/results_corruption/OLMo-2-1124-7B/rand_pos_succ/wikitext_rand_pos_successive_0/" \
    --chunk_id "0" \
    --chunk_size "10" \
    --total_sentences "100"


# echo "--- Finished GPU Job Corrupted Evaluation Chunk $SLURM_ARRAY_TASK_ID ---"
# date