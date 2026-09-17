#!/bin/bash
#SBATCH --job-name=eval-gpu  # Job name
#SBATCH --output=slurm_logs/eval_%A_%a.out # Standard output and error log
#SBATCH --nodes=1                # Run all processes on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=64G                # Job memory request
#SBATCH --time=36:00:00          # Time limit 
#SBATCH --gres=gpu:1

# TODO
ENV_PATH='/data/samanthafs/projects/lab_mbaroni/ckervadec/conda_envs/llm-trace'
LLM_STRACE_PATH='/homes/users/ckervadec/llm-strace'

# --- This script is submitted by submit_jobs.sh ---
# --- It expects $SLURM_ARRAY_TASK_ID to be set (as chunk_id) ---

echo "--- Starting GPU Job Stage 3 (Reco Error) Chunk $SLURM_ARRAY_TASK_ID ---"
date

# Activate the Python environment
# load miniconda
ml modulepath/UPF/apps
ml Miniconda3
# load conda env
source activate $ENV_PATH

cd $LLM_STRACE_PATH

# Get arguments from the environment variables set in submit_jobs.sh
: "${STRACE_DIR:?STRACE_DIR not set}"
: "${FINAL_DIR:?FINAL_DIR not set}"
: "${CHUNK_SIZE:?CHUNK_SIZE not set}"
: "${TOTAL_SENTENCES:?TOTAL_SENTENCES not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
: "${CHECKPOINT:?CHECKPOINT not set}"
: "${CPU_OFFLOAD:?CPU_OFFLOAD not set}"

# Initialize the command in an array
CMD_ARGS=(
    --model_name "$MODEL_NAME"
    --chunk_id "$SLURM_ARRAY_TASK_ID"
    --chunk_size "$CHUNK_SIZE"
    --total_sentences "$TOTAL_SENTENCES"
    # --strace_dir "$STRACE_DIR"
    --final_dir "$FINAL_DIR"
    --checkpoint "$CHECKPOINT"
)

# Conditionally add the flag
if [ "$CPU_OFFLOAD" = "1" ]; then
    CMD_ARGS+=(--cpu_offload)
fi

# Run the Stage 3 GPU-bound Python script
python 3_evaluate_gpu.py "${CMD_ARGS[@]}"
# python 3_patch_nucleus_tokens.py "${CMD_ARGS[@]}"
# python 3_extract_last_hidden.py "${CMD_ARGS[@]}"

echo "--- Finished GPU Job Stage 3 (Reco Error) Chunk $SLURM_ARRAY_TASK_ID ---"
date

