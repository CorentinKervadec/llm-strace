#!/bin/bash
#SBATCH --job-name=strace-cpu    # Job name
#SBATCH --output=slurm_logs/strata_%A_%a.out # Standard output and error log
#SBATCH --nodes=1                # Run all processes on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=64G                # Job memory request
#SBATCH --time=05:00:00          # Time limit (4 hours, adjust for 100 sentences)
#SBATCH --partition=high-cpu          # Run on the 'cpu' partition (adjust as needed)

# Explicitly set Hugging Face cache variables
export HF_HOME="/scratch/lab_mbaroni/shared_data/hf_cache"
export TRANSFORMERS_CACHE="/scratch/lab_mbaroni/shared_data/hf_cache"
export HF_HUB_CACHE="/scratch/lab_mbaroni/shared_data/hf_cache"

# verify environment variable in job logs
echo "Using HF_HOME: $HF_HOME"


# TODO
ENV_PATH='/data/samanthafs/projects/lab_mbaroni/ckervadec/conda_envs/llm-trace'
LLM_STRACE_PATH='/homes/users/ckervadec/llm-strace'

# --- This script is submitted by submit_jobs.sh ---
# --- It expects $SLURM_ARRAY_TASK_ID to be set (as chunk_id) ---

echo "--- Starting CPU Job Chunk $SLURM_ARRAY_TASK_ID ---"
date

# Activate the Python environment
# load miniconda
ml modulepath/UPF/apps
ml Miniconda3

# Initialize Conda for non-interactive bash subshells
eval "$(conda shell.bash hook)"
conda activate $ENV_PATH

cd $LLM_STRACE_PATH

# Get arguments from the environment variables set in submit_jobs.sh
: "${INTERMEDIATE_DIR:?INTERMEDIATE_DIR not set}"
: "${STRACE_DIR:?STRACE_DIR not set}"
: "${CHUNK_SIZE:?CHUNK_SIZE not set}"
: "${TOTAL_SENTENCES:?TOTAL_SENTENCES not set}"
: "${IMPORTANCE:?IMPORTANCE not set}"

# Run the CPU-bound Python script
python 2_stratification_cpu.py \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --chunk_size $CHUNK_SIZE \
    --total_sentences $TOTAL_SENTENCES \
    --intermediate_dir $INTERMEDIATE_DIR \
    --strace_dir $STRACE_DIR \
    --importance $IMPORTANCE \
    --cleanup

echo "--- Finished CPU Job Chunk $SLURM_ARRAY_TASK_ID ---"
date

