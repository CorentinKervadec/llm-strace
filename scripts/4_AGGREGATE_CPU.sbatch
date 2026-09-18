#!/bin/bash
#SBATCH --job-name=aggregate    # Job name
#SBATCH --output=slurm_logs/strata_%A_%a.out # Standard output and error log
#SBATCH --nodes=1                # Run all processes on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=64G                # Job memory request
#SBATCH --time=05:00:00          # Time limit (4 hours, adjust for 100 sentences)
#SBATCH --partition=high-cpu          # Run on the 'cpu' partition (adjust as needed)

# TODO
ENV_PATH='/data/samanthafs/projects/lab_mbaroni/ckervadec/conda_envs/llm-trace'
LLM_STRACE_PATH='/homes/users/ckervadec/llm-strace'

# Arguments
BASE_RESULT_DIR=$1
LENGTH=$2
MAX_FILES=$3

echo "--- Starting CPU strace aggregation ---"
date

# Activate the Python environment
# load miniconda
ml modulepath/UPF/apps
ml Miniconda3

# Initialize Conda for non-interactive bash subshells
eval "$(conda shell.bash hook)"
conda activate $ENV_PATH

cd $LLM_STRACE_PATH

python analyse/aggregate_straces.py \
    --base_dir $LLM_STRACE_PATH/$BASE_RESULT_DIR \
    --length $LENGTH \
    --max_files $MAX_FILES \
    --components


echo "--- Finished CPU Job ---"
date

