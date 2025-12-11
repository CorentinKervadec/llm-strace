#!/bin/bash
#SBATCH --job-name=ctx_search    # Job name
#SBATCH --output=slurm_logs/ctx_%A_%a.out # Standard output and error log
#SBATCH --nodes=1                # Run all processes on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=64G                # Job memory request
#SBATCH --time=00:20:00          # Time limit
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --partition=alien
#SBATCH --qos=alien

# Loading CUDA module
module --ignore-cache load CUDA/11.8

# Activate the Python environment
source activate unnatural_prompt

# The arguments are passed from the manager script
MODEL_NAME=$1
CONTEXT_LEN=$2

echo "[$MODEL_NAME] Running trial with context: $CONTEXT_LEN"

# Run the python worker
python find_max_context_worker.py --model_name $MODEL_NAME --context_len $CONTEXT_LEN


# Capture the exit code of python and pass it to Slurm
EXIT_CODE=$?
exit $EXIT_CODE