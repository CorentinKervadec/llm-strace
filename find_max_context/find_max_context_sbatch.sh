#!/bin/bash
#SBATCH --job-name=ctx_search
#SBATCH --output=slurm_logs/ctx_%A_%a.out # Standard output and error log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G 
#SBATCH --time=00:20:00          # Time limit
#SBATCH --partition=high-gpu
#SBATCH --gres=gpu:1 

# load miniconda
ml modulepath/UPF/apps
ml Miniconda3

# load conda env
source activate /data/samanthafs/projects/lab_mbaroni/ckervadec/conda_envs/llm-trace

cd /homes/users/ckervadec/llm-strace/find_max_context

# The arguments are passed from the manager script
MODEL_NAME=$1
CONTEXT_LEN=$2

echo "[$MODEL_NAME] Running trial with context: $CONTEXT_LEN"

# Run the python worker
python find_max_context_worker.py --model_name $MODEL_NAME --context_len $CONTEXT_LEN


# Capture the exit code of python and pass it to Slurm
EXIT_CODE=$?
exit $EXIT_CODE