#!/bin/bash
#SBATCH --job-name=generate-color    # Job name
#SBATCH --output=slurm_logs/generate_%A_%a.out # Standard output and error log
#SBATCH --nodes=1                # Run all processes on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=64G                # Job memory request
#SBATCH --time=15:00:00          # Time limit (2 hours, adjust for 100 sentences)
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --partition=alien
#SBATCH --qos=alien

echo "Starting slurm job $SLURM_JOB_NAME.$SLURM_JOB_ID"


################## P A T H S ##################

JOB_NAME="$SLURM_JOB_NAME.$SLURM_JOB_ID"


cd $HOME

echo ''
echo '[paths] Job name: '$JOB_NAME
echo ''

################## M O D U L E S ##################

# Loading CUDA module
module --ignore-cache load CUDA/11.8

# Activate the Python environment
source activate unnatural_prompt

################## R U N ##################

# Run the Python script
python llm-strace/analyse/generate_colored_text.py --model_name "Qwen/Qwen3-4B-Base" --prompt "Continue the description of the weather today:\nThe sky is gray. Clouds drift overhead. Rain falls. The wind blows softly through the trees that line the street. Puddles form on the pavement that" --length 200 --output_prefix qwen3_4B_colored --context_length 40


echo ''
echo 'Run is over.'
