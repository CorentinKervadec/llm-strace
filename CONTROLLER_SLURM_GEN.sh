#!/bin/bash
#SBATCH --job-name=exp_controller
#SBATCH --output=controller_%j.out
#SBATCH --error=controller_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=high-cpu        # Run on CPU, it's just a lightweight script


echo "Starting Experiment Controller on $(hostname)"
date

# Activate your environment
# source venv/bin/activate
source activate unnatural_prompt

# Run the python orchestrator
python run_multi_model_gen_experiment.py

echo "Controller finished."
date