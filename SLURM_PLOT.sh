#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=plot_strace      # Name of the job
#SBATCH --output=slurm_logs/plot_strace.out # Standard output log
#SBATCH --partition=short             # Request a CPU partition
#SBATCH --ntasks=1                  # Run on a single task
#SBATCH --cpus-per-task=4           # Request 4 CPUs (loading/aggregating can be parallelized)
#SBATCH --mem=32G                   # Request 16GB of memory (loading many files can be memory-intensive)
#SBATCH --time=01:00:00             # 1-hour time limit (adjust if needed)

# --- Job Execution ---
echo "Plotting job started at: $(date)"
echo "Running on host: $SLURMD_NODENAME"
echo "CPUs requested: $SLURM_CPUS_PER_TASK"

# Create the logs directory if it doesn't exist
mkdir -p slurm_logs

# 1. Load your cluster's environment modules (if needed)
# module load python/3.10
# module load ...

# Activate the Python environment
source activate unnatural_prompt

# Get arguments from the environment variables set in submit_jobs.sh
: "${FINAL_DIR:?FINAL_DIR not set}"
: "${PDF_FILE:?PDF_FILE not set}"

# 3. Run the plotting script
# Make sure plot_strace_results.py is in the same directory
# or provide the full path to it.
echo "Running Python plotting script..."
python analyse/plot_straces.py \
    --result_dir $FINAL_DIR \
    --pdf_name $PDF_FILE

echo "Python script finished."
echo "Job finished at: $(date)"
