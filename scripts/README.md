# Slurm Job Management & Automation (scripts/)

This directory contains Slurm batch scripts and the master orchestration utility MASTER_SLURM_DATASET.sh, designed to execute end-to-end s-Trace extraction and evaluation pipelines in parallel across GPU/CPU compute cluster nodes.

--------------------------------------------------------------------------------

## Master Launcher (MASTER_SLURM_DATASET.sh)

The master script automates directory management, dataset chunking, array sizing, and task dependency chaining (--dependency=aftercorr) across all three main computational pipeline stages.

### Syntax

./MASTER_SLURM_DATASET.sh <partition> <CPU_OFFLOAD> <model_name> <checkpoint> \
                          <importance> <dataset_name> <split> <nb_data> \
                          <chunk_size> [start_stage] [end_stage]

### Argument Reference

1. partition    : Target Slurm GPU queue (e.g., "high-gpu")
2. CPU_OFFLOAD  : Enable HF accelerate CPU offloading (0 or 1)
3. model_name   : Hugging Face model identifier (e.g., "allenai/OLMo-2-0425-1B")
4. checkpoint   : Git commit, tag, or branch revision (e.g., "main")
5. importance   : Edge weight attribution metric (e.g., "L1-norm")
6. dataset_name : Name prefix of input file inside data/ (e.g., "wikitext")
7. split        : Dataset split identifier ("none" or integer suffix like "40")
8. nb_data      : Total number of sentences to process (e.g., 5000)
9. chunk_size   : Number of sentences assigned per array task (e.g., 100)
10. start_stage : (Optional) Initial stage to execute (1, 2, or 3; Default: 1)
11. end_stage   : (Optional) Final stage to execute (1, 2, or 3; Default: 3)

--------------------------------------------------------------------------------

## Execution Workflow

1. Task Array Division: Calculates total chunks needed (N = ceil(nb_data / chunk_size)) and schedules Slurm job arrays throttled to a maximum of 50 concurrent tasks.
2. Sequential Dependency Chaining: Stage 2 (CPU) array tasks automatically wait for their corresponding Stage 1 (GPU) chunk task to complete using --dependency=aftercorr. Stage 3 (GPU) waits on Stage 2.
3. Mid-Pipeline Execution: Setting start_stage to 2 or 3 triggers pre-flight validation on intermediate .npz files (check_files) before submitting remaining steps.

--------------------------------------------------------------------------------

## Examples

- Run Complete Pipeline (Stages 1 through 3):
  ./MASTER_SLURM_DATASET.sh high-gpu 0 "allenai/OLMo-2-0425-1B" main L1-norm "wikitext" 40 5000 100

- Run Stage 1 Only (Graph Population):
  ./MASTER_SLURM_DATASET.sh high-gpu 0 "allenai/OLMo-2-0425-1B" main L1-norm "wikitext" 40 5000 100 1 1

- Resume Pipeline from Stage 3 (Reconstruction Evaluation):
  ./MASTER_SLURM_DATASET.sh high-gpu 0 "allenai/OLMo-2-0425-1B" main L1-norm "wikitext" 40 5000 100 3 3

--------------------------------------------------------------------------------

## Directory Structure & Logs

Output directories and logs are structured systematically:

- Results Path: results_<IMPORTANCE>_<DATASET_NAME>/<SANITIZED_MODEL_NAME>/<CHECKPOINT>/
- Log Location: Slurm array logs are stored in <BASE_OUTPUT_DIR>/slurm_logs/ as:
  - Stage 1: 1_gpu_%A_%a.out
  - Stage 2: 2_cpu_%A_%a.out
  - Stage 3: 3_gpu_%A_%a.out