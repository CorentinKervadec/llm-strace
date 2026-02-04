#!/bin/bash

#-----------------------------------------------------------------------
# Master SLURM Generation Experiment Script
#
# Usage: ./GEN_MASTER_SLURM.sh <model_name> <prompts_file> <importance_mode> <strace_mode> <max_new_tokens> <seed> [start_stage] [end_stage]
#
# Arguments:
#   model_name:      (Required) e.g., "mistralai/Mistral-7B-v0.1"
#   prompts_file:    (Required) Path to the CSV file containing prompts.
#   importance_mode: (Required) how to compute the importance
#   strace_mode      (Required) how to extrace the trace
#   max_new_tokens   (Required) number of token generated 
#   seed:            (Required) Random seed
#   start_stage:     (Optional) Stage to start from (1-3). Default: 1
#   end_stage:       (Optional) Stage to end on (1-3). Default: 3
#
# Example:
#   ./GEN_MASTER_SLURM.sh "mistralai/Mistral-7B-v0.1" "data/prompts.csv" "norm" "threshold" 20 75
#-----------------------------------------------------------------------

set -e # Exit immediately if any command fails

# --- 1. Input Validation ---
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 <model_name> <prompts_file> <importance_mode> <strace_mode> <max_new_tokens> <random_seed> [start_stage] [end_stage]"
    exit 1
fi

export MODEL_NAME=$1
# Get absolute path for prompts file
export PROMPTS_FILE=$(readlink -f "$2")
export IMPORTANCE_MODE=$3
export STRACE_MODE=$4
export MAX_NEW_TOKENS=$5
export SEED=$6
START_STAGE=${7:-1}
END_STAGE=${8:-3}

# --- 2. Fixed Parameters (Adjust here if needed) ---
export BATCH_SIZE=1
export P_SAMPLE=0.6
MAX_CONCURRENT_JOBS=100
EXCLUDED_NODES="node044,node042"

echo "--- Configuration ---"
echo "Model: $MODEL_NAME"
echo "Prompts File: $PROMPTS_FILE"
echo "Importance: $IMPORTANCE_MODE"
echo "Strace: $STRACE_MODE"
echo "Max tokens: $MAX_NEW_TOKENS"
echo "Seed: $SEED"
echo "Stages: $START_STAGE to $END_STAGE"
echo "---------------------"

# --- 3. Directory Setup ---
# Sanitize model name (remove slashes)
SANITIZED_MODEL_NAME=${MODEL_NAME##*/}
BASE_DIR="$(pwd)/generation_results/${SANITIZED_MODEL_NAME}/seed_${SEED}"

export INTERMEDIATE_DIR="${BASE_DIR}/intermediate_graphs"
export STRACE_DIR="${BASE_DIR}/intermediate_straces"
export FINAL_DIR="${BASE_DIR}/final_straces"
LOG_DIR="${BASE_DIR}/slurm_logs_gen"

mkdir -p $INTERMEDIATE_DIR
mkdir -p $STRACE_DIR
mkdir -p $FINAL_DIR
mkdir -p $LOG_DIR

# --- 4. Array Calculation ---
# Count lines in prompts file
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Error: Prompts file '$PROMPTS_FILE' not found."
    exit 1
fi

TOTAL_LINES=$(wc -l < "$PROMPTS_FILE")
# Assuming CSV with header, subtract 1. If no header, remove the subtraction.
NUM_PROMPTS=$((TOTAL_LINES - 1)) 

if [ "$NUM_PROMPTS" -le 0 ]; then
    echo "Error: Prompts file seems empty or only has header."
    exit 1
fi

# Array range 0 to N-1 (Python indices for prompts)
ARRAY_RANGE="0-$((NUM_PROMPTS - 1))%${MAX_CONCURRENT_JOBS}"

echo "Detected $NUM_PROMPTS prompts."
echo "Slurm Array Range: $ARRAY_RANGE"

# --- 5. Job Submission ---

LAST_JOB_ID=""

# --- Stage 1: Generation & Extraction (GPU) ---
if [ "$START_STAGE" -le 1 ] && [ "$END_STAGE" -ge 1 ]; then
    echo "Submitting Stage 1 (Generation)..."
    GPU_JOB_ID=$(sbatch --parsable \
        --array=$ARRAY_RANGE \
        --export=ALL \
        --output="${LOG_DIR}/1_gpu_%A_%a.out" \
        --exclude=$EXCLUDED_NODES \
        --job-name="${SANITIZED_MODEL_NAME}_strace_gpu" \
        1_GEN_SLURM_GPU.sh)
    
    if [ -z "$GPU_JOB_ID" ]; then echo "Error submitting Stage 1"; exit 1; fi
    echo "  -> Job ID: $GPU_JOB_ID"
    LAST_JOB_ID=$GPU_JOB_ID
fi

# --- Stage 2: Stratification (CPU) ---
if [ "$START_STAGE" -le 2 ] && [ "$END_STAGE" -ge 2 ]; then
    echo "Submitting Stage 2 (Stratification)..."
    
    DEP_FLAG=""
    if [ ! -z "$LAST_JOB_ID" ]; then
        # 'aftercorr' allows prompt N to start Stage 2 as soon as prompt N finishes Stage 1
        DEP_FLAG="--dependency=aftercorr:${LAST_JOB_ID}"
    fi
    
    CPU_JOB_ID=$(sbatch --parsable \
        --array=$ARRAY_RANGE \
        $DEP_FLAG \
        --export=ALL \
        --output="${LOG_DIR}/2_cpu_%A_%a.out" \
        --exclude=$EXCLUDED_NODES \
        --job-name="${SANITIZED_MODEL_NAME}_strace_cpu" \
        2_GEN_SLURM_CPU.sh)

    if [ -z "$CPU_JOB_ID" ]; then echo "Error submitting Stage 2"; exit 1; fi
    echo "  -> Job ID: $CPU_JOB_ID"
    LAST_JOB_ID=$CPU_JOB_ID
fi

# --- Stage 3: Evaluation (GPU) ---
if [ "$START_STAGE" -le 3 ] && [ "$END_STAGE" -ge 3 ]; then
    echo "Submitting Stage 3 (Evaluation)..."
    
    DEP_FLAG=""
    if [ ! -z "$LAST_JOB_ID" ]; then
        DEP_FLAG="--dependency=aftercorr:${LAST_JOB_ID}"
    fi
    
    GPU_JOB_ID_2=$(sbatch --parsable \
        --array=$ARRAY_RANGE \
        $DEP_FLAG \
        --export=ALL \
        --output="${LOG_DIR}/3_gpu_%A_%a.out" \
        --exclude=$EXCLUDED_NODES \
        --job-name="${SANITIZED_MODEL_NAME}_eval_gpu" \
        3_GEN_SLURM_GPU.sh)

    if [ -z "$GPU_JOB_ID_2" ]; then echo "Error submitting Stage 3"; exit 1; fi
    echo "  -> Job ID: $GPU_JOB_ID_2"
fi

echo "All jobs submitted."