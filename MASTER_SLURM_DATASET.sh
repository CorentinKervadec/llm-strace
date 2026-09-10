#!/bin/bash

#-----------------------------------------------------------------------
# Master SLURM Job Submission Script
#
# Usage: ./submit_jobs.sh <partition> <model_name> <importance> <strace> <dataset_name> <split> <nb_data> <chunk_size> [start_stage] [end_stage]
#
# Arguments:
#   1. partition:    (Required) e.g., "alien"
#   2. model_name:   (Required) e.g., "allenai/OLMo-2-0425-1B"
#   3. importance:   (Required) e.g., "norm"
#   4. strace:       (Required) e.g., "threshold"
#   5. dataset_name: (Required) e.g., "c4" or "wikitext"
#   6. split:        (Required) e.g., "0" or "none" (use "none" or "" if no split)
#   7. nb_data:      (Required) e.g., 10000
#   8. chunk_size:   (Required) e.g., 50
#   9. start_stage:  (Optional) Default: 1
#   10. end_stage:   (Optional) Default: 4
#
# Example:
#   ./submit_jobs.sh alien "allenai/OLMo-2-0425-1B" norm threshold c4 0 10000 50
#-----------------------------------------------------------------------

set -e # Exit immediately if any command fails

# --- 1. Input Validation ---
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$7" ] || [ -z "$8" ] || [ -z "$9" ] || [ -z "$10" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 <partition> <CPU_OFFLOAD> <model_name> <checkpoint> <importance> <strace> <dataset_name> <split> <nb_data> <chunk_size> [start_stage] [end_stage]"
    exit 1
fi

PARTITION=$1
CPU_OFFLOAD=$2
MODEL_NAME=$3
CHECKPOINT=$4
IMPORTANCE=$5
STRACE=$6
DATASET_NAME=$7
SPLIT=$8
NB_DATA=$9
CHUNK_SIZE=${10}
START_STAGE=${11:-1}
END_STAGE=${12:-4}

MAX_CONCURRENT_JOBS=50
EXCLUDED_NODES="node044,node042"

# Handle Split Logic for Folders
if [ "$SPLIT" == "none" ] || [ -z "$SPLIT" ]; then
    SPLIT_SUFFIX=""
    SPLIT_VAL=""
    # echo "No Split specified."
else
    SPLIT_SUFFIX="_${SPLIT}"
    SPLIT_VAL=$SPLIT
    # echo "Split detected: $SPLIT"
fi

echo "--- Configuration ---"
echo "GPU Partition:   $PARTITION"
echo "Model Name:      $MODEL_NAME"
echo "Checkpoint:      $CHECKPOINT"
echo "Importance:      $IMPORTANCE"
echo "Strace:          $STRACE"
echo "Dataset Name:    $DATASET_NAME"
echo "Split:           $SPLIT"
echo "NB Data:         $NB_DATA"
echo "Chunk Size:      $CHUNK_SIZE"
echo "Start Stage:     $START_STAGE"
echo "End Stage:       $END_STAGE"
echo "---------------------"

PARTITION_FLAGS="--partition=$PARTITION --qos=alien"

# --- 2. Configuration & File/Directory Setup ---

SANITIZED_MODEL_NAME=${MODEL_NAME##*/}
echo "Sanitized Model Name: $SANITIZED_MODEL_NAME"

DATASET_FILE="$(pwd)/data/${DATASET_NAME}_${SPLIT_VAL}.txt"

echo "Dataset File: $DATASET_FILE"

# First, get the line count (SAFELY)
if [ -f "$DATASET_FILE" ]; then
    LINE_COUNT=$(wc -l < "$DATASET_FILE")
else
    echo "Warning: Data file not found: $DATASET_FILE. Defaulting to 0 lines."
    LINE_COUNT=0
fi

if [ "$NB_DATA" -gt "$LINE_COUNT" ]; then
  TOTAL_SENTENCES=$LINE_COUNT
else
  TOTAL_SENTENCES=$NB_DATA
fi

if [ "$TOTAL_SENTENCES" -eq 0 ]; then
    echo "Error: TOTAL_SENTENCES is 0. No data to process."
    exit 1
fi

NUM_CHUNKS=$(((TOTAL_SENTENCES + CHUNK_SIZE - 1) / CHUNK_SIZE))
ARRAY_RANGE="0-$((NUM_CHUNKS - 1))%${MAX_CONCURRENT_JOBS}"

echo "Total Sentences: $TOTAL_SENTENCES"
echo "Total Chunks (Jobs): $NUM_CHUNKS"
echo "Slurm Array Range: $ARRAY_RANGE"

# --- Directory and Data Configuration ---
export MODEL_NAME=$MODEL_NAME
export CHECKPOINT=$CHECKPOINT
export IMPORTANCE=$IMPORTANCE
export STRACE=$STRACE
export DATA_FILE=$DATASET_FILE
export CHUNK_SIZE=$CHUNK_SIZE
export TOTAL_SENTENCES=$TOTAL_SENTENCES
# Exporting specific variables for python scripts
export DATASET_NAME=$DATASET_NAME 
export SPLIT=$SPLIT_VAL
# CPU OFFLOAD to save GPU memory
export CPU_OFFLOAD=$CPU_OFFLOAD

# Define all directory paths
# Base directory uses just the DATASET_NAME
BASE_OUTPUT_DIR="$(pwd)/results_${IMPORTANCE}_${STRACE}_${DATASET_NAME}_emnlp/${SANITIZED_MODEL_NAME}"

# Subdirectories use the SPLIT_SUFFIX (e.g., _S0 or empty)
# Note: I removed ${SENTENCE_LENGTH} from these paths as requested
export INTERMEDIATE_DIR="${BASE_OUTPUT_DIR}/${CHECKPOINT}/intermediate_graphs${SPLIT_SUFFIX}"
# export INTERMEDIATE_DIR="${BASE_OUTPUT_DIR}/final_straces${SPLIT_SUFFIX}" # we reuse the graph already computed
export STRACE_DIR="${BASE_OUTPUT_DIR}/${CHECKPOINT}/intermediate_straces${SPLIT_SUFFIX}"
# export STRACE_DIR="${BASE_OUTPUT_DIR}/final_straces${SPLIT_SUFFIX}"
# export STRACE_DIR="${BASE_OUTPUT_DIR}/final_straces${SPLIT_SUFFIX}"
export FINAL_DIR="${BASE_OUTPUT_DIR}/${CHECKPOINT}/final_straces${SPLIT_SUFFIX}"
# export FINAL_DIR="${BASE_OUTPUT_DIR}/final_straces${SPLIT_SUFFIX}_size"
export PDF_FILE="${BASE_OUTPUT_DIR}/${CHECKPOINT}/strace_analysis_plots${SPLIT_SUFFIX}.pdf"
LOG_DIR="${BASE_OUTPUT_DIR}/slurm_logs${SPLIT_SUFFIX}"

echo "Base Output Dir: $BASE_OUTPUT_DIR"
echo "Intermediate Dir: $INTERMEDIATE_DIR"
echo "Strace Dir: $STRACE_DIR"
echo "Final Dir: $FINAL_DIR"

# Create all directories
mkdir -p $INTERMEDIATE_DIR
mkdir -p $STRACE_DIR
mkdir -p $FINAL_DIR
mkdir -p $LOG_DIR

# --- 3. Validation Function ---

check_files() {
    local dir=$1
    local expected_count=$2
    local stage_name=$3

    echo "Validating prerequisite for $stage_name..."
    if [ ! -d "$dir" ]; then
        echo "Error: Prerequisite directory $dir does not exist."
        exit 1
    fi

    local file_count=$(find "$dir" -maxdepth 1 -type f -name '*.npz' | wc -l)
    
    if [ "$file_count" -lt "$expected_count" ]; then
        echo "Error: Prerequisite check failed for $stage_name."
        echo "  Found $file_count files in $dir, but expected $expected_count."
        read -p "Do you want to continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    echo "  Validation successful: Found $file_count files in $dir."
}


# --- 4. Job Submission ---

LAST_JOB_ID=""

# --- Stage 1: GPU Population ---
if [ "$START_STAGE" -le 1 ] && [ "$END_STAGE" -ge 1 ]; then
    echo "Submitting Stage 1: GPU Population..."

    GPU_JOB_ID=$(sbatch --parsable \
        --array=$ARRAY_RANGE \
        $PARTITION_FLAGS \
        --export=ALL,INTERMEDIATE_DIR \
        --output="${LOG_DIR}/1_gpu_%A_%a.out" \
        --exclude=$EXCLUDED_NODES \
        --job-name="${SANITIZED_MODEL_NAME}_strace_gpu" \
        1_SLURM_GPU.sh)

    if [ -z "$GPU_JOB_ID" ]; then
        echo "Error: Failed to submit GPU job. Exiting."
        exit 1
    fi
    echo "  -> Stage 1 submitted with Array ID: $GPU_JOB_ID"
    LAST_JOB_ID=$GPU_JOB_ID
fi

# --- Stage 2: CPU Analysis ---
if [ "$START_STAGE" -le 2 ] && [ "$END_STAGE" -ge 2 ]; then
    echo "Submitting Stage 2: CPU Analysis..."
    
    DEP_FLAG=""
    if [ ! -z "$LAST_JOB_ID" ]; then
        DEP_FLAG="--dependency=aftercorr:${LAST_JOB_ID}"
        echo "  -> Will run after corresponding Stage 1 jobs."
    elif [ "$START_STAGE" -eq 2 ]; then
        check_files $INTERMEDIATE_DIR $TOTAL_SENTENCES "Stage 2"
    fi

    CPU_JOB_ID=$(sbatch --parsable \
        --array=$ARRAY_RANGE \
        $DEP_FLAG \
        --export=ALL,INTERMEDIATE_DIR,STRACE_DIR \
        --output="${LOG_DIR}/2_cpu_%A_%a.out" \
        --exclude=$EXCLUDED_NODES \
        --job-name="${SANITIZED_MODEL_NAME}_strace_cpu" \
        2_SLURM_CPU.sh)

    if [ -z "$CPU_JOB_ID" ]; then
        echo "Error: Failed to submit CPU job. Exiting."
        exit 1
    fi
    echo "  -> Stage 2 submitted with Array ID: $CPU_JOB_ID"
    LAST_JOB_ID=$CPU_JOB_ID
fi

# --- Stage 3: GPU Reconstruction ---
if [ "$START_STAGE" -le 3 ] && [ "$END_STAGE" -ge 3 ]; then
    echo "Submitting Stage 3: GPU Reconstruction..."
    
    DEP_FLAG=""
    if [ ! -z "$LAST_JOB_ID" ]; then
        DEP_FLAG="--dependency=aftercorr:${LAST_JOB_ID}"
        echo "  -> Will run after corresponding Stage 2 jobs."
    elif [ "$START_STAGE" -eq 3 ]; then
        check_files $STRACE_DIR $TOTAL_SENTENCES "Stage 3"
        # echo "Skipping check file..."
    fi

    GPU_JOB_ID_2=$(sbatch --parsable \
        --array=$ARRAY_RANGE \
        $DEP_FLAG \
        $PARTITION_FLAGS \
        --export=ALL,STRACE_DIR,FINAL_DIR \
        --output="${LOG_DIR}/3_gpu_%A_%a.out" \
        --exclude=$EXCLUDED_NODES \
        --job-name="${SANITIZED_MODEL_NAME}_eval_gpu" \
        3_SLURM_GPU.sh)

    if [ -z "$GPU_JOB_ID_2" ]; then
        echo "Error: Failed to submit GPU Stage 3 job. Exiting."
        exit 1
    fi
    echo "  -> Stage 3 submitted with Array ID: $GPU_JOB_ID_2"
    LAST_JOB_ID=$GPU_JOB_ID_2
fi

# --- Stage 4: Plotting ---
if [ "$START_STAGE" -le 4 ] && [ "$END_STAGE" -ge 4 ]; then
    echo "Submitting Stage 4: Plotting..."

    DEP_FLAG=""
    if [ ! -z "$LAST_JOB_ID" ]; then
        DEP_FLAG="--dependency=afterany:${LAST_JOB_ID}"
        echo "  -> Will run after all Stage 3 jobs have finished."
    elif [ "$START_STAGE" -eq 4 ]; then
        check_files $FINAL_DIR $TOTAL_SENTENCES "Stage 4 (Plotting)"
    fi
    
    PLOT_JOB_ID=$(sbatch --parsable \
        $DEP_FLAG \
        --export=ALL,FINAL_DIR,PDF_FILE \
        --output="${LOG_DIR}/4_plot_%j.out" \
        --exclude=$EXCLUDED_NODES \
        --job-name="${SANITIZED_MODEL_NAME}_plot" \
        SLURM_PLOT.sh)
    
    if [ -z "$PLOT_JOB_ID" ]; then
        echo "Error submitting Stage 4 plotting job."
        exit 1
    fi
    echo "  -> Stage 4 Plotting Job ID: $PLOT_JOB_ID"
    LAST_JOB_ID=$PLOT_JOB_ID
fi

echo ""
echo "All requested jobs submitted successfully."
echo "Check '$LOG_DIR' for logs."
echo "Run 'squeue -u $USER' to monitor."