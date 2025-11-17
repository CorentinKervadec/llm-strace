#!/bin/bash

#-----------------------------------------------------------------------
# Master SLURM Job Submission Script
#
# Usage: ./submit_jobs.sh <model_name> <sentence_length> <nb_data> <chunk_size> [start_stage] [end_stage]
#
# Arguments:
#   model_name:      (Required) e.g., "allenai/OLMo-2-0425-1B"
#   sentence_length: (Required) e.g., 30
#   nb_data:         (Required) e.g., 10000
#   chunk_size:      (Required) e.g., 50
#   start_stage:     (Optional) Stage to start from (1-4). Default: 1
#   end_stage:       (Optional) Stage to end on (1-4). Default: 4
#
# Example (Run all stages):
#   ./submit_jobs.sh 30 10000 50 "allenai/OLMo-2-0425-1B"
#
# Example (Run only Stage 2 and 3):
#   ./submit_jobs.sh 30 10000 50 "allenai/OLMo-2-0425-1B" 2 3
#-----------------------------------------------------------------------

set -e # Exit immediately if any command fails

# --- 1. Input Validation ---
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 <model_name> <sentence_length> <nb_data> <chunk_size> [start_stage] [end_stage]"
    exit 1
fi

MODEL_NAME=$1
SENTENCE_LENGTH=$2
NB_DATA=$3
CHUNK_SIZE=$4
START_STAGE=${5:-1}  # Default to 1 if not provided
END_STAGE=${6:-4}    # Default to 4 if not provided

MAX_CONCURRENT_JOBS=50 # Fairness: Don't run more than 50 jobs at once
EXCLUDED_NODES="node044,node042"

echo "--- Configuration ---"
echo "Model Name: $MODEL_NAME"
echo "Sentence Length: $SENTENCE_LENGTH"
echo "NB Data: $NB_DATA"
echo "Chunk Size: $CHUNK_SIZE"
echo "Start Stage: $START_STAGE"
echo "End Stage: $END_STAGE"
echo "---------------------"


# --- 2. Configuration & File/Directory Setup ---

# Sanitize the model name: get the last part after the final '/'
SANITIZED_MODEL_NAME=${MODEL_NAME##*/}
echo "Sanitized Model Name: $SANITIZED_MODEL_NAME"

DATASET_FILE="$(pwd)/data/${SENTENCE_LENGTH}_data.txt"

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

# --- Handle 0 sentences case ---
if [ "$TOTAL_SENTENCES" -eq 0 ]; then
    echo "Error: TOTAL_SENTENCES is 0. No data to process."
    exit 1
fi

# Calculate total number of chunks (jobs)
# Use integer arithmetic: (A + B - 1) / B
NUM_CHUNKS=$(((TOTAL_SENTENCES + CHUNK_SIZE - 1) / CHUNK_SIZE))
ARRAY_RANGE="0-$((NUM_CHUNKS - 1))%${MAX_CONCURRENT_JOBS}"

echo "Total Sentences: $TOTAL_SENTENCES"
echo "Total Chunks (Jobs): $NUM_CHUNKS"
echo "Slurm Array Range: $ARRAY_RANGE"

# --- Directory and Data Configuration ---
export MODEL_NAME=$MODEL_NAME
export DATA_FILE=$DATASET_FILE
export CHUNK_SIZE=$CHUNK_SIZE
export TOTAL_SENTENCES=$TOTAL_SENTENCES

# Define all directory paths
# Define all directory paths with SANITIZED_MODEL_NAME
BASE_OUTPUT_DIR="$(pwd)/results/${SANITIZED_MODEL_NAME}"
export INTERMEDIATE_DIR="${BASE_OUTPUT_DIR}/intermediate_graphs_${SENTENCE_LENGTH}"
export STRACE_DIR="${BASE_OUTPUT_DIR}/intermediate_straces_${SENTENCE_LENGTH}"
export FINAL_DIR="${BASE_OUTPUT_DIR}/final_straces_${SENTENCE_LENGTH}"
export PDF_FILE="${BASE_OUTPUT_DIR}/strace_analysis_plots_${SENTENCE_LENGTH}.pdf"
LOG_DIR="${BASE_OUTPUT_DIR}/slurm_logs_${SENTENCE_LENGTH}"

# Create all directories
mkdir -p $INTERMEDIATE_DIR
mkdir -p $STRACE_DIR
mkdir -p $FINAL_DIR
mkdir -p $LOG_DIR

# --- 3. Validation Function ---

# Function to check if the prerequisite files exist for a given stage
# Usage: check_files <directory_to_check> <expected_file_count> <stage_name>
check_files() {
    local dir=$1
    local expected_count=$2
    local stage_name=$3

    echo "Validating prerequisite for $stage_name..."
    if [ ! -d "$dir" ]; then
        echo "Error: Prerequisite directory $dir does not exist."
        echo "Cannot skip to $stage_name. Please run previous stages."
        exit 1
    fi

    # Count files (e.g., .npz)
    # This counts one file *per sentence*
    local file_count=$(find "$dir" -maxdepth 1 -type f -name '*.npz' | wc -l)
    
    if [ "$file_count" -lt "$expected_count" ]; then
        echo "Error: Prerequisite check failed for $stage_name."
        echo "  Found $file_count files in $dir, but expected $expected_count (one per sentence)."
        echo "  Cannot skip to $stage_name. Please re-run previous stages."
        exit 1
    fi
    echo "  Validation successful: Found $file_count files in $dir."
}


# --- 4. Job Submission ---

LAST_JOB_ID="" # This will hold the ID of the *previous* stage

# --- Stage 1: GPU Population ---
if [ "$START_STAGE" -le 1 ] && [ "$END_STAGE" -ge 1 ]; then
    echo "Submitting Stage 1: GPU Population..."
    GPU_JOB_ID=$(sbatch --parsable \
        --array=$ARRAY_RANGE \
        --export=ALL,INTERMEDIATE_DIR \
        --output="${LOG_DIR}/1_gpu_%A_%a.out" \
        --exclude=$EXCLUDED_NODES \
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
    
    # Set dependency if Stage 1 was just submitted
    DEP_FLAG=""
    if [ ! -z "$LAST_JOB_ID" ]; then
        # Use aftercorr for efficient array-to-array dependency
        # stage_2_job[i] will run just after stage_1_job[i]
        DEP_FLAG="--dependency=aftercorr:${LAST_JOB_ID}"
        echo "  -> Will run after corresponding Stage 1 jobs."
    elif [ "$START_STAGE" -eq 2 ]; then
        # If we are *starting* here, validate previous stage
        check_files $INTERMEDIATE_DIR $TOTAL_SENTENCES "Stage 2"
    fi

    CPU_JOB_ID=$(sbatch --parsable \
        --array=$ARRAY_RANGE \
        $DEP_FLAG \
        --export=ALL,INTERMEDIATE_DIR,STRACE_DIR \
        --output="${LOG_DIR}/2_cpu_%A_%a.out" \
        --exclude=$EXCLUDED_NODES \
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
    fi

    GPU_JOB_ID_2=$(sbatch --parsable \
        --array=$ARRAY_RANGE \
        $DEP_FLAG \
        --export=ALL,STRACE_DIR,FINAL_DIR \
        --output="${LOG_DIR}/3_gpu_%A_%a.out" \
        --exclude=$EXCLUDED_NODES \
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
        # Use afterany: run plot *after all* stage 3 jobs are finished
        DEP_FLAG="--dependency=afterany:${LAST_JOB_ID}"
        echo "  -> Will run after all Stage 3 jobs have finished."
    elif [ "$START_STAGE" -eq 4 ]; then
        check_files $FINAL_DIR $TOTAL_SENTENCES "Stage 4 (Plotting)"
    fi
    
    # This is a single job, not an array
    PLOT_JOB_ID=$(sbatch --parsable \
        $DEP_FLAG \
        --export=ALL,FINAL_DIR,PDF_FILE \
        --output="${LOG_DIR}/4_plot_%j.out" \
        --exclude=$EXCLUDED_NODES \
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

