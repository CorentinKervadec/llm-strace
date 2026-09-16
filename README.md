# 🌌 llm-strace: Tracing Computation Density in LLMs

This repository contains the code for the EMNLP paper **"Tracing Computation Density in LLMs"**. The `s-Trace` method efficiently estimates a subgraph of size $s$ that approximates a full model output, uncovering the "construction" and "refinement" phases of language model computation.

## Setup

We recommend creating a fresh Conda environment before installing the requirements. 

```bash
conda create -n llm-trace python=3.10 -y
conda activate llm-trace
pip install -r requirements.txt
```

*(Alternatively, you can run `bash install_requirements.sh` which sets up the environment and runs a small end-to-end test.)*

## Usage

The extraction and evaluation of `s-Trace` is divided into three stages:

### Stage 1: Graph Population (GPU)
Populates the complete computation graph with edge importance scores.
```bash
python 1_extraction_gpu.py \
    --model_name Qwen/Qwen3-0.6B-Base \
    --chunk_id 0 \
    --chunk_size 10 \
    --total_sentences 5000 \
    --data_file data/wikitext_40.txt \
    --intermediate_dir results/intermediate_graphs \
    --checkpoint main
```

### Stage 2: s-Trace Extraction (CPU)
Extracts computational subgraphs ("s-Traces") from the full graph based on a set of target sizes.
```bash
python 2_stratification_cpu.py \
    --chunk_id 0 \
    --chunk_size 10 \
    --total_sentences 5000 \
    --intermediate_dir results/intermediate_graphs \
    --strace_dir results/intermediate_straces
```

### Stage 3: Evaluation (GPU)
Evaluates the predictive power of each extracted s-Trace by measuring the reconstruction error (Total Variation distance).
```bash
python 3_evaluate_gpu.py \
    --model_name Qwen/Qwen3-0.6B-Base \
    --chunk_id 0 \
    --chunk_size 10 \
    --total_sentences 5000 \
    --strace_dir results/intermediate_straces \
    --final_dir results/final_straces \
    --checkpoint main
```

## Large Scale Execution

For running across a compute cluster, you can use the provided Slurm master script:
```bash
./MASTER_SLURM_DATASET.sh <partition> <CPU_OFFLOAD> <model_name> <checkpoint> <importance> <strace> <dataset_name> <split> <nb_data> <chunk_size>
```
