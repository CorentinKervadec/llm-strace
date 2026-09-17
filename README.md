# 🌌 llm-strace: Tracing Computation Density in LLMs

This repository contains the code for the [EMNLP paper **"Tracing Computation Density in LLMs"**](https://arxiv.org/abs/2605.27033). The `s-Trace` method efficiently estimates a subgraph of size $s$ that approximates a full model output, uncovering the "construction" and "refinement" phases of language model computation.

## Setup

We recommend creating a fresh Conda environment before installing the requirements. 

```bash
conda create -n llm-trace python=3.10 -y
conda activate llm-trace
pip install -r requirements.txt
```

## Usage

The extraction and evaluation of `s-Trace` is divided into three stages:

### Parallel Chunking Parameters

For efficiency and parallel cluster execution (e.g., Slurm array jobs), the target dataset is divided into manageable chunks across all three pipeline scripts using three parameters:
```
    --chunk_id: The 0-indexed identifier for the specific chunk/job being processed.

    --chunk_size: The number of sentences or dataset entries assigned to each chunk.

    --total_sentences: The total number of sentences contained in the target dataset.
```

In what follows, we use the small LLM `Qwen/Qwen3-0.6B-Base` to process the first 10 instances of the `wikitext_40.txt`, as an example.

### Stage 1: Graph Population (GPU)
Populates the complete computation graph with edge importance scores.
```bash
python 1_extraction_gpu.py \
    --model_name Qwen/Qwen3-0.6B-Base \
    --chunk_id 0 \
    --chunk_size 10 \
    --total_sentences 5000 \
    --data_file data/wikitext_40.txt \
    --intermediate_dir results/intermediate_graphs
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
    --final_dir results/final_straces
```

### Section 4.4: Hidden Representation Extraction (GPU)

Extracts last-token hidden representations across trace sizes and baseline conditions (random/inverse subgraphs) to reproduce the representation geometry and alignment analysis presented in **Section 4.4** and **Figure 5** of the paper.
```bash
python 3_extract_last_hidden.py \
    --model_name Qwen/Qwen3-0.6B-Base \
    --chunk_id 0 \
    --chunk_size 10 \
    --total_sentences 5000 \
    --final_dir results/final_straces
```

## Large Scale Execution

For running across a compute cluster, you can use the provided Slurm master script in scripts/. Note: this script is provided as an example, you will need to adapt it to your specific computation infrastructure.
```bash
./scripts/MASTER_SLURM_DATASET.sh <partition> <CPU_OFFLOAD> <model_name> <checkpoint> <importance> <dataset_name> <split> <nb_data> <chunk_size> [start_stage] [end_stage]
```

### Note on Hardware & Context Windows: 

VRAM consumption varies across model architectures and sequence lengths. You can run the benchmarking launcher in find_max_context/ (find_max_context_launcher.sh) to determine the exact model parameter sizes and maximum context lengths supported by your GPU hardware without incurring Out-Of-Memory (OOM) errors.  

## Citation

If you use this code, please cite:
```
@article{kervadec2026tracing,
  title={Tracing Computation Density in LLMs},
  author={Kervadec, Corentin and Lysova, Iuliia and Macocco, Iuri and Baroni, Marco and Boleda, Gemma},
  journal={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026}
}
```
