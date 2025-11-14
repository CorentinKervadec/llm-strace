# llm-strace: Stratified Tracing for Language Models

llm-strace is a research framework for analyzing Large Language Models by decomposing their computation into "strata," or minimal subgraphs. It is designed for large-scale, reproducible experiments on HPC clusters using Slurm.

This framework allows you to:
1. Model an LLM's forward pass as a complete, directed computation graph.
2. Assign an "importance" score to every edge (representing attention heads, MLP blocks, and residual connections).
3. "Stratify" this graph by applying a 'nucleus' filter (like Top-P) to create a series of subgraphs, or "strata".
4. Evaluate the predictive power of each stratum by running the model "masked" to only that subgraph.

This allows us to analyze questions like, **"What is the minimal set of components required for the model to make a specific prediction?"**

## Project Structure

The project is split into the core library (`src/`) and a set of executable scripts for running the pipeline.

```
├── MASTER_SLURM.sh         # Master script to launch the full 4-stage pipeline on Slurm.
├── main_test.py            # A local script for testing/debugging one sentence at a time.
│
├── 1_extraction_gpu.py     # Stage 1: (GPU) Runs a forward pass to populate the full graph.
├── 2_stratification_cpu.py # Stage 2: (CPU) Applies graph algorithms to extract strata.
├── 3_evaluate_gpu.py       # Stage 3: (GPU) Runs reconstruction/evaluation on each stratum.
│
├── 1_SLURM_GPU.sh          # Slurm batch scripts for each stage
├── 2_SLURM_CPU.sh
├── 3_SLURM_GPU.sh
├── SLURM_PLOT.sh
│
├── analyse/                # Scripts for plotting and analyzing final results.
│   ├── plot_straces.py
│   └── ...
│
├── data/                   # Input data files (e.g., 10_data.txt, 20_data.txt)
│
└── src/                    # The core Python library.
    ├── llm_hooked/         # Model-agnostic and specific classes for hooking models.
    ├── llm_graph/          # Classes for building the NetworkX computation graph.
    └── llm_trace/          # The main LLM_STRACE class that orchestrates the pipeline.
```

## Setup and Installation

1. Clone the repository:
```
git clone [https://github.com/your-username/llm-strace.git](https://github.com/your-username/llm-strace.git)
cd llm-strace
```
2. Create a virtual environment and install dependencies:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

There are two primary ways to run this project: a simple local test for debugging, or the full-scale Slurm pipeline for research.

### 1. Local Testing (main_test.py)

For development and debugging, you can run the entire 3-stage pipeline on a single sentence using main_test.py. This script runs locally (not on Slurm) and prints results to the console.

Example for testing with Mistral-7B:
`python main_test.py --model_name "mistralai/Mistral-7B-v0.1"`

Example for testing with OLMo:
`python main_test.py --model_name "allenai/OLMo-2-0425-1B"`

### 2. Large-Scale Execution (MASTER_SLURM.sh)

This is the main workflow for the project, designed to run on an HPC cluster using Slurm. The MASTER_SLURM.sh script automates the entire 4-stage, multi-dependency pipeline.

Usage:
`./MASTER_SLURM.sh <sentence_length> <nb_data> <chunk_size> <model_name> [start_stage] [end_stage]`

Arguments:
- `<sentence_length>`: The length prefix of the input data file to use (e.g., 20 for 20_data.txt).
- `<nb_data>`: The total number of sentences to process from the data file.
- `<chunk_size>`: The number of sentences to process per Slurm job.
- `<model_name>`: The Hugging Face name of the model (e.g., "allenai/OLMo-2-0425-1B").
- `[start_stage]` (Optional): The stage to start from (1-4). Default: 1.
- `[end_stage]` (Optional): The stage to end on (1-4). Default: 4.Example (Run all stages for 10,000 sentences):./MASTER_SLURM.sh 20 10000 50 "allenai/OLMo-2-0425-1B"

Example (Run only Stage 3 and 4, assuming Stages 1 & 2 are complete):
`./MASTER_SLURM.sh 20 10000 50 "allenai/OLMo-2-0425-1B" 3 4`

#### Slurm Pipeline Stages

The MASTER_SLURM.sh script orchestrates four distinct stages, creating a robust and efficient pipeline.

**Stage 1: Graph Population**
- Script: `1_extraction_gpu.py`
- Job: `1_SLURM_GPU.sh`
- Description: A GPU-heavy array job. Each job takes a chunk of sentences, loads the specified model, and performs a full forward pass to generate and save the complete, weighted computation graph for each sentence.
- Output: `[sanitized_model_name]/intermediate_graphs_[len]/...`

**Stage 2: Stratification**
- Script: `2_stratification_cpu.py`
- Job: `2_SLURM_CPU.sh`
- Description: A CPU-heavy array job. Each job loads the full graphs from Stage 1 and runs graph-traversal algorithms ('nucleus' or 'threshold') to extract all strata.
- Output: `[sanitized_model_name]/intermediate_straces_[len]/...`

**Stage 3: Evaluation**
- Script: `3_evaluate_gpu.py`
- Job: `3_SLURM_GPU.sh`
- Description: A GPU-heavy array job. Each job loads the strata from Stage 2, re-loads the model, and runs masked forward passes to calculate the reconstruction error (Loss, Entropy, TV, etc.) for each stratum.
- Output: `[sanitized_model_name]/final_straces_[len]/...`

**Stage 4: Plotting & Analysis**
- Script: `analyse/plot_straces.py`
- Job: `SLURM_PLOT.sh`
- Description: A single CPU job that runs after all Stage 3 jobs are complete. It aggregates results from all thousands of sentence files and generates a final PDF report with aggregated plots.
- Output: `[sanitized_model_name]/strace_analysis_plots_[len].pdf`