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

### Model Supported

We have already included the following LLMs to our framework.

| Model Family | Included Variants |
| :--- | :--- |
| **Mistral** | `mistralai/Mistral-7B-v0.1` |
| **OLMo 2** | `allenai/OLMo-2-0425-1B`, `allenai/OLMo-2-1124-7B`, `allenai/OLMo-2-1124-13B` |
| **Qwen 3** | `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base`, `Qwen/Qwen3-8B-Base` |
| **Qwen 2.5** | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B`, `Qwen/Qwen2.5-32B` |
| **Qwen 2** | `Qwen/Qwen2-0.5B`, `Qwen/Qwen2-1.5B`, `Qwen/Qwen2-7B` |
| **Qwen 2** | `Qwen/Qwen2-0.5B`, `Qwen/Qwen2-1.5B`, `Qwen/Qwen2-7B` |
| **Llama-3.1** | `meta-llama/Llama-3.1-8B` |
| **Deepseek-LLM** | `deepseek-ai/deepseek-llm-7b-base`|
| **Phi**| `microsoft/phi-4`|

--------------------------------------------------------------------------------

## Output Format

For each instance in the dataset, Stage 3 output a `strace_final_{sid}.npz` file. Let's review what you can find inside it.

The `strace_final_{sid}.npz` file contains the raw execution output, subgraph size arrays, reconstruction metrics, and nucleus recovery data for an individual instance evaluated across multiple graph pruning regimes.

### File Keys & Data Structures

When loaded via `numpy.load(..., allow_pickle=True)`, the `.npz` container exposes the following primary arrays and objects:

#### 1. `strata_rel_size` (or `size`)
- Type: `np.ndarray` (shape: `(N_strata,)`, dtype: `float64`)
- Description: Array of relative subgraph edge densities s = |E_sub| / |E_full| evaluated across discrete strata steps (ranging from 10^-5 to 1.0).

#### 2. `tv` / `tv_trace` / `tv_inv` / `tv_random`
- Type: `np.ndarray` (shape: `(N_strata,)`, dtype: `float64`)
- Description: Reconstruction error measured in Total Variation (TV) distance between full-model target logit distribution and the pruned model output at each density step s.

#### 3. `nucleus_<K>` (e.g., `nucleus_60`, `nucleus_10`, `nucleus_1`)
- Type: `np.ndarray` storing a pickled dictionary (accessible via `.item()`)
- Description: Tracks top-k nucleus logit recovery predictions across different extraction regimes.
- Internal Structure:
  ```python
  {
      "trace": {
          "only": [...],     # Target s-trace subgraph predictions
          "inverse": [...]   # Inverse (complementary/pruned) edge predictions
      },
      "random": {
          "only": [...]      # Baseline randomly pruned subgraph predictions
      }
  }
  ```

#### 4. `entropy`
- Type: `float`
- Description: Prediction entropy of the target model when evaluating the full, unpruned sentence graph.

### Loading Example

```python
import numpy as np

# Load the s-trace output file
sid = 1
data = np.load(f"strace_final_{sid}.npz", allow_pickle=True)

# 1. Access relative trace densities
rel_sizes = data["strata_rel_size"]  # Shape: (N_strata,)

# 2. Access nucleus recovery stats
nucleus_60 = data["nucleus_60"].item()
trace_only_preds = nucleus_60["trace"]["only"]
trace_inv_preds  = nucleus_60["trace"]["inverse"]
random_preds     = nucleus_60["random"]["only"]
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
