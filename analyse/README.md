# Analysis & Data Aggregation (analyse/)

This directory contains the scripts to postprocess and aggregate raw per-sentence s-trace evaluation outputs (.npz files) generated across language models into  datasets and statistical correlation reports for analysis.

--------------------------------------------------------------------------------

## How To aggregate_straces.py

Use this script to aggregate the results of the Stage 3. The aggregate_straces.py script performs three crucial aggregation tasks:

1. Logarithmic Domain Alignment: Interpolates sentence-specific graph sizes s = |E_sub| / |E_full| onto a standardized 150-point logarithmic grid s in [10^-5, 1.0] (COMMON_SIZE) to allow direct cross-sentence and cross-model averaging.
2. Topological & Component Tracking: Quantifies how structural network elements (Attention heads, MLP blocks, Residual streams, and depth quartiles) enter the s-trace as subgraph density increases.
3. Statistical Correlation: Computes Spearman rank correlations (rho) to measure the relationship between prediction entropy and computation density (AUC-TV), as well as sentence difficulty consistency across different models.

--------------------------------------------------------------------------------

## Usage
```bash
python analyse/aggregate_straces.py \
    --base_dir $LLM_STRACE_PATH/$BASE_RESULT_DIR \
    --length $LENGTH \
    --max_files $MAX_FILES \
    --components
```

### Argument Reference

- --base_dir   (str)  : Root directory containing per-model Stage 3 result folders.
- --length     (str)  : Data chunk/length identifier suffix (e.g., "40").
- --max_files  (int)  : Maximum number of sentence .npz files processed per model (Default: 500).
- --components (flag) : (Optional) Enables detailed transformer module-level tracking (head_H_LL, mlp_LL). This is quite slow...

The script will automatically find and process the LLMs that are in the base_dir and aggregate the data into one output file.

--------------------------------------------------------------------------------

## Output Files

Executing the script generates four processed pickle files and a text correlation report:

- processed_data_agg_`<len>`_`<files>`.pkl       : Consolidated TV distance curves (s-trace, inverse, random) and Top-k nucleus recovery sizes.
- processed_data_sentences_`<len>`_`<files>`.pkl : Extremal sentence samples (lowest 5 and highest 5 entropy sentences per model).
- processed_data_topology_`<len>`_`<files>`.pkl  : Structural edge proportions (Attention, MLP, Residuals, and 4 layer depth quartiles) vs density s.
- processed_data_components_`<len>`_`<files>`.pkl: Module-level cumulative presence over density s (only generated when --components is set).
- correlations_report_`<len>`__`<files>`.txt     : Spearman rank correlations (rho) for entropy vs AUC-TV and pairwise cross-model sentence rankings.

--------------------------------------------------------------------------------

## Data Structures of Generated .pkl Files

### 1. processed_data_agg_<len>_<files>.pkl
```json
{
    "aggregate": {
        "<model_name>": {
            "size": np.ndarray,      # (150,) Logarithmic density grid s in [10^-5, 1.0]
            "tv_trace": np.ndarray,   # (150,) Mean TV distance for target s-trace subgraphs
            "tv_inv": np.ndarray,     # (150,) Mean TV distance for inverse (pruned) subgraphs
            "tv_random": np.ndarray   # (150,) Mean TV distance for random baseline subgraphs
        },
        ...
    },
    "nucleus": {
        "<model_name>": {
            1: [float, ...],         # Subgraph size s required to recover 1% top logits across sentences
            5: [float, ...],         # ... 5% recovery
            10: [float, ...],        # ... 10% recovery
            20: [float, ...],
            40: [float, ...],
            60: [float, ...],
            80: [float, ...],
            90: [float, ...]
        },
        ...
    }
}
```

### 2. processed_data_sentences_<len>_<files>.pkl
```json
{
    "<model_name>": {
        "low": [                      # 5 lowest prediction entropy sentences
            {
                "file": str,          # Sentence filename (e.g., "sent_0042.npz")
                "entropy": float,     # Full unpruned model prediction entropy
                "auc": float,         # AUC-TV (Area Under the TV Curve)
                "size": np.ndarray,   # Raw relative sizes for this sentence
                "tv": np.ndarray      # Raw TV distances for this sentence
            },
            ...
        ],
        "high": [ ... ]               # 5 highest prediction entropy sentences
    }
}
```

### 3. processed_data_topology_<len>_<files>.pkl
```json
{
    "<model_name>": {
        "size": np.ndarray,           # (150,) Logarithmic density grid s
        "attn_prop": np.ndarray,      # (150,) Proportion of Attention edges in s-trace at size s
        "mlp_prop": np.ndarray,       # (150,) Proportion of MLP edges in s-trace at size s
        "res_prop": np.ndarray,       # (150,) Proportion of Residual edges in s-trace at size s
        "layer_props": [
            np.ndarray,               # (150,) Group 0 edge proportion (Q1: Earliest layers)
            np.ndarray,               # (150,) Group 1 edge proportion (Q2: Early-mid layers)
            np.ndarray,               # (150,) Group 2 edge proportion (Q3: Late-mid layers)
            np.ndarray                # (150,) Group 3 edge proportion (Q4: Deepest layers)
        ]
    }
}
```

### 4. processed_data_components_<len>_<files>.pkl
```json
{
    "<model_name>": {
        "size": np.ndarray,           # (150,) Logarithmic density grid s
        "comp_counts": {
            "head_0_L0": np.ndarray,  # (150,) Cumulative sentence activation count at size s
            "mlp_L2": np.ndarray,
            "res_attn_L1": np.ndarray,
            ...
        }
    }
}
```

### 5. data_aggregated_emnlp26.pkl
In addition, you will find in this folder the `data_aggregated_emnlp26.pkl` file, containing all consolidated experimental data evaluated in the EMNLP 2026 paper across all models and datasets.

--------------------------------------------------------------------------------

## How To process_hidden_states.py
This script ggregates continuous hidden states across graph extraction regimes (`trace_only`, `random_only`, `trace_inverse`) to evaluate trajectory dynamics, structural alignment, and geometric metrics across trace strata.

### Key Features**
* **Trajectory Dynamics**: Tracks step-wise L2 velocity, consecutive cosine distance, and Menger curvature across trace strata.
* **Distance & Alignment**: Computes mean pairwise Cosine and Euclidean (L2) distance matrices, state magnitudes, and Linear Centered Kernel Alignment (CKA) using optimized Gram matrix centering.
* **Information Imbalance**: Calculates asymmetric k-nearest-neighbor rank imbalance matrices across strata using `dadapy`.
* **Dimensionality Reduction**: Generates 2D t-SNE projections sampled across configuration representations.

### CLI Options
* `--hidden_dir` (Required): Path to directory containing `hidden_strata_*.npz` files.
* `--strace_dir` (Required): Path to directory containing `strace_final_*.npz` files.
* `--output_dir` (Default: `"processed_data"`): Directory where precomputed matrices and projections are saved.
* `--max_tsne_samples` (Default: `3000`): Maximum vector sample size per configuration for t-SNE execution.
* `--ii_k` (Default: `1`): Number of nearest neighbors (k) for Information Imbalance calculations.
* **Outputs**: `aggregated_matrices.npz`, `tsne_embeddings_summary.tsv`
--------------------------------------------------------------------------------

## Visualization Scripts for Model Graph & Trajectory Analysis

Plotting scripts to reproduce the figures from the EMNLP 2026 paper using these generated .pkl files have been added to the `plots/` folder.