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

## Visualization Scripts for Model Graph & Trajectory Analysis

Plotting scripts to reproduce the figures from the EMNLP 2026 paper using these generated .pkl files have been added to the `plots/` folder. It contains Python plotting scripts for analyzing graph trace sizes, reconstruction errors, layer/component distributions, Spearman correlations, and hidden representation dynamics across language models.

### Prerequisites & Dependencies

Required Python libraries:
* matplotlib
* seaborn
* pandas
* numpy
* scipy
* scienceplots (optional; falls back to standard styles if unavailable)

LaTeX environment (optional):
* TeX Live / MacTeX (installed at standard paths like `/Library/TeX/texbin`) for rendering TeX labels in `scienceplots`.

---

### Script Descriptions

* **plot_trace_vs_tv.py**
  * **Purpose**: Plots Reconstruction Error (Total Variation) against Relative Trace Size across models.
  * **Key Features**: Highlights two key execution regimes—Construction Phase (1e-4 to 1e-2) and Refinement Phase (1e-2 to 1.0)—and exports structured plot data to CSV.
  * **Input Data**: `processed_data_agg_40.pkl`
  * **Outputs**: `fig1_trace_vs_tv.pdf`, `raw_data_trace_vs_tv.csv`

* **plot_trace_vs_tv_inverse.py**
  * **Purpose**: Plots TV reconstruction error using inverse trace metrics (`tv_inv`) as a function of trace size.
  * **Input Data**: `processed_data_agg_40_5000.pkl`
  * **Outputs**: `fig_trace_vs_tv_inverse.pdf`

* **plot_trace_vs_random.py**
  * **Purpose**: Compares trace extraction algorithms against a baseline random edge removal strategy.
  * **Key Features**: Plots trace performance (solid lines) directly alongside random baseline performance (dashed lines) for each evaluated model.
  * **Input Data**: `processed_data_agg_40.pkl`
  * **Outputs**: `figA_trace_vs_random.pdf`

* **plot_trace_vs_nu.py**
  * **Purpose**: Evaluates required relative graph size against nucleus targets ($k$).
  * **Key Features**: Computes median values with 25th and 75th percentile error bounds across nucleus sampling targets ($k \in \{1, 5, 10, 20, 40, 60, 80, 90\}$). Includes horizontal phase annotations.
  * **Input Data**: `processed_data_agg_40.pkl`
  * **Outputs**: `fig2_nucleus_size.pdf`

* **plot_pairwise_correlations.py**
  * **Purpose**: Parses Spearman rank correlation ($\rho$) logs between model pairs and displays a model similarity matrix.
  * **Key Features**: Generates a lower-triangle heatmap highlighting pairwise correlations between LLMs (Llama 3.1, Mistral, OLMo 2, Qwen 2.5/3, DeepSeek, Phi-4).
  * **Outputs**: `heatmap_rho.pdf`

* **plot_layer_distribution.py**
  * **Purpose**: Visualizes how model edge allocation shifts across depth (Initial, Early-mid, Late-mid, and Last layers) as trace size expands.
  * **Key Features**: Generates multi-page stacked area plots per model, with dashed vertical overlays representing nucleus target medians ($k=1, 40, 80$).
  * **CLI Options**: `--length` (Default: `"40"`)
  * **Input Data**: `processed_data_topology_{length}.pkl`, `processed_data_agg_{length}.pkl`
  * **Outputs**: `fig4_layer_composition_with_nu_{length}.pdf`

* **plot_component_distribution.py**
  * **Purpose**: Analyzes edge composition partitioned by architectural component (Residual, Attention, MLP).
  * **Key Features**: Multi-page PDF stackplots showing component proportions across graph sizes, overlaid with nucleus target thresholds ($k=1, 40, 80$).
  * **CLI Options**: `--length` (Default: `"40"`)
  * **Input Data**: `processed_data_topology_{length}.pkl`, `processed_data_agg_{length}.pkl`
  * **Outputs**: `fig5_component_composition_with_nu_{length}.pdf`

* **plot_component_frequency.py**
  * **Purpose**: Generates Lorenz curves (cumulative distribution) showing edge concentration across components.
  * **Key Features**: Uses continuous logarithmic color scaling (`plasma`) across trace sizes with visual highlights at $s=10^{-3}$ and phase markers.
  * **CLI Options**: `--length` (Default: `"32"`)
  * **Input Data**: `processed_data_components_{length}_1000.pkl`
  * **Outputs**: `fig6_component_frequency_{length}.pdf`

* **plot_hidden_analysis.py**
  * **Purpose**: Comprehensive geometric and trajectory analysis of hidden state representations.
  * **Key Features**:
    1. Distance Matrices: Heatmaps for Cosine Distance, L2 Distance, CKA Alignment, and Information Imbalance.
    2. Trajectory Dynamics: 4-panel grid evaluating L2 Velocity ($\Vert \Delta h \Vert$), Cosine Distance ($1 - S_c$), Menger Curvature ($\kappa_M$), and Magnitude ($\Vert h \Vert$).
    3. Exports a Markdown table comparing CKA alignment against final trace sizes.
  * **CLI Options**: `--data_dir` (Default: `"processed_data"`), `--output_plots_dir` (Default: `"analysis_plots"`)
  * **Input Data**: `aggregated_matrices.npz`, `tsne_embeddings_summary.tsv`
  * **Outputs**: PDF plots and `cka_analysis_vs_final.md` saved into the specified output directory.

---

### Execution Examples

Run scripts with standard defaults:
```bash
python plot_trace_vs_tv.py
python plot_pairwise_correlations.py
python plot_hidden_analysis.py --data_dir ./processed_data --output_plots_dir ./analysis_plots
```