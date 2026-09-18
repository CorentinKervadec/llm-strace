## Visualization Scripts for Model Graph & Trajectory Analysis
This folder contains Python plotting scripts for analyzing graph trace sizes, reconstruction errors, layer/component distributions, Spearman correlations, and hidden representation dynamics across language models.

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