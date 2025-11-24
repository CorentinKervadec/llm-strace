import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr, gaussian_kde
import glob
from collections import defaultdict
import warnings
import re
from matplotlib.lines import Line2D # Import Line2D for custom legend

# Filter out common Matplotlib/NumPy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Data Loading Functions ---

def load_token_counts(file_path):
    """
    Load a text file where each line contains tab-separated fields: token_id, count.
    """
    if not file_path or not os.path.exists(file_path):
        return None
    
    print(f"  Loading token counts from: {os.path.basename(file_path)}")
    token_counts = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            start_idx = 1 if "count" in lines[0].lower() else 0
            for line in lines[start_idx:]: 
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        token_str, count = parts[0], parts[1]
                        token_counts[token_str] = int(count)
                except ValueError:
                    continue
    except Exception as e:
        print(f"  Warning: Failed to load token counts: {e}")
        return None
        
    return token_counts

def parse_olmo_checkpoint(model_dir_name):
    """
    Parses OLMo directory names to extract training progress.
    Returns:
        tuple: (stage_int, tokens_float, display_name)
    """
    name = os.path.basename(model_dir_name)
    
    # Regex for intermediate checkpoints
    match = re.search(r'stage(\d+)-step(\d+)-tokens(\d+\.?\d*)B', name)
    
    if match:
        stage = int(match.group(1))
        tokens = float(match.group(3))
        return stage, tokens, f"S{stage}-{tokens}B"
    
    # regex when the name also include the ingredient (likely for stage 2)
    match_ingredient = re.search(r'stage(\d+)-ingredient(\d+)-step(\d+)-tokens(\d+\.?\d*)B', name)
    if match_ingredient:
        stage = int(match_ingredient.group(1))
        ingredient = int(match_ingredient.group(2))
        step = int(match_ingredient.group(3))
        tokens = float(match_ingredient.group(4))
        return stage, tokens, f"Stage {stage} ({tokens}B)"

    # Regex for just step/tokens if stage is missing (sometimes happens)
    match_step = re.search(r'step(\d+)-tokens(\d+\.?\d*)B', name)
    if match_step:
        tokens = float(match_step.group(2))
        return 1, tokens, f"S1-{tokens}B"

    # Base model (Final)
    if name == "OLMo-2-0425-1B" or name.endswith("OLMo-2-0425-1B"):
        return 99, 9999.0, "Final"
        
    return -1, -1, name # Unknown format

def get_olmo_color_map(sorted_models):
    """
    Creates a color mapping where:
    - Hue = Stage
    - Shade/Alpha = Progress (Tokens) within that stage
    """
    # 1. Group models by stage
    stage_groups = defaultdict(list)
    for m_name in sorted_models:
        stage, tokens, display_name = parse_olmo_checkpoint(m_name)
        if stage != -1:
            stage_groups[stage].append((tokens, m_name, display_name))
    
    model_to_color = {}
    
    # Use a qualitative colormap for Stages (e.g., Set1)
    base_cmap = plt.get_cmap("Set1") 
    
    sorted_stages = sorted(stage_groups.keys())
    
    for i, stage in enumerate(sorted_stages):
        if stage == 99:
            base_color = np.array([0.0, 0.0, 0.0, 1.0]) # Black for Final
        else:
            base_color = np.array(base_cmap(i % 8))
        
        # Sort checkpoints within stage by tokens
        checkpoints = sorted(stage_groups[stage], key=lambda x: x[0])
        
        # Assign shades (using alpha channel: Light -> Dark)
        for j, (tokens, m_name, _) in enumerate(checkpoints):
            if len(checkpoints) > 1:
                # Progress alpha from 0.4 (lightest) to 1.0 (darkest/most saturated)
                intensity = 0.4 + 0.6 * (j / (len(checkpoints) - 1))
            else:
                intensity = 1.0
            
            color = base_color.copy()
            color[3] = intensity # Set alpha
            model_to_color[m_name] = color
            
    return model_to_color

def load_model_data(model_dir, dir_suffix, count_file_path=None, load_individual_traces=False):
    """
    Loads data for a single model. If load_individual_traces is True, returns 
    a dictionary mapping filename to a tuple of (tv_array, size_array).
    """
    full_path = os.path.join(model_dir, dir_suffix)
    
    file_paths = glob.glob(os.path.join(full_path, '*.npz'))
    if not file_paths:
         file_paths = glob.glob(os.path.join(model_dir, '*.npz'))

    if len(file_paths) < 500 and not load_individual_traces: 
        return None

    if load_individual_traces:
        print(f"  Loading individual TV traces for {os.path.basename(model_dir)}...")
        traces = {}
        for f_path in file_paths:
             try:
                data = np.load(f_path, allow_pickle=True)
                rel_size = data['strata_rel_size']
                tv = data['strata_reco_tv'].item()['trace']['only']
                traces[os.path.basename(f_path)] = (tv, rel_size)
             except Exception:
                continue
        return traces
    
    # Standard loading for aggregate plots
    print(f"  Loading {len(file_paths)} files from {os.path.basename(model_dir)}...")

    token_counts = load_token_counts(count_file_path)
    norm_token_counts = {}
    if token_counts:
        total_count = sum(token_counts.values())
        norm_token_counts = {k: float(v)/total_count for k, v in token_counts.items()}

    sentence_data = {'loss': [], 'entropy': [], 'pred_token_count': [], 'avg_rel_size': [], 'auc': {}}
    
    metric_keys = ['strata_reco_tv', 'strata_reco_nu']
    eval_modes = ['trace']
    sub_modes = ['only']

    for f_path in file_paths:
        try:
            data = np.load(f_path, allow_pickle=True)
            
            rel_size = data['strata_rel_size']
            sort_idx = np.argsort(rel_size)
            size_sorted = np.array(rel_size)[sort_idx]
            sentence_data['avg_rel_size'].append(np.mean(size_sorted))
            
            for key in metric_keys:
                for mode in eval_modes:
                    for sub_mode in sub_modes:
                        full_key = f"{key}_{mode}_{sub_mode}"
                        raw_metric = data[key].item()[mode][sub_mode]
                        metric_sorted = np.array(raw_metric)[sort_idx]
                        metric_auc = np.trapz(metric_sorted, size_sorted)
                        
                        if full_key not in sentence_data['auc']:
                            sentence_data['auc'][full_key] = []
                        sentence_data['auc'][full_key].append(metric_auc)
            
            full_idx = np.argmax(rel_size)
            s_loss = data["strata_loss"].item()['trace']['only'][full_idx]
            s_entropy = data["strata_entropy"].item()['trace']['only'][full_idx]
            
            sentence_data["loss"].append(s_loss)
            sentence_data["entropy"].append(s_entropy)

            if norm_token_counts:
                nucleus_60 = data["nucleus_60"].item()['trace']['only'][full_idx]
                freqs = [norm_token_counts[tkn] for tkn in nucleus_60 if tkn in norm_token_counts]
                if freqs:
                    sentence_data["pred_token_count"].append(np.mean(freqs))
                else:
                    sentence_data["pred_token_count"].append(np.nan)

        except Exception:
            continue

    n_valid = np.sum(np.isfinite(sentence_data["loss"]) & np.isfinite(sentence_data["entropy"]))
    if n_valid < 500: 
        print(f"... Only {n_valid} valid points: skipping.")
        return None
            
    return sentence_data

# --- 2. Plotting Functions ---

def plot_training_trajectory(all_models_data, model_colors, output_pdf):
    """
    Scatter plot showing evolution of Mean AUC vs Training step index.
    """
    print(f"Generating Training Trajectory Plots -> {output_pdf}...")
    
    first_model = next(iter(all_models_data.values()))
    auc_keys = list(first_model['auc'].keys())
    
    # Sort models by (Stage, Tokens)
    sorted_models = sorted(all_models_data.keys(), key=lambda x: parse_olmo_checkpoint(x)[:2])
    
    with PdfPages(output_pdf) as pdf:
        for auc_key in auc_keys:
            fig, ax = plt.subplots(figsize=(14, 9))
            
            checkpoint_handles = []
            checkpoint_labels = []
            
            # 1. Plot Checkpoints
            for i, model_name in enumerate(sorted_models):
                data = all_models_data[model_name]
                if auc_key not in data['auc']: continue
                
                x_val = i
                y_val = np.nanmean(data['auc'][auc_key])
                y_err = np.nanstd(data['auc'][auc_key])
                
                color = model_colors[model_name]
                _, _, display_name = parse_olmo_checkpoint(model_name)
                
                handle = ax.errorbar(x_val, y_val, yerr=y_err, fmt='o', color=color, 
                                     capsize=5, markersize=8, alpha=0.8, 
                                     label=display_name, zorder=2)
                
                checkpoint_handles.append(handle[0]) 
                checkpoint_labels.append(display_name)
                
                if "Final" in display_name:
                    ax.text(x_val, y_val, "Final", fontsize=9, ha='center', va='bottom', fontweight='bold')

            # 2. Plot Trajectory Lines
            current_stage = -1
            stage_x = []
            stage_y = []
            
            for model_name in sorted_models:
                stage, _, _ = parse_olmo_checkpoint(model_name)
                data = all_models_data[model_name]
                x_val = sorted_models.index(model_name)
                y_val = np.nanmean(data['auc'][auc_key])
                
                if stage != current_stage:
                    if stage_x:
                        ax.plot(stage_x, stage_y, color='gray', linestyle='--', alpha=0.3, zorder=1)
                    current_stage = stage
                    stage_x = [x_val]
                    stage_y = [y_val]
                else:
                    stage_x.append(x_val)
                    stage_y.append(y_val)
            
            if stage_x:
                 trajectory_handle = ax.plot(stage_x, stage_y, color='gray', linestyle='--', alpha=0.3, zorder=1)

            
            # --- CUSTOM EXPLICIT LEGEND ---
            legend_handles = checkpoint_handles
            legend_labels = checkpoint_labels
            
            if 'trajectory_handle' in locals():
                legend_handles.append(Line2D([0], [0], color='gray', linestyle='--'))
                legend_labels.append('Training Trajectory')
            
            ax.legend(legend_handles, legend_labels, title="Model Checkpoint", 
                      bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)

            # Clean X labels
            labels = [parse_olmo_checkpoint(m)[2] for m in sorted_models]
                
            ax.set_title(f"Evolution of {auc_key} over Training Steps")
            ax.set_xlabel("Training Checkpoint Order (Early $\\to$ Late)")
            ax.set_ylabel(f"Mean AUC ($\\mathrm{{{auc_key.replace('_', ' ').replace('strata reco', 'TV/NU')}}}$)")
            ax.set_xticks(range(len(sorted_models)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            pdf.savefig(fig)
            plt.close(fig)

def plot_auc_distributions_olmo(all_models_data, model_colors, output_pdf):
    """
    Generates KDE plots, colored by stage/step, with explicit legend.
    """
    print(f"Generating Distribution Plots -> {output_pdf}...")
    
    first_model = next(iter(all_models_data.values()))
    auc_keys = list(first_model['auc'].keys())
    sorted_models = sorted(all_models_data.keys(), key=lambda x: parse_olmo_checkpoint(x)[:2])
    
    with PdfPages(output_pdf) as pdf:
        for auc_key in auc_keys:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            legend_handles = []
            legend_labels = []
            
            for model_name in sorted_models:
                data = all_models_data[model_name]
                if auc_key not in data['auc']: continue
                
                aucs = np.array(data['auc'][auc_key])
                if len(aucs) < 2 or np.std(aucs) == 0: continue
                    
                try:
                    density = gaussian_kde(aucs)
                    x_min, x_max = aucs.min(), aucs.max()
                    xs = np.linspace(x_min, x_max, 200)
                    ys = density(xs)
                    ys_norm = ys / ys.max()

                    color = model_colors[model_name]
                    _, _, display_name = parse_olmo_checkpoint(model_name)
                    
                    line, = ax.plot(xs, ys_norm, color=color, linewidth=2, label=display_name)
                    
                    legend_handles.append(line)
                    legend_labels.append(display_name)
                    
                except Exception:
                    continue

            ax.set_title(f"Normalized Distribution of AUC Scores: {auc_key}")
            ax.set_xlabel(f"AUC ($\\mathrm{{{auc_key.replace('_', ' ').replace('strata reco', 'TV/NU')}}}$)")
            ax.set_ylabel("Normalized Density (Peak = 1.0)")
            
            ax.legend(legend_handles, legend_labels, title="Model Checkpoint (S=Stage, B=Billion Tokens)", 
                      bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
            
            ax.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            pdf.savefig(fig)
            plt.close(fig)


def plot_sentence_tv_evolution(all_model_traces, model_colors, sorted_models, output_pdf):
    """
    Plots the TV vs Graph Size curve for the same 10 sentences across all checkpoints.
    """
    print(f"Generating Sentence TV Evolution Plots -> {output_pdf}...")
    
    if not all_model_traces:
        print("No model trace data available for sentence analysis.")
        return
        
    # 1. Find intersection of file names across all models
    common_files = None
    for model_name, traces in all_model_traces.items():
        current_files = set(traces.keys())
        if common_files is None:
            common_files = current_files
        else:
            common_files = common_files.intersection(current_files)
            
    if not common_files:
        print("No common sentence files found across all checkpoints. Cannot plot evolution.")
        return
        
    # Select the first 10 common files based on alphabetical order (proxy for consistency)
    selected_files = sorted(list(common_files))[:10]
    
    print(f"Plotting TV evolution for {len(selected_files)} common sentences.")

    with PdfPages(output_pdf) as pdf:
        for file_name in selected_files:
            fig, ax = plt.subplots(figsize=(10, 8))
            legend_handles = []
            legend_labels = []
            
            # Plot each checkpoint's TV curve for this sentence
            for model_name in sorted_models:
                if model_name not in all_model_traces or file_name not in all_model_traces[model_name]:
                    continue
                    
                # Data: (tv_array, size_array)
                tv_data, size_data = all_model_traces[model_name][file_name]
                
                # Sort data by size
                sort_idx = list(np.argsort(size_data))
                
                size_sorted = size_data[sort_idx]
                tv_sorted = np.array(tv_data)[sort_idx]

                color = model_colors[model_name]
                _, _, display_name = parse_olmo_checkpoint(model_name)

                # Plot the TV trace
                line, = ax.plot(size_sorted, tv_sorted, color=color, linewidth=2, label=display_name)
                
                legend_handles.append(line)
                legend_labels.append(display_name)
            
            # --- Final Plot Styling ---
            ax.set_title(f"Sentence TV Evolution: {file_name}")
            ax.set_xlabel("Relative Graph Size")
            ax.set_ylabel("Total Variation ($\mathrm{TV = strata\\_reco\\_tv}$)")
            
            # Ensure X-axis is 0 to 1
            ax.set_xlim(0, 1.05) 
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Explicit legend for all checkpoints
            ax.legend(legend_handles, legend_labels, title="Model Checkpoint", 
                      bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
            
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            pdf.savefig(fig)
            plt.close(fig)


# --- 3. Main ---

def main():
    parser = argparse.ArgumentParser(description="OLMo Evolution Analysis")
    parser.add_argument('--base_result_dir', type=str, required=True, 
                        help='Directory containing model subfolders.')
    parser.add_argument('--dir', type=str, required=True, 
                        help='Directory inside model folder containing the npz (e.g., final_straces_20).')
    parser.add_argument('--token_count_dir', type=str, default=None,
                        help='Optional directory containing token counts.')
    parser.add_argument('--output_prefix', type=str, default="olmo_evolution", 
                        help='Prefix for output PDFs')
    args = parser.parse_args()

    if not os.path.isdir(args.base_result_dir):
        print(f"Error: {args.base_result_dir} not found.")
        return

    # 1. Find only OLMo directories
    all_dirs = os.listdir(args.base_result_dir)
    olmo_dirs = [d for d in all_dirs if "OLMo-2-0425-1B" in d and os.path.isdir(os.path.join(args.base_result_dir, d))]
    
    if not olmo_dirs:
        print("No OLMo-2-0425-1B directories found.")
        return

    print(f"Found {len(olmo_dirs)} OLMo checkpoints. Processing...")
    
    all_models_data = {} # For aggregate AUC/Corr plots
    all_model_traces = {} # For individual sentence plots

    # Prepare colors and sorting order *before* loading all data
    sorted_models = sorted(olmo_dirs, key=lambda x: parse_olmo_checkpoint(x)[:2])
    model_colors = get_olmo_color_map(sorted_models)

    for model_name in sorted_models:
        full_path = os.path.join(args.base_result_dir, model_name)
        
        # Token count handling (only needed for aggregate)
        count_file = None
        if args.token_count_dir:
            base_counts = glob.glob(os.path.join(args.token_count_dir, "*OLMo-2-0425-1B*token_counts.tsv"))
            if base_counts:
                count_file = base_counts[0]
        
        print(f"\n--- Processing: {model_name} ---")
        
        # Load Aggregate Data
        model_data = load_model_data(full_path, args.dir, count_file, load_individual_traces=False)
        if model_data:
            all_models_data[model_name] = model_data
        
        # Load Individual Traces for TV evolution plot
        model_traces = load_model_data(full_path, args.dir, load_individual_traces=True)
        if model_traces:
            all_model_traces[model_name] = model_traces
        
    if not all_models_data:
        print("No valid aggregate data loaded. Skipping AUC/Trajectory plots.")
    else:
        # 3. Plots (Aggregate)
        plot_training_trajectory(all_models_data, model_colors, f"{args.output_prefix}_trajectory.pdf")
        plot_auc_distributions_olmo(all_models_data, model_colors, f"{args.output_prefix}_distributions.pdf")

    if not all_model_traces:
        print("No valid trace data loaded. Skipping Sentence TV plots.")
    else:
        # 4. Plots (Individual Sentence TV)
        plot_sentence_tv_evolution(all_model_traces, model_colors, sorted_models, f"{args.output_prefix}_sentence_tv_evolution.pdf")


if __name__ == "__main__":
    main()

# import argparse
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# from matplotlib.backends.backend_pdf import PdfPages
# from scipy.stats import pearsonr, gaussian_kde
# import glob
# from collections import defaultdict
# import warnings
# import re

# # --- 1. Data Loading Functions ---

# def load_token_counts(file_path):
#     """
#     Load a text file where each line contains tab-separated fields: token_id, count.
#     """
#     if not file_path or not os.path.exists(file_path):
#         return None
    
#     print(f"  Loading token counts from: {os.path.basename(file_path)}")
#     token_counts = {}
#     try:
#         with open(file_path, 'r') as f:
#             lines = f.readlines()
#             start_idx = 1 if "count" in lines[0].lower() else 0
#             for line in lines[start_idx:]: 
#                 try:
#                     parts = line.strip().split('\t')
#                     if len(parts) >= 2:
#                         token_str, count = parts[0], parts[1]
#                         token_counts[token_str] = int(count)
#                 except ValueError:
#                     continue
#     except Exception as e:
#         print(f"  Warning: Failed to load token counts: {e}")
#         return None
        
#     return token_counts

# def parse_olmo_checkpoint(model_dir_name):
#     """
#     Parses OLMo directory names to extract training progress.
#     Expected formats:
#     1. 'OLMo-2-0425-1B' (The final model -> Stage 3/Final)
#     2. 'OLMo-2-0425-1B_stage[S]-step[T]-tokens[K]B'
    
#     Returns:
#         tuple: (stage_int, tokens_float, display_name)
#     """
#     name = os.path.basename(model_dir_name)
    
#     # Regex for intermediate checkpoints
#     # Matches: ...stage1-step1000-tokens5B...
#     match = re.search(r'stage(\d+)-step(\d+)-tokens(\d+\.?\d*)B', name)
    
#     if match:
#         stage = int(match.group(1))
#         step = int(match.group(2))
#         tokens = float(match.group(3))
#         return stage, tokens, f"Stage {stage} ({tokens}B)"
    
#     # regex when the name also include the ingredient (likely for stage 2)
#     match_ingredient = re.search(r'stage(\d+)-ingredient(\d+)-step(\d+)-tokens(\d+\.?\d*)B', name)
#     if match_ingredient:
#         stage = int(match_ingredient.group(1))
#         ingredient = int(match_ingredient.group(2))
#         step = int(match_ingredient.group(3))
#         tokens = float(match_ingredient.group(4))

#         return stage, tokens, f"Stage {stage} ({tokens}B)"

#     # Regex for just step/tokens if stage is missing (sometimes happens)
#     match_step = re.search(r'step(\d+)-tokens(\d+\.?\d*)B', name)
#     if match_step:
#         # Assume Stage 1 if not specified, or infer based on token count?
#         # For now, let's assume Stage 1
#         tokens = float(match_step.group(2))
#         return 1, tokens, f"Stage 1 ({tokens}B)"

#     # Base model (Final)
#     if name == "OLMo-2-0425-1B" or name.endswith("OLMo-2-0425-1B"):
#         # Treat as Stage 3 (or whatever is after Stage 2) with high token count
#         return 99, 9999.0, "Final Model"
        
#     return -1, -1, name # Unknown format

# def get_olmo_color_map(sorted_models):
#     """
#     Creates a color mapping where:
#     - Hue = Stage
#     - Shade/Alpha = Progress (Tokens) within that stage
#     """
#     # 1. Group models by stage
#     stage_groups = defaultdict(list)
#     for m_name in sorted_models:
#         stage, tokens, _ = parse_olmo_checkpoint(m_name)
#         if stage != -1:
#             stage_groups[stage].append((tokens, m_name))
    
#     model_to_color = {}
    
#     # 2. Assign colors
#     # Use a qualitative colormap for Stages (e.g., Set1 or Dark2)
#     # We skip a few to ensure distinct colors
#     base_cmap = plt.get_cmap("Set1") 
    
#     sorted_stages = sorted(stage_groups.keys())
    
#     for i, stage in enumerate(sorted_stages):
#         # Get base color for this stage
#         # If stage is 99 (Final), make it black or distinct
#         if stage == 99:
#             base_color = np.array([0.0, 0.0, 0.0, 1.0]) # Black
#         else:
#             base_color = np.array(base_cmap(i % 8))
        
#         # Sort checkpoints within stage by tokens
#         checkpoints = sorted(stage_groups[stage], key=lambda x: x[0])
        
#         # Assign shades
#         # We vary alpha or lightness. Let's vary Alpha for simplicity in overlap,
#         # or blend with white for lightness.
        
#         for j, (tokens, m_name) in enumerate(checkpoints):
#             # Progress from 0.3 to 1.0 intensity
#             if len(checkpoints) > 1:
#                 intensity = 0.3 + 0.7 * (j / (len(checkpoints) - 1))
#             else:
#                 intensity = 1.0
            
#             # Create color: base_color with adjusted alpha/intensity
#             # Actually, varying alpha makes it hard to see in scatter plots.
#             # Let's darken it. Light -> Dark.
#             # Or just use the alpha channel if plotting supports it well.
#             color = base_color.copy()
#             color[3] = intensity # Set alpha
#             model_to_color[m_name] = color
            
#     return model_to_color

# def load_model_data(model_dir, dir_suffix, count_file_path=None):
#     """
#     Loads data for a single model.
#     """
#     full_path = os.path.join(model_dir, dir_suffix)
    
#     # Find all .npz files
#     file_paths = glob.glob(os.path.join(full_path, '*.npz'))
    
#     # Retry simply inside the folder if suffix logic failed
#     if not file_paths:
#          file_paths = glob.glob(os.path.join(model_dir, '*.npz'))

#     if len(file_paths) < 500: # Lower threshold for intermediate checkpoints?
#         print(f"  Skipping {os.path.basename(model_dir)}: Not enough files ({len(file_paths)})")
#         return None

#     print(f"  Loading {len(file_paths)} files from {os.path.basename(model_dir)}...")

#     # Load token counts if available
#     token_counts = load_token_counts(count_file_path)
#     norm_token_counts = {}
#     if token_counts:
#         total_count = sum(token_counts.values())
#         norm_token_counts = {k: float(v)/total_count for k, v in token_counts.items()}

#     sentence_data = {'loss': [], 'entropy': [], 'pred_token_count': [], 'avg_rel_size': [], 'auc': {}}
    
#     metric_keys = ['strata_reco_tv', 'strata_reco_nu']
#     eval_modes = ['trace']
#     sub_modes = ['only']

#     for f_path in file_paths:
#         try:
#             data = np.load(f_path, allow_pickle=True)
            
#             rel_size = data['strata_rel_size']
#             sort_idx = np.argsort(rel_size)
#             size_sorted = np.array(rel_size)[sort_idx]
#             sentence_data['avg_rel_size'].append(np.mean(size_sorted))
            
#             for key in metric_keys:
#                 for mode in eval_modes:
#                     for sub_mode in sub_modes:
#                         full_key = f"{key}_{mode}_{sub_mode}"
#                         raw_metric = data[key].item()[mode][sub_mode]
#                         metric_sorted = np.array(raw_metric)[sort_idx]
#                         metric_auc = np.trapz(metric_sorted, size_sorted)
                        
#                         if full_key not in sentence_data['auc']:
#                             sentence_data['auc'][full_key] = []
#                         sentence_data['auc'][full_key].append(metric_auc)
            
#             full_idx = np.argmax(rel_size)
#             s_loss = data["strata_loss"].item()['trace']['only'][full_idx]
#             s_entropy = data["strata_entropy"].item()['trace']['only'][full_idx]
            
#             sentence_data["loss"].append(s_loss)
#             sentence_data["entropy"].append(s_entropy)

#             if norm_token_counts:
#                 nucleus_60 = data["nucleus_60"].item()['trace']['only'][full_idx]
#                 freqs = [norm_token_counts[tkn] for tkn in nucleus_60 if tkn in norm_token_counts]
#                 if freqs:
#                     sentence_data["pred_token_count"].append(np.mean(freqs))
#                 else:
#                     sentence_data["pred_token_count"].append(np.nan)

#         except Exception:
#             continue

#     n_valid = np.sum(np.isfinite(sentence_data["loss"]) & np.isfinite(sentence_data["entropy"]))
#     if n_valid < 500: 
#         print(f"... Only {n_valid} valid points: skipping.")
#         return None
            
#     return sentence_data

# # --- 2. Plotting Functions ---

# def plot_training_trajectory(all_models_data, model_colors, output_pdf):
#     """
#     Scatter plot showing evolution of Mean AUC vs Mean Loss over training steps.
#     """
#     print(f"Generating Training Trajectory Plots -> {output_pdf}...")
    
#     first_model = next(iter(all_models_data.values()))
#     auc_keys = list(first_model['auc'].keys())
    
#     # Sort models by (Stage, Tokens)
#     sorted_models = sorted(all_models_data.keys(), key=lambda x: parse_olmo_checkpoint(x)[:2])

#     with PdfPages(output_pdf) as pdf:
#         for auc_key in auc_keys:
#             fig, ax = plt.subplots(figsize=(14, 9))
            
#             xs = []
#             ys = []
#             colors = []
#             sizes = []
            
#             for model_name in sorted_models:
#                 data = all_models_data[model_name]
#                 if auc_key not in data['auc']: continue
                
#                 # X: Mean Loss (Proxy for performance)
#                 # OR use Token Count as X-axis? Let's use Token Count from name
#                 stage, tokens, _ = parse_olmo_checkpoint(model_name)
                
#                 # Let's try plotting Metric vs Training Tokens (Cumulative)
#                 # This requires normalizing tokens across stages, which is hard.
#                 # Let's just stick to AUC (Y) vs Loss (X) to see the Pareto frontier?
#                 # OR AUC (Y) vs Training Step (X - simplified)
                
#                 # Using training sequence index as X
#                 x_val = sorted_models.index(model_name)
                
#                 y_val = np.nanmean(data['auc'][auc_key])
#                 y_err = np.nanstd(data['auc'][auc_key])
                
#                 color = model_colors[model_name]
                
#                 ax.errorbar(x_val, y_val, yerr=y_err, fmt='o', color=color, capsize=5, alpha=0.8)
                
#                 # Annotate stages
#                 if "Final" in model_name:
#                     ax.text(x_val, y_val, "Final", fontsize=9, ha='center', va='bottom', fontweight='bold')
#                 elif tokens > 0: # Only label some points to avoid clutter
#                      # ax.text(x_val, y_val, f"{tokens}B", fontsize=8, ha='center', va='bottom')
#                      pass

#             # Connect the dots to show trajectory
#             # Group indices by stage to draw separate lines
#             current_stage = -1
#             stage_x = []
#             stage_y = []
            
#             for model_name in sorted_models:
#                 stage, _, _ = parse_olmo_checkpoint(model_name)
#                 if stage != current_stage:
#                     if stage_x:
#                         ax.plot(stage_x, stage_y, color='gray', linestyle='--', alpha=0.3)
#                     current_stage = stage
#                     stage_x = []
#                     stage_y = []
                
#                 data = all_models_data[model_name]
#                 x_val = sorted_models.index(model_name)
#                 y_val = np.nanmean(data['auc'][auc_key])
#                 stage_x.append(x_val)
#                 stage_y.append(y_val)
            
#             # Plot last segment
#             if stage_x:
#                  ax.plot(stage_x, stage_y, color='gray', linestyle='--', alpha=0.3)

#             # Custom Legend for Stages
#             from matplotlib.lines import Line2D
#             legend_elements = [
#                 Line2D([0], [0], marker='o', color='w', label='Stage 1', markerfacecolor=plt.get_cmap("Set1")(0), markersize=10),
#                 Line2D([0], [0], marker='o', color='w', label='Stage 2', markerfacecolor=plt.get_cmap("Set1")(1), markersize=10),
#                 Line2D([0], [0], marker='o', color='w', label='Final', markerfacecolor='black', markersize=10),
#                 Line2D([0], [0], color='gray', linestyle='--', label='Training Trajectory')
#             ]
            
#             ax.set_title(f"Evolution of {auc_key} over Training Steps")
#             ax.set_xlabel("Training Checkpoint Order (Early -> Late)")
#             ax.set_ylabel(f"Mean AUC")
#             ax.set_xticks(range(len(sorted_models)))
            
#             # Clean X labels
#             labels = []
#             for m in sorted_models:
#                 s, t, _ = parse_olmo_checkpoint(m)
#                 if s == 99: labels.append("Final")
#                 else: labels.append(f"S{s}-{t}B")
#             ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            
#             ax.grid(True, linestyle='--', alpha=0.5)
#             ax.legend(handles=legend_elements, title="Training Phase")
            
#             plt.tight_layout()
#             pdf.savefig(fig)
#             plt.close(fig)

# def plot_auc_distributions_olmo(all_models_data, model_colors, output_pdf):
#     """
#     Generates KDE plots, colored by stage/step.
#     """
#     print(f"Generating Distribution Plots -> {output_pdf}...")
    
#     first_model = next(iter(all_models_data.values()))
#     auc_keys = list(first_model['auc'].keys())
#     sorted_models = sorted(all_models_data.keys(), key=lambda x: parse_olmo_checkpoint(x)[:2])
    
#     with PdfPages(output_pdf) as pdf:
#         for auc_key in auc_keys:
#             fig, ax = plt.subplots(figsize=(12, 8))
            
#             for model_name in sorted_models:
#                 data = all_models_data[model_name]
#                 if auc_key not in data['auc']: continue
                
#                 aucs = np.array(data['auc'][auc_key])
#                 if len(aucs) < 2 or np.std(aucs) == 0: continue
                    
#                 try:
#                     density = gaussian_kde(aucs)
#                     x_min, x_max = aucs.min(), aucs.max()
#                     xs = np.linspace(x_min, x_max, 200)
#                     ys = density(xs)
#                     ys_norm = ys / ys.max()

#                     color = model_colors[model_name]
                    
#                     # Plot
#                     ax.plot(xs, ys_norm, color=color, linewidth=2)
#                     # No fill to reduce clutter with many lines
#                     # ax.fill_between(xs, ys_norm, color=color, alpha=0.1)
                    
#                 except Exception:
#                     continue

#             ax.set_title(f"Distribution Evolution: {auc_key}")
#             ax.set_xlabel(f"AUC")
#             ax.set_ylabel("Normalized Density")
            
#             # Add a text box explaining the color gradient
#             text = "Color: Stage (Blue=1, Orange=2)\nShade: Token Count (Light=Early, Dark=Late)"
#             props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#             ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
#                     verticalalignment='top', bbox=props)
            
#             ax.grid(True, linestyle='--', alpha=0.6)
            
#             pdf.savefig(fig)
#             plt.close(fig)


# # --- 3. Main ---

# def main():
#     parser = argparse.ArgumentParser(description="OLMo Evolution Analysis")
#     parser.add_argument('--base_result_dir', type=str, required=True, 
#                         help='Directory containing model subfolders.')
#     parser.add_argument('--dir', type=str, required=True, 
#                         help='Directory inside model folder containing the npz (e.g., final_straces_20).')
#     parser.add_argument('--token_count_dir', type=str, default=None,
#                         help='Optional directory containing token counts.')
#     parser.add_argument('--output_prefix', type=str, default="olmo_evolution", 
#                         help='Prefix for output PDFs')
#     args = parser.parse_args()

#     if not os.path.isdir(args.base_result_dir):
#         print(f"Error: {args.base_result_dir} not found.")
#         return

#     # 1. Find only OLMo directories
#     all_dirs = os.listdir(args.base_result_dir)
#     olmo_dirs = [d for d in all_dirs if "OLMo-2-0425-1B" in d and os.path.isdir(os.path.join(args.base_result_dir, d))]
    
#     if not olmo_dirs:
#         print("No OLMo-2-0425-1B directories found.")
#         return

#     print(f"Found {len(olmo_dirs)} OLMo checkpoints. Processing...")
    
#     all_models_data = {}

#     for model_name in sorted(olmo_dirs):
#         full_path = os.path.join(args.base_result_dir, model_name)
        
#         # Token count handling
#         count_file = None
#         if args.token_count_dir:
#             # For intermediate steps, they usually share vocab with base model
#             count_file = os.path.join(args.token_count_dir, "olmo2_1B_token_counts.tsv")
        
#         print(f"\n--- Processing: {model_name} ---")
#         model_data = load_model_data(full_path, args.dir, count_file)
        
#         if model_data:
#             all_models_data[model_name] = model_data
#         else:
#             print(f"  Skipping {model_name} (Invalid).")

#     if not all_models_data:
#         print("No valid data loaded.")
#         return

#     # 2. Prepare Colors
#     # Sort strictly for coloring
#     sorted_models = sorted(all_models_data.keys(), key=lambda x: parse_olmo_checkpoint(x)[:2])
#     model_colors = get_olmo_color_map(sorted_models)

#     # 3. Plots
#     plot_training_trajectory(all_models_data, model_colors, f"{args.output_prefix}_trajectory.pdf")
#     plot_auc_distributions_olmo(all_models_data, model_colors, f"{args.output_prefix}_distributions.pdf")

# if __name__ == "__main__":
#     main()