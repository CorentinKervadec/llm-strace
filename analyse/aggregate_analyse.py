import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
import glob
from collections import defaultdict
import warnings
import re
from scipy.stats import pearsonr, gaussian_kde

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
            # Simple heuristic to skip header if it exists
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

def parse_model_info(model_name):
    """
    Parses model name to extract Family and Size.
    Expected format: "[Family]-*[Size]B*" (e.g., Qwen3-0.6B-Base)
    """
    # Remove path if present to get just the directory name
    name = os.path.basename(model_name)
    
    # Regex to find size: looks for number (int or float) followed by 'B'
    # e.g., 7B, 0.6B, 70B. Case insensitive.
    match = re.search(r'[-_]?(\d+\.?\d*)B[-_]?', name, re.IGNORECASE)
    
    if match:
        size_str = match.group(1)
        try:
            size = float(size_str)
        except ValueError:
            size = 1.0 # Fallback
        
        # Family is the part before the size match
        start_idx = match.start()
        family = name[:start_idx].strip('-_')
        if not family:
            family = "Unknown"
    else:
        # Fallback if no "B" size found
        size = 1.0 
        family = name
        
    return family, size

def load_model_data(model_dir, count_file_path=None):
    """
    Loads data for a single model.
    """

    # Find all .npz files
    file_paths = glob.glob(os.path.join(model_dir, '*.npz'))
    if len(file_paths) < 1000:
        print(f"  Skipping {os.path.basename(model_dir)}: Not enough files ({len(file_paths)} < 1000)")
        return None

    print(f"  Loading {len(file_paths)} files from {os.path.basename(model_dir)}...")

    # Load token counts if available
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
            
            # Sort by relative size for AUC calculation
            rel_size = data['strata_rel_size']
            sort_idx = np.argsort(rel_size)
            size_sorted = np.array(rel_size)[sort_idx]
            
            # Store average relative size for X-axis in scatter plot
            sentence_data['avg_rel_size'].append(np.mean(size_sorted))
            
            # 1. Calculate AUCs for metrics
            for key in metric_keys:
                for mode in eval_modes:
                    for sub_mode in sub_modes:
                        full_key = f"{key}_{mode}_{sub_mode}"
                        
                        # Extract and sort
                        raw_metric = data[key].item()[mode][sub_mode]
                        metric_sorted = np.array(raw_metric)[sort_idx]
                        
                        # Compute AUC
                        metric_auc = np.trapz(metric_sorted, size_sorted)
                        
                        if full_key not in sentence_data['auc']:
                            sentence_data['auc'][full_key] = []
                        sentence_data['auc'][full_key].append(metric_auc)
            
            # 2. Extract Full Model Metrics
            full_idx = np.argmax(rel_size)
            
            s_loss = data["strata_loss"].item()['trace']['only'][full_idx]
            s_entropy = data["strata_entropy"].item()['trace']['only'][full_idx]
            
            sentence_data["loss"].append(s_loss)
            sentence_data["entropy"].append(s_entropy)

            # 3. Calculate Token Count Frequency
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
    if n_valid < 666: 
        print(f"... Only {n_valid} loss & entropy: skipping.")
        return None # model not valid
            
    return sentence_data

# --- 2. Plotting Functions ---

def plot_correlation_summary(all_models_data, output_pdf):
    """
    Generates Bar Plots comparing correlations across models.
    """
    print(f"Generating Correlation Summary Bar Plots -> {output_pdf}...")
    
    first_model = next(iter(all_models_data.values()))
    y_keys = ['loss', 'entropy']
    if any(all_models_data[m]['pred_token_count'] for m in all_models_data):
        y_keys.append('pred_token_count')
    x_keys = list(first_model['auc'].keys())

    with PdfPages(output_pdf) as pdf:
        for y_key in y_keys:
            for x_key in x_keys:
                
                model_names = []
                correlations = []
                p_values = []
                
                for model_name in sorted(all_models_data.keys()):
                    data = all_models_data[model_name]
                    
                    if y_key not in data or not data[y_key]: continue
                    if x_key not in data['auc']: continue
                        
                    y_arr = np.array(data[y_key])
                    x_arr = np.array(data['auc'][x_key])
                    
                    mask = np.isfinite(y_arr) & np.isfinite(x_arr)
                    if np.sum(mask) < 2: continue
                        
                    r, p = pearsonr(x_arr[mask], y_arr[mask])
                    
                    model_names.append(model_name)
                    correlations.append(r)
                    p_values.append(p)
                
                if not model_names: continue

                fig, ax = plt.subplots(figsize=(max(8, len(model_names)*1.5), 6))
                
                x_pos = np.arange(len(model_names))
                bars = ax.bar(x_pos, correlations, color='skyblue', edgecolor='black', alpha=0.7)
                
                ax.set_ylabel(f"Pearson Correlation (r)")
                ax.set_title(f"Correlation: {y_key} vs. AUC({x_key})")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                ax.axhline(0, color='black', linewidth=0.8)
                ax.grid(axis='y', linestyle='--', alpha=0.5)

                for bar, p_val in zip(bars, p_values):
                    height = bar.get_height()
                    offset = 0.02 if height >= 0 else -0.05
                    ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                            f"p={p_val:.1e}",
                            ha='center', va='bottom' if height >= 0 else 'top', 
                            fontsize=9, rotation=0, color='darkred')
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

def plot_model_size_scatter(all_models_data, output_pdf):
    """
    Generates Scatter Plots summarizing models.
    X-axis: Model Size (Parameters in Billions)
    Y-axis: Mean AUC (+ Std Dev as error bars)
    Color: Model Family
    """
    print(f"Generating Model Size Scatter Plots -> {output_pdf}...")
    
    # 1. Organize data by metric (AUC keys)
    first_model = next(iter(all_models_data.values()))
    auc_keys = list(first_model['auc'].keys())
    
    # 2. Prepare Families and Colors
    all_families = set()
    for m_name in all_models_data.keys():
        fam, _ = parse_model_info(m_name)
        all_families.add(fam)
    
    unique_families = sorted(list(all_families))
    colors = cm.get_cmap('tab10', len(unique_families))
    fam_to_color = {fam: colors(i) for i, fam in enumerate(unique_families)}

    with PdfPages(output_pdf) as pdf:
        for auc_key in auc_keys:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare grouped data for plotting: family -> list of points
            grouped_data = defaultdict(lambda: {'x': [], 'y': [], 'yerr': [], 'names': []})

            for model_name in sorted(all_models_data.keys()):
                data = all_models_data[model_name]
                if auc_key not in data['auc']: continue
                
                # X: Model Size
                family, size_b = parse_model_info(model_name)
                
                # Y: Mean AUC
                aucs = np.array(data['auc'][auc_key])
                mask = np.isfinite(aucs)
                y_val = np.mean(aucs[mask])
                y_err = np.std(aucs[mask])
                
                # Store grouped
                grouped_data[family]['x'].append(size_b)
                grouped_data[family]['y'].append(y_val)
                grouped_data[family]['yerr'].append(y_err)
                grouped_data[family]['names'].append(model_name)
            
            # Plot each family series
            for family, f_data in grouped_data.items():
                color = fam_to_color[family]
                
                # Plot points with error bars
                # fmt='o' plots markers, capsize adds caps to error bars
                ax.errorbar(f_data['x'], f_data['y'], yerr=f_data['yerr'], 
                            fmt='o', label=family, color=color, 
                            capsize=5, markersize=8, alpha=0.8)
                
                # Add text labels for specific models next to their dots
                for x, y, name in zip(f_data['x'], f_data['y'], f_data['names']):
                    # Offset slightly to avoid overlapping the dot
                    ax.text(x, y, name, fontsize=8, ha='left', va='bottom', 
                            rotation=15, alpha=0.7)

            ax.set_title(f"Model Comparison: Mean AUC({auc_key}) vs Model Size")
            ax.set_xlabel("Model Size (Parameters in Billions)")
            ax.set_ylabel(f"Mean AUC ({auc_key})")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(title="Model Family")
            
            # Optional: Log scale X if sizes span orders of magnitude (e.g. 0.5B to 70B)
            # ax.set_xscale('log') 
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def plot_detailed_scatter(all_models_data, output_pdf):
    """
    Generates the original scatter plots with regression lines for every model.
    """
    print(f"Generating Detailed Scatter Plots -> {output_pdf}...")
    
    with PdfPages(output_pdf) as pdf:
        for model_name in sorted(all_models_data.keys()):
            data = all_models_data[model_name]
            
            y_keys = ['loss', 'entropy']
            if data['pred_token_count']: y_keys.append('pred_token_count')
            x_keys = list(data['auc'].keys())

            for y_key in y_keys:
                for x_key in x_keys:
                    try:
                        y_arr = np.array(data[y_key])
                        x_arr = np.array(data['auc'][x_key])
                        
                        mask = np.isfinite(y_arr) & np.isfinite(x_arr)
                        y_arr = y_arr[mask]
                        x_arr = x_arr[mask]
                        
                        if len(y_arr) < 2: continue
                        
                        # Stats
                        r, p = pearsonr(x_arr, y_arr)
                        m, b = np.polyfit(x_arr, y_arr, 1)
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(x_arr, y_arr, alpha=0.3, label=f"n={len(x_arr)}")
                        ax.plot(x_arr, m*x_arr + b, color='red', label=f"Fit: r={r:.3f}")
                        
                        ax.set_title(f"[{model_name}]\n{y_key} vs AUC({x_key})")
                        ax.set_xlabel(f"AUC ({x_key})")
                        ax.set_ylabel(y_key)
                        
                        stats_text = f"r = {r:.3f}\np = {p:.2e}"
                        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                                bbox=dict(facecolor='white', alpha=0.8), va='top')
                        
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        pdf.savefig(fig)
                        plt.close(fig)
                        
                    except Exception:
                        continue

def plot_auc_distributions(all_models_data, output_pdf):
    """
    Generates KDE plots showing the distribution of AUC scores for each model.
    Includes normalization (peak=1) for easier comparison.
    """
    print(f"Generating AUC Distribution Plots -> {output_pdf}...")
    
    first_model = next(iter(all_models_data.values()))
    auc_keys = list(first_model['auc'].keys())
    
    # Prepare colors for models
    model_names = sorted(all_models_data.keys())
    colors = cm.get_cmap('tab10', len(model_names))
    
    with PdfPages(output_pdf) as pdf:
        for auc_key in auc_keys:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for i, model_name in enumerate(model_names):
                data = all_models_data[model_name]
                if auc_key not in data['auc']: continue
                
                aucs = np.array(data['auc'][auc_key])
                
                # Skip if too few points or zero variance for KDE
                if len(aucs) < 2 or np.std(aucs) == 0:
                    print(f"  Warning: Skipping KDE for {model_name} - insufficient data/variance.")
                    continue
                    
                try:
                    # Compute Kernel Density Estimate
                    density = gaussian_kde(aucs)
                    
                    # Define range for the plot (slightly padded)
                    x_min, x_max = aucs.min(), aucs.max()
                    padding = (x_max - x_min) * 0.2
                    if padding == 0: padding = 0.1 # avoid zero range
                    xs = np.linspace(x_min - padding, x_max + padding, 200)
                    
                    # Evaluate density
                    ys = density(xs)
                    
                    # --- NORMALIZE: Scale peak to 1.0 ---
                    if ys.max() > 0:
                        ys_norm = ys / ys.max()
                    else:
                        ys_norm = ys

                    color = colors(i)
                    # Plot the density line
                    ax.plot(xs, ys_norm, color=color, label=model_name, linewidth=2)
                    # Fill the area
                    ax.fill_between(xs, ys_norm, color=color, alpha=0.2)
                    
                except Exception as e:
                    print(f"  Error computing KDE for {model_name}: {e}")
                    continue

            ax.set_title(f"Normalized Distribution of AUC Scores: {auc_key}")
            ax.set_xlabel(f"AUC ({auc_key})")
            ax.set_ylabel("Normalized Density (Peak = 1.0)")
            ax.legend(title="Model")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            pdf.savefig(fig)
            plt.close(fig)


# --- 3. Main ---

def main():
    parser = argparse.ArgumentParser(description="Multi-Model Correlation Analysis")
    parser.add_argument('--base_result_dir', type=str, required=True, 
                        help='Directory containing model subfolders (e.g. ./results)')
    parser.add_argument('--dir', type=str, required=True, 
                        help='Directory containing the npz (e.g., ./final_straces_20).')
    parser.add_argument('--token_count_dir', type=str, default=None,
                        help='Optional directory containing {model_name}_token_counts.tsv files.')
    parser.add_argument('--output_prefix', type=str, default="analysis", 
                        help='Prefix for output PDFs')
    args = parser.parse_args()

    if not os.path.isdir(args.base_result_dir):
        print(f"Error: {args.base_result_dir} not found.")
        return

    # Find potential model directories
    model_dirs = [d for d in os.listdir(args.base_result_dir) 
                  if os.path.isdir(os.path.join(args.base_result_dir, d))]
    
    all_models_data = {}

    print(f"Found {len(model_dirs)} directories. Processing...")

    for model_name in sorted(model_dirs):
        full_path = os.path.join(args.base_result_dir, model_name, args.dir)
        
        # Try to find a token count file if a directory was provided
        count_file = None
        if args.token_count_dir:
            # Heuristic: look for file starting with model name or exact match
            # e.g. "OLMo-7B_token_counts.tsv"
            possible_files = glob.glob(os.path.join(args.token_count_dir, f"*{model_name.replace('-', '_').lower()}*token_counts.tsv"))
            if possible_files:
                count_file = possible_files[0]
        
        print(f"\n--- Processing: {model_name} ---")
        model_data = load_model_data(full_path, count_file)
        
        if model_data:
            all_models_data[model_name] = model_data
        else:
            print(f"  Skipping {model_name} (Invalid or empty).")

    if not all_models_data:
        print("No valid data loaded.")
        return

    # Generate Plots
    plot_auc_distributions(all_models_data, f"{args.output_prefix}_auc_distributions.pdf")
    plot_correlation_summary(all_models_data, f"{args.output_prefix}_correlations_summary.pdf")
    plot_model_size_scatter(all_models_data, f"{args.output_prefix}_model_size_comparison.pdf")
    plot_detailed_scatter(all_models_data, f"{args.output_prefix}_scatter_details.pdf")

if __name__ == "__main__":
    main()