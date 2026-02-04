import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For coloring models
from matplotlib.backends.backend_pdf import PdfPages
import glob
from collections import defaultdict
import warnings

import random

def plot_auc_filtered_sentences(all_model_paths: dict, output_pdf: str, n_pool=1000, n_select=5):
    """
    1. Samples 'n_pool' sentences.
    2. Computes AUC for Strata Size vs TV.
    3. Plots the 'n_select' Lowest, Highest, and Average (Mean) AUC sentences.
    """
    print(f"Generating AUC Analysis PDF (Lowest/Average/Highest): {output_pdf}...")
    
    # Font Configuration
    TITLE_SIZE = 10
    LABEL_SIZE = 14
    TICK_SIZE = 12
    ANNOTATE_SIZE = 14 

    with PdfPages(output_pdf) as pdf:
        for model_name, model_dir_path in all_model_paths.items():
            
            # --- STEP 1: LOAD & COMPUTE AUC ---
            file_paths = glob.glob(os.path.join(model_dir_path, '*.npz'))
            if not file_paths:
                continue
                
            # Sample pool of files (e.g., 1000)
            n_sample_actual = min(len(file_paths), n_pool)
            pool_files = random.sample(file_paths, n_sample_actual)
            
            print(f"  Computing AUC for {n_sample_actual} files in {model_name}...")
            
            records = []
            
            for f_path in pool_files:
                try:
                    data = np.load(f_path, allow_pickle=True)
                    
                    x_size = data['strata_rel_size']
                    y_tv = data['strata_reco_tv'].item()['trace']['only']
                    
                    if 'strata_tau' in data:
                        taus = data['strata_tau']
                    else:
                        taus = np.zeros_like(x_size)
                    
                    # Sort for AUC computation and plotting
                    sort_idx = np.argsort(x_size)
                    x_plot = np.array(x_size)[sort_idx]
                    y_plot = np.array(y_tv)[sort_idx]
                    t_plot = np.array(taus)[sort_idx]
                    
                    # Compute AUC using Trapezoidal rule
                    auc_val = np.trapz(y_plot, x_plot)
                    
                    records.append({
                        'file': os.path.basename(f_path),
                        'sent': data['input'],
                        'x': x_plot,
                        'y': y_plot,
                        't': t_plot,
                        'auc': auc_val
                    })
                except Exception:
                    continue
            
            if not records:
                continue

            # --- STEP 2: SELECT INTERESTING SENTENCES ---
            # Sort by AUC
            records.sort(key=lambda r: r['auc'])
            
            # 1. Lowest AUC
            lowest = records[:n_select]
            for r in lowest: r['group'] = 'Lowest'
            
            # 2. Highest AUC
            highest = records[-n_select:]
            for r in highest: r['group'] = 'Highest'
            
            # 3. Average (Mean) AUC
            mean_auc = np.mean([r['auc'] for r in records])
            # Sort by distance to the mean
            closest_to_mean = sorted(records, key=lambda r: abs(r['auc'] - mean_auc))
            average = closest_to_mean[:n_select]
            for r in average: r['group'] = 'Average'
            
            # Combine unique entries (handle cases where N is small and lists overlap)
            # We add them to a specific list to plot in order
            final_selection = []
            seen_files = set()
            
            # Add in specific order: Lowest -> Average -> Highest
            for group_list in [lowest, average, highest]:
                for rec in group_list:
                    if rec['file'] not in seen_files:
                        final_selection.append(rec)
                        seen_files.add(rec['file'])

            print(f"  Plotting {len(final_selection)} selected sentences...")

            # --- STEP 3: PLOT ---
            for rec in final_selection:
                fig, ax = plt.subplots(figsize=(12, 9))
                
                x_plot = rec['x']
                y_plot = rec['y']
                t_plot = rec['t']
                
                # 1. Plot Main Line & Fill
                ax.plot(x_plot, y_plot, color='firebrick', linewidth=2.5, alpha=0.9, zorder=2)
                ax.fill_between(x_plot, y_plot, color='firebrick', alpha=0.15, zorder=1)
                ax.scatter(x_plot, y_plot, color='firebrick', s=60, zorder=3)
                
                # 2. Annotate Tau <= 0.005
                for j in range(len(x_plot)):
                    tau_val = t_plot[j]
                    
                    if tau_val > 0.005:
                        continue

                    # Staggering Logic
                    angles = [90, 270, 45, 135, 225, 315] 
                    angle = angles[j % len(angles)]
                    dist = 50 if j % 2 == 0 else 80
                    
                    off_x = dist * np.cos(np.radians(angle))
                    off_y = dist * np.sin(np.radians(angle))
                    
                    tau_str = f"{tau_val:.2g}" if tau_val < 0.001 else f"{tau_val:.4f}"

                    ax.annotate(
                        f"τ={tau_str}",
                        xy=(x_plot[j], y_plot[j]),
                        xytext=(off_x, off_y),
                        textcoords='offset points',
                        fontsize=ANNOTATE_SIZE,
                        fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
                        arrowprops=dict(arrowstyle='->', color='black', linewidth=0.8, shrinkB=5),
                        zorder=5
                    )
                
                # 3. Dynamic Title with AUC Stats
                title_str = (f"Model: {model_name} | AUC: {rec['auc']:.4f}\n"
                             f"Sentence: {rec['sent']}\n")
                
                ax.set_title(title_str, fontsize=TITLE_SIZE, pad=15)
                ax.set_xlabel("Relative Trace Size", fontsize=LABEL_SIZE)
                ax.set_ylabel("Reconstruction Error", fontsize=LABEL_SIZE)
                
                ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
                ax.set_xlim(0, 1.05)
                ax.set_ylim(bottom=0)
                ax.grid(True, linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Successfully saved AUC-filtered plots to {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Multi-Model Plotting")
    parser.add_argument('--base_result_dir', type=str, required=True, 
                        help='Base directory containing model sub-folders (e.g., ./results).')
    parser.add_argument('--dir', type=str, required=True, 
                        help='Directory containing the npz (e.g., ./final_straces_20).')
    parser.add_argument('--pdf_name', type=str, required=True, 
                        help='Name of the output PDF (e.g., ./all_models_comparison.pdf).')
    parser.add_argument('--min_files', type=int, default=1000,
                        help='Minimum number of .npz files required to include a model (default: 1000).')
    args = parser.parse_args()

    all_models_data = {}

    # 1. Find all sub-directories (models) in the base directory
    if not os.path.isdir(args.base_result_dir):
        print(f"Error: Base result directory not found: {args.base_result_dir}")
        return

    model_dirs = [d for d in os.listdir(args.base_result_dir) 
                  if os.path.isdir(os.path.join(args.base_result_dir, d))]

    if not model_dirs:
        print(f"Error: No model sub-folders found in {args.base_result_dir}")
        return

    print(f"Found {len(model_dirs)} potential model directories...")

    model_paths_for_sampling = {}

    # 2. Load and aggregate data for each model
    for model_name in sorted(model_dirs):

        if "stage" in model_name or "Qwen3-8B-Base" not in model_name:
            continue # skip intermediate training checkpoints

        model_dir_path = os.path.join(args.base_result_dir, model_name, args.dir)
        print(f"\n--- Loading data for model: {model_name} ---")

        # --- VALIDATION: Check for minimum file count ---
        try:
            file_paths = glob.glob(os.path.join(model_dir_path, '*.npz'))
            file_count = len(file_paths)
        except Exception as e:
            print(f"  Error counting files in {model_dir_path}: {e}. Skipping model.")
            continue
        
        if file_count < args.min_files:
            print(f"  Warning: Model discarded. Found {file_count} files, "
                  f"but the minimum required is {args.min_files}.")
            continue
            
        model_paths_for_sampling[model_name] = model_dir_path

    plot_auc_filtered_sentences(model_paths_for_sampling, 'samples_' + args.pdf_name)
if __name__ == "__main__":
    main()