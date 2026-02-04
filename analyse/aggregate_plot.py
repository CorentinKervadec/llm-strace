import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For coloring models
from matplotlib.backends.backend_pdf import PdfPages
import glob
from collections import defaultdict
import warnings
from matplotlib.patches import Ellipse

FONTSIZE=22

def load_single_model_data(model_dir: str):
    """
    Loads all .npz files from a SINGLE model's directory, extracts and aggregates data.

    Returns:
        tuple: (static_data, aggregated_data)
        - static_data: dict with data assumed constant (e.g., 'tau', 'index')
        - aggregated_data: dict with 'mean' and 'std' for all variable metrics
    """
    
    # Find all .npz files
    file_paths = glob.glob(os.path.join(model_dir, '*.npz'))
    if not file_paths:
        return None, None
    
    all_data = defaultdict(list)
    static_data = {}
    
    metric_keys = ['strata_reco_tv', 'strata_reco_nu', 'strata_loss', 'strata_entropy']
    eval_modes = ['trace',]# 'random']
    sub_modes = ['only',]# 'inverse']

    for i, f_path in enumerate(file_paths):
        try:
            data = np.load(f_path, allow_pickle=True)
            
            # Sort data by stratum index (CRITICAL)
            sort_key = np.argsort(data['strata_index'])
            
            if 'tau' not in static_data:
                static_data['index'] = data['strata_index'][sort_key]
                static_data['tau'] = data['strata_tau'][sort_key]
            
            rel_size_sorted = data['strata_rel_size'][sort_key]
            all_data['rel_size'].append(rel_size_sorted)
            
            for key in metric_keys:
                for mode in eval_modes:
                    for sub_mode in sub_modes:
                        full_key = f"{key}_{mode}_{sub_mode}"
                        try:
                            metric_list = data[key].item()[mode][sub_mode]
                            all_data[full_key].append(np.array(metric_list)[sort_key])
                        except KeyError:
                            # print(f"    Warning: Key {key}[{mode}][{sub_mode}] not found in {f_path}")
                            pass

        except Exception as e:
            print(f"    Warning: Could not load or process file {f_path}. Error: {e}")
            continue
            
    if not all_data:
        return None, None
        
    # --- Aggregate all collected data ---
    aggregated_data = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        for key, data_list in all_data.items():
            try:
                data_array = np.array(data_list)
                if data_array.ndim == 1:
                     aggregated_data[f'mean_{key}'] = data_array
                     aggregated_data[f'std_{key}'] = np.zeros_like(data_array)
                else:
                    aggregated_data[f'mean_{key}'] = np.mean(data_array, axis=0)
                    aggregated_data[f'std_{key}'] = np.std(data_array, axis=0)
            except Exception as e:
                print(f"    Error aggregating {key}: {e}")

    return static_data, aggregated_data

def plot_oval_uncertainty_results(all_models_data: dict, output_pdf: str):
    """
    Generates a PDF with one model per page.
    Features: Radial leader lines, non-overlapping boxes, and clipped X-axis at 1.0.
    """
    print(f"Generating Final Oval Uncertainty PDF: {output_pdf}...")
    
    num_models = len(all_models_data)
    colors = cm.get_cmap('tab10', max(num_models, 10))
    
    y_metric = 'strata_reco_tv'
    mode, sub_mode = ('trace', 'only')

    # Font Settings
    TITLE_SIZE, LABEL_SIZE, TICK_SIZE, ANNOTATE_SIZE = 22, 22, 22, 22

    with PdfPages(output_pdf) as pdf:
        for i, (model_name, data) in enumerate(all_models_data.items()):
            fig, ax = plt.subplots(figsize=(14, 10))
            fig.suptitle(f'{model_name}', fontsize=TITLE_SIZE, fontweight='bold')
            
            x_mean = data['agg']['mean_rel_size']
            x_std = data['agg']['std_rel_size']
            tau_values = data['static']['tau']
            
            mean_key = f"mean_{y_metric}_{mode}_{sub_mode}"
            std_key = f"std_{y_metric}_{mode}_{sub_mode}"

            if mean_key not in data['agg']:
                continue
            
            y_mean = data['agg'][mean_key]
            y_std = data['agg'][std_key]

            # 1. Scatter points
            ax.scatter(x_mean, y_mean, color=colors(i), s=80, alpha=1.0, zorder=5, edgecolors='black', linewidth=0.7)
            
            # 2. Add Ovals and Smart Labels
            for j in range(len(x_mean)):
                # Draw Ellipse
                ellipse = Ellipse(
                    xy=(x_mean[j], y_mean[j]),
                    width=2 * x_std[j],
                    height=2 * y_std[j],
                    edgecolor=colors(i),
                    facecolor=colors(i),
                    alpha=0.15,
                    linewidth=0.8,
                    zorder=2
                )
                ax.add_patch(ellipse)

                # --- RADIAL STAGGERING LOGIC TO PREVENT OVERLAP ---
                angles = [45, 135, 225, 315, 90, 270] # Preferred directions
                angle = angles[j % len(angles)]
                
                # Alternate distance
                dist = 35 if (j // len(angles)) % 2 == 0 else 60 
                
                # Convert polar to cartesian
                off_x = dist * np.cos(np.radians(angle))
                off_y = dist * np.sin(np.radians(angle))

                tau_val = tau_values[j]
                ax.annotate(
                    f"τ={tau_val:.2g}",
                    xy=(x_mean[j], y_mean[j]),
                    xytext=(off_x, off_y),
                    textcoords='offset points',
                    fontsize=ANNOTATE_SIZE,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    zorder=6,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec=colors(i), lw=0.5),
                    arrowprops=dict(
                        arrowstyle='-', 
                        color='black', 
                        linewidth=0.6, 
                        alpha=0.7,
                        connectionstyle="arc3,rad=0"
                    )
                )

            ax.set_title(f'Reconstruction Error vs. Relative Trace Size', fontsize=18, pad=20)
            ax.set_xlabel('Relative Trace Size', fontsize=LABEL_SIZE)
            ax.set_ylabel('Reconstruction Error (Total Variation Distance)', fontsize=LABEL_SIZE)
            ax.set_xlim(left=0, right=1.0)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Successfully saved oval plots to {output_pdf}")

def plot_multi_model_results(all_models_data: dict, output_pdf: str):
    """
    Generates plots comparing all models on the same axes.
    Separates Trace and Random onto different pages.
    """
    
    print(f"Plotting standard multi-model results (Original) to {output_pdf}...")
    
    num_models = len(all_models_data)
    colors = cm.get_cmap('tab10', max(num_models, 10))
    
    metrics_to_plot = [
        ('strata_reco_tv', 'Reconstruction Error (Total Variation Distance)'),
        ('strata_reco_nu', 'Nucleus Reconstruction'),
        ('strata_loss', 'Reconstruction Loss'),
        ('strata_entropy', 'Reconstruction Entropy')
    ]
    x_axes_to_plot = [
        ('tau', 'Tau Threshold'),
        ('rel_size', 'Relative Trace Size')
    ]
    eval_combinations = [
        ('trace', 'only', 'Trace - Only'),
        ('trace', 'inverse', 'Trace - Inverse'),
        ('random', 'only', 'Random - Only'),
        ('random', 'inverse', 'Random - Inverse')
    ]

    with PdfPages(output_pdf) as pdf:
        
        # --- 1. Plot 1: Tau vs. Relative Size (All Models) ---
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, (model_name, data) in enumerate(all_models_data.items()):
            x_data = data['static']['tau']
            y_mean = data['agg']['mean_rel_size']
            y_std = data['agg']['std_rel_size']
            
            ax.plot(x_data, y_mean, label=model_name, color=colors(i), marker='o', markersize=3, alpha=0.8)
            ax.fill_between(x_data, y_mean - y_std, y_mean + y_std, color=colors(i), alpha=0.1)
        
        ax.set_title('Stratum Relative Size vs. Tau Threshold (All Models)')
        ax.set_xlabel('Tau Threshold')
        ax.set_ylabel('Relative Stratum Size')
        ax.invert_xaxis()
        ax.legend(title="Model")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xscale('log')
        pdf.savefig(fig)
        plt.close(fig)

        # --- 2. Main plots ---
        for x_key, x_label in x_axes_to_plot:
            for y_key, y_label in metrics_to_plot:
                for mode, sub_mode, eval_label in eval_combinations:
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.set_title(f'{y_label} vs. {x_label} ({eval_label})')
                    
                    for i, (model_name, data) in enumerate(all_models_data.items()):
                        if x_key == 'tau':
                            x_data = data['static']['tau']
                        else:
                            x_data = data['agg']['mean_rel_size']
                        
                        mean_key = f"mean_{y_key}_{mode}_{sub_mode}"
                        if mean_key not in data['agg']: continue
                        y_mean = data['agg'][mean_key]

                        ax.plot(x_data, y_mean, label=model_name, color=colors(i), marker='o', markersize=3, alpha=0.8)

                    if x_key == 'tau': ax.invert_xaxis()
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.legend(title="Model")
                    ax.grid(True, linestyle='--', alpha=0.6)

                    pdf.savefig(fig)
                    plt.close(fig)

    print("Successfully saved multi-model PDF.")

def plot_multi_model_results_2(all_models_data: dict, output_pdf: str):
    """
    Generates plots where Random and Trace results are shown on the SAME graph.
    Solid line = Trace
    Dashed line = Random
    """
    print(f"Plotting Comparison (Trace vs Random) to {output_pdf}...")

    num_models = len(all_models_data)
    colors = cm.get_cmap('tab10', max(num_models, 10))

    metrics_to_plot = [
        ('strata_reco_tv', 'Reconstruction Error (TV Distance)'),
        ('strata_reco_nu', 'Nucleus Reconstruction'),
        ('strata_loss', 'Reconstruction Loss'),
        ('strata_entropy', 'Reconstruction Entropy')
    ]
    x_axes_to_plot = [
        ('tau', 'Tau Threshold'),
        ('rel_size', 'Relative Trace Size')
    ]
    # We loop over sub-modes only, effectively merging Trace/Random onto one plot
    sub_modes = [
        ('only', 'Only'), 
        ('inverse', 'Inverse')
    ]

    with PdfPages(output_pdf) as pdf:
        for x_key, x_label in x_axes_to_plot:
            for y_key, y_label in metrics_to_plot:
                for sub_mode_key, sub_mode_label in sub_modes:
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Title reflects that we are comparing Trace vs Random
                    ax.set_title(f'{y_label} vs. {x_label}\n(Trace [Solid] vs Random [Dashed] - {sub_mode_label})')

                    # Helper to avoid duplicate legend entries if needed, 
                    # but simple label logic usually suffices.
                    lines = [] 
                    
                    for i, (model_name, data) in enumerate(all_models_data.items()):
                        
                        # Determine X Data
                        if x_key == 'tau':
                            x_data = data['static']['tau']
                        else:
                            x_data = data['agg']['mean_rel_size']

                        # --- PLOT TRACE (Solid) ---
                        trace_mean_key = f"mean_{y_key}_trace_{sub_mode_key}"
                        if trace_mean_key in data['agg']:
                            y_trace = data['agg'][trace_mean_key]
                            line, = ax.plot(x_data, y_trace, 
                                            color=colors(i), linestyle='-', 
                                            marker='o', markersize=3, 
                                            label=f"{model_name} (Trace)")
                        
                        # --- PLOT RANDOM (Dashed) ---
                        random_mean_key = f"mean_{y_key}_random_{sub_mode_key}"
                        if random_mean_key in data['agg']:
                            y_random = data['agg'][random_mean_key]
                            ax.plot(x_data, y_random, 
                                    color=colors(i), linestyle='--', 
                                    marker='x', markersize=3, alpha=0.7,
                                    label=f"{model_name} (Random)")

                    if x_key == 'tau':
                        ax.invert_xaxis()
                        ax.set_xscale('log')

                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    
                    # Place legend outside if too many models
                    ax.legend(title="Model & Mode", bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, linestyle='--', alpha=0.6)
                    
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

    print(f"Successfully saved Trace vs Random comparison PDF to {output_pdf}")

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

    # 2. Load and aggregate data for each model
    for model_name in sorted(model_dirs):

        if "stage" in model_name:
            continue 

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
            
        static_data, aggregated_data = load_single_model_data(model_dir_path)
        
        # 3. Validation
        if static_data and aggregated_data:
            all_models_data[model_name] = {
                'static': static_data,
                'agg': aggregated_data
            }
            print(f"Successfully loaded and aggregated data for {model_name}.")
        else:
            print(f"Warning: No valid data loaded for {model_name}. Skipping this model.")

    # 4. Plot if we have data
    if all_models_data:
        # Original plots
        plot_multi_model_results(all_models_data, args.pdf_name)
        plot_oval_uncertainty_results(all_models_data, 'oval_' + args.pdf_name)
        
        # NEW FUNCTION CALL
        # plot_multi_model_results_2(all_models_data, 'trace_vs_random_' + args.pdf_name)
    else:
        print("Error: No data was successfully loaded from any model directory.")

if __name__ == "__main__":
    main()