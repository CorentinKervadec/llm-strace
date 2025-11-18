import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For coloring models
from matplotlib.backends.backend_pdf import PdfPages
import glob
from collections import defaultdict
import warnings

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
        # This is a valid case (e.g., folder exists but is empty), return None
        return None, None

    # ... (rest of the function is identical to your old load_and_aggregate_data) ...
    
    all_data = defaultdict(list)
    static_data = {}
    
    metric_keys = ['strata_reco_tv', 'strata_reco_nu', 'strata_loss', 'strata_entropy']
    eval_modes = ['trace', 'random']
    sub_modes = ['only', 'inverse']

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
                            print(f"    Warning: Key {key}[{mode}][{sub_mode}] not found in {f_path}")

        except Exception as e:
            print(f"    Warning: Could not load or process file {f_path}. Error: {e}")
            continue
            
    if not all_data:
        # Valid case if .npz files were empty/corrupt
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
                    aggregated_data[f'mean_{key}'] = np.nanmean(data_array, axis=0)
                    aggregated_data[f'std_{key}'] = np.nanstd(data_array, axis=0)
            except Exception as e:
                print(f"    Error aggregating {key}: {e}")

    return static_data, aggregated_data


def plot_multi_model_results(all_models_data: dict, output_pdf: str):
    """
    Generates plots comparing all models on the same axes.
    
    Args:
        all_models_data: A dict where keys are model names and values are
                         {'static': static_data, 'agg': aggregated_data}
        output_pdf: Path to save the multi-page PDF.
    """
    
    print(f"Plotting results for {len(all_models_data)} models and saving to {output_pdf}...")
    
    # Get a color map
    num_models = len(all_models_data)
    colors = cm.get_cmap('tab10', max(num_models, 10))
    
    # Define plotting loops
    metrics_to_plot = [
        ('strata_reco_tv', 'Total Variation (TV)'),
        ('strata_reco_nu', 'Nucleus Reconstruction'),
        ('strata_loss', 'Reconstruction Loss'),
        ('strata_entropy', 'Reconstruction Entropy')
    ]
    x_axes_to_plot = [
        ('tau', 'Tau Threshold'),
        ('rel_size', 'Relative Stratum Size')
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
        
        pdf.savefig(fig)
        plt.close(fig)

        # --- 2. Main plots: One page per metric combination ---
        for x_key, x_label in x_axes_to_plot:
            for y_key, y_label in metrics_to_plot:
                for mode, sub_mode, eval_label in eval_combinations:
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # This plot's title
                    plot_title = f'{y_label} vs. {x_label} ({eval_label})'
                    ax.set_title(plot_title)
                    
                    # Plot all models on this single plot
                    for i, (model_name, data) in enumerate(all_models_data.items()):
                        
                        # Get x-axis data
                        if x_key == 'tau':
                            x_data = data['static']['tau']
                        else:
                            x_data = data['agg']['mean_rel_size']
                        
                        # Get y-axis data
                        mean_key = f"mean_{y_key}_{mode}_{sub_mode}"
                        std_key = f"std_{y_key}_{mode}_{sub_mode}"

                        if mean_key not in data['agg']:
                            print(f"Warning: Model '{model_name}' missing data for {mean_key}")
                            continue
                            
                        y_mean = data['agg'][mean_key]
                        y_std = data['agg'][std_key]

                        ax.plot(x_data, y_mean, label=model_name, color=colors(i), marker='o', markersize=3, alpha=0.8)
                        ax.fill_between(x_data, y_mean - y_std, y_mean + y_std, color=colors(i), alpha=0.1)

                    if x_key == 'tau':
                        ax.invert_xaxis() # Invert for Tau
                    
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.legend(title="Model")
                    ax.grid(True, linestyle='--', alpha=0.6)
                    
                    pdf.savefig(fig)
                    plt.close(fig)

    print("Successfully saved multi-model PDF.")

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
        plot_multi_model_results(all_models_data, args.pdf_name)
    else:
        print("Error: No data was successfully loaded from any model directory.")

if __name__ == "__main__":
    main()