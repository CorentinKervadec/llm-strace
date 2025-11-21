import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
from collections import defaultdict
import warnings

def load_and_aggregate_data(results_dir: str):
    """
    Loads all .npz files from the results_dir, extracts and aggregates data.

    Returns:
        tuple: (static_data, aggregated_data)
        - static_data: dict with data assumed constant (e.g., 'tau', 'index')
        - aggregated_data: dict with 'mean' and 'std' for all variable metrics
    """
    
    # Check if directory exists
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return None, None

    # Find all .npz files
    file_paths = glob.glob(os.path.join(results_dir, '*.npz'))
    if not file_paths:
        print(f"Error: No .npz files found in {results_dir}")
        return None, None

    print(f"Found {len(file_paths)} result files. Loading...")

    # To store data from all files
    all_data = defaultdict(list)
    static_data = {}
    
    # Define the metrics and their nested structure
    metric_keys = ['strata_reco_tv', 'strata_reco_nu', 'strata_loss', 'strata_entropy']
    eval_modes = ['trace', 'random']
    sub_modes = ['only', 'inverse']

    for i, f_path in enumerate(file_paths):
        try:
            # Load the .npz file
            data = np.load(f_path, allow_pickle=True)
            
            # --- Store static data (assumed same across all files) ---
            # These are already numpy arrays, no conversion needed
            if 'tau' not in static_data:
                static_data['tau'] = data['strata_tau']
                static_data['index'] = data['strata_index']
            
            # --- Store variable data for aggregation ---
            all_data['rel_size'].append(data['strata_rel_size'])
            
            for key in metric_keys:
                for mode in eval_modes:
                    for sub_mode in sub_modes:
                        # e.g., 'strata_reco_tv_trace_only'
                        full_key = f"{key}_{mode}_{sub_mode}"
                        try:
                            # Use .item() to extract the Python dict from the np array
                            all_data[full_key].append(data[key].item()[mode][sub_mode])
                        except KeyError:
                            print(f"Warning: Key {key}[{mode}][{sub_mode}] not found in {f_path}")
        except Exception as e:
            print(f"Warning: Could not load or process file {f_path}. Error: {e}")
            continue
            
    if not all_data:
        print("Error: No data was successfully loaded.")
        return None, None
        
    print("Aggregation complete. Calculating mean and std...")
    
    # --- Aggregate all collected data ---
    aggregated_data = {}
    with warnings.catch_warnings():
        # Suppress "Degrees of freedom <= 0 for slice" warning if only 1 file
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        for key, data_list in all_data.items():
            try:
                data_array = np.array(data_list)
                aggregated_data[f'mean_{key}'] = np.nanmean(data_array, axis=0)
                aggregated_data[f'std_{key}'] = np.nanstd(data_array, axis=0)
            except Exception as e:
                print(f"Error aggregating {key}: {e}")

    return static_data, aggregated_data


def plot_results(static_data, aggregated_data, output_pdf):
    """
    Generates all 9 plots and saves them to a multi-page PDF.
    """
    
    print(f"Plotting results and saving to {output_pdf}...")
    
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
        ('trace', 'only', 'Trace - Only', 'blue', '-'),
        ('trace', 'inverse', 'Trace - Inverse', 'green', '--'),
        ('random', 'only', 'Random - Only', 'orange', '-'),
        ('random', 'inverse', 'Random - Inverse', 'red', '--')
    ]

    with PdfPages(output_pdf) as pdf:
        
        # --- 1. Plot 1: Tau vs. Relative Size (Special Case) ---
        fig, ax = plt.subplots(figsize=(10, 7))
        x_data = static_data['tau']
        y_mean = aggregated_data['mean_rel_size']
        y_std = aggregated_data['std_rel_size']
        
        ax.plot(x_data, y_mean, label='Mean Relative Size', color='purple', marker='o', markersize=4)
        ax.fill_between(x_data, y_mean - y_std, y_mean + y_std, color='purple', alpha=0.15, label='Std. Dev.')
        
        ax.set_title('Stratum Relative Size vs. Tau Threshold')
        ax.set_xlabel('Tau Threshold')
        ax.set_ylabel('Relative Stratum Size')
        ax.invert_xaxis() # Tau is more intuitive from 1.0 down to 0
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        pdf.savefig(fig)
        plt.close(fig)

        # --- 2. Plots 2-9 (Looping) ---
        for x_key, x_label in x_axes_to_plot:
            for y_key, y_label in metrics_to_plot:
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Get x-axis data
                if x_key == 'tau':
                    x_data = static_data['tau']
                    ax.invert_xaxis() # Invert for Tau
                else:
                    # Use mean relative size for the x-axis
                    x_data = aggregated_data['mean_rel_size']

                # Plot all 4 evaluation combinations
                for mode, sub_mode, plot_label, color, style in eval_combinations:
                    
                    mean_key = f"mean_{y_key}_{mode}_{sub_mode}"
                    std_key = f"std_{y_key}_{mode}_{sub_mode}"

                    if mean_key not in aggregated_data:
                        print(f"Warning: Missing data for {mean_key}")
                        continue
                        
                    y_mean = aggregated_data[mean_key]
                    y_std = aggregated_data[std_key]

                    ax.plot(x_data, y_mean, label=plot_label, color=color, linestyle=style, marker='o', markersize=4)
                    ax.fill_between(x_data, y_mean - y_std, y_mean + y_std, color=color, alpha=0.15)

                ax.set_title(f'{y_label} vs. {x_label}')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend(title="Evaluation Mode")
                ax.grid(True, linestyle='--', alpha=0.6)
                
                pdf.savefig(fig)
                plt.close(fig)
    print("Successfully saved PDF.")

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Plotting")
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to load result graphs from.')
    parser.add_argument('--pdf_name', type=str, required=True, help='Name of the output pdf.')
    args = parser.parse_args()

    static_data, aggregated_data = load_and_aggregate_data(args.result_dir)
    
    if static_data and aggregated_data:
        plot_results(static_data, aggregated_data, args.pdf_name)

if __name__ == "__main__":
    main()