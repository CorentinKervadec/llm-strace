import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
import glob
from collections import defaultdict
import warnings

def load_token_counts(file_path):
    """
    Load a text file where each line contains tab-separated fields: token_id, count.

    Args:
        file_path (str): Path to the text file.

    Returns:
        dict: A dictionary where keys are token IDs (int) and values are counts (int).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    token_counts = {}
    with open(file_path, 'r') as f:
        for line in f.readlines()[1:]: # skip header
            try:
                token_str, count = line.split('\t')
                token_counts[token_str] = int(count)
            except ValueError:
                print(f"Warning: Skipping malformed line: {line.strip()}")
    return token_counts

def load_and_aggregate_data(results_dir: str, count_file: str):
    """
    Loads all .npz files from the results_dir, extracts and aggregates data.

    Returns:
        tuple: (static_data, aggregated_data)
        - static_data: dict with data assumed constant (e.g., 'tau', 'index')
        - aggregated_data: dict with 'mean' and 'std' for all variable metrics
    """
    
    # load token count stats
    token_counts = load_token_counts(count_file)
    total_count = sum([v for v in token_counts.values()])
    token_counts = {k:float(v)/total_count for k,v in token_counts.items()} # normalise
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

    sentence_data = {'loss': [], 'entropy': [], 'pred_token_count': [], 'auc':{}}
    
    # Define the metrics and their nested structure
    metric_keys = ['strata_reco_tv', 'strata_reco_nu']
    eval_modes = ['trace',]# 'random']
    sub_modes = ['only',]# 'inverse']

    for i, f_path in enumerate(file_paths):
        try:
            # Load the .npz file
            data = np.load(f_path, allow_pickle=True)
            
            # sort for auc
            sort_size_indices = np.argsort(data['strata_rel_size'])
            size_sorted = np.array(data['strata_rel_size'])[sort_size_indices]
            
            for key in metric_keys:
                for mode in eval_modes:
                    for sub_mode in sub_modes:
                        # e.g., 'strata_reco_tv_trace_only'
                        full_key = f"{key}_{mode}_{sub_mode}"
                        
                        metric_sorted = np.array(data[key].item()[mode][sub_mode])[sort_size_indices]
                        metric_auc = np.trapz(metric_sorted, size_sorted)
                        if full_key not in sentence_data["auc"]:
                            sentence_data["auc"][full_key] = []    
                        sentence_data["auc"][full_key].append(metric_auc)
            # full model metrics (i.e. the last stratum)
            sentence_data["loss"].append(data["strata_loss"].item()['trace']['only'][-1])
            sentence_data["entropy"].append(data["strata_entropy"].item()['trace']['only'][-1])
            nucleus_60 = data["nucleus_60"].item()['trace']['only'][-1]
            # I ignore tokens when I don't have the count -> not super clean
            nucleus_60_count_avg = np.array([token_counts[tkn] for tkn in nucleus_60 if tkn in token_counts]).mean()
            sentence_data["pred_token_count"].append(nucleus_60_count_avg)


        except Exception as e:
            print(f"Warning: Could not load or process file {f_path}. Error: {e}")
            continue
            

    return sentence_data


def plot_regression_and_correlation(sentence_data, output_pdf="regression_analysis.pdf"):
    """
    Generates regression plots and calculates correlations for the given data.

    Y-axes are taken from ['loss', 'entropy', 'pred_token_count'].
    X-axes are taken from the keys in sentence_data['auc'].

    Args:
        sentence_data (dict): A dictionary with the structure:
            {
                'loss': [L1, L2, ...],
                'entropy': [E1, E2, ...],
                'pred_token_count': [C1, C2, ...],
                'auc': {
                    'metric1': [AUC1_1, AUC1_2, ...],
                    'metric2': [AUC2_1, AUC2_2, ...]
                }
            }
        output_pdf (str): The filename for the output PDF.
    """
    print(f"Generating regression plots and saving to {output_pdf}...")

    y_keys = ['loss', 'entropy', 'pred_token_count']
    
    if 'auc' not in sentence_data or not sentence_data['auc']:
        print("Error: 'auc' key is missing or empty in sentence_data.")
        return
        
    x_keys = list(sentence_data['auc'].keys())
    # Open a multi-page PDF file
    with PdfPages(output_pdf) as pdf:
        # Loop over every Y-metric
        for y_key in y_keys:
            # Loop over every X-metric (AUCs)
            for x_key in x_keys:
                
                print(f"  Plotting {y_key} vs. AUC({x_key})...")
                
                try:
                    # 1. Extract and validate data
                    if y_key not in sentence_data:
                        print(f"    Warning: Y-key '{y_key}' not in sentence_data. Skipping.")
                        continue
                        
                    y_data = np.array(sentence_data[y_key])
                    x_data = np.array(sentence_data['auc'][x_key])

                    if len(y_data) == 0 or len(x_data) == 0:
                        print(f"    Warning: No data for {y_key} or {x_key}. Skipping plot.")
                        continue
                    
                    if len(y_data) != len(x_data):
                        print(f"    Warning: Mismatched data lengths for {y_key} ({len(y_data)}) "
                              f"and {x_key} ({len(x_data)}). Skipping.")
                        continue
                    
                    # Remove any NaNs or Infs
                    mask = np.isfinite(x_data) & np.isfinite(y_data)
                    x_data = x_data[mask]
                    y_data = y_data[mask]
                    
                    if len(x_data) < 2:
                        print("    Warning: Not enough finite data points to plot. Skipping.")
                        continue

                    # 2. Calculate Correlation
                    # Check for constant data (which breaks correlation)
                    if np.all(x_data == x_data[0]) or np.all(y_data == y_data[0]):
                        r, p = (np.nan, np.nan)
                        text_str = "Correlation: N/A (constant data)"
                    else:
                        r, p = pearsonr(x_data, y_data)
                        text_str = f"Pearson's r: {r:.3f}\np-value: {p:.3e}"

                    # 3. Calculate Regression Line
                    m, b = np.polyfit(x_data, y_data, 1)
                    reg_line = m * x_data + b

                    # 4. Create Plot
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.scatter(x_data, y_data, alpha=0.5, label="Data points")
                    ax.plot(x_data, reg_line, color='red', label=f"Fit: y = {m:.2f}x + {b:.2f}")

                    # 5. Set Labels and Title
                    ax.set_xlabel(f"AUC ({x_key})")
                    ax.set_ylabel(y_key)
                    ax.set_title(f"Regression: {y_key} vs. AUC ({x_key})")

                    # 6. Add Correlation Text Box
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
                            verticalalignment='top', bbox=props)

                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.6)
                    
                    # 7. Save to PDF
                    pdf.savefig(fig)
                    plt.close(fig)
                
                except Exception as e:
                    print(f"Error plotting {y_key} vs {x_key}: {e}")
                    
    print(f"All plots saved to {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Plotting")
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to load result graphs from.')
    parser.add_argument('--count_file', type=str, required=True, help='')
    parser.add_argument('--pdf_name', type=str, required=True, help='Name of the output pdf.')
    args = parser.parse_args()

    # count_file = "olmo2_1B_token_counts.tsv"
    # count_file = "mistral_7B_token_counts.tsv"

    sentence_data = load_and_aggregate_data(args.result_dir, args.count_file)
    
    plot_regression_and_correlation(sentence_data, args.pdf_name)

if __name__ == "__main__":
    main()