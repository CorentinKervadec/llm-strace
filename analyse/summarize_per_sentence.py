import argparse
import os
import numpy as np
import glob
import re
import csv
from collections import defaultdict
import warnings

# Suppress NumPy warnings (often related to NaNs in data)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Core Data Extraction Logic ---

def extract_metrics_from_npz(data):
    """
    Extracts AUC (Total Variation), Full Model Loss, and Full Model Entropy 
    from a single loaded .npz file data structure.
    
    Args:
        data (np.lib.npyio.NpzFile): The loaded data from one trace file.
        
    Returns:
        dict: {'auc_tv': float, 'loss': float, 'entropy': float} or None on error.
    """
    try:
        # --- 1. Preparation: Get relative sizes and sorting index ---
        rel_size = data['strata_rel_size']
        if len(rel_size) < 2 or np.max(rel_size) <= 0:
            return None # Insufficient data for AUC or full model metrics

        # Sort by relative size for correct AUC calculation
        sort_idx = np.argsort(rel_size)
        size_sorted = np.array(rel_size)[sort_idx]

        # --- 2. Calculate AUC for Total Variation ---
        # The metric is accessed via strata_reco_tv['trace']['only']
        # The key for AUC calculation
        tv_key = 'strata_reco_tv'
        
        raw_tv = data[tv_key].item()['trace']['only']
        tv_sorted = np.array(raw_tv)[sort_idx]
        
        # Compute Area Under the Curve (Trapz method)
        auc_tv = np.trapz(tv_sorted, size_sorted)

        # --- 3. Extract Full Model Metrics (Loss and Entropy) ---
        # These are the metrics for the stratum closest to size 1.0 (the full model).
        full_idx = np.argmax(rel_size)
        
        # Access the full model loss and entropy values at the max size index
        loss_key = "strata_loss"
        entropy_key = "strata_entropy"
        
        full_loss = data[loss_key].item()['trace']['only'][full_idx]
        full_entropy = data[entropy_key].item()['trace']['only'][full_idx]
        
        return {
            'auc_tv': auc_tv,
            'loss': full_loss,
            'entropy': full_entropy
        }
    except Exception as e:
        # print(f"Warning: Failed to process file due to error: {e}")
        return None

def load_model_data(model_name, model_dir, min_files):
    """
    Loads and processes all trace files for a single model.
    
    Args:
        model_dir (str): Path to the model's result directory.
        
    Returns:
        dict: {sentence_id: {'auc_tv': float, 'loss': float, 'entropy': float}}
    """

    file_paths = glob.glob(os.path.join(model_dir, '*.npz'))
    
    if not file_paths:
        print(f"  Skipping {model_name}: No .npz files found in {model_dir}.")
        return {}
    
    if len(file_paths) < min_files:
        print(f"  Skipping {model_name}: Only {len(file_paths)} files found in {model_dir}.")
        return {}
    
    print(f"  Processing {len(file_paths)} traces for model: {model_name}...")
    
    sentence_metrics = {}
    
    for f_path in file_paths:
        try:
            # Extract sentence_id (filename without extension)
            sentence_id = os.path.splitext(os.path.basename(f_path))[0]
            
            data = np.load(f_path, allow_pickle=True)
            metrics = extract_metrics_from_npz(data)
            
            if metrics:
                sentence_metrics[sentence_id] = metrics
                
        except Exception as e:
            # print(f"  Warning: Could not load or process {os.path.basename(f_path)}: {e}")
            continue
            
    return sentence_metrics

# --- 2. Data Aggregation and Output ---

def write_tsv_summary(aggregated_data, output_file):
    """
    Writes the aggregated data to a TSV file.
    
    Args:
        aggregated_data (dict): {'sentence_id': {'model_name': {metrics}}}
        output_file (str): Path to the output .tsv file.
    """
    if not aggregated_data:
        print("No aggregated data to write. Aborting.")
        return

    # Get all unique model names and metric keys
    all_models = sorted(list(set(m_name for metrics_by_model in aggregated_data.values() 
                                for m_name in metrics_by_model.keys())))
    
    # Assuming all metric keys are consistent (auc_tv, loss, entropy)
    metric_keys = ['auc_tv', 'loss', 'entropy']
    
    # Construct Header
    header = ['sentence_id']
    for model_name in all_models:
        for m_key in metric_keys:
            header.append(f"{model_name}_{m_key}")

    print(f"\nWriting {len(aggregated_data)} sentences to {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)

        # Sort sentence IDs for predictable output order
        for sentence_id in sorted(aggregated_data.keys()):
            row = [sentence_id]
            metrics_by_model = aggregated_data[sentence_id]
            
            for model_name in all_models:
                model_metrics = metrics_by_model.get(model_name, {})
                
                # Fill in the row data. Use empty string/NaN if model data is missing for this sentence.
                for m_key in metric_keys:
                    value = model_metrics.get(m_key, np.nan)
                    row.append(f"{value:.6f}" if not np.isnan(value) else "") # Format to 6 decimal places or empty

            writer.writerow(row)
            
    print(f"Successfully created summary file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Per-Sentence Multi-Model Metrics Summarization.")
    parser.add_argument('--base_result_dir', type=str, required=True, 
                        help='Directory containing model subfolders (e.g., ./results).')
    parser.add_argument('--dir', type=str, required=True, 
                        help='Directory containing the npz (e.g., ./final_straces_20).')
    parser.add_argument('--output_file', type=str, default="per_sentence_summary.tsv", 
                        help='Name of the output TSV file.')
    parser.add_argument('--min_files', type=int, default=1000,
                        help='Minimum number of .npz files required to include a model (default: 1000).')
    args = parser.parse_args()

    if not os.path.isdir(args.base_result_dir):
        print(f"Error: Base results directory '{args.base_result_dir}' not found.")
        return

    # Find potential model directories
    model_dirs = [os.path.join(args.base_result_dir, d) 
                  for d in os.listdir(args.base_result_dir) 
                  if os.path.isdir(os.path.join(args.base_result_dir, d))]
    
    # Structure: {sentence_id: {model_name: {metrics}}}
    aggregated_data = defaultdict(lambda: defaultdict(dict))

    print(f"Found {len(model_dirs)} model directories. Starting processing...")

    for model_path in sorted(model_dirs):
        model_name = os.path.basename(model_path)

        if "stage" in model_name:
            continue # skip intermediate training checkpoints
        
        sentence_metrics = load_model_data(model_name, os.path.join(model_path, args.dir), args.min_files)
        
        # Aggregate results for this model
        for sentence_id, metrics in sentence_metrics.items():
            aggregated_data[sentence_id][model_name] = metrics

    if not aggregated_data:
        print("No data was successfully loaded across all models.")
        return

    # Final output generation
    write_tsv_summary(aggregated_data, args.output_file)

if __name__ == "__main__":
    main()