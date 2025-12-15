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
        dict: {'auc_tv': float, 'raw_size': list(int), 'entropy': float} or None on error.
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
        
        # Access the full model entropy value at the max size index
        entropy_key = "strata_entropy"
        
        full_entropy = data[entropy_key].item()['trace']['only'][full_idx]
        
        raw_size = np.array(data['strata_raw_size'])
        
        return {
            'auc_tv': auc_tv,
            'entropy': full_entropy,
            'raw_size': raw_size
        }
    except Exception as e:
        print(f"Warning: Failed to process file due to error: {e}")
        return None

def load_model_data(model_name, prompt_name, model_dir, min_files):
    """
    Loads and processes all trace files for a single model.
    
    Args:
        model_dir (str): Path to the model's result directory.
        
    Returns:
        dict: {sentence_id: {'auc_tv': float, 'loss': float, 'entropy': float}}
    """

    file_paths = glob.glob(os.path.join(model_dir, '*.npz'))

    # Sort by the numeric suffix in filenames like "strace_final_X.npz" (ascending, so X=0 first).
    def _extract_index(path):
        name = os.path.basename(path)
        m = re.search(r'strace_final_(\d+)\.npz$', name)
        if m:
            return int(m.group(1))
        m2 = re.search(r'_(\d+)\.npz$', name)
        if m2:
            return int(m2.group(1))
        return float('inf')

    file_paths = sorted(file_paths, key=_extract_index)
    
    if not file_paths:
        print(f"  Skipping {model_name}|{prompt_name}: No .npz files found in {model_dir}.")
        return {}
    
    if len(file_paths) < min_files:
        print(f"  Skipping {model_name}|{prompt_name}: Only {len(file_paths)} files found in {model_dir}.")
        return {}

    print(f"  Processing {len(file_paths)} traces for model: {model_name}|{prompt_name}...")
    
    sentence_metrics = {}
    
    for f_path in file_paths:
        try:
            # Extract sentence_id (filename without extension)
            step = _extract_index(f_path)

            data = np.load(f_path, allow_pickle=True)
            metrics = extract_metrics_from_npz(data)
            
            if metrics:
                sentence_metrics[step] = metrics

                
        except Exception as e:
            print(f"  Warning: Could not load or process {os.path.basename(f_path)}: {e}")
            continue
            
    return sentence_metrics

# --- 2. Data Aggregation and Output ---

def write_tsv_summary(aggregated_data, output_file):
    """
    Writes the aggregated data to a TSV file.

    Args:
        aggregated_data (dict): {'prompt_name':{'model_name': {'step': {metrics}}}}
        output_file (str): Path to the output .tsv file.
    """
    if not aggregated_data:
        print("No aggregated data to write. Aborting.")
        return

    metric_keys = ['auc_tv', 'entropy', 'raw_size']
    header = ['prompt_name', 'model_name', 'step'] + metric_keys

    def step_sort_key(s):
        # try numeric sort, fall back to string
        try:
            return (0, int(s))
        except Exception:
            return (1, s)

    rows_written = 0
    print(f"\nWriting summary to {output_file}...")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)

        for prompt_name in sorted(aggregated_data.keys()):
            models = aggregated_data[prompt_name]
            for model_name in sorted(models.keys()):
                steps = models[model_name]
                for step in sorted(steps.keys(), key=step_sort_key):
                    metrics = steps.get(step, {}) or {}
                    row = [prompt_name, model_name, step]
                    for k in metric_keys:
                        v = metrics.get(k, None)
                        # Format numeric values to 6 decimals, otherwise empty.
                        # If v is a list/tuple/np.ndarray, write the full list.
                        formatted = ""
                        if v is None:
                            formatted = ""
                        elif isinstance(v, (list, tuple, np.ndarray)):
                            try:
                                formatted = str(list(v))
                            except Exception:
                                formatted = str(v)
                        else:
                            try:
                                fv = float(v)
                                if np.isfinite(fv):
                                    formatted = f"{fv:.6f}"
                                else:
                                    formatted = ""
                            except Exception:
                                formatted = ""
                        row.append(formatted)
                    writer.writerow(row)
                    rows_written += 1

    print(f"Successfully created summary file: {output_file} with {rows_written} rows.")


def main():
    parser = argparse.ArgumentParser(description="Per-Sentence Multi-Model Metrics Summarization.")
    parser.add_argument('--base_result_dir', type=str, required=True, 
                        help='Directory containing model subfolders (e.g., ./generation/results).')
    parser.add_argument('--output_file', type=str, default="per_sentence_summary.tsv", 
                        help='Name of the output TSV file.')
    parser.add_argument('--min_files', type=int, default=20,
                        help='Minimum number of .npz files required to include a prompt (default: 20).')
    args = parser.parse_args()

    if not os.path.isdir(args.base_result_dir):
        print(f"Error: Base results directory '{args.base_result_dir}' not found.")
        return

    # Find potential model directories
    model_dirs = [os.path.join(args.base_result_dir, d) 
                  for d in os.listdir(args.base_result_dir) 
                  if os.path.isdir(os.path.join(args.base_result_dir, d))]
    
    # Structure: {model_name: {sentence_id: {metrics}}}
    aggregated_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict) ))

    print(f"Found {len(model_dirs)} model directories. Starting processing...")

    for model_path in sorted(model_dirs):
        model_name = os.path.basename(model_path)
        print('model_name', model_name)
        print(os.listdir(os.path.join(model_path, 'final_straces')))
        # Find potential model directories
        prompt_dirs = [os.path.join(model_path, 'final_straces', d) 
                  for d in os.listdir(os.path.join(model_path, 'final_straces')) 
                  if os.path.isdir(os.path.join(model_path, 'final_straces', d))]

        print(f"Found {len(prompt_dirs)} prompt directories. Starting processing...")

        for prompt_path in prompt_dirs:
            prompt_name = os.path.basename(prompt_path)
            print('prompt_name', prompt_name)

            sentence_metrics = load_model_data(model_name, prompt_name, prompt_path, args.min_files)
        
            # Aggregate results for this model
            for step, metrics in sentence_metrics.items():
                aggregated_data[prompt_name][model_name][step] = metrics

    if not aggregated_data:
        print("No data was successfully loaded across all models and prompt.")
        return

    # Final output generation
    write_tsv_summary(aggregated_data, args.output_file)

if __name__ == "__main__":
    main()