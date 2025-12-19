import argparse
import os
import numpy as np
import glob
import re
import csv
from collections import defaultdict
import warnings
import concurrent.futures
from tqdm import tqdm

# Suppress NumPy warnings (often related to NaNs in data)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Core Data Extraction Logic ---

def extract_metrics_from_npz(data):
    """
    Extracts AUC (Total Variation), Full Model Loss, and Full Model Entropy 
    from a single loaded .npz file data structure.
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
        tv_key = 'strata_reco_tv'
        
        raw_tv = data[tv_key].item()['trace']['only']
        tv_sorted = np.array(raw_tv)[sort_idx]
        
        # Compute Area Under the Curve (Trapz method)
        auc_tv = np.trapz(tv_sorted, size_sorted)

        # --- 3. Extract Full Model Metrics (Loss and Entropy) ---
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
        return None

def _extract_index_from_path(path):
    """Helper to extract the numeric step/index from a filename."""
    name = os.path.basename(path)
    m = re.search(r'strace_final_(\d+)\.npz$', name)
    if m:
        return int(m.group(1))
    m2 = re.search(r'_(\d+)\.npz$', name)
    if m2:
        return int(m2.group(1))
    return float('inf')

def process_single_file(f_path):
    """Worker function for multiprocessing."""
    try:
        step = _extract_index_from_path(f_path)
        data = np.load(f_path, allow_pickle=True)
        metrics = extract_metrics_from_npz(data)
        if metrics:
            return step, metrics
    except Exception:
        return None
    return None

def load_generation_tsv(prompt_path):
    """
    Loads generation.tsv from the sibling 'intermediate_graphs' directory.
    Ignores the 'next_token_str' to avoid parsing errors.
    """
    try:
        # Construct path: .../seed/final_straces/prompt -> .../seed/intermediate_graphs/prompt/generation.tsv
        seed_dir = os.path.dirname(os.path.dirname(prompt_path))
        prompt_name = os.path.basename(prompt_path)
        tsv_path = os.path.join(seed_dir, 'intermediate_graphs', prompt_name, 'generation.tsv')

        if not os.path.exists(tsv_path):
            return {}

        tsv_data = {}
        with open(tsv_path, 'r', encoding='utf-8') as f:
            # Skip header
            _ = f.readline()
            
            for i, line in enumerate(f):
                line = line.strip('\n')
                if not line:
                    continue
                
                if line.startswith('BREAK'):
                    continue

                parts = line.split('\t')
                
                # We need at least the 4 numeric columns at the end.
                # If the string column contained tabs, len(parts) > 5.
                # We simply grab the last 4 elements safely.
                if len(parts) < 5:
                    continue 

                # Right-to-Left parsing for safety
                token_id = parts[-4]
                token_prob = parts[-3]
                nuc_id = parts[-2]
                ent = parts[-1]

                tsv_data[i] = {
                    'next_token_id': token_id,
                    'next_token_prob': token_prob,
                    'nucleus_token_id': nuc_id,
                    'gen_entropy': ent 
                }
        return tsv_data
    except Exception as e:
        print(f"Warning parsing TSV {tsv_path}: {e}")
        return {}

def load_model_data_parallel(model_name, prompt_name, seed, prompt_path, min_files):
    """
    Loads .npz files (parallel) and generation.tsv (sequential) and merges them.
    """
    file_paths = glob.glob(os.path.join(prompt_path, '*.npz'))
    
    if not file_paths or len(file_paths) < min_files:
        return {}

    # 1. Load TSV data for this prompt
    tsv_metrics = load_generation_tsv(prompt_path)

    # 2. Load NPZ data in parallel
    file_paths = sorted(file_paths, key=_extract_index_from_path)
    sentence_metrics = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_single_file, file_paths), 
            total=len(file_paths), 
            leave=False, 
            desc=f"    Parsing {prompt_name[:15]}..."
        ))

    # 3. Merge Data
    for res in results:
        if res is not None:
            step, npz_data = res
            
            combined = npz_data.copy()
            
            # Merge with TSV data if available for this step
            if step in tsv_metrics:
                combined.update(tsv_metrics[step])
            else:
                combined.update({
                    'next_token_id': '', 'next_token_prob': '', 
                    'nucleus_token_id': '', 'gen_entropy': ''
                })

            sentence_metrics[step] = combined

    return sentence_metrics

# --- 2. Data Aggregation and Output ---

def write_tsv_summary(aggregated_data, output_file):
    """
    Writes the aggregated data to a TSV file.
    """
    if not aggregated_data:
        print(f"No aggregated data to write for {output_file}.")
        return

    # Columns to export (Removed next_token_str)
    metric_keys = [
        'auc_tv', 'entropy', 'gen_entropy', 
        'next_token_id', 'next_token_prob', 'nucleus_token_id', 
        'raw_size'
    ]
    header = ['prompt_name', 'seed', 'step'] + metric_keys

    def step_sort_key(s):
        try:
            return (0, int(s))
        except Exception:
            return (1, s)

    rows_written = 0
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)

        for prompt_name in sorted(aggregated_data.keys()):
            models = aggregated_data[prompt_name]
            for seed_key in sorted(models.keys()):
                steps = models[seed_key]
                for step in sorted(steps.keys(), key=step_sort_key):
                    metrics = steps.get(step, {}) or {}
                    row = [prompt_name, seed_key, step]
                    
                    for k in metric_keys:
                        v = metrics.get(k, None)
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
                                    # Don't format IDs as floats
                                    if k.endswith('_id'):
                                        formatted = str(v)
                                    else:
                                        formatted = f"{fv:.6f}"
                                else:
                                    formatted = ""
                            except Exception:
                                formatted = str(v)
                        row.append(formatted)
                    writer.writerow(row)
                    rows_written += 1

    print(f"  -> Saved: {output_file} ({rows_written} rows)")


def main():
    parser = argparse.ArgumentParser(description="Per-Sentence Multi-Model Metrics Summarization.")
    parser.add_argument('--base_result_dir', type=str, required=True, 
                        help='Directory containing model subfolders.')
    parser.add_argument('--output_prefix', type=str, default="per_sentence_summary", 
                        help='Prefix for the output TSV file.')
    parser.add_argument('--min_files', type=int, default=20,
                        help='Minimum number of .npz files required to include a prompt.')
    args = parser.parse_args()

    if not os.path.isdir(args.base_result_dir):
        print(f"Error: Base results directory '{args.base_result_dir}' not found.")
        return

    model_dirs = [os.path.join(args.base_result_dir, d) 
                  for d in os.listdir(args.base_result_dir) 
                  if os.path.isdir(os.path.join(args.base_result_dir, d))]
    
    print(f"Found {len(model_dirs)} model directories.")

    # Outer loop over models
    for model_path in tqdm(sorted(model_dirs), desc="Models"):
        model_name = os.path.basename(model_path)
        aggregated_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        seed_dirs = [os.path.join(model_path, d) 
                  for d in os.listdir(model_path) 
                  if os.path.isdir(os.path.join(model_path, d))]
        
        has_data = False

        for seed_path in seed_dirs:
            seed = os.path.basename(seed_path).split('_')[-1]
            final_straces_path = os.path.join(seed_path, 'final_straces')
            
            if not os.path.isdir(final_straces_path):
                continue

            prompt_dirs = [os.path.join(final_straces_path, d) 
                    for d in os.listdir(final_straces_path) 
                    if os.path.isdir(os.path.join(final_straces_path, d))]

            for prompt_path in tqdm(prompt_dirs, desc=f"  Seeds in {model_name}", leave=False):
                prompt_name = os.path.basename(prompt_path)

                sentence_metrics = load_model_data_parallel(
                    model_name, prompt_name, seed, prompt_path, args.min_files
                )
            
                if sentence_metrics:
                    has_data = True
                    for step, metrics in sentence_metrics.items():
                        aggregated_data[prompt_name][seed][step] = metrics

        if has_data:
            output_path = f"{args.output_prefix}_{model_name}.tsv"
            write_tsv_summary(aggregated_data, output_path)
        else:
            print(f"  No valid data found for model {model_name}")

if __name__ == "__main__":
    main()