"""
Trace Metric Aggregator & TSV Exporter
======================================
Processes instance-level .npz evaluation outputs across models and extracts 
per-trace metrics (Total Variation, Surprisal, Entropy, Loss, Trace Size) into a 
structured TSV file.

TSV Columns:
- sentence_id
- model name
- trace_size
- trace_entropy
- trace_loss
- tv
- surprisal
"""

import argparse
import glob
import os
import re
import traceback
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("default", category=RuntimeWarning)


def extract_sentence_id(filename):
    """
    Safely extracts integer sentence ID from filename. 
    Falls back to string filename if no integer is found.
    """
    basename = os.path.basename(filename)
    match = re.search(r'\d+', basename)
    if match:
        return int(match.group(0))
    return basename


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-trace TV, surprisal, entropy, and loss metrics into TSV."
    )
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Root directory storing per-model evaluation outputs.')
    parser.add_argument('--length', type=str, required=True,
                        help='Data chunk length identifier (e.g., 20, 40).')
    parser.add_argument('--max_files', type=int, default=500,
                        help='Maximum number of sentence .npz files to aggregate per model.')
    parser.add_argument('--output_tsv', type=str, default="trace_metrics_aggregated.tsv",
                        help='Output TSV file path.')
    args = parser.parse_args()

    model_dirs = [
        d for d in os.listdir(args.base_dir) 
        if os.path.isdir(os.path.join(args.base_dir, d, 'main'))
    ]

    if not model_dirs:
        print(f"Error: No valid model directories found under '{args.base_dir}'.")
        return

    print(f"[AGGREGATOR] Found {len(model_dirs)} models to process in '{args.base_dir}'.")

    records = []

    for model_name in sorted(model_dirs):
        print(f"\n--- Processing Model: {model_name} ---")

        target_dir = os.path.join(args.base_dir, model_name, 'main', f'final_straces_{args.length}')
        npz_files = glob.glob(os.path.join(target_dir, '*.npz'))[:args.max_files]

        if not npz_files:
            print(f"  [WARN] No .npz files found in {target_dir}. Skipping.")
            continue

        processed_count = 0

        for f in tqdm(npz_files, desc=f"  Extracting traces for {model_name}"):
            try:
                data = np.load(f, allow_pickle=True)

                sentence_id = extract_sentence_id(f)

                # Extract relative subgraph sizes and compute sort permutation
                rel_size = np.asarray(data['strata_rel_size'])
                sort_idx = np.argsort(rel_size)
                sorted_size = rel_size[sort_idx]

                # Extract full traces aligned by sort order
                tv_trace = np.array(data['strata_reco_tv'].item()['trace']['only'])[sort_idx]
                loss_trace = np.array(data['strata_loss'].item()['trace']['only'])[sort_idx]
                entropy_trace = np.array(data['strata_entropy'].item()['trace']['only'])[sort_idx]

                # Unpack step for each trace size sample
                for step_idx in range(len(sorted_size)):
                    t_size = float(sorted_size[step_idx])
                    t_entropy = float(entropy_trace[step_idx])
                    t_loss = float(loss_trace[step_idx])
                    t_tv = float(tv_trace[step_idx])

                    # Skip invalid step entries
                    if np.isnan(t_entropy) or np.isnan(t_loss) or np.isnan(t_tv):
                        continue

                    records.append({
                        'sentence_id': sentence_id,
                        'model name': model_name,
                        'trace_size': t_size,
                        'trace_entropy': t_entropy,
                        'trace_loss': t_loss,
                        'tv': t_tv,
                        'surprisal': t_loss,  # Surprisal corresponds to sequence trace loss
                    })

                processed_count += 1

            except Exception as e:
                print(f"  [ERROR] Processing file {f}: {e}")

        print(f"  [SUMMARY] {model_name}: Extracted trace records from {processed_count} files.")

    # Export collected records to TSV
    if records:
        output_df = pd.DataFrame(records)
        
        # Enforce column schema
        column_order = [
            'sentence_id',
            'model name',
            'trace_size',
            'trace_entropy',
            'trace_loss',
            'tv',
            'surprisal'
        ]
        output_df = output_df[column_order]

        output_df.to_csv(args.output_tsv, sep='\t', index=False)
        print(f"\n[COMPLETE] Exported {len(output_df)} trace records to '{args.output_tsv}'.")
    else:
        print("\n[WARN] No records were generated. TSV output omitted.")


if __name__ == "__main__":
    main()