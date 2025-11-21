import argparse
import os
import numpy as np
import glob
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

# Suppress warnings that often occur when dealing with log scales on histograms
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Data Loading and Aggregation ---

def extract_weights(model_dir, npz_dir, min_files):
    """
    Loads all .npz files for a single model and extracts all edge weights, 
    categorized by their type.

    Args:
        model_dir (str): Path to the model's result directory.

    Returns:
        dict: {edge_type: list of weights}
    """
    model_name = os.path.basename(model_dir)
    results_glob = glob.glob(os.path.join(model_dir, npz_dir))
    
    if not results_glob:
        print(f"  Skipping {model_name}: No {npz_dir} subfolder found.")
        return {}
        
    results_dir = results_glob[0]
    file_paths = glob.glob(os.path.join(results_dir, '*.npz'))
    
    if not file_paths:
        print(f"  Skipping {model_name}: No .npz files found in {results_dir}.")
        return {}
        
    if len(file_paths) < min_files:
        print(f"  Skipping {model_name}: Only {len(file_paths)} files found in {results_dir}.")
        return {}

    print(f"  Processing {len(file_paths)} traces for model: {model_name}...")
    
    # Structure: {edge_type: [w1, w2, w3, ...]}
    all_weights_by_type = defaultdict(list)
    
    for f_path in tqdm(file_paths, desc=f"[{model_name}] Extracting weights..."):
        try:
            data = np.load(f_path, allow_pickle=True)
            
            # The 'weights' key contains all raw edge importance scores
            weights = data['weights']
            
            # 'edge_name_map' contains the string names (e.g., 'attn_q-input')
            edge_maps = data['edge_name_map']
            
            # 'name_ids' links weights to the names in 'edge_name_map'
            name_ids = data['name_ids']
            
            # Iterate through all weights and categorize them
            for w, name_id in zip(weights, name_ids):
                edge_name = edge_maps[name_id]                

                edge_type = edge_name.split('_')[0].split('-')[0]                
                
                all_weights_by_type[edge_type].append(w)
                    
        except Exception as e:
            print(f"  Warning: Could not load or process {os.path.basename(f_path)}: {e}")
            continue
            
    # Convert lists to NumPy arrays for efficiency
    return {k: np.array(v) for k, v in all_weights_by_type.items()}

# --- 2. Plotting Logic ---

def create_plots(model_name, weights_by_type, output_dir):
    """
    Generates and saves the overall and per-type distribution plots.
    """
    if not weights_by_type:
        print(f"No weights found for {model_name}. Skipping plots.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall Distribution Plot
    all_weights = np.concatenate(list(weights_by_type.values()))
    # Filter out zero weights for log scale
    all_weights_filtered = all_weights[all_weights > 0]
    
    if len(all_weights_filtered) > 100:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            all_weights_filtered, 
            bins=np.logspace(np.log10(all_weights_filtered.min()), np.log10(all_weights_filtered.max()), 100),
            kde=True,
            stat="density",
            palette="viridis",
            log_scale=True
        )
        plt.xscale('log')
        plt.title(f'Overall Edge Weight Distribution for {model_name}', fontsize=14)
        plt.xlabel('Absolute Edge Weight (Log Scale)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_overall_distribution.png'))
        plt.close()
        print(f"  -> Saved overall plot.")
    else:
        print(f"  -> Too few non-zero weights ({len(all_weights_filtered)}) for overall plot.")


    # 2. Per-Type Distribution Plot (Top N types)
    # Determine top 10 edge types by count for cleaner visualization
    type_counts = {k: len(v) for k, v in weights_by_type.items()}
    sorted_types = sorted(type_counts.keys(), key=lambda k: type_counts[k], reverse=True)
    top_n_types = sorted_types[:10]
    
    if top_n_types:
        plt.figure(figsize=(12, 8))
        
        # Prepare data for Seaborn kdeplot
        plot_data = []
        for type_name in top_n_types:
            weights = weights_by_type[type_name]
            weights_filtered = weights[weights > 0]
            if len(weights_filtered) > 50: # Only plot types with enough data points
                for w in weights_filtered:
                    plot_data.append({'Weight': w, 'Edge Type': type_name})

        if plot_data:
            df = np.array(plot_data, dtype=[('Weight', 'f8'), ('Edge Type', 'U50')])
            
            sns.kdeplot(
                data=df, 
                x='Weight', 
                hue='Edge Type', 
                log_scale=True, 
                fill=True, 
                alpha=.4, 
                linewidth=1.5,
                palette="tab10"
            )
            
            plt.xscale('log')
            plt.title(f'Edge Weight Distribution by Type (Top {len(top_n_types)} Types) for {model_name}', fontsize=14)
            plt.xlabel('Absolute Edge Weight (Log Scale)', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend(title='Edge Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_type_distribution.png'))
            plt.close()
            print(f"  -> Saved per-type plot.")
        else:
            print("  -> Insufficient filtered data to create per-type plot.")


# --- 3. Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Plot edge weight distributions from LLM trace NPZ files."
    )
    parser.add_argument('--base_result_dir', type=str, required=True, 
                        help='Directory containing model subfolders (e.g., ./results).')
    parser.add_argument('--dir', type=str, required=True, 
                        help='Directory containing the npz (e.g., ./final_straces_20).')
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="weight_plots", 
        help='Directory to save the generated plots.'
    )
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
    
    if not model_dirs:
        print("No model subdirectories found.")
        return

    print(f"Found {len(model_dirs)} model directories. Starting processing...")
    
    for model_path in sorted(model_dirs):
        model_name = os.path.basename(model_path)

        if "stage" in model_name:
            continue #skip intermediate checkpoints

        match = re.search(r'[-_]?(\d+\.?\d*)B[-_]?', model_name, re.IGNORECASE)
        model_size = float(match[1])
        if model_size > 2:
            continue # skip large models

        print(f"\nProcessing model: {model_name}")
        
        # 1. Load and aggregate all weights for the model
        weights_by_type = extract_weights(model_path, args.dir, args.min_files)
        
        # 2. Create and save the plots
        create_plots(model_name, weights_by_type, args.output_dir)

if __name__ == "__main__":
    main()