"""
Section 4 & 5: Subgraph Topology & Computation Density Aggregator
=================================================================
This script processes instance-level .npz evaluation outputs generated across models 
to construct plot for data visualization and statistical analysis.

Outputs:
-------------------------------------------
1. Subgraph Density Domain Alignment (COMMON_SIZE):
   Standardizes varying edge-count subgraphs across sentences onto a logarithmic 
   grid of 150 points from s = 10^-5 to s = 1.0 (100% density).

2. Total Variation (TV) Reconstruction Curves:
   - Target s-Trace ('tv_trace'): Reconstruction error of the extracted minimal subgraph.
   - Inverse Subgraph ('tv_inv'): Reconstruction error of the complementary pruned graph.
   - Random Baseline ('tv_rand'): Reconstruction error of uniformly random edge pruning.

3. Structural Component Topology & Depth Quartiles:
   Tracks the sequential entry of Attention heads, MLP modules, and Residual 
   connections into the s-trace, as well as depth distribution across 4 layer 
   quartiles (Layer Group 0 to 3).

4. Statistical Correlations:
   Evaluates Spearman rank correlation between full prediction entropy and 
   computation density (AUC-TV), as well as sentence difficulty consistency 
   across model pairs.
"""

import argparse
import glob
import itertools
import os
import pickle
import re
import traceback
import warnings
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

# Suppress expected numerical warnings during edge-case interpolation steps
warnings.filterwarnings("default", category=RuntimeWarning)

# Standardized output directory and logarithmic subgraph density domain s in [10^-5, 1.0]
SAVE_FOLDER = "./"
COMMON_SIZE = np.logspace(-5, 0, 150)


# =============================================================================
# 1. Array Validation & Interpolation Helpers
# =============================================================================

def check_array(name, arr, file, stats):
    """
    Validates array integrity, logging NaN or Inf anomalies for dataset quality audits.
    """
    arr = np.asarray(arr)
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()

    if n_nan > 0 or n_inf > 0:
        stats["nan_counts"][name] += int(n_nan)
        stats["inf_counts"][name] += int(n_inf)
        stats["files_with_nan"].add(file)

    return arr


def safe_interp(x, xp, fp, stats, file, name):
    """
    Safely interpolates irregular metric traces (xp, fp) onto the uniform COMMON_SIZE grid (x).
    Validates monotonic ascending order of density points (xp).
    """
    if len(xp) == 0 or len(fp) == 0:
        stats["empty_arrays"] += 1
        return None

    if not np.all(np.diff(xp) >= 0):
        stats["non_monotonic"] += 1
        return None

    try:
        out = np.interp(x, xp, fp)
        if not np.all(np.isfinite(out)):
            stats["bad_interp"] += 1
            return None
        return out
    except Exception:
        stats["interp_failures"] += 1
        return None


def safe_mean(arr_list, stats):
    """
    Computes point-wise mean across sentence curves mapped to the common density domain.
    """
    if len(arr_list) == 0:
        stats["empty_means"] += 1
        return np.full_like(COMMON_SIZE, np.nan)
    return np.mean(arr_list, axis=0)


# =============================================================================
# 2. Structural Topology & Component Mapping
# =============================================================================

def get_component_names(data):
    """
    Maps edge node target IDs to specific Transformer architectural components 
    (e.g., Attention Heads 'head_H_LL', MLP modules 'mlp_LL', Residual streams 'res_attn_LL').
    """
    edge_name_map = data['edge_name_map']
    name_ids = data['name_ids']
    edges = data['edges']
    targets = edges[:, 1]
    n_tokens = data['graph_attrs'].item().get('n_tokens', 1)

    # Derive layer index from target node position
    block_idx = targets // n_tokens
    layers = np.where(block_idx == 0, 0, (block_idx - 1) // 2)
    names = [str(edge_name_map[nid]).lower() for nid in name_ids]

    comp_names = []
    for i, name in enumerate(names):
        l = layers[i]
        if "residual" in name:
            suffix = "attn" if "attention" in name else "mlp"
            comp_names.append(f"res_{suffix}_L{l}")
        elif "mlp" in name:
            comp_names.append(f"mlp_L{l}")
        elif "attention" in name:
            match = re.search(r'h(\d+)', name)
            h_id = match.group(1) if match else "X"
            comp_names.append(f"head_{h_id}_L{l}")
        else:
            comp_names.append(f"other_L{l}")
            
    return np.array(comp_names)


def extract_topology(data, stats, file):
    """
    Extracts structural breakdown as subgraph grows:
      - Proportion of Attention, MLP, and Residual edges.
      - Proportion of edges originating across 4 layer depth quartiles.
    """
    try:
        edge_name_map = data['edge_name_map']
        name_ids = data['name_ids']
        names = [str(edge_name_map[nid]).lower() for nid in name_ids]

        # Component classification masks
        attn_mask = np.array(['attention' in n and 'residual' not in n for n in names])
        mlp_mask = np.array(['mlp' in n and 'residual' not in n for n in names])
        res_mask = np.array(['residual' in n for n in names])

        edges = data['edges']
        targets = edges[:, 1]
        n_tokens = data['graph_attrs'].item().get('n_tokens', 1)

        block_idx = targets // n_tokens
        layers = np.where(block_idx == 0, 0, (block_idx - 1) // 2)
        max_layer = int(np.max(layers))

        # Order edges according to attribution importance
        strata_index = data['strata_index']
        sort_idx = np.argsort(strata_index)
        sorted_strata = strata_index[sort_idx]

        stratum_assignments = data['stratum']
        order = np.argsort(stratum_assignments)
        sorted_assignments = stratum_assignments[order]

        # Cumulative component edge accumulation
        attn_cumsum = np.cumsum(attn_mask[order])
        mlp_cumsum = np.cumsum(mlp_mask[order])
        res_cumsum = np.cumsum(res_mask[order])
        total_cumsum = np.cumsum(np.ones_like(attn_cumsum))

        idxs = np.searchsorted(sorted_assignments, sorted_strata, side='right') - 1
        idxs = np.clip(idxs, 0, len(attn_cumsum) - 1)

        safe_total = np.where(total_cumsum[idxs] == 0, 1, total_cumsum[idxs])

        # Partition network depth into 4 equal layer quartiles
        num_layers = max_layer + 1
        layer_categories = np.array_split(np.arange(num_layers), 4)
        group_counts = np.zeros((4, len(sorted_strata)))

        for g_idx, group in enumerate(layer_categories):
            mask = np.isin(layers[order], group).astype(np.int32)
            group_counts[g_idx] = np.cumsum(mask)[idxs]

        return {
            'sort_idx': sort_idx,
            'attn_prop': attn_cumsum[idxs] / safe_total,
            'mlp_prop': mlp_cumsum[idxs] / safe_total,
            'res_prop': res_cumsum[idxs] / safe_total,
            'layer_group_props': group_counts / safe_total
        }
    except Exception:
        stats["topology_failures"] += 1
        return None


# =============================================================================
# 3. Main Data Aggregation Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Section 4/5: Process evaluation files to compute curves, topology, and correlations."
    )
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Root directory storing per-model evaluation outputs.')
    parser.add_argument('--length', type=str, required=True,
                        help='Data chunk length identifier (e.g., 20, 40).')
    parser.add_argument('--max_files', type=int, default=500,
                        help='Maximum number of sentence .npz files to aggregate per model.')
    parser.add_argument('--components', action='store_true',
                        help='Enable detailed module-level (head/MLP) tracking.')
    args = parser.parse_args()

    data_agg, data_nu, data_sentences, data_topology = {}, {}, {}, {}
    data_components = {}
    model_sentence_auc, model_sentence_entropy = {}, {}
    reports = {}

    model_dirs = [
        d for d in os.listdir(args.base_dir) 
        if os.path.isdir(os.path.join(args.base_dir, d, 'main'))
    ]

    print(f"[AGGREGATOR] Found {len(model_dirs)} models to process in '{args.base_dir}'.")

    for model_name in sorted(model_dirs):
        print(f"\n--- Processing Model: {model_name} ---")

        stats = defaultdict(int)
        stats.update({
            "nan_counts": defaultdict(int),
            "inf_counts": defaultdict(int),
            "files_with_nan": set()
        })

        error_count = 0
        target_dir = os.path.join(args.base_dir, model_name, 'main', f'final_straces_{args.length}')
        npz_files = glob.glob(os.path.join(target_dir, '*.npz'))[:args.max_files]

        if not npz_files:
            print(f"  [WARN] No .npz files found in {target_dir}. Skipping.")
            continue

        model_agg_trace_tv, model_agg_random_tv, model_agg_inv_tv = [], [], []
        nu_targets = [1, 5, 10, 20, 40, 60, 80, 90]
        nucleus_sizes = {k: [] for k in nu_targets}

        sentence_data = []
        model_sentence_auc[model_name], model_sentence_entropy[model_name] = {}, {}

        topo_attn, topo_mlp, topo_res = [], [], []
        topo_layers = {i: [] for i in range(4)}
        model_comp_agg = {}

        for f in tqdm(npz_files, desc=f"  Aggregating {model_name}"):
            try:
                data = np.load(f, allow_pickle=True)

                # --- Extract and sort relative subgraph sizes ---
                rel_size = check_array("rel_size", data['strata_rel_size'], f, stats)
                sort_idx = np.argsort(rel_size)
                sorted_size = rel_size[sort_idx]

                # --- Total Variation Traces (Target, Inverse, Random) ---
                tv_trace = check_array("tv_trace", np.array(data['strata_reco_tv'].item()['trace']['only'])[sort_idx], f, stats)
                tv_inv = check_array("tv_trace", np.array(data['strata_reco_tv'].item()['trace']['inverse'])[sort_idx], f, stats)
                tv_rand = check_array("tv_rand", np.array(data['strata_reco_tv'].item()['random']['only'])[sort_idx], f, stats)

                # Interpolate TV distance onto logarithmic common grid
                interp_trace = safe_interp(COMMON_SIZE, sorted_size, tv_trace, stats, f, "tv_trace")
                interp_inv = safe_interp(COMMON_SIZE, sorted_size, tv_inv, stats, f, "tv_inv")
                interp_rand = safe_interp(COMMON_SIZE, sorted_size, tv_rand, stats, f, "tv_rand")

                if interp_trace is None or interp_inv is None or interp_rand is None:
                    continue

                model_agg_trace_tv.append(interp_trace)
                model_agg_inv_tv.append(interp_inv)
                model_agg_random_tv.append(interp_rand)

                # --- Nucleus Recovery Thresholds (k%) ---
                nu_trace = check_array("nu_trace", np.array(data['strata_reco_nu'].item()['trace']['only'])[sort_idx], f, stats)
                for k in nu_targets:
                    valid = np.where(nu_trace >= k)[0]
                    if len(valid) > 0:
                        nucleus_sizes[k].append(sorted_size[valid[0]])

                # --- Sentence-Level Metrics & Computation Density (AUC-TV) ---
                entropy_curve = check_array("entropy", data['strata_entropy'].item()['trace']['only'], f, stats)
                if len(entropy_curve) == 0:
                    continue

                full_entropy = float(entropy_curve[np.argmax(rel_size)])
                auc_val = np.trapz(tv_trace, np.log10(np.clip(sorted_size, 1e-8, 1.0)))

                fname = os.path.basename(f)
                model_sentence_auc[model_name][fname] = auc_val
                model_sentence_entropy[model_name][fname] = full_entropy

                sentence_data.append({
                    'file': fname,
                    'entropy': full_entropy,
                    'auc': auc_val,
                    'size': sorted_size,
                    'tv': tv_trace
                })

                # --- Structural Topology Accumulation ---
                topo = extract_topology(data, stats, f)
                if topo is not None:
                    align = np.argsort(topo['sort_idx'])[sort_idx]

                    for name, src, dest in [
                        ("attn", topo['attn_prop'], topo_attn),
                        ("mlp", topo['mlp_prop'], topo_mlp),
                        ("res", topo['res_prop'], topo_res)
                    ]:
                        interp = safe_interp(COMMON_SIZE, sorted_size, src[align], stats, f, name)
                        if interp is not None:
                            dest.append(interp)

                    for g in range(4):
                        interp = safe_interp(COMMON_SIZE, sorted_size, topo['layer_group_props'][g][align], stats, f, f"layer_{g}")
                        if interp is not None:
                            topo_layers[g].append(interp)

                # --- Detailed Component Activation ---
                if args.components:
                    comp_names = get_component_names(data)
                    stratum_assignments = data['stratum']
                    unique_f_comps = np.unique(comp_names)

                    for c_name in unique_f_comps:
                        comp_mask = (comp_names == c_name)
                        min_assign = np.min(stratum_assignments[comp_mask])

                        s_idx = np.searchsorted(data['strata_index'][sort_idx], min_assign)
                        if s_idx >= len(sorted_size):
                            continue

                        entry_size = sorted_size[s_idx]
                        interp_presence = (COMMON_SIZE >= entry_size).astype(int)

                        if c_name not in model_comp_agg:
                            model_comp_agg[c_name] = interp_presence.copy()
                        else:
                            model_comp_agg[c_name] += interp_presence

            except Exception as e:
                error_count += 1
                print(f"  [ERROR] File {f}: {e}")
                traceback.print_exc()

        # --- Aggregate Model Results ---
        data_agg[model_name] = {
            "size": COMMON_SIZE,
            "tv_trace": safe_mean(model_agg_trace_tv, stats),
            "tv_inv": safe_mean(model_agg_inv_tv, stats),
            "tv_random": safe_mean(model_agg_random_tv, stats)
        }

        data_nu[model_name] = nucleus_sizes

        sentence_data.sort(key=lambda x: x['entropy'])
        data_sentences[model_name] = {
            "low": sentence_data[:5],
            "high": sentence_data[-5:]
        }

        data_topology[model_name] = {
            "size": COMMON_SIZE,
            "attn_prop": safe_mean(topo_attn, stats),
            "mlp_prop": safe_mean(topo_mlp, stats),
            "res_prop": safe_mean(topo_res, stats),
            "layer_props": [safe_mean(topo_layers[g], stats) for g in range(4)]
        }

        if args.components:
            data_components[model_name] = {
                "size": COMMON_SIZE,
                "comp_counts": model_comp_agg
            }

        reports[model_name] = {
            "errors": error_count,
            "files": len(npz_files),
            "nan_files": len(stats['files_with_nan']),
            "nan_counts": dict(stats["nan_counts"]),
            "inf_counts": dict(stats["inf_counts"])
        }

        print(f"  [SUMMARY] {model_name}: Processed {len(npz_files)} files, Errors: {error_count}")

    # =============================================================================
    # 4. Save Processed Pickles
    # =============================================================================
    print("\n[SAVING] Exporting aggregated data pickles...")
    for tag, d in [
        ("agg", data_agg),
        ("sentences", data_sentences),
        ("topology", data_topology),
        ("components", data_components if args.components else None)
    ]:
        if d is not None:
            save_path = f"{SAVE_FOLDER}processed_data_{tag}_{args.length}_{args.max_files}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(d if tag != "agg" else {"aggregate": d, "nucleus": data_nu}, f)
            print(f"  Saved -> {save_path}")

    report_path = f"{SAVE_FOLDER}robustness_report_{args.length}_{args.max_files}.pkl"
    with open(report_path, 'wb') as f:
        pickle.dump(reports, f)

    # =============================================================================
    # 5. Generate Statistical Correlation Report
    # =============================================================================
    corr_report_file = f"{SAVE_FOLDER}correlations_report_{args.length}__{args.max_files}.txt"
    print(f"\n[CORRELATIONS] Writing Spearman rank correlation analysis to {corr_report_file}...")

    with open(corr_report_file, 'w') as f_out:
        f_out.write(f"=== EMNLP Correlation Analysis Report (Length: {args.length}) ===\n\n")

        # 1. Full Prediction Entropy vs TV AUC (Per Model)
        f_out.write("1. Rank Correlation: Full Prediction Entropy vs Computation Density (AUC-TV)\n")
        f_out.write("-" * 75 + "\n")
        for model in sorted(model_sentence_auc.keys()):
            files = list(model_sentence_auc[model].keys())
            if not files:
                continue

            auc_list = [model_sentence_auc[model][fl] for fl in files]
            ent_list = [model_sentence_entropy[model][fl] for fl in files]

            rho, pval = spearmanr(ent_list, auc_list)
            f_out.write(f"{model:<35} | Rho: {rho:+.4f} | p-value: {pval:.4e} | N: {len(files)}\n")

        f_out.write("\n\n")

        # 2. Pairwise Model TV AUC Correlation across Shared Sentences
        f_out.write("2. Pairwise Model Rank Correlation: AUC-TV across identical sentences\n")
        f_out.write("-" * 75 + "\n")

        models_list = sorted(list(model_sentence_auc.keys()))
        auc_correlations = []

        for m1, m2 in itertools.combinations(models_list, 2):
            common_files = sorted(list(set(model_sentence_auc[m1].keys()) & set(model_sentence_auc[m2].keys())))
            if len(common_files) < 2:
                continue

            auc_m1 = [model_sentence_auc[m1][fl] for fl in common_files]
            auc_m2 = [model_sentence_auc[m2][fl] for fl in common_files]

            rho, pval = spearmanr(auc_m1, auc_m2)
            auc_correlations.append(rho)
            f_out.write(f"{m1} vs {m2}\n")
            f_out.write(f"   -> Rho: {rho:+.4f} | p-value: {pval:.4e} | Common Sentences: {len(common_files)}\n\n")

        if auc_correlations:
            f_out.write("-" * 75 + "\n")
            f_out.write("PAIRWISE AUC CORRELATION SUMMARY\n")
            f_out.write(f"Average Rho: {np.mean(auc_correlations):+.4f}\n")
            f_out.write(f"Min Rho:     {np.min(auc_correlations):+.4f}\n")
            f_out.write(f"Max Rho:     {np.max(auc_correlations):+.4f}\n")
            f_out.write(f"Std Dev:     {np.std(auc_correlations):.4f}\n")

    print(f"[COMPLETE] Processing finished successfully.\n")


if __name__ == "__main__":
    main()