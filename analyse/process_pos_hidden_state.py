import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


def compute_hidden_trajectory(h_states):
    """
    Computes consecutive step distances and distances to the full trace size
    for hidden state representations.
    """
    h = np.nan_to_num(np.asarray(h_states, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(h, axis=1)

    # 1. Consecutive Step-by-Step Trajectory (Length: num_strata - 1)
    h_t = h[:-1]
    h_tp1 = h[1:]

    norm_t = norms[:-1]
    norm_tp1 = norms[1:]
    norm_prod = np.where(norm_t * norm_tp1 == 0, 1e-12, norm_t * norm_tp1)

    dot_consec = np.sum(h_t * h_tp1, axis=1)
    consec_cosine = np.clip(1.0 - (dot_consec / norm_prod), 0.0, 2.0)
    consec_l2 = np.linalg.norm(h_tp1 - h_t, axis=1)

    # 2. Convergence to Full Trace (Length: num_strata)
    h_full = h[-1:]
    norm_full = norms[-1:]
    norm_prod_full = np.where(norms * norm_full == 0, 1e-12, norms * norm_full)

    dot_full = np.sum(h * h_full, axis=1)
    full_trace_cosine = np.clip(1.0 - (dot_full / norm_prod_full), 0.0, 2.0)
    full_trace_l2 = np.linalg.norm(h - h_full, axis=1)

    return {
        'hidden_norm': norms,
        'consec_cosine': consec_cosine,
        'consec_l2': consec_l2,
        'full_trace_cosine': full_trace_cosine,
        'full_trace_l2': full_trace_l2,
    }


def extract_strata_metric(s_data, key, cfg, pred_keys, configs, num_strata, full_trace_pad_val=0.0):
    """
    Extracts a 1D per-stratum metric trajectory (e.g. 'strata_reco_tv' or 'strata_reco_nu')
    for a given configuration from strace_final_{sid}.npz.
    """
    if key not in s_data:
        return None

    raw = s_data[key]
    arr = None

    if raw.dtype == object:
        obj = raw.item()
        if isinstance(obj, dict):
            if cfg in obj:
                arr = obj[cfg]
            else:
                pk1, pk2 = pred_keys[cfg]
                if pk1 in obj and isinstance(obj[pk1], dict) and pk2 in obj[pk1]:
                    arr = obj[pk1][pk2]
    elif isinstance(raw, np.ndarray):
        if raw.ndim == 1 and cfg == configs[0]:
            arr = raw
        elif raw.ndim == 2:
            cfg_idx = configs.index(cfg)
            if cfg_idx < raw.shape[0]:
                arr = raw[cfg_idx]

    if arr is None:
        return None

    arr = np.asarray(arr, dtype=np.float64).ravel()

    if len(arr) == num_strata - 1:
        arr = np.append(arr, full_trace_pad_val)
    elif len(arr) != num_strata:
        return None

    return arr


def compute_knn_pos_accuracy(X, labels, k=10, metric='cosine'):
    """
    Computes Leave-One-Out k-NN classification accuracy for POS labels.
    """
    n = len(labels)
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return np.nan

    if metric == 'cosine':
        norms = np.linalg.norm(X, axis=1)
        if np.any(norms == 0):
            X = X.copy()
            X[norms == 0] = 1e-12

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)[:, 1:]  # exclude self at index 0

    neighbor_labels = labels[indices]
    correct = 0
    for i in range(n):
        vals, counts = np.unique(neighbor_labels[i], return_counts=True)
        pred = vals[np.argmax(counts)]
        if pred == labels[i]:
            correct += 1

    return correct / n


def compute_per_stratum_embeddings(stratum_store, configs, max_samples=3000, perplexity=30.0, random_state=42):
    """
    Runs per-stratum dimensionality reduction (t-SNE and PCA) and computes
    POS discriminability metrics (k-NN accuracy and Silhouette score) at each stratum.
    """
    embedding_records = []
    separability_records = []
    rng = np.random.default_rng(random_state)

    for cfg in configs:
        strata_indices = sorted(stratum_store[cfg].keys())
        if not strata_indices:
            continue

        # Sample a consistent set of Text_IDs across all strata for this config
        # so the exact same sentences are tracked across strata
        all_sids = sorted({item[0] for t in strata_indices for item in stratum_store[cfg][t]})
        if len(all_sids) > max_samples:
            chosen_sids = set(rng.choice(all_sids, size=max_samples, replace=False).tolist())
        else:
            chosen_sids = set(all_sids)

        print(f"\nComputing per-stratum embeddings & POS separability for '{cfg}' "
              f"({len(chosen_sids)} sentences across {len(strata_indices)} strata)...")

        for t in tqdm(strata_indices, desc=f"Embeddings ({cfg})"):
            items = [it for it in stratum_store[cfg][t] if it[0] in chosen_sids]
            if len(items) < 5:
                continue

            sids = [it[0] for it in items]
            pos_labels = np.array([it[1] for it in items])
            rel_sizes = np.array([it[2] for it in items], dtype=np.float64)
            X = np.nan_to_num(np.vstack([it[3] for it in items]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

            n_samples = X.shape[0]
            mean_rel_size = float(np.mean(rel_sizes))

            # 1. PCA (2D)
            if X.shape[1] >= 2 and np.std(X) > 0:
                pca = PCA(n_components=2, random_state=random_state)
                coords_pca = pca.fit_transform(X)
            else:
                coords_pca = np.zeros((n_samples, 2), dtype=np.float64)

            # 2. t-SNE (2D)
            eff_perplexity = min(float(perplexity), max(5.0, (n_samples - 1) / 3.0))
            if eff_perplexity >= n_samples:
                eff_perplexity = max(1.0, (n_samples - 1) / 2.0)

            try:
                tsne = TSNE(
                    n_components=2,
                    perplexity=eff_perplexity,
                    init='pca' if np.std(X) > 0 else 'random',
                    learning_rate='auto',
                    random_state=random_state
                )
                coords_tsne = tsne.fit_transform(X)
            except Exception:
                coords_tsne = coords_pca.copy()

            # 3. Quantitative POS Discriminability Metrics
            unique_pos, pos_counts = np.unique(pos_labels, return_counts=True)
            n_pos_classes = len(unique_pos)
            majority_baseline = float(np.max(pos_counts) / n_samples)

            knn_acc_hd = compute_knn_pos_accuracy(X, pos_labels, k=10, metric='cosine')
            knn_acc_2d = compute_knn_pos_accuracy(coords_tsne, pos_labels, k=10, metric='euclidean')

            if 2 <= n_pos_classes < n_samples and np.std(X) > 0:
                try:
                    sil_hd = float(silhouette_score(X, pos_labels, metric='cosine'))
                except Exception:
                    sil_hd = np.nan
                try:
                    sil_2d = float(silhouette_score(coords_tsne, pos_labels, metric='euclidean'))
                except Exception:
                    sil_2d = np.nan
            else:
                sil_hd = np.nan
                sil_2d = np.nan

            separability_records.append({
                'config': cfg,
                'stratum': t,
                'mean_rel_size': mean_rel_size,
                'n_samples': n_samples,
                'n_pos_classes': n_pos_classes,
                'majority_baseline_acc': majority_baseline,
                'knn_pos_acc_highdim': knn_acc_hd,
                'knn_pos_acc_2d': knn_acc_2d,
                'silhouette_highdim': sil_hd,
                'silhouette_2d': sil_2d,
            })

            for i in range(n_samples):
                embedding_records.append({
                    'config': cfg,
                    'stratum': t,
                    'mean_rel_size': mean_rel_size,
                    'rel_size': float(rel_sizes[i]),
                    'Text_ID': sids[i],
                    'POS': pos_labels[i],
                    'tsne_1': float(coords_tsne[i, 0]),
                    'tsne_2': float(coords_tsne[i, 1]),
                    'pca_1': float(coords_pca[i, 0]),
                    'pca_2': float(coords_pca[i, 1]),
                })

    return pd.DataFrame(embedding_records), pd.DataFrame(separability_records)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-POS trajectory metrics, Total Variation, Nucleus Reconstruction, and per-stratum 2D embeddings (t-SNE & PCA)."
    )
    parser.add_argument('--hidden_dir', type=str, required=True,
                        help="Directory containing hidden_strata_X.npz files")
    parser.add_argument('--strace_dir', type=str, required=True,
                        help="Directory containing strace_final_X.npz files")
    parser.add_argument('--pos_csv', type=str, required=True,
                        help="Path to metadata CSV containing Text_ID and POS columns")
    parser.add_argument('--pos_col', type=str, default="POS",
                        help="Column name in metadata CSV for POS tag (e.g., 'POS' or 'last_POS')")
    parser.add_argument('--output_dir', type=str, default="processed_pos_data",
                        help="Output directory for processed metrics and embeddings")
    parser.add_argument('--min_pos_count', type=int, default=5,
                        help="Minimum unique sentence count to retain a POS category")
    parser.add_argument('--max_tsne_samples', type=int, default=3000,
                        help="Max sentences per config to embed across strata")
    parser.add_argument('--tsne_perplexity', type=float, default=30.0,
                        help="Perplexity parameter for per-stratum t-SNE")
    parser.add_argument('--random_state', type=int, default=42,
                        help="Random seed for subsampling, PCA, and t-SNE")
    parser.add_argument('--skip_embeddings', action='store_true',
                        help="Skip per-stratum t-SNE/PCA embedding computation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading metadata from {args.pos_csv} (using POS column '{args.pos_col}')...")
    df_meta = pd.read_csv(args.pos_csv)

    if 'Text_ID' not in df_meta.columns or args.pos_col not in df_meta.columns:
        raise ValueError(f"The provided CSV must contain 'Text_ID' and '{args.pos_col}' columns.")

    df_meta_clean = df_meta[['Text_ID', args.pos_col]].dropna().copy()
    df_meta_clean['Text_ID'] = df_meta_clean['Text_ID'].astype(int)
    df_meta_clean[args.pos_col] = df_meta_clean[args.pos_col].astype(str)
    df_meta_clean = df_meta_clean[df_meta_clean[args.pos_col].str.lower() != 'nan']

    pos_map = dict(zip(df_meta_clean['Text_ID'], df_meta_clean[args.pos_col]))

    strace_files = glob.glob(os.path.join(args.strace_dir, 'strace_final_*.npz'))
    sentence_ids = sorted([int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in strace_files])

    configs = ['trace_only', 'random_only', 'trace_inverse']
    pred_keys = {
        'trace_only': ('trace', 'only'),
        'random_only': ('random', 'only'),
        'trace_inverse': ('trace', 'inverse')
    }

    records = []
    stratum_hidden_store = {cfg: {} for cfg in configs}

    print("Extracting hidden representations, 'strata_reco_tv', and 'strata_reco_nu' trajectories...")
    for sid in tqdm(sentence_ids, desc="Processing Sentences"):
        if sid not in pos_map:
            continue

        pos_tag = pos_map[sid]
        s_path = os.path.join(args.strace_dir, f'strace_final_{sid}.npz')
        h_path = os.path.join(args.hidden_dir, f'hidden_strata_{sid}.npz')

        if not os.path.exists(s_path):
            continue

        has_hidden_file = os.path.exists(h_path)

        try:
            s_data = np.load(s_path, allow_pickle=True)
            h_data = np.load(h_path) if has_hidden_file else None

            rel_sizes = np.array(s_data['strata_rel_size'], dtype=np.float64)
            num_strata = len(rel_sizes)

            for cfg in configs:
                has_h = (h_data is not None) and (cfg in h_data) and (h_data[cfg].shape[0] == num_strata)
                tv_arr = extract_strata_metric(
                    s_data, 'strata_reco_tv', cfg, pred_keys, configs, num_strata, full_trace_pad_val=0.0
                )
                nu_arr = extract_strata_metric(
                    s_data, 'strata_reco_nu', cfg, pred_keys, configs, num_strata, full_trace_pad_val=100.0
                )

                has_tv = tv_arr is not None
                has_nu = nu_arr is not None

                if not (has_h or has_tv or has_nu):
                    continue

                if has_h:
                    h_mat = np.nan_to_num(np.asarray(h_data[cfg], dtype=np.float64))
                    h_traj = compute_hidden_trajectory(h_mat)
                    if not args.skip_embeddings:
                        for t in range(num_strata):
                            stratum_hidden_store[cfg].setdefault(t, []).append(
                                (sid, pos_tag, rel_sizes[t], h_mat[t])
                            )
                else:
                    h_traj = None

                tv_step_diff = np.abs(np.diff(tv_arr)) if has_tv else None
                nu_step_diff = np.diff(nu_arr) if has_nu else None

                # 1. Consecutive Step Metrics (step t -> t+1)
                for t in range(num_strata - 1):
                    records.append({
                        'Text_ID': sid,
                        'POS': pos_tag,
                        'config': cfg,
                        'metric_type': 'consecutive',
                        'step_or_stratum': t,
                        'rel_size_start': rel_sizes[t],
                        'rel_size_end': rel_sizes[t + 1],
                        'delta_rel_size': rel_sizes[t + 1] - rel_sizes[t],
                        'hidden_cosine_dist': h_traj['consec_cosine'][t] if has_h else np.nan,
                        'hidden_l2_dist': h_traj['consec_l2'][t] if has_h else np.nan,
                        'hidden_norm': h_traj['hidden_norm'][t] if has_h else np.nan,
                        'reco_tv_dist': tv_step_diff[t] if has_tv else np.nan,
                        'reco_nu_score': nu_step_diff[t] if has_nu else np.nan,
                    })

                # 2. Full-Trace Convergence Metrics (stratum t -> t_max)
                for t in range(num_strata):
                    records.append({
                        'Text_ID': sid,
                        'POS': pos_tag,
                        'config': cfg,
                        'metric_type': 'full_trace_convergence',
                        'step_or_stratum': t,
                        'rel_size_start': rel_sizes[t],
                        'rel_size_end': rel_sizes[-1],
                        'delta_rel_size': rel_sizes[-1] - rel_sizes[t],
                        'hidden_cosine_dist': h_traj['full_trace_cosine'][t] if has_h else np.nan,
                        'hidden_l2_dist': h_traj['full_trace_l2'][t] if has_h else np.nan,
                        'hidden_norm': h_traj['hidden_norm'][t] if has_h else np.nan,
                        'reco_tv_dist': tv_arr[t] if has_tv else np.nan,
                        'reco_nu_score': nu_arr[t] if has_nu else np.nan,
                    })

        except Exception as e:
            print(f"Skipping sentence {sid} due to error: {e}")

    df_results = pd.DataFrame(records)

    if df_results.empty:
        print("No valid matches found between NPZ files and CSV metadata.")
        return

    # Filter out rare POS tags with fewer than min_pos_count unique sentences
    pos_counts = df_results.groupby('POS')['Text_ID'].nunique()
    valid_pos = set(pos_counts[pos_counts >= args.min_pos_count].index)
    df_filtered = df_results[df_results['POS'].isin(valid_pos)].copy()

    # Save raw granular trajectory records
    raw_csv_path = os.path.join(args.output_dir, 'per_pos_trajectory_granular.csv')
    df_filtered.to_csv(raw_csv_path, index=False)
    print(f"Saved granular trajectory metrics to '{raw_csv_path}'.")

    # Aggregate by Config, Metric Type, POS, and Stratum/Step
    print("Aggregating hidden representation, TV distance, and Nucleus Reconstruction metrics per POS tag...")
    summary_df = df_filtered.groupby(
        ['config', 'metric_type', 'POS', 'step_or_stratum']
    ).agg(
        n_samples=('Text_ID', 'nunique'),
        mean_rel_size_start=('rel_size_start', 'mean'),
        mean_hidden_cosine_dist=('hidden_cosine_dist', 'mean'),
        std_hidden_cosine_dist=('hidden_cosine_dist', 'std'),
        mean_hidden_l2_dist=('hidden_l2_dist', 'mean'),
        std_hidden_l2_dist=('hidden_l2_dist', 'std'),
        mean_hidden_norm=('hidden_norm', 'mean'),
        mean_reco_tv_dist=('reco_tv_dist', 'mean'),
        std_reco_tv_dist=('reco_tv_dist', 'std'),
        mean_reco_nu_score=('reco_nu_score', 'mean'),
        std_reco_nu_score=('reco_nu_score', 'std'),
    ).reset_index()

    summary_csv_path = os.path.join(args.output_dir, 'per_pos_trajectory_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved aggregated per-POS summary table to '{summary_csv_path}'.")

    # Compute per-stratum 2D embeddings (t-SNE & PCA) and POS discriminability metrics
    if not args.skip_embeddings:
        filtered_store = {
            cfg: {
                t: [it for it in items if it[1] in valid_pos]
                for t, items in strata_dict.items()
            }
            for cfg, strata_dict in stratum_hidden_store.items()
        }
        df_emb, df_sep = compute_per_stratum_embeddings(
            filtered_store,
            configs=configs,
            max_samples=args.max_tsne_samples,
            perplexity=args.tsne_perplexity,
            random_state=args.random_state
        )

        if not df_emb.empty:
            emb_path = os.path.join(args.output_dir, 'per_stratum_embeddings.tsv')
            df_emb.to_csv(emb_path, sep='\t', index=False)
            print(f"Saved per-stratum 2D embeddings (t-SNE & PCA) to '{emb_path}'.")

        if not df_sep.empty:
            sep_path = os.path.join(args.output_dir, 'per_stratum_pos_separability.csv')
            df_sep.to_csv(sep_path, index=False)
            print(f"Saved per-stratum POS discriminability metrics to '{sep_path}'.")


if __name__ == "__main__":
    main()