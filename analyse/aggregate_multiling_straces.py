import os
import glob
import numpy as np
import pickle
import argparse
import traceback
from tqdm import tqdm
import warnings
from scipy.stats import spearmanr
import itertools
from collections import defaultdict

warnings.filterwarnings("default", category=RuntimeWarning)

SAVE_FOLDER = "./"
COMMON_SIZE = np.logspace(-5, 0, 150)
LANGS = ['ar', 'es', 'ru']
BASE_DIR_TEMPLATE = "/homes/users/ckervadec/scratch/llm-strace/results_L1-norm_wiki-{lang}"

# =========================
# Safety Helpers
# =========================

def check_array(name, arr, file, stats):
    arr = np.asarray(arr)
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()

    if n_nan > 0 or n_inf > 0:
        stats["nan_counts"][name] += int(n_nan)
        stats["inf_counts"][name] += int(n_inf)
        stats["files_with_nan"].add(file)

    return arr

def safe_interp(x, xp, fp, stats, file, name):
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
    if len(arr_list) == 0:
        stats["empty_means"] += 1
        return np.full_like(COMMON_SIZE, np.nan)
    return np.mean(arr_list, axis=0)

# =========================
# Main Pipeline
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=str, required=True, help="Sequence length identifier (e.g., 40)")
    parser.add_argument('--max_files', type=int, default=500)
    args = parser.parse_args()

    # Dictionary to store final agg data: data_agg[model_name][lang]
    data_agg = defaultdict(dict)
    # Store AUCs for correlation: lang_sentence_auc[model_name][lang][file_name]
    lang_sentence_auc = defaultdict(lambda: defaultdict(dict))
    reports = defaultdict(dict)

    # 1. Discover all unique model directories across language results
    all_model_names = set()
    for lang in LANGS:
        lang_dir = BASE_DIR_TEMPLATE.format(lang=lang)
        if os.path.exists(lang_dir):
            dirs = [d for d in os.listdir(lang_dir) if os.path.isdir(os.path.join(lang_dir, d, 'main'))]
            all_model_names.update(dirs)

    if not all_model_names:
        print(f"Error: No model directories found across language folders.")
        return

    print(f"Discovered {len(all_model_names)} models across languages: {sorted(list(all_model_names))}")

    # 2. Iterate over models and languages
    for model_name in sorted(all_model_names):
        print(f"\n--- Processing Model: {model_name} ---")

        for lang in LANGS:
            lang_dir = BASE_DIR_TEMPLATE.format(lang=lang)
            target_dir = os.path.join(lang_dir, model_name, 'main', f'final_straces_{args.length}')
            
            stats = defaultdict(int)
            stats.update({
                "nan_counts": defaultdict(int),
                "inf_counts": defaultdict(int),
                "files_with_nan": set()
            })
            
            error_count = 0
            model_agg_trace_tv = []
            
            if not os.path.exists(target_dir):
                print(f"[WARNING] Directory missing for {model_name} ({lang}): {target_dir}")
                continue
                
            npz_files = glob.glob(os.path.join(target_dir, '*.npz'))[:args.max_files]
            if not npz_files:
                continue

            for f in tqdm(npz_files, desc=f"Language: {lang}"):
                try:
                    data = np.load(f, allow_pickle=True)

                    rel_size = check_array("rel_size", data['strata_rel_size'], f, stats)
                    sort_idx = np.argsort(rel_size)
                    sorted_size = rel_size[sort_idx]

                    # Extract Total Variation (TV) array
                    tv_trace = check_array("tv_trace", np.array(data['strata_reco_tv'].item()['trace']['only'])[sort_idx], f, stats)
                    
                    interp_trace = safe_interp(COMMON_SIZE, sorted_size, tv_trace, stats, f, "tv_trace")
                    if interp_trace is None:
                        continue

                    model_agg_trace_tv.append(interp_trace)

                    # -------- SENTENCE METRICS (AUC) --------
                    auc_val = np.trapz(tv_trace, np.log10(np.clip(sorted_size, 1e-8, 1.0)))
                    fname = os.path.basename(f)
                    lang_sentence_auc[model_name][lang][fname] = auc_val

                except Exception as e:
                    error_count += 1
                    print(f"[ERROR] {f}: {e}")

            # Aggregate per language
            if model_agg_trace_tv:
                data_agg[model_name][lang] = {
                    "size": COMMON_SIZE,
                    "tv_trace": safe_mean(model_agg_trace_tv, stats),
                }

            reports[model_name][lang] = {
                "errors": error_count,
                "files": len(npz_files),
                "nan_files": len(stats['files_with_nan'])
            }

    # -------- SAVE AGGREGATED DATA --------
    out_pkl = f"{SAVE_FOLDER}processed_data_agg_multiling_{args.length}_{args.max_files}.pkl"
    with open(out_pkl, 'wb') as f:
        pickle.dump({"aggregate": dict(data_agg)}, f)
    print(f"\n[SUMMARY] Saved aggregated multilingual data to {out_pkl}")

    # =========================================================================
    # CORRELATION REPORT GENERATION (Per Model, Between Languages)
    # =========================================================================
    report_file = f"{SAVE_FOLDER}correlations_report_multiling_{args.length}_{args.max_files}.txt"
    with open(report_file, 'w') as f_out:
        f_out.write(f"=== Multilingual Extraction Correlation Report (Length: {args.length}) ===\n\n")
        
        for model_name in sorted(data_agg.keys()):
            f_out.write(f"Model: {model_name}\n")
            f_out.write("-" * 70 + "\n")
            
            available_langs = list(lang_sentence_auc[model_name].keys())
            
            for l1, l2 in itertools.combinations(available_langs, 2):
                # Intersect to make sure we only compare on the exact same sentence IDs
                files_l1 = set(lang_sentence_auc[model_name][l1].keys())
                files_l2 = set(lang_sentence_auc[model_name][l2].keys())
                common_files = sorted(list(files_l1 & files_l2))
                
                if len(common_files) < 2:
                    continue
                    
                auc_l1 = [lang_sentence_auc[model_name][l1][fl] for fl in common_files]
                auc_l2 = [lang_sentence_auc[model_name][l2][fl] for fl in common_files]
                
                rho, pval = spearmanr(auc_l1, auc_l2)
                f_out.write(f"{l1:>5} vs {l2:<5} | Rho: {rho:+.4f} | p-value: {pval:.4e} | N: {len(common_files)}\n")
            
            f_out.write("\n")

    print(f"Success! Processed data saved, and correlations written to {report_file}")

if __name__ == "__main__":
    main()