import transformers
from transformers import AutoTokenizer
import os
import glob
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import traceback
from collections import defaultdict

model2hf_mapping = {
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "OLMo-2-1124-7B": "allenai/OLMo-2-1124-7B",
    "OLMo-2-1124-13B": "allenai/OLMo-2-1124-13B",
    "Qwen3-8B-Base": "Qwen/Qwen3-8B-Base",
    "Qwen3-14B-Base": "Qwen/Qwen3-14B-Base",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen2.5-14B": "Qwen/Qwen2.5-14B",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "deepseek-llm-7b-base": "deepseek-ai/deepseek-llm-7b-base",
    "phi-4": "microsoft/phi-4",
    "Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "Llama-2-13b-hf": "meta-llama/Llama-2-13b-hf"
}


def main():
    parser = argparse.ArgumentParser(description="Process multilingual model data and output a TSV file.")
    parser.add_argument('--base_dir_template', type=str, default="/home/ckervadec/llm-strace/results_L1-norm_wiki-{lang}",
                        help="Template for language directories with {lang} placeholder")
    parser.add_argument('--langs', nargs='+', default=['ar', 'es', 'ru'], help="Languages to process (e.g., ar es ru)")
    parser.add_argument('--length', type=str, required=True, help="Length suffix for the directory (e.g., '40')")
    parser.add_argument('--max_files', type=int, default=10000, help="Maximum files per model per language")
    parser.add_argument('--output_tsv', type=str, default="full_dataset_metrics_multiling.tsv")
    args = parser.parse_args()

    records = []

    # 1. Discover all unique model directories across language folders
    all_model_names = set()
    for lang in args.langs:
        lang_dir = args.base_dir_template.format(lang=lang)
        if os.path.exists(lang_dir):
            dirs = [d for d in os.listdir(lang_dir) if os.path.isdir(os.path.join(lang_dir, d, 'main'))]
            all_model_names.update(dirs)

    if not all_model_names:
        print("Error: No model directories found across specified language folders.")
        return

    print(f"Discovered {len(all_model_names)} models across languages: {sorted(list(all_model_names))}")

    # 2. Iterate over models and languages
    for model_name in sorted(all_model_names):
        if model_name not in model2hf_mapping:
            print(f"[WARNING] Model '{model_name}' not found in model2hf_mapping. Skipping.")
            continue

        hf_model = model2hf_mapping[model_name]
        print(f"\n--- Processing Model: {model_name} ({hf_model}) ---")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_model)
            vocab = tokenizer.get_vocab()
        except Exception as e:
            print(f"[ERROR] Could not load tokenizer for {hf_model}: {e}")
            continue

        for lang in args.langs:
            lang_dir = args.base_dir_template.format(lang=lang)
            target_dir = os.path.join(lang_dir, model_name, 'main', f'final_straces_{args.length}')
            
            if not os.path.exists(target_dir):
                print(f"[WARNING] Directory missing for {model_name} ({lang}): {target_dir}")
                continue

            npz_files = glob.glob(os.path.join(target_dir, '*.npz'))[:args.max_files]
            if not npz_files:
                print(f"No files found in {target_dir}")
                continue

            save_count = 0
            for f in tqdm(npz_files, desc=f"Language: {lang}"):
                try:
                    data = np.load(f, allow_pickle=True)
                    
                    # 1. Filename and Sentence ID
                    fname = os.path.basename(f)
                    sentence_id = int(fname.split('_')[-1].split('.')[0])
                    
                    # 2. AUC Calculation
                    rel_size = np.asarray(data['strata_rel_size'])
                    sort_idx = np.argsort(rel_size)
                    sorted_size = rel_size[sort_idx]
                    
                    tv_trace = np.array(data['strata_reco_tv'].item()['trace']['only'])[sort_idx]
                    # AUC formula: trapz over log-scaled size
                    auc_val = np.trapz(tv_trace, np.log10(np.clip(sorted_size, 1e-8, 1.0)))
                    
                    # 3. Entropy/Loss Calculation
                    loss_trace = data['strata_loss'].item()['trace']['only']
                    full_loss = float(loss_trace[np.argmax(rel_size)])

                    entropy_trace = data['strata_entropy'].item()['trace']['only']
                    full_entropy = float(entropy_trace[np.argmax(rel_size)])
                    
                    # 4. Prediction
                    prediction_trace = data['nucleus_60'].item()['trace']['only']
                    full_prediction = prediction_trace[np.argmax(rel_size)]
                    
                    # Tokenize/map to vocab IDs to prevent formatting issue in TSV
                    bow_special_character = 'Ġ'
                    full_prediction_token = []
                    for p in full_prediction:
                        if p in vocab:
                            full_prediction_token.append(vocab[p])
                        elif p.startswith(' '):
                            new_p = bow_special_character + p[1:]
                            if new_p in vocab:
                                full_prediction_token.append(vocab[new_p])
                            else:
                                print(f"[WARNING] Token not in vocab: {p}")
                    
                    # Check for None, NaN, or corrupted values
                    if (pd.isna(full_entropy) or pd.isna(full_loss) or pd.isna(auc_val)):
                        continue
                    
                    records.append({
                        'language': lang,
                        'sentence id': sentence_id,
                        'model name': hf_model,
                        'entropy': full_entropy,
                        'loss': full_loss,
                        'auc': auc_val,
                        'prediction': full_prediction_token,
                    })

                    save_count += 1
                    
                except Exception as e:
                    print(f"[ERROR] Skipping {f}: {e}")

            print(f"> Successfully processed {save_count} records for {model_name} [{lang}]")

    # Convert to DataFrame and Save as TSV
    if records:
        df = pd.DataFrame(records)
        df.to_csv(args.output_tsv, sep='\t', index=False)
        print(f"\nSuccessfully saved {len(df)} total records to {args.output_tsv}")
    else:
        print("No data was processed.")

if __name__ == "__main__":
    main()