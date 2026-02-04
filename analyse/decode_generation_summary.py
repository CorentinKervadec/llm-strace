import argparse
import sys
import html
import re
from transformers import AutoTokenizer
import pandas as pd

def decode_summary(df, tokenizer, prompts_df):
    
    # --- 2. Process Data ---
    if 'seed' in df.columns:
        grouper = ['prompt_name', 'seed']
    else:
        grouper = ['prompt_name']

    grouped = df.groupby(grouper)

    for name, group in grouped:
        if isinstance(name, tuple):
            prompt_id_str = str(name[0])
            seed_str = str(name[1])
            display_name = f"{name[0]} (Seed: {name[1]})"
        else:
            prompt_id_str = str(name)
            display_name = name

        # --- Extract Prompt Prefix ---
        prompt_prefix_text = "[Prefix not found]"
        try:
            if "_" in prompt_id_str:
                idx = int(prompt_id_str.split('_')[-1])
                if 0 <= idx < len(prompts_df):
                    prompt_prefix_text = prompts_df.iloc[idx]['prompt']
                else:
                    prompt_prefix_text = f"[Index {idx} out of bounds]"
            else:
                 prompt_prefix_text = "[Invalid format]"
        except Exception:
            pass

        print(f"[PROMPT={prompt_id_str}; SEED={seed_str}]")
        print(f"* Prompt prefix:", prompt_prefix_text)

        group = group.sort_values('step')
        group = group[(group['step'] >= 0) & (group['step'] <= 29)]

        for _, row in group.iterrows():
            token_id = int(row['next_token_id'])
            auc_val = row.get('auc_tv')
            ent_val = row.get('gen_entropy')
            prob_val = row.get('next_token_prob')
            
            # --- Decode Main Token ---
            decoded_text = tokenizer.convert_ids_to_tokens(token_id)

            bow = decoded_text.startswith('\u2581') or decoded_text.startswith('Ġ') # begining of word
            
            # --- Decode Nucleus ---
            raw_nucleus = str(row.get('nucleus_token_id'))
            decoded_nucleus = []
            if raw_nucleus and raw_nucleus.lower() != 'nan':
                # Use regex to find all numbers regardless of formatting (brackets, commas, etc)
                nuc_ids = re.findall(r'\d+', raw_nucleus)
                
                for aid_str in nuc_ids:
                    try:
                        aid = int(aid_str)
                        alt_dec = tokenizer.convert_ids_to_tokens([aid])
                        
                        decoded_nucleus.append(f"{alt_dec}")
                    except Exception:
                        pass

            print_lines = f"[Step: {row['step']}] Token: {decoded_text}, BOW: {bow}, ID: {token_id}, AUC: {auc_val:.4f}, Entropy: {ent_val:.4f}, Prob: {prob_val:.4f}, Nucleus: [{','.join(decoded_nucleus)}]"
            print(print_lines)
        input('PRESS ANY KEY TO CONTINUE TO THE NEXT PROMPT...')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Input TSV")
    parser.add_argument('--prompt_file', type=str, required=True, help="Input CSV with prompt")
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception:
        sys.exit(1)

    try:
        df = pd.read_csv(args.data, sep='\t')
        prompts_df = pd.read_csv(args.prompt_file)
        decode_summary(df, tokenizer, prompts_df)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()