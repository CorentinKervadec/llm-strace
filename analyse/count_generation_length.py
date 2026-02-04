import argparse
import sys
import re
from transformers import AutoTokenizer
import pandas as pd
import re
import numpy as np

def decode_summary(df):
    """
    Extracts the 30-step generation data for each Prompt/Seed combination.
    """
    # --- 1. Initialize Container for Extracted Data ---
    # Dictionary structure: { (prompt_id, seed): [step_0_data, step_1_data, ... step_29_data] }
    all_sequences = {}

    # --- 2. Process Data ---
    if 'seed' in df.columns:
        grouper = ['prompt_name', 'seed']
    else:
        grouper = ['prompt_name']

    grouped = df.groupby(grouper)

    for name, group in grouped:
        # Resolve names and seeds
        if isinstance(name, tuple):
            prompt_id_str = str(name[0])
            seed_str = str(name[1])
        else:
            prompt_id_str = str(name)
            seed_str = "None" 

        # --- Prepare List for this Sequence (Indices 0 to 29) ---
        sequence_list = [None] * 30 

        # Filter for relevant steps
        group = group.sort_values('step')
        group = group[(group['step'] >= 0) & (group['step'] <= 29)]

        flag_is_valid = True
        for _, row in group.iterrows():
            step_idx = int(row['step'])
            
            try:
                token_id = int(row['next_token_id'])
                auc_val = row.get('auc_tv')
                ent_val = row.get('gen_entropy')
                prob_val = row.get('next_token_prob')
            except ValueError:
                flag_is_valid = False
                break
            
            # Pack the data into a dictionary
            step_data = {
                'token_id': token_id,
                'auc_val': auc_val,
                'ent_val': ent_val,
                'prob_val': prob_val
            }
            
            # Assign to the specific list index matching the step
            if 0 <= step_idx < 30:
                sequence_list[step_idx] = step_data

        # Store the completed list for this prompt/seed
        if flag_is_valid:
            key = (prompt_id_str, seed_str)
            all_sequences[key] = sequence_list

    return all_sequences

def get_last_sentence_end(text):
    """
    Finds the character index of the end of the last complete sentence.
    Handles:
    - Standard punctuation (., ?, !)
    - Spaces before punctuation (e.g., "word .")
    - Common abbreviations (Mr., Dr., etc.) that do not end a sentence.
    """
    # Common abbreviations that end with a dot but aren't sentence ends.
    # Add more here if needed.
    abbreviations = {'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'St', 'etc'}
    
    # Regex explanation:
    # \s* : Matches optional whitespace before punctuation
    # [.!?]     : Matches the punctuation marks
    # (?=\s|$|['"\n\t])  : Positive lookahead - ensures punct is followed by whitespace, quotes, newline, tab, or end of string
    pattern = re.compile(r'\s*[.!?](?=\s|$|[\'"\n\t])')
    
    last_valid_end = -1
    
    # Iterate through all punctuation matches in the text
    for match in pattern.finditer(text):
        end_idx = match.end()
        
        # Check the word preceding this punctuation
        # We take the text up to the match start, strip trailing space, and split to get the last word
        preceding_text = text[:match.start()].rstrip()
        if not preceding_text:
            continue
            
        last_word = preceding_text.split()[-1]
        
        # If the preceding word is NOT in our abbreviation list, this is a valid sentence end
        if last_word not in abbreviations:
            last_valid_end = end_idx

    return last_valid_end

def merge_generation_and_prompt(generation_data, df_prompt, tokenizer):
    """
    Merges generation steps with prompt text and truncates to the last full sentence.
    """
    merged_results = {}

    for (prompt_id, seed), sequence_list in generation_data.items():
        prompt_text = "[Prefix not found]"
        
        # --- 1. Retrieve Prompt Text ---
        try:
            if "_" in prompt_id:
                idx = int(prompt_id.split('_')[-1])
            else:
                idx = int(prompt_id)
            
            if 0 <= idx < len(df_prompt):
                prompt_text = df_prompt.iloc[idx]['prompt']
            else:
                prompt_text = f"[Index {idx} out of bounds]"
        except Exception:
            pass

        # --- 2. Extract Generation Tokens ---
        # Filter out None steps (in case some steps were missing in the input file)
        valid_steps = [s for s in sequence_list if s is not None]
        gen_token_ids = [s['token_id'] for s in valid_steps]
        
        # Decode the full generation to text
        full_gen_text = tokenizer.decode(gen_token_ids)
        
        # --- 3. Smart Truncation Logic ---
        cutoff_char_len = get_last_sentence_end(full_gen_text)
        
        truncated_gen_token_ids = []
        truncated_steps_data = []
        
        if cutoff_char_len == -1:
            # No valid sentence end found. 
            # Option A: Keep everything (uncomment below)
            # truncated_gen_token_ids = gen_token_ids
            # truncated_steps_data = valid_steps
            
            # Option B: Return empty string (strict mode) - CURRENT CHOICE
            truncated_gen_text = "" 
        else:
            # We found a valid end index in characters. 
            # Now we must find which tokens correspond to this text length.
            # We decode incrementally to find the cut-off point.
            
            truncated_gen_text = full_gen_text[:cutoff_char_len]
            
            # Incremental check to align tokens with the character count
            for i in range(1, len(gen_token_ids) + 1):
                sub_tokens = gen_token_ids[:i]
                sub_text = tokenizer.decode(sub_tokens)
                
                # Check if this subset of tokens covers our truncated text
                if len(sub_text) >= cutoff_char_len:
                    truncated_gen_token_ids = sub_tokens
                    truncated_steps_data = valid_steps[:i]
                    break
        
        # --- 4. Final Merge (Prompt + Truncated Generation) ---
        prompt_tokens = tokenizer.encode(prompt_text)
        
        # Full (Untruncated)
        full_combined_tokens = prompt_tokens + gen_token_ids
        full_combined_text = tokenizer.decode(full_combined_tokens)
        
        # Truncated
        trunc_combined_tokens = prompt_tokens + truncated_gen_token_ids
        trunc_combined_text = tokenizer.decode(trunc_combined_tokens)

        merged_results[(prompt_id, seed)] = {
            'prompt_id': prompt_id,
            'seed': seed,
            'prompt_text': prompt_text,
            'full_text': full_combined_text,
            'trunc_text': trunc_combined_text,
            'steps_data': valid_steps,
            'trunc_steps_data': truncated_steps_data
        }
        
    return merged_results

def compute_average_lengths(merged_data):
    """
    Computes the average word count for full vs truncated sequences.
    We count words by splitting on whitespace.
    """
    full_lengths = []
    trunc_lengths = []
    
    for key, data in merged_data.items():
        # Total text (Prompt + Gen)
        full_lengths.append(len(data['full_text'].split()))
        trunc_lengths.append(len(data['trunc_text'].split()))
        
    stats = {
        'total_docs': len(merged_data),
        'avg_len_full_total': np.mean(full_lengths) if full_lengths else 0,
        'avg_len_trunc_total': np.mean(trunc_lengths) if trunc_lengths else 0,
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Input TSV (Generation data)")
    parser.add_argument('--prompt_file', type=str, required=True, help="Input CSV (Prompt text)")
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    try:
        # 1. Load Files
        df_gen = pd.read_csv(args.data, sep='\t')
        df_prompt = pd.read_csv(args.prompt_file)
        
        # 2. Extract Generation Data (Steps 0-29)
        print("Decoding summary...")
        extracted_data = decode_summary(df_gen)
        
        # 3. Merge with Prompt Text
        print("Merging with prompt text...")
        final_dataset = merge_generation_and_prompt(extracted_data, df_prompt, tokenizer)
        
        print(f"\nProcessing complete. Merged {len(final_dataset)} sequences.")


        # --- Compute Statistics ---
        stats = compute_average_lengths(final_dataset)
        
        print("\n" + "="*40)
        print("      STATISTICS (Word Counts)")
        print("="*40)
        print(f"Number of Sequences processed: {stats['total_docs']}")
        print("-" * 40)
        print(f"Average Length (Prompt + Generation):")
        print(f"  Before Truncation: {stats['avg_len_full_total']:.2f} words")
        print(f"  After Truncation:  {stats['avg_len_trunc_total']:.2f} words")
        print("="*40)


        # --- Example: Verify the first entry ---
        if final_dataset:
            for j in range(3):
                first_key = list(final_dataset.keys())[j]
                entry = final_dataset[first_key]
                
                print(f"\n--- Result for {first_key} ---")
                print(f"Prompt Text: {entry['full_text']}")
                print(f"Trunca Text: {entry['trunc_text']}")
                # Print first 3 steps as a check
                steps = entry['steps_data']
                for i in range(3):
                    if steps[i]:
                        print(f"Step {i}: TokenID={steps[i]['token_id']}, AUC={steps[i]['auc_val']}")
                    else:
                        print(f"Step {i}: [Missing]")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()