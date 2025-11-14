import argparse
import csv
from collections import Counter
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Iterator, Dict, Optional, List
import sys

def load_all_sentences(data_file: str) -> Iterator[str]:
    """
    Efficiently loads all 'input' sentences from the .tsv data file,
    one by one.
    """
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    parts = line.strip().split('\t')
                    # Expects format: {sentence}\t{...}
                    if len(parts) >= 1:
                        # We only need the first column (the sentence) for token counting
                        yield parts[0]
                    else:
                        print(f"[load_all_sentences] Warning: Skipping empty or malformed line {i+1}.")
                except Exception as e:
                    print(f"[load_all_sentences] Error processing line {i+1}: {e}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error opening data file {data_file}: {e}", file=sys.stderr)
        sys.exit(1)

def compute_statistics(
    tokenizer: PreTrainedTokenizer,
    sentence_iterator: Iterator[str]
) -> Counter:
    """
    Tokenizes all sentences and computes token frequency.
    """
    token_counts = Counter()
    
    for sentence in sentence_iterator:
        if not sentence:
            continue
        
        # tokenize() splits a string into a list of token strings
        try:
            tokens = tokenizer.tokenize(sentence)
            token_counts.update(tokens)
        except Exception as e:
            print(f"Error tokenizing sentence: '{sentence}'. Error: {e}")
            
    return token_counts

def save_statistics(
    token_counts: Counter,
    output_file: str
) -> None:
    """
    Saves the sorted token counts to a .tsv file.
    """
    if not token_counts:
        print("Warning: No tokens were counted. Output file will be empty.")
        # We can still create an empty file with headers
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            
            # Write header
            writer.writerow(['token', 'count'])
            
            # Write data, sorted from most common to least common
            for token, count in token_counts.most_common():
                if token[0] == 'Ġ': # ugly fix
                    token = token.replace('Ġ',' ')
                if token[0] == '_': # ugly fix
                    token = token.replace('_',' ')
                writer.writerow([token, count])
                
    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function to parse arguments, load data, run statistics,
    and save results.
    """
    parser = argparse.ArgumentParser(
        description="Compute token frequency statistics from a sentence dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the Hugging Face model to load the tokenizer (e.g., 'bert-base-uncased')."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the input .tsv data file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output .tsv statistics file."
    )
    
    args = parser.parse_args()
    
    # --- 1. Load Tokenizer ---
    print(f"Loading tokenizer for '{args.model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error: Could not load tokenizer '{args.model_name}'. {e}", file=sys.stderr)
        sys.exit(1)
    
    # --- 2. Load Sentences ---
    print(f"Loading sentences from '{args.data_file}'...")
    sentence_iterator = load_all_sentences(args.data_file)
    
    # --- 3. Compute Statistics ---
    print("Computing token statistics... (This may take a while)")
    token_counts = compute_statistics(tokenizer, sentence_iterator)
    
    if not token_counts:
        print("Warning: No tokens were found or processed.")
    else:
        print(f"Found {len(token_counts)} unique tokens.")
        
    # --- 4. Save Statistics ---
    print(f"Saving statistics to '{args.output_file}'...")
    save_statistics(token_counts, args.output_file)
    
    print("Done.")

if __name__ == "__main__":
    main()