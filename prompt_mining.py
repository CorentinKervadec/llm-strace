import argparse
import io
import tokenize
import keyword
import csv
import re
import random
from collections import defaultdict

import spacy
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
#               CONFIGURATION
# ==========================================

# 1. POS Mapping (Spacy -> Target Categories)
# Maps standard Spacy POS tags to the fine-grained linguistic categories requested.
# 'ignore' indicates tags we do not want to target as the final word of a prompt.
POS_MAP_SPACY = {
    # Target Categories
    'NOUN': 'NOUN', 
    'PROPN': 'PROPN', 
    'PUNCT': 'PUNCT', 
    'VERB': 'VERB',
    'DET': 'DET', 
    'ADJ': 'ADJ', 
    'PRON': 'PRON', 
    'AUX': 'AUX', 
    'ADV': 'ADV',
    
    # Merged Categories
    'ADP': 'ADP+PART',  'PART': 'ADP+PART',
    'CCONJ': 'CCONJ+SCONJ', 'SCONJ': 'CCONJ+SCONJ',
    
    # Ignored Categories
    'NUM': 'ignore', 'X': 'ignore', 'INTJ': 'ignore', 'SYM': 'ignore', 'SPACE': 'ignore'
}

# 2. Domain Configuration
# Defines which HuggingFace datasets to use for each domain.
# 'strict_casing': If True, enforces that sentences must start with an uppercase letter.
#                  Set to False for 'bookcorpus' because it is entirely lowercase.
DOMAIN_CONFIG = {
    'book': {
        'path': 'bookcorpus', 'name': 'plain_text', 'col': 'text', 'split': 'train', 
        'strict_casing': False, 'skip':5000
    },
    'news': {
        'path': 'cnn_dailymail', 'name': '3.0.0', 'col': 'article', 'split': 'validation', 
        'strict_casing': True, 'skip':400
    },
    'wiki': {
        'path': 'wikitext', 'name': 'wikitext-103-v1', 'col': 'text', 'split': 'validation', 
        'strict_casing': True, 'skip':20
    },
    # Uncomment to enable Web/Code domains if needed
    # 'web':  {'path': 'allenai/c4', 'name': 'en', 'col': 'text', 'split': 'validation', 'strict_casing': True},
    # 'code': {'path': 'bigcode/the-stack', 'data_dir': 'data/python', 'col': 'content', 'split': 'train'}
}

# 3. Pronoun Configuration (For Self-Contained Checks)
# Used to detect sentences where a pronoun (He/She) appears without a preceding noun.
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves'
}
DEMONSTRATIVES = {'this', 'that', 'these', 'those'}


# ==========================================
#             HELPER FUNCTIONS
# ==========================================

def load_spacy_model():
    """Safely loads the Spacy English model, disabling heavy components not needed."""
    try:
        # We need the tagger (for POS) and parser (for sentence boundaries).
        # We disable lemmatizer and textcat to speed up processing.
        return spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
    except OSError:
        print("Error: Spacy model not found. Run: python -m spacy download en_core_web_sm")
        exit(1)

def is_self_contained(span):
    """
    Checks if a text span is 'self-contained'.
    
    Heuristic:
    - If a 3rd person pronoun (he/she/it) or demonstrative (this/that) is used as a pronoun,
      we check if a Proper Noun (PROPN) or Noun (NOUN) appeared recently before it.
    - If no noun is found in the preceding context (lookback window), the prompt is rejected.
    """
    for i, token in enumerate(span):
        token_lower = token.text.lower()
        
        # 1. Check 3rd Person Pronouns
        if token_lower in THIRD_PERSON_PRONOUNS:
            found_noun = False
            # Look backwards up to 4 tokens to find a reference
            for j in range(max(0, i - 4), i):
                if span[j].pos_ in ['PROPN']: # Strict: require Proper Noun (e.g., "Alice") for clarity
                    found_noun = True
                    break
            if not found_noun:
                return False
                
        # 2. Check Demonstratives (only if tagged as PRON, e.g., "This is...")
        if token_lower in DEMONSTRATIVES and token.pos_ == 'PRON':
            found_noun = False
            for j in range(max(0, i - 4), i):
                if span[j].pos_ in ['NOUN', 'PROPN']:
                    found_noun = True
                    break
            if not found_noun:
                return False
                
    return True

def reconstruct_span(span, fix_casing=False):
    out_text = ""
    FIX_WORDS = {
        'i': 'I', "i'm": "I'm", "i'll": "I'll", "i've": "I've", "i'd": "I'd",
        'mr': 'Mr', 'mrs': 'Mrs', 'ms': 'Ms', 'dr': 'Dr', 'prof': 'Prof', 
        'st': 'St', 'usa': 'USA', 'uk': 'UK', 'nato': 'NATO', 'fbi': 'FBI'
    }
    for token in span:
        text = token.text
        if fix_casing:
            if text.lower() in FIX_WORDS: text = FIX_WORDS[text.lower()]
            elif token.pos_ == 'PROPN': text = text.title()
        out_text += text + token.whitespace_
    return out_text

def clean_text(text):
    """
    Fixed Version: Prioritizes normalizing "" artifacts and removing
    spaces inside quotes (e.g. " word " -> "word").
    """
    if not text: return text
    
    # --- 1. Early Normalization (Critical) ---
    # Fix 'Moses' escaped quotes: \ ' -> '
    text = re.sub(r'\\\s*([\'"])', r'\1', text) 
    
    # Wiki/Book Artifact: Convert double-double quotes to standard quotes immediately.
    # This turns '"" as being ""' -> '" as being "'
    text = text.replace('""', '"')

    # --- 2. Fix Mojibake ---
    replacements = {
        '‚Äôs': "'s",  '‚Äô': "'",   '‚Äì': "-",   '‚Äî': "—",
        'â€™': "'",    'â€œ': '"',   'â€': '"',
        'â€“': '-',    'â€”': '—'
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # --- 3. Inner Quote Spacing (The Fix) ---
    # Removes spaces *inside* quotes.
    # Matches: " + space + text + space + " -> "text"
    text = re.sub(r'"\s+([^"]+?)\s+"', r'"\1"', text)
    # Matches: ' + space + text + space + ' -> 'text'
    text = re.sub(r"'\s+([^']+?)\s+'", r"'\1'", text)

    # --- 4. Smart Split (Run-on Quotes) ---
    # Fix "he'made" -> "he 'made" (but preserve "don't", "it's")
    text = re.sub(r"([a-zA-Z])'(?!s\b|t\b|m\b|ll\b|ve\b|re\b|d\b)([a-zA-Z]+)", r"\1 '\2", text)

    # --- 5. Punctuation Attachment ---
    # Fix "will, '" -> "will,'" (Attach closing quote to punctuation)
    text = re.sub(r"([,.;?!])\s+'", r"\1'", text)
    text = re.sub(r"([,.;?!])\s+\"", r'\1"', text)

    # --- 6. Numeric & Unit Artifacts ---
    text = re.sub(r'(\d)\s*°\s*(\d)', r'\1°\2', text)
    text = re.sub(r'(\d)\s*:\s*(\d)', r'\1:\2', text)
    text = re.sub(r'(\d)\s+%', r'\1%', text)
    text = re.sub(r'\$\s+(\d)', r'$\1', text)
    text = re.sub(r'(\d)\s+m\b', r'\1m', text)

    # --- 7. Domain Specific Fixes ---
    text = re.sub(r'\s*—\s*', '—', text) # Standardize em-dashes
    text = re.sub(r"(:)'([A-Z])", r"\1 '\2", text) # News artifact
    text = re.sub(r'([a-zA-Z])"\s+([vV]e|[mM]|[sS]|[tT]|[rR]e|[lL]l|[dD])\b', r"\1'\2", text) # Book elisions

    # --- 8. Final Polish ---
    text = re.sub(r'(\w)\s*/\s*(\w)', r'\1/\2', text)
    text = re.sub(r'(\w)\s*[‑–-]\s*(\w)', r'\1-\2', text)
    
    # Parentheses & Punctuation
    text = re.sub(r'([(])\s+', r'\1', text)
    text = re.sub(r'\s+([)])', r'\1', text)
    text = re.sub(r'\)\s*([A-Za-z])', r') \1', text) 
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Contractions Re-glue (Safety net)
    text = re.sub(r"\s+(['’](?:re|s|m|ll|ve|d|t))\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(n['’]t)", r"\1", text, flags=re.IGNORECASE)
    
    # Moses Cleanup
    text = text.replace('`` ', '"').replace(' ``', '"').replace(" ''", '"').replace("''", '"')
    
    # Trimming
    text = re.sub(r'\s+', ' ', text)
    if len(text) > 0 and text[0].islower(): 
        text = text[0].upper() + text[1:]
        
    return text.strip()

def is_clean_sentence_start(sent, strict_casing=True):
    """
    Filters sentences to ensure they are high-quality starts.
    Rejects sentences starting with punctuation, conjunctions, or garbage tokens.
    """
    first_token = sent[0]
    first_text = first_token.text
    full_text = sent.text
    
    # Reject common garbage in web/wiki text
    if '<unk>' in full_text or '@-@' in full_text or '@.@' in full_text or '@,@' in full_text: return False

    # Chapter Heading Filter (BookCorpus bias)
    if first_text.lower() == 'chapter': return False
    if first_text.isdigit() and len(sent) < 10: return False # Likely a page number or header
    
    # Reject starting with punctuation
    if first_token.pos_ == 'PUNCT' or first_text in ['"', "'", '(', '[', '-', '`', '``']: return False
    
    # Reject starting with coordinating conjunctions (fragments)
    if first_text.lower() in ['and', 'but', 'or', 'because', 'so']: return False
    
    # Reject lowercase starts (unless strict_casing is False, e.g. for BookCorpus)
    if strict_casing and not first_text[0].isupper(): return False
    
    return True

# ==========================================
#             CATEGORY MAPPERS
# ==========================================

def get_spacy_category(token):
    """Maps a Spacy token to our target categories or 'ignore'."""
    return POS_MAP_SPACY.get(token.pos_, 'ignore')

def get_code_category(token_type, token_string):
    """Maps Python token types to structural code categories."""
    if token_type == tokenize.NAME:
        if keyword.iskeyword(token_string): return 'KEYWORD'     # if, else, def
        return 'IDENTIFIER'                                      # variables, function names
    if token_type == tokenize.STRING or token_type == tokenize.NUMBER: 
        return 'LITERAL'                                         # "string", 42
    if token_type == tokenize.OP: 
        return 'OPERATOR'                                        # +, =, .
    return 'ignore'


# ==========================================
#             MINING LOGIC
# ==========================================

def mine_text_domain(dataset_iter, nlp, buckets, needed_counts, min_len, max_len, strict_casing, skip):
    """
    Mines prompts from Natural Language datasets (Books, Wiki, News).
    """
    skips_remaining = 0
    
    for item in dataset_iter:
        # Fast-forward mechanism: Skip X docs after finding a prompt to avoid topic clusters
        if skips_remaining > 0:
            skips_remaining -= 1
            continue

        # Extract text field (varies by dataset)
        text = item.get('text', item.get('article', ''))
        if not text or len(text) < 100: continue
        
        doc = nlp(text)
        
        for sent in doc.sents:
            # 1. Check basic sentence length
            if len(sent) <= min_len: continue
            
            # 2. Check sentence start quality
            if not is_clean_sentence_start(sent, strict_casing=strict_casing): continue

            # 3. Scanning Window
            # Look for a valid 'cut point' within the [min_len, max_len] range
            search_end = min(len(sent), max_len + 1)
            search_start = min_len
            
            found_category = None
            cut_index = -1
            
            # Scan tokens in the allowed window
            for i in range(search_start, search_end):
                token = sent[i]
                cat = get_spacy_category(token)
                
                # Check if this token's category is needed
                if cat != 'ignore' and cat in buckets and needed_counts[cat] > 0:
                    found_category = cat
                    cut_index = i
                    break 

            if found_category:
                prompt_span = sent[0 : cut_index + 1]
                
                # 4. Check Semantic Consistency (Self-contained)
                if not is_self_contained(prompt_span):
                    continue

                # 5. Reconstruct and Clean
                needs_casing_fix = not strict_casing
                raw_text = reconstruct_span(prompt_span, fix_casing=needs_casing_fix)
                final_text = clean_text(raw_text)
                
                # 6. Save Prompt
                entry = {
                    'domain': 'text', 
                    'category': found_category,
                    'last_pos': sent[cut_index].pos_,
                    'length_tokens': len(prompt_span),
                    'prompt': final_text
                }
                buckets[found_category].append(entry)
                needed_counts[found_category] -= 1
                
                # Trigger skip to ensure diversity
                skips_remaining = random.randint(skip/2, skip)
                yield 1
                break # Move to next document
        
        # Stop if all buckets for this domain are full
        if sum(needed_counts.values()) <= 0: return

def mine_code_domain(dataset_iter, buckets, needed_counts, min_len, max_len):
    """
    Mines prompts from Code datasets (Python).
    Uses the 'tokenize' library instead of Spacy.
    """
    skips_remaining = 0
    for item in dataset_iter:
        if skips_remaining > 0:
            skips_remaining -= 1
            continue

        code = item.get('content', '')
        if not code or len(code) < 50: continue

        # Attempt to tokenize the code file
        try:
            tokens_gen = tokenize.tokenize(io.BytesIO(code.encode('utf-8')).readline)
            # Filter out encoding markers and comments
            all_tokens = [t for t in tokens_gen if t.type not in (tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL, tokenize.COMMENT)]
        except (tokenize.TokenError, IndentationError):
            continue

        current_line = []
        for token in all_tokens:
            if token.type == tokenize.NEWLINE:
                # Process the completed line
                if len(current_line) > min_len:
                    # Filter: Skip lines starting with operators (e.g. decorators or continuations)
                    if current_line[0].type == tokenize.OP: 
                        current_line = [] 
                        continue

                    search_end = min(len(current_line), max_len + 1)
                    found_cat = None
                    cut_idx = -1
                    
                    # Scan window
                    for i in range(min_len, search_end):
                        t = current_line[i]
                        cat = get_code_category(t.type, t.string)
                        if cat != 'ignore' and cat in buckets and needed_counts[cat] > 0:
                            found_cat = cat
                            cut_idx = i
                            break
                    
                    if found_cat:
                        # Untokenize to get string back
                        subset = current_line[0 : cut_idx+1]
                        prompt_text = tokenize.untokenize(subset).strip()
                        # Clean artifacts from untokenize
                        prompt_text = prompt_text.replace(' .', '.').replace(' (', '(')

                        entry = {
                            'domain': 'code', 
                            'category': found_cat,
                            'last_pos': 'CODE',
                            'length_tokens': len(subset),
                            'prompt': prompt_text
                        }
                        buckets[found_cat].append(entry)
                        needed_counts[found_cat] -= 1
                        skips_remaining = random.randint(10, 50)
                        yield 1
                        break 
                current_line = []
            elif token.type not in (tokenize.INDENT, tokenize.DEDENT):
                current_line.append(token)
                
        if sum(needed_counts.values()) <= 0: return


# ==========================================
#               MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count_per_domain', type=int, default=100, help="Total prompts to collect per domain.")
    parser.add_argument('--output', type=str, default="meta_prompts.csv", help="Output CSV filename.")
    args = parser.parse_args()

    # Load Spacy once
    nlp = load_spacy_model()
    nlp.max_length = 2_000_000 # Handle large documents
    
    # Prompt Length Constraints (in tokens/words)
    MIN_LEN = 15 
    MAX_LEN = 25
    
    # Max number of times to refill dataset if we run out of data before filling buckets
    MAX_RETRIES = 100  

    # Define Targets per domain
    TEXT_TARGETS = ['NOUN', 'PROPN', 'PUNCT', 'ADP+PART', 'VERB', 'DET', 'ADJ', 'PRON', 'AUX', 'CCONJ+SCONJ', 'ADV']
    CODE_TARGETS = ['KEYWORD', 'IDENTIFIER', 'LITERAL', 'OPERATOR']

    final_data = []

    # Iterate over each configured domain
    for domain, cfg in DOMAIN_CONFIG.items():
        print(f"\n--- Mining Domain: {domain} ---")
        
        # Setup buckets
        target_cats = CODE_TARGETS if domain == 'code' else TEXT_TARGETS
        buckets = {k: [] for k in target_cats}
        # In this config, we want 'count_per_domain' TOTAL prompts (e.g. 100), 
        # so we divide the requirement among the categories.
        # (Note: Use args.count_per_domain as strict total if you prefer)
        needed = {k: args.count_per_domain for k in buckets} # User Logic: N per bucket? Or N total? 
        # Based on previous prompt logic: "count_per_domain" seemed to apply per bucket in your snippet?
        # Let's assume you want 'count_per_domain' TOTAL.
        # NOTE: Reverting to your specific request in previous turn where you printed `needed` 
        # implying you wanted the args.count to apply to the buckets.
        
        pbar = tqdm(total=sum(needed.values()), desc=f"{domain}")
        
        retry_count = 0
        
        # --- RETRY LOOP ---
        # If the dataset stream ends but buckets aren't full, we reload and shuffle again.
        while sum(needed.values()) > 0 and retry_count < MAX_RETRIES:
            
            if retry_count > 0:
                print(f"  [Retry {retry_count}/{MAX_RETRIES}] Refilling dataset. Missing: {sum(needed.values())} items.")
            
            try:
                # Load Dataset (Streaming Mode)
                ds = load_dataset(cfg['path'], cfg.get('name'), split=cfg['split'], streaming=True)
                # Shuffle with a new seed every retry to ensure fresh data
                current_seed = 42 + retry_count
                ds = ds.shuffle(seed=current_seed, buffer_size=5000)
            except Exception as e:
                print(f"  Error loading dataset on retry {retry_count}: {e}")
                break

            # Select appropriate miner
            if domain == 'code':
                miner = mine_code_domain(ds, buckets, needed, MIN_LEN, MAX_LEN)
            else:
                strict_casing = cfg.get('strict_casing', True)
                miner = mine_text_domain(ds, nlp, buckets, needed, MIN_LEN, MAX_LEN, strict_casing, cfg['skip'])
            
            # Execute mining
            for _ in miner:
                pbar.update(1)
            
            retry_count += 1
            
        pbar.close()
        
        # Warning if still missing data
        if sum(needed.values()) > 0:
            print(f"  Warning: Could not fill all buckets for {domain} after {MAX_RETRIES} passes.")
            print(f"  Missing counts: {needed}")

        # Collect results
        for cat, items in buckets.items():
            for item in items:
                item['domain'] = domain
                final_data.append(item)

    # Save to CSV
    print(f"\nSaving {len(final_data)} prompts to {args.output}...")
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['domain', 'category', 'last_pos', 'length_tokens', 'prompt']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_data)
    print("Done.")

if __name__ == "__main__":
    main()