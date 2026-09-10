import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

# The repository ID
REPO_ID = "allenai/OLMo-2-1124-7B"

# Your specific log-spaced checkpoints
checkpoints = [
    # "stage1-step150-tokens1B",
    "stage1-step600-tokens3B",
    "stage1-step900-tokens4B",
    "stage1-step4000-tokens17B",
    "stage1-step15000-tokens63B",
    "stage1-step39000-tokens164B",
    "stage1-step89000-tokens374B",
    "stage1-step199000-tokens835B",
    "stage1-step431000-tokens1808B",
    "stage1-step928646-tokens3896B",
    "main"
]

def verify_and_cache():
    for revision in checkpoints:
        print(f"\n{'='*50}")
        print(f"Checking/Downloading: {revision}")
        print(f"{'='*50}")
        
        try:
            # 1. Download/Cache the full snapshot (weights + config + tokenizer)
            # This ensures every file is present in your local cache.
            snapshot_path = snapshot_download(
                repo_id=REPO_ID, 
                revision=revision,
                # Avoid downloading unnecessary files if they exist (like .msgpack or .safetensors if you only need one)
                # ignore_patterns=["*.msgpack", "*.bin"] # Uncomment if you only want safetensors
            )
            print(f"✓ Snapshot cached at: {snapshot_path}")

            # 2. Lightweight verification
            # We load the config and tokenizer just to be 100% sure the files are valid.
            AutoConfig.from_pretrained(REPO_ID, revision=revision)
            AutoTokenizer.from_pretrained(REPO_ID, revision=revision)
            print(f"✓ Configuration and Tokenizer verified for {revision}")

        except Exception as e:
            print(f"✗ Failed to process {revision}: {e}")

if __name__ == "__main__":
    verify_and_cache()
    print("\nAll specified checkpoints have been checked/cached.")
# import re
# import numpy as np
# from huggingface_hub import list_repo_refs

# def get_log_spaced_checkpoints(repo_id="allenai/OLMo-2-1124-7B", num_samples=10, prefix="stage1-"):
#     # 1. Fetch all repository branches
#     try:
#         refs = list_repo_refs(repo_id)
#         branches = [b.name for b in refs.branches]
#     except Exception as e:
#         return f"Error fetching from Hugging Face: {e}"

#     # 2. Filter for stage 1 and extract step numbers for sorting
#     # We use a list of tuples: (step_number, branch_name)
#     stage1_checkpoints = []
#     for name in branches:
#         if name.startswith(prefix):
#             step_match = re.search(r"step(\d+)", name)
#             if step_match:
#                 stage1_checkpoints.append((int(step_match.group(1)), name))

#     # 3. Sort checkpoints numerically by step
#     stage1_checkpoints.sort(key=lambda x: x[0])
    
#     if not stage1_checkpoints:
#         return f"No checkpoints found starting with '{prefix}'."

#     # 4. Calculate Logarithmic Spacing
#     # We find indices in the list that are logarithmically spaced
#     num_available = len(stage1_checkpoints)
    
#     if num_available <= num_samples:
#         # If we have fewer than requested, just return all of them
#         selected = [cp[1] for cp in stage1_checkpoints]
#     else:
#         # np.geomspace(1, N, 10) gives 10 numbers from 1 to N log-spaced
#         # We subtract 1 to get 0-based indices
#         indices = np.geomspace(1, num_available, num_samples)
#         indices = np.round(indices).astype(int) - 1
        
#         # Ensure indices are unique (geomspace can produce duplicates if N is small)
#         unique_indices = sorted(list(set(indices)))
#         selected = [stage1_checkpoints[i][1] for i in unique_indices]

#     return selected

# if __name__ == "__main__":
#     # Change prefix to "step" if "stage1-" returns no results
#     PREFIX = "stage1-" 
#     checkpoints = get_log_spaced_checkpoints(num_samples=10, prefix=PREFIX)
    
#     print(f"--- Selected {len(checkpoints)} Log-Spaced Checkpoints ({PREFIX}*) ---")
#     for cp in checkpoints:
#         print(cp)