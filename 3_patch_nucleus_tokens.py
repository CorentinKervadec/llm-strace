import argparse
import os
import time
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer

# Import required tools from your codebase
from src.llm_trace.llm_trace_2 import load_from_file_light, get_nucleus
from src.llm_trace.mask_utils import prepare_mask
from src.test.unit_tests import MaskingError
from src.modified_transformers.utils import get_model_class, identify_model_type
from accelerate import cpu_offload

def compute_minimal_nucleus_tkn(strace, do_random=True, do_inverse=True):
    """
    A highly stripped-down version of compute_stratum_reconstruction_error
    designed solely to compute and store strata_nucleus_60_tkn.
    """
    # 1. Reset specifically the target metric so we start fresh
    strace.strata_nucleus_60_tkn = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
    device = strace.llm.device

    for i, stratum_index in tqdm(enumerate(strace.strata_index), desc=f"[STRACE] Fast Patching strata", total=len(strace.strata_index)):
        for random in [False, True]:
            if not do_random and random: 
                continue
            if random:
                stratum_size = strace.strata_raw_size[i]
                stratum = strace.graph.get_random_connected_subgraph(stratum_size)
            else:
                def filter_edges(u, v, k):  
                    return strace.graph[u][v][k].get('stratum') <= stratum_index
                stratum = nx.subgraph_view(strace.graph, filter_edge=filter_edges)

            stratum_size = stratum.get_size()
            if not strace.strata_raw_size[i] == stratum_size:
                raise MaskingError(f"Stratum {stratum_index} size mismatch: expected {strace.strata_raw_size[i]}, got {stratum_size}")

            for inverse in [True, False]:
                if not do_inverse and inverse:
                    continue

                graph_mask, nb_non_masked_edges, _ = prepare_mask(
                    graph=stratum, 
                    seq_len=strace.graph.graph['n_tokens'],
                    nb_head=strace.graph.graph['n_heads'], 
                    n_layers=strace.graph.graph['n_layers'], 
                    inverse=inverse, 
                    keep_residual=True if (inverse or random) else False
                )
                
                # --- FAST FORWARD PASS ---
                with torch.no_grad():
                    output = strace.llm(
                        input_ids=strace.input_prepared[0].to(device),
                        attention_mask=strace.input_prepared[1].to(device), 
                        graph_mask=graph_mask, 
                        build_graph=None, 
                        unit_test=False, 
                        attn_implementation="eager"
                    )
                
                # --- ONLY compute the required tokens (Skipping Surprisal, TV, Nu, etc.) ---
                graph_logits = output.logits.view(-1, strace.llm.config.vocab_size).cpu()
                nucleus_indices = get_nucleus(graph_logits[-1].cpu(), 60)

                key_tuple = ('random' if random else 'trace', 'inverse' if inverse else 'only') 
                
                # Extract as native int to prevent weird tensor scaling issues during numpy serialization
                tkn_list = [token_id.item() for token_id in nucleus_indices[:5]]
                strace.strata_nucleus_60_tkn[key_tuple[0]][key_tuple[1]].append(tkn_list)


def main():
    parser = argparse.ArgumentParser(description="Fast Patch: Minimally compute and inject nucleus_60_tkn")
    parser.add_argument('--model_name', type=str, required=True, help='HF name of the llm.')
    parser.add_argument('--chunk_id', type=int, required=True, help='Slurm array task ID, used as chunk index.')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of sentences to process per job.')
    parser.add_argument('--total_sentences', type=int, required=True, help='Total number of sentences in the dataset.')
    parser.add_argument('--final_dir', type=str, required=True, help='Directory where final strace .npz results are saved.')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offload')
    parser.add_argument('--checkpoint', type=str, default='main')
    args = parser.parse_args()

    half_precision = True

    print(f"[PATCH-JOB CHUNK {args.chunk_id}] Loading model...")
    start_time = time.time()
    
    model_type = identify_model_type(args.model_name)
    hf_constructor = get_model_class(model_type)
    if args.cpu_offload:
        llm = hf_constructor.from_pretrained(
            args.model_name,
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=torch.float16 if half_precision else torch.float32,
            revision=args.checkpoint
        )
        llm = cpu_offload(llm, execution_device="cuda:0")
    else:
        llm = hf_constructor.from_pretrained(
            args.model_name,
            device_map="auto",
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=torch.float16 if half_precision else torch.float32,
            revision=args.checkpoint
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"[PATCH-JOB CHUNK {args.chunk_id}] Time to initialize model: {time.time() - start_time:.2f} s")

    start_index = args.chunk_id * args.chunk_size
    end_index = min((args.chunk_id + 1) * args.chunk_size, args.total_sentences)
    print(f"[PATCH-JOB CHUNK {args.chunk_id}] Fast-Patching sentences from {start_index} to {end_index - 1}")

    for sentence_index in range(start_index, end_index):
        final_file_path = os.path.join(args.final_dir, f'strace_final_{sentence_index}.npz')
        
        if not os.path.exists(final_file_path):
            print(f"[PATCH-JOB {sentence_index}] WARNING: No file found at {final_file_path}. Skipping.")
            continue

        strace = load_from_file_light(final_file_path, llm, tokenizer)
        if strace is None:
            continue
            
        print(f"[PATCH-JOB {sentence_index}] Executing fast forward passes...")
        start_eval_time = time.time()
        
        # Call our ultra-lightweight monkey-patched function instead
        compute_minimal_nucleus_tkn(strace, do_random=False, do_inverse=False)
        
        print(f"[PATCH-JOB {sentence_index}] Compute time: {time.time() - start_eval_time:.2f} s")

        # --- Minimal Patching Step ---
        try:
            original_data = dict(np.load(final_file_path, allow_pickle=True))
            original_data['nucleus_60_tkn'] = strace.strata_nucleus_60_tkn
            np.savez_compressed(final_file_path, **original_data)
            print(f"[PATCH-JOB {sentence_index}] NPZ successfully updated.")
        except Exception as e:
            print(f"[PATCH-JOB {sentence_index}] ERROR while patching file: {e}")

if __name__ == "__main__":
    main()