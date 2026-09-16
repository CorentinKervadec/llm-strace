"""
Section 4.4: Hidden State Representation Extraction
===================================================
This script extracts the last-token hidden representations across different trace sizes (s-trace)
and subgraph conditions (target trace vs. random baseline, pruned vs. inverse subgraphs).

It directly supports Section 4.4 ("Geometry of the traces' representation space") and Figure 5 
of the paper, saving compressed representation matrices used to analyze representation 
geometry and alignment as target subgraph density (s-trace size) scales.
"""

import argparse
import os
import time
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate import cpu_offload

from src.llm_trace.llm_trace import load_from_file_light
from src.llm_trace.mask_utils import prepare_mask
from src.test.unit_tests import MaskingError
from src.modified_transformers.utils import get_model_class, identify_model_type


def compute_and_save_hidden_strace(strace, final_dir, sentence_index, captured_hidden, do_random=True, do_inverse=True):
    """
    Executes forward passes across all trace sizes and sub-conditions, extracting the final 
    token's hidden representation via PyTorch forward hooks. Saves all matrices to a single .npz file.
    """
    device = strace.llm.device
    
    # Storage structure for representations across density thresholds
    hidden_data = {
        'trace': {'inverse': [], 'only': []},
        'random': {'inverse': [], 'only': []}
    }

    for i, stratum_index in tqdm(
        enumerate(strace.strata_index), 
        desc=f"[HIDDEN EXTRACT | IDX {sentence_index}] Strace Pass", 
        total=len(strace.strata_index),
        leave=False
    ):
        for is_random in [False, True]:
            if not do_random and is_random: 
                continue

            if is_random:
                stratum_size = strace.strata_raw_size[i]
                stratum = strace.graph.get_random_connected_subgraph(stratum_size)
            else:
                def filter_edges(u, v, k):  
                    return strace.graph[u][v][k].get('stratum') <= stratum_index
                stratum = nx.subgraph_view(strace.graph, filter_edge=filter_edges)

            # Verify that the extracted subgraph edge count matches the raw stratum size
            stratum_size = stratum.get_size()
            if strace.strata_raw_size[i] != stratum_size:
                raise MaskingError(
                    f"Stratum {stratum_index} size mismatch: expected {strace.strata_raw_size[i]}, got {stratum_size}"
                )

            for is_inverse in [True, False]:
                if not do_inverse and is_inverse:
                    continue

                # Construct attention graph mask for the current stratum condition
                graph_mask, _, _ = prepare_mask(
                    graph=stratum, 
                    seq_len=strace.graph.graph['n_tokens'],
                    nb_head=strace.graph.graph['n_heads'], 
                    n_layers=strace.graph.graph['n_layers'], 
                    inverse=is_inverse, 
                    keep_residual=True if (is_inverse or is_random) else False
                )
                
                captured_hidden.clear()

                # Execute forward pass with targeted graph mask
                with torch.no_grad():
                    _ = strace.llm(
                        input_ids=strace.input_prepared[0].to(device),
                        attention_mask=strace.input_prepared[1].to(device), 
                        graph_mask=graph_mask, 
                        build_graph=None, 
                        unit_test=False, 
                        attn_implementation="eager"
                    )
                
                if 'state' not in captured_hidden or captured_hidden['state'] is None:
                    raise RuntimeError(f"Forward hook failed to capture hidden states at stratum {stratum_index}.")
                
                # Extract last token representation (shape: batch_size=1, seq_len, hidden_dim)
                last_token_hidden = captured_hidden['state'][0, -1, :].cpu().numpy()

                type_key = 'random' if is_random else 'trace'
                mode_key = 'inverse' if is_inverse else 'only'
                hidden_data[type_key][mode_key].append(last_token_hidden)

    # Package matrices (num_strace, hidden_dim) into a single dictionary
    save_dict = {}
    for type_key in ['trace', 'random']:
        for mode_key in ['inverse', 'only']:
            if hidden_data[type_key][mode_key]:
                save_dict[f"{type_key}_{mode_key}"] = np.array(hidden_data[type_key][mode_key])
                
    # Save output representation file
    if save_dict:
        hidden_file_path = os.path.join(final_dir, f'hidden_strace_{sentence_index}.npz')
        np.savez_compressed(hidden_file_path, **save_dict)
        print(f"  > Saved representation matrices to {hidden_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Section 4.4: Extract last-token hidden representations across trace sizes.")
    parser.add_argument('--model_name', type=str, required=True, help='Hugging Face model identifier.')
    parser.add_argument('--chunk_id', type=int, required=True, help='Slurm array task ID (used as chunk index).')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of sentences to process per job.')
    parser.add_argument('--total_sentences', type=int, required=True, help='Total size of the dataset.')
    parser.add_argument('--final_dir', type=str, required=True, help='Directory containing evaluated Stage 3 .npz results.')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offload for large models.')
    parser.add_argument('--checkpoint', type=str, default='main')
    args = parser.parse_args()

    half_precision = True

    # --- 1. Model Initialization ---
    print(f"[HIDDEN EXTRACT | CHUNK {args.chunk_id}] Initializing model...")
    start_time = time.time()
    
    target_dtype = torch.float16 if half_precision else torch.float32
    model_type = identify_model_type(args.model_name)
    hf_constructor = get_model_class(model_type)

    if args.cpu_offload:
        llm = hf_constructor.from_pretrained(
            args.model_name,
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=target_dtype,
            revision=args.checkpoint
        )
        llm = cpu_offload(llm, execution_device="cuda:0")
    else:
        llm = hf_constructor.from_pretrained(
            args.model_name,
            device_map="auto",
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=target_dtype,
            revision=args.checkpoint
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"[HIDDEN EXTRACT] {args.model_name} loaded in {time.time() - start_time:.2f} s")

    # --- 2. Register Forward Hook on Final Representation Layer ---
    target_layer = None
    if hasattr(llm, 'model') and hasattr(llm.model, 'norm'):  # LLaMA / Mistral / Qwen
        target_layer = llm.model.norm
    elif hasattr(llm, 'transformer') and hasattr(llm.transformer, 'ln_f'):  # GPT / OLMo
        target_layer = llm.transformer.ln_f
    elif hasattr(llm, 'model') and hasattr(llm.model, 'final_layernorm'):  # Gemma
        target_layer = llm.model.final_layernorm

    if target_layer is None:
        raise RuntimeError("Could not locate final layer norm module to attach hidden representation hook.")

    captured_hidden = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured_hidden['state'] = output[0].detach()
        else:
            captured_hidden['state'] = output.detach()

    hook_handle = target_layer.register_forward_hook(hook_fn)
    print(f"[HIDDEN EXTRACT] Forward hook registered on: {target_layer.__class__.__name__}")

    # --- 3. Calculate Data Chunk Bounds ---
    start_index = args.chunk_id * args.chunk_size
    end_index = min((args.chunk_id + 1) * args.chunk_size, args.total_sentences)
    print(f"[HIDDEN EXTRACT | CHUNK {args.chunk_id}] Processing dataset subset: Sentences {start_index} to {end_index - 1}")

    # --- 4. Process Chunk ---
    for sentence_index in range(start_index, end_index):
        final_file_path = os.path.join(args.final_dir, f'strace_final_{sentence_index}.npz')
        hidden_file_path = os.path.join(args.final_dir, f'hidden_strace_{sentence_index}.npz')

        # Skip if hidden representations have already been extracted
        if os.path.exists(hidden_file_path):
            print(f"[HIDDEN EXTRACT | IDX {sentence_index}] Hidden representations already extracted. Skipping.")
            continue

        # Check if input Stage 3 evaluated file exists
        if not os.path.exists(final_file_path):
            print(f"[HIDDEN EXTRACT | IDX {sentence_index}] WARNING: Missing Stage 3 file at {final_file_path}. Skipping.")
            continue

        strace = load_from_file_light(final_file_path, llm, tokenizer)
        if strace is None:
            continue

        print(f"\n[HIDDEN EXTRACT | IDX {sentence_index}] Extracting last-token representations...")
        t_start = time.time()
        try:
            compute_and_save_hidden_strace(
                strace=strace,
                final_dir=args.final_dir,
                sentence_index=sentence_index,
                captured_hidden=captured_hidden,
                do_random=True,
                do_inverse=True
            )
            print(f"  > Representations extracted in {time.time() - t_start:.2f} s")
        except Exception as e:
            print(f"[HIDDEN EXTRACT | IDX {sentence_index}] ERROR during extraction: {e}")

    # Cleanly detach hook
    hook_handle.remove()
    print(f"\n[HIDDEN EXTRACT | CHUNK {args.chunk_id}] Complete.")


if __name__ == "__main__":
    main()