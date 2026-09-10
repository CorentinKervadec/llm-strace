import argparse
import os
import time
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer

# Import required tools from your codebase
from src.llm_trace.llm_trace_2 import load_from_file_light
from src.llm_trace.mask_utils import prepare_mask
from src.test.unit_tests import MaskingError
from src.modified_transformers.utils import get_model_class, identify_model_type
from accelerate import cpu_offload

def compute_and_save_hidden_strata(strace, final_dir, sentence_index, captured_hidden, do_random=True, do_inverse=True):
    """
    Loops through all strata sizes and configurations, performs a forward pass with 
    the corresponding graph mask, and extracts the last token hidden representation.
    Saves all configurations for the given sentence into a single compressed .npz file.
    """
    device = strace.llm.device
    
    # Initialize dictionary to collect hidden representations
    hidden_data = {
        'trace': {'inverse': [], 'only': []},
        'random': {'inverse': [], 'only': []}
    }

    for i, stratum_index in tqdm(enumerate(strace.strata_index), desc=f"[STRACE] Extracting hidden strata", total=len(strace.strata_index)):
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
                
                # Clear any hidden states from previous iterations
                captured_hidden.clear()

                # --- FAST FORWARD PASS ---
                with torch.no_grad():
                    _ = strace.llm(
                        input_ids=strace.input_prepared[0].to(device),
                        attention_mask=strace.input_prepared[1].to(device), 
                        graph_mask=graph_mask, 
                        build_graph=None, 
                        unit_test=False, 
                        attn_implementation="eager"
                    )
                
                # Fetch the hidden state captured passively by our hook
                if 'state' in captured_hidden and captured_hidden['state'] is not None:
                    last_layer_hidden = captured_hidden['state']
                else:
                    raise RuntimeError(f"[ERROR] Forward hook failed to capture hidden states at stratum {stratum_index}!")
                
                # Extract the last token's representation (assuming batch size of 1)
                # Shape expected from hook: (batch_size, sequence_length, hidden_dimension)
                last_token_hidden = last_layer_hidden[0, -1, :].cpu().numpy()

                key_tuple = ('random' if random else 'trace', 'inverse' if inverse else 'only') 
                hidden_data[key_tuple[0]][key_tuple[1]].append(last_token_hidden)

    # Convert the lists of 1D arrays into 2D arrays (num_strata, hidden_dimension)
    save_dict = {}
    for type_key in ['trace', 'random']:
        for mode_key in ['inverse', 'only']:
            if hidden_data[type_key][mode_key]:
                save_dict[f"{type_key}_{mode_key}"] = np.array(hidden_data[type_key][mode_key])
                
    # Save all condition matrices into one unified file per sentence
    if save_dict:
        hidden_file_path = os.path.join(final_dir, f'hidden_strata_{sentence_index}.npz')
        np.savez_compressed(hidden_file_path, **save_dict)
        print(f"[EXTRACT-JOB {sentence_index}] Saved hidden representations to {hidden_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and save last-token hidden layers across strata conditions using hooks.")
    parser.add_argument('--model_name', type=str, required=True, help='HF name of the llm.')
    parser.add_argument('--chunk_id', type=int, required=True, help='Slurm array task ID, used as chunk index.')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of sentences to process per job.')
    parser.add_argument('--total_sentences', type=int, required=True, help='Total number of sentences in the dataset.')
    parser.add_argument('--final_dir', type=str, required=True, help='Directory where final strace .npz results are saved.')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offload')
    parser.add_argument('--checkpoint', type=str, default='main')
    args = parser.parse_args()

    half_precision = True

    print(f"[EXTRACT-JOB CHUNK {args.chunk_id}] Loading model...")
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
    print(f"[EXTRACT-JOB CHUNK {args.chunk_id}] Time to initialize model: {time.time() - start_time:.2f} s")

    # --- REGISTER FORWARD HOOK ON THE FINAL LAYER ---
    # We dynamically find the final block layer or layer norm depending on architecture
    target_layer = None
    if hasattr(llm, 'model') and hasattr(llm.model, 'norm'): # Llama / Mistral
        target_layer = llm.model.norm
    elif hasattr(llm, 'transformer') and hasattr(llm.transformer, 'ln_f'): # GPT-style
        target_layer = llm.transformer.ln_f
    
    if target_layer is None:
        raise RuntimeError("Could not find a valid final layer or layer-norm to hook into.")

    captured_hidden = {}
    def hook_fn(module, input, output):
        # Outputs from a layer block could be a tuple; Layer Norms return a Tensor directly
        if isinstance(output, tuple):
            captured_hidden['state'] = output[0].detach()
        else:
            captured_hidden['state'] = output.detach()

    hook_handle = target_layer.register_forward_hook(hook_fn)
    print(f"[EXTRACT-JOB] Successfully registered forward hook on: {target_layer.__class__.__name__}")

    # --- RUN EXTRACTION CHUNKS ---
    start_index = args.chunk_id * args.chunk_size
    end_index = min((args.chunk_id + 1) * args.chunk_size, args.total_sentences)
    print(f"[EXTRACT-JOB CHUNK {args.chunk_id}] Extracting hidden states for sentences from {start_index} to {end_index - 1}")

    for sentence_index in range(start_index, end_index):
        final_file_path = os.path.join(args.final_dir, f'strace_final_{sentence_index}.npz')
        
        if not os.path.exists(final_file_path):
            print(f"[EXTRACT-JOB {sentence_index}] WARNING: No file found at {final_file_path}. Skipping.")
            continue

        strace = load_from_file_light(final_file_path, llm, tokenizer)
        if strace is None:
            continue
            
        print(f"[EXTRACT-JOB {sentence_index}] Executing stratified forward passes...")
        start_eval_time = time.time()
        
        try:
            compute_and_save_hidden_strata(strace, args.final_dir, sentence_index, captured_hidden, do_random=True, do_inverse=True)
            print(f"[EXTRACT-JOB {sentence_index}] Total compute time: {time.time() - start_eval_time:.2f} s")
        except Exception as e:
            print(f"[EXTRACT-JOB {sentence_index}] ERROR processing file: {e}")

    # Remove the hook cleanly when script completes execution
    hook_handle.remove()

if __name__ == "__main__":
    main()