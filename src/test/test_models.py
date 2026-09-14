# tests/test_models.py
import pytest
import torch
import gc
import time
import networkx as nx
from transformers import AutoTokenizer
import traceback
import os

from src.test.unit_tests import PrecisionError, TopologyError, CausalLeakageError, MaskingError
from src.modified_transformers.utils import get_model_class, identify_model_type
from src.llm_trace.mask_utils import prepare_mask

N_SENTENCES=5

# --- Data Loading Logic ---
def load_sentences_wikitext(data_file, n_sentences):
    """Loads the first N valid sentences from the .tsv data file."""
    if not os.path.exists(data_file):
        print(f"Warning: Dataset file not found at {data_file}")
        return []
        
    sentences = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(sentences) >= n_sentences:
                break
            try:
                parts = line.strip().split('\t')
                # Expects format: {sentence}\t{last word}\t{...}\t{next word}\t{...}
                if len(parts) >= 4:
                    input_sentence = parts[0]
                    gt_next = parts[3]
                    sentences.append({"input": input_sentence, "gt_next": gt_next, "id": i})
                else:
                    print(f"[load_sentences_wikitext] Error: Line {i} has fewer than 4 columns. Line: '{line.strip()}'")
            except Exception as e:
                print(f"[load_sentences_wikitext] Error processing line {i}: {e}")
                
    return sentences

# --- Pytest Configuration Hooks ---

def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters based on loaded datasets."""
    if "test_case" in metafunc.fixturenames:
        
        datasets = {
            # "short": "data/wikitext_10.txt",
            "medium": "data/wikitext_20.txt",
            # "long": "data/wikitext_50.txt"
        }
        
        test_cases = []
        test_ids = []
        
        for length_category, path in datasets.items():
            loaded_data = load_sentences_wikitext(path, N_SENTENCES)
            for item in loaded_data:
                test_cases.append((length_category, item))
                test_ids.append(f"{length_category}-line_{item['id']}")
                
        metafunc.parametrize("test_case", test_cases, ids=test_ids)

# --- Model Configuration ---

AVAILABLE_MODELS = [
    # "Qwen/Qwen2.5-7B",
    'Qwen/Qwen3-8B-Base',
    'Qwen/Qwen3-14B-Base',
    # 'allenai/OLMo-2-0425-1B',
    "allenai/OLMo-2-1124-7B", 
    "allenai/OLMo-2-1124-13B",
    "meta-llama/Llama-3.1-8B",

]

# --- The Test Function ---
@pytest.mark.parametrize("model_path", AVAILABLE_MODELS)
@pytest.mark.parametrize("dtype_str", ["float32"]) #["float32", "float16"])
def test_model_reconstruction(model_path, test_case, dtype_str, record_property):
    # test_case is a tuple injected by pytest_generate_tests: (length_category, dict_item)
    length_category, item = test_case
    sentence = item["input"]
    gt_next = item["gt_next"] # Ground truth next token, if you want to test accuracy later

    target_dtype = torch.float32 if dtype_str == "float32" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_type = identify_model_type(model_path)
    ModelClass = get_model_class(model_type)
    
    model = None
    try:
        model = ModelClass.from_pretrained(model_path, device_map="auto", attn_implementation="eager", torch_dtype=target_dtype)
        device = model.device
        inputs = tokenizer(sentence, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        n_tokens = inputs['input_ids'].shape[1]
        record_property("tokens", n_tokens)
        record_property("model", model_path.split('/')[-1])
        record_property("dtype", dtype_str)
        record_property("dataset", length_category)

        start_time = time.perf_counter()
        
        # 1. Forward Pass (Catches PrecisionErrors from unit_test.py)
        # Note: Added attn_implementation="eager" to ensure causal_mask is generated!
        with torch.no_grad():
            output, graph_obj = model(**inputs, build_graph='norm', unit_test=True, attn_implementation="eager")
        graph_obj.remove_disconnected_nodes()

        end_time = time.perf_counter()
        record_property("time", end_time - start_time)

        assert output is not None
        assert graph_obj is not None

        # 2. Topology Test
        nx_graph = graph_obj.G if hasattr(graph_obj, 'G') else graph_obj
        if not nx.is_directed_acyclic_graph(nx_graph):
            raise TopologyError("The computational graph contains a cycle!")

        # 3. Graph Dimensions Test
        n_layers = model.config.num_hidden_layers
        theoretical_nodes = n_tokens # input nodes
        theoretical_nodes += n_layers * 2 * n_tokens # all layers 
        theoretical_nodes -= 2* (n_tokens - 1) # remove the disconnected nodes
        if len(nx_graph.nodes) != theoretical_nodes:
           raise TopologyError(f"Node count mismatch! Expected {theoretical_nodes}, got {len(nx_graph.nodes)}")
        
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        
        # Calculate theoretical edges
        attention_edges = n_heads * (n_tokens * (n_tokens + 1) // 2) + n_tokens # residual + attention
        mlp_edges = 2 * n_tokens # residual + MLP
        theoretical_edges = n_layers * (attention_edges + mlp_edges)
        theoretical_edges -= 2* (n_tokens - 1) # remove disconnected mlp edges + residual
        theoretical_edges -= n_heads * ((n_tokens-1) * ((n_tokens-1) + 1) // 2) + n_tokens -1# remove the disconnected attention edges + residual
        if len(nx_graph.edges) != theoretical_edges:
           raise TopologyError(f"Edges count mismatch! Expected {theoretical_edges}, got {len(nx_graph.edges)}")
        
        record_property("status_msg", "Passed successfully.")

    except PrecisionError as e:
        record_property("failure_type", "Precision Error (Float Truncation)")
        record_property("status_msg", str(e))
        pytest.fail(str(e))
        
    except TopologyError as e:
        record_property("failure_type", "Topological Graph Error")
        record_property("status_msg", str(e))
        pytest.fail(str(e))

    except torch.cuda.OutOfMemoryError as e:
        record_property("failure_type", "CUDA Out of Memory")
        record_property("status_msg", "GPU ran out of VRAM during execution.")
        pytest.fail("OOM Error")
        
    except Exception as e:
        # Catch-all for unexpected crashes
        error_trace = traceback.format_exc().splitlines()[-1]
        record_property("failure_type", "Unexpected Exception")
        record_property("status_msg", error_trace)
        pytest.fail(error_trace)

    finally:
        # 1. Delete all explicitly created heavy objects
        if 'model' in locals() and model is not None:
            del model
        if 'inputs' in locals():
            del inputs
        if 'output' in locals():
            del output
        if 'graph_obj' in locals():
            del graph_obj
        if 'nx_graph' in locals():
            del nx_graph
            
        # 2. Force garbage collection to destroy objects before emptying cache
        gc.collect()
        
        # 3. Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

@pytest.mark.parametrize("model_path", AVAILABLE_MODELS)
def test_graph_masking_consistency(model_path, test_case, record_property):
    """Tests if masking disconnected nodes preserves the exact root node logits."""
    length_category, item = test_case
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ModelClass = get_model_class(identify_model_type(model_path))
    model = ModelClass.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float32)
    inputs = tokenizer(item["input"], return_tensors='pt').to(model.device)
    
    try:
        # 1. Get baseline output and full graph
        output_baseline, graph_obj = model(**inputs, build_graph='norm', unit_test=False)
        baseline_logits = output_baseline.logits[0, -1] # Root node
        
        # 2. Prune disconnected nodes
        graph_obj.remove_disconnected_nodes()
        
        # 3. Build masks from the pruned graph
        n_layers = model.config.num_hidden_layers
        n_heads = getattr(model.config, "num_attention_heads", 0) 
        seq_len = inputs['input_ids'].shape[1]
        
        masks, _, _ = prepare_mask(
            graph=graph_obj, 
            seq_len=seq_len, 
            nb_head=n_heads, 
            n_layers=n_layers, 
            inverse=False, 
            keep_residual=False
        )
        
        # 4. Run model WITH masks
        output_masked = model(**inputs, build_graph=None, graph_mask=masks)
        masked_logits = output_masked.logits[0, -1]
        
        # 5. Check Consistency (They should be mathematically identical)
        if not torch.allclose(baseline_logits, masked_logits, atol=1e-5):
            max_diff = (baseline_logits - masked_logits).abs().max().item()
            raise MaskingError(f"Pruning disconnected nodes altered root logits! Max drift: {max_diff:.5f}")
            
    except Exception as e:
        record_property("failure_type", type(e).__name__)
        record_property("status_msg", str(e))
        pytest.fail(str(e))
    
    finally:
        if 'model' in locals() and model is not None:
            del model
        for var in ['inputs', 'output_baseline', 'baseline_logits', 'output_masked', 'masked_logits', 'masks', 'graph_obj']:
            if var in locals():
                del locals()[var]
                
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()