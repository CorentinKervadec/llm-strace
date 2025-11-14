from src.llm_graph.mistral_graph_nx import Mistral_Graph_NX
from src.llm_graph.olmo_graph_nx import Olmo2_Graph_NX
from src.llm_graph.qwen_graph_nx import Qwen3_Graph_NX

GRAPH_CONSTRUCTOR = {
    'mistralai/Mistral-7B-v0.1': Mistral_Graph_NX,
    'allenai/OLMo-2-0425-1B': Olmo2_Graph_NX,
    "allenai/OLMo-2-1124-7B": Olmo2_Graph_NX,
    "allenai/OLMo-2-1124-13B": Olmo2_Graph_NX,
    "Qwen/Qwen3-0.6B-Base": Qwen3_Graph_NX,
    "Qwen/Qwen3-1.7B-Base": Qwen3_Graph_NX,
    "Qwen/Qwen3-4B-Base": Qwen3_Graph_NX,
    "Qwen/Qwen3-8B-Base": Qwen3_Graph_NX,
}

def get_graph_constructor(model_name):
    if model_name in GRAPH_CONSTRUCTOR:
        return GRAPH_CONSTRUCTOR[model_name]
    else:
        raise NotImplementedError(f"LLM_Graph not implemented for model {model_name}.")
