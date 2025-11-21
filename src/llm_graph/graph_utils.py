from src.llm_graph.mistral_graph_nx import Mistral_Graph_NX
from src.llm_graph.olmo_graph_nx import Olmo2_Graph_NX
from src.llm_graph.qwen_graph_nx import Qwen_Graph_NX
from src.llm_graph.gemma_graph_nx import Gemma3_Graph_NX

GRAPH_CONSTRUCTOR = {
    'mistralai/Mistral-7B-v0.1': Mistral_Graph_NX,
    'allenai/OLMo-2-0425-1B': Olmo2_Graph_NX,
    "allenai/OLMo-2-1124-7B": Olmo2_Graph_NX,
    "allenai/OLMo-2-1124-13B": Olmo2_Graph_NX,
    "Qwen/Qwen3-0.6B-Base": Qwen_Graph_NX,
    "Qwen/Qwen3-1.7B-Base": Qwen_Graph_NX,
    "Qwen/Qwen3-4B-Base": Qwen_Graph_NX,
    "Qwen/Qwen3-8B-Base": Qwen_Graph_NX,
    "google/gemma-3-270m": Gemma3_Graph_NX,
    "Qwen/Qwen2.5-0.5B": Qwen_Graph_NX,
    "Qwen/Qwen2.5-1.5B": Qwen_Graph_NX,
    "Qwen/Qwen2.5-3B": Qwen_Graph_NX,
    "Qwen/Qwen2.5-7B": Qwen_Graph_NX,
    "Qwen/Qwen2.5-14B": Qwen_Graph_NX,
    "Qwen/Qwen2.5-32B": Qwen_Graph_NX,
    "Qwen/Qwen2-0.5B": Qwen_Graph_NX,
    "Qwen/Qwen2-1.5B": Qwen_Graph_NX,
    "Qwen/Qwen2-7B": Qwen_Graph_NX,
}

def get_graph_constructor(model_name):
    for key_model, constructor in GRAPH_CONSTRUCTOR.items():
        if model_name.startswith(key_model):
            return constructor
    raise NotImplementedError(f"LLM_Graph not implemented for model {model_name}.")