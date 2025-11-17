from src.llm_hooked.mistral_hooked import Mistral_Hooked
from src.llm_hooked.olmo_hooked import Olmo2_Hooked
from src.llm_hooked.qwen_hooked import Qwen3_Hooked, Qwen2_Hooked
from src.llm_hooked.gemma_hooked import Gemma3_Hooked

HOOKED_CONSTRUCTOR = {
    'mistralai/Mistral-7B-v0.1': Mistral_Hooked,
    'allenai/OLMo-2-0425-1B': Olmo2_Hooked,
    'allenai/OLMo-2-1124-7B': Olmo2_Hooked,
    "allenai/OLMo-2-1124-13B": Olmo2_Hooked,
    "Qwen/Qwen3-0.6B-Base": Qwen3_Hooked,
    "Qwen/Qwen3-1.7B-Base": Qwen3_Hooked,
    "Qwen/Qwen3-4B-Base": Qwen3_Hooked,
    "Qwen/Qwen3-8B-Base": Qwen3_Hooked,
    "google/gemma-3-270m": Gemma3_Hooked,
    "Qwen/Qwen2.5-0.5B": Qwen2_Hooked,
    "Qwen/Qwen2.5-1.5B": Qwen2_Hooked,
    "Qwen/Qwen2.5-3B": Qwen2_Hooked,
    "Qwen/Qwen2.5-7B": Qwen2_Hooked,
    "Qwen/Qwen2.5-14B": Qwen2_Hooked,
    "Qwen/Qwen2.5-32B": Qwen2_Hooked,
    "Qwen/Qwen2-0.5B": Qwen2_Hooked,
    "Qwen/Qwen2-1.5B": Qwen2_Hooked,
    "Qwen/Qwen2-7B": Qwen2_Hooked,
}

def get_hooked_constructor(model_name):
    if model_name in HOOKED_CONSTRUCTOR:
        return HOOKED_CONSTRUCTOR[model_name]
    else:
        raise NotImplementedError(f"LLM_Hooked not implemented for model {model_name}.")
