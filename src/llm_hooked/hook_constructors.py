from src.llm_hooked.mistral_hooked import Mistral_Hooked
from src.llm_hooked.olmo_hooked import Olmo2_Hooked
from src.llm_hooked.qwen_hooked import Qwen3_Hooked

HOOKED_CONSTRUCTOR = {
    'mistralai/Mistral-7B-v0.1': Mistral_Hooked,
    'allenai/OLMo-2-0425-1B': Olmo2_Hooked,
    'allenai/OLMo-2-1124-7B': Olmo2_Hooked,
    "allenai/OLMo-2-1124-13B": Olmo2_Hooked,
    "Qwen/Qwen3-0.6B-Base": Qwen3_Hooked,
    "Qwen/Qwen3-1.7B-Base": Qwen3_Hooked,
    "Qwen/Qwen3-4B-Base": Qwen3_Hooked,
    "Qwen/Qwen3-8B-Base": Qwen3_Hooked,
}

def get_hooked_constructor(model_name):
    if model_name in HOOKED_CONSTRUCTOR:
        return HOOKED_CONSTRUCTOR[model_name]
    else:
        raise NotImplementedError(f"LLM_Hooked not implemented for model {model_name}.")
