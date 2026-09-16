import torch
from transformers import AutoModelForCausalLM, AutoConfig

def get_model_class(model_type):
    """Get corresponding model class based on model type"""
    try:
        if model_type == "olmo2":
            from .modeling_olmo2 import Olmo2ForCausalLM
            print(f"Successfully imported Olmo2ForCausalLM")
            return Olmo2ForCausalLM
        elif model_type == "qwen3":
            from .modeling_qwen3 import Qwen3ForCausalLM
            print(f"Successfully imported Qwen3ForCausalLM")
            return Qwen3ForCausalLM
        elif model_type == "qwen2":
            from .modeling_qwen2 import Qwen2ForCausalLM
            print(f"Successfully imported Qwen2ForCausalLM")
            return Qwen2ForCausalLM
        elif model_type == "gemma3":
            from .modeling_gemma3 import Gemma3ForCausalLM
            print(f"Successfully imported Gemma3ForCausalLM")
            return Gemma3ForCausalLM
        elif model_type == "llama":
            from .modeling_llama import LlamaForCausalLM
            print(f"Successfully imported LlamaForCausalLM")
            return LlamaForCausalLM
        elif model_type == "mistral":
            from .modeling_mistral import MistralForCausalLM
            print(f"Successfully imported MistralForCausalLM")
            return MistralForCausalLM
        elif model_type == "phi3":
            from .modeling_phi3 import Phi3ForCausalLM
            print(f"Successfully importedPhi3ForCausalLM")
            return Phi3ForCausalLM
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except ImportError as e:
        print(f"Failed to import {model_type} model: {e}")
        raise
    except Exception as e:
        print(f"Unknown error occurred while getting {model_type} model class: {e}")
        raise

def identify_model_type(model_path: str) -> str:
    """Automatically identify model type"""
    model_name = model_path.lower()
    
    if "olmo-2" in model_name:
        return "olmo2"
    elif "qwen3-" in model_name:
        return "qwen3"
    elif "qwen2" in model_name:
        return "qwen2"
    elif "llama" in model_name:
        return "llama"
    elif "mistral" in model_name:
        return "mistral"
    elif "deepseek-llm" in model_name:
        return "llama" # deepseek llm use the llama code (LlamaForCausalLM)
    elif "phi-4" in model_name:
        return "phi3" # phi4 is based on phi3 code
    else:
        return None
