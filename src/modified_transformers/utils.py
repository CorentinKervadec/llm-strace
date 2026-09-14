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
    elif "gemma-3" in model_name:
        return "gemma3"
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

def load_gemma3(model_name, half_precision, cuda):
    """
    Loading a pretrained Gemma3 can cause problems because of "language_model" missing in the mapping keys.
    """
    # 1. Load the official model (on CPU to save GPU memory during the conversion)
    print("Loading official checkpoint...")
    official_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="cpu",
        torch_dtype=torch.float16 if half_precision else torch.float32,
    )
    official_state_dict = official_model.state_dict()

    # 2. Create the mapping by renaming the keys
    print("Mapping checkpoint keys...")
    mapped_state_dict = {}
    for key, tensor in official_state_dict.items():
        # The official checkpoint has keys like: model.language_model.layers.4...
        # Our custom model expects: model.layers.4...
        if "language_model." in key:
            new_key = key.replace("language_model.", "")
            mapped_state_dict[new_key] = tensor
        else:
            # Pass through any keys that don't need changing (e.g., lm_head.weight)
            mapped_state_dict[key] = tensor

    # Free up memory by deleting the original model object
    del official_model

    # 3. Load the config and force eager attention
    multimodal_config = AutoConfig.from_pretrained(
        model_name,
        attn_implementation="eager"
    )
    # Extract the purely text-based configuration!
    text_config = multimodal_config.text_config
    # (Optional but safe) Ensure the eager attention flag trickles down to the text config
    text_config._attn_implementation = "eager"

    # 4. Instantiate our custom model from the configuration (initializes with random weights)
    print("Instantiating custom architecture...")
    from .modeling_gemma3 import Gemma3ForCausalLM
    # We temporarily tell PyTorch to create new tensors in float16. 
    # If we don't do this, Gemma3ForCausalLM(config) creates an empty model in float32 
    # which takes up ~16GB of RAM and might crash your system before loading weights.
    torch.set_default_dtype(torch.float16)
    custom_model = Gemma3ForCausalLM(text_config)
    torch.set_default_dtype(torch.float32) # Revert back to default float32 just to be safe
    print(f"Successfully imported Gemma3ForCausalLM")

    # 5. Load the remapped weights into our custom model
    print("Loading mapped weights into custom model...")
    # strict=False is highly recommended here! It tells PyTorch to ignore any 
    # vision-related weights (like vision encoders) from the official checkpoint 
    # that do not exist in your pure-text CausalLM.
    missing_keys, unexpected_keys = custom_model.load_state_dict(mapped_state_dict, strict=False)

    print("Missing keys (should be empty or expected):", missing_keys)
    print("Model loaded successfully!")

    # Move to GPU if needed
    if cuda:
        custom_model = custom_model.to("cuda")
    
    return custom_model