try:
    from .modeling_olmo2 import Olmo2ForCausalLM, Olmo2Model, Olmo2PreTrainedModel
    _has_olmo2 = True
except ImportError:
    _has_olmo2 = False

try:
    from .modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model, Qwen3PreTrainedModel
    _has_qwen3 = True
except ImportError:
    _has_qwen3 = False

try:
    from .modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model, Qwen2PreTrainedModel
    _has_qwen2 = True
except ImportError:
    _has_qwen2 = False

try:
    from .modeling_gemma3 import Gemma3ForCausalLM, Gemma3TextModel, Gemma3PreTrainedModel
    _has_gemma3 = True
except ImportError:
    _has_gemma3 = False

try:
    from .modeling_llama import LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel
    _has_llama = True
except ImportError:
    _has_llama = False

try:
    from .modeling_mistral import MistralForCausalLM, MistralModel, MistralPreTrainedModel
    _has_mistral = True
except ImportError:
    _has_mistral = False

try:
    from .modeling_phi3 import Phi3ForCausalLM, Phi3Model, Phi3PreTrainedModel
    _has_phi3 = True
except ImportError:
    _has_phi3 = False


# Build list of available models
__all__ = []

if _has_olmo2:
    __all__.extend(["Olmo2ForCausalLM", "Olmo2Model", "Olmo2PreTrainedModel"])

if _has_qwen3:
    __all__.extend(["Qwen3ForCausalLM", "Qwen3Model", "Qwen3PreTrainedModel"])

if _has_qwen2:
    __all__.extend(["Qwen2ForCausalLM", "Qwen2Model", "Qwen2PreTrainedModel"])

if _has_gemma3:
    __all__.extend(["Gemma3ForCausalLM", "Gemma3TextModel", "Gemma3PreTrainedModel"])

if _has_llama:
    __all__.extend(["LlamaForCausalLM", "LlamaModel", "LlamaPreTrainedModel"])

if _has_mistral:
    __all__.extend(["MistralForCausalLM", "MistralModel", "MistralPreTrainedModel"])

if _has_phi3:
    __all__.extend(["Phi3ForCausalLM", "Phi3Model", "Phi3PreTrainedModel"])