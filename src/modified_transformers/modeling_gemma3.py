# Modified by Corentk

# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

import torch.nn.init as torch_init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs, can_return_tuple, logging
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config, Gemma3TextConfig

# === MODIFICATION START: Custom Imports for Mechanistic Interpretability ===
from typing import Optional, Union, Tuple
from src.llm_graph.llm_graph_nx_light import LLM_Graph_NX
from src.llm_hooked.utils import get_real_weight_from_offloaded_module, linearize_rms_norm, apply_linearized_norm
from src.test.unit_tests import test_reconstruction, CausalLeakageError
import inspect
# === MODIFICATION END ===

logger = logging.get_logger(__name__)

# === MODIFICATION START: Batch Size for Attention Decomposition ===
# Batch sizes for iteration (adjust these if you still hit OOM or want faster CPU transfers)
DECOMPOSE_BATCH_SIZE = 16
# === MODIFICATION END ===

# === MODIFICATION START: Replace HF init with the pytorch one ===
class init:
    """A shim to backport HF's new initialization module to older transformers versions."""
    zeros_ = torch_init.zeros_
    constant_ = torch_init.constant_
    normal_ = torch_init.normal_
    
    @staticmethod
    def copy_(dst, src):
        with torch.no_grad():
            dst.copy_(src)
# === MODIFICATION END ===

@dataclass
# === MODIFICATION START: Comment out missing decorator ===
# @auto_docstring  removed because it was causing an error
# === MODIFICATION END ===
class Gemma3ModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: torch.FloatTensor | None = None


@dataclass
# === MODIFICATION START: Comment out missing decorator ===
# @auto_docstring  removed because it was causing an error
# === MODIFICATION END ===
class Gemma3CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    # === MODIFICATION START: Add mask application to MLP ===
    def forward(self, x, mask=None):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # Apply the mask if provided: zero out masked positions along the sequence dimension
        if mask is not None:
            down_proj = torch.einsum('bsd,s->bsd', down_proj, mask.to(down_proj.device).to(down_proj.dtype))
        return down_proj
    # === MODIFICATION END ===


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma3TextConfig, device=None, layer_type=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.layer_types = list(set(config.layer_types))
        self.rope_type = {}
        for layer_type in self.layer_types:
            # rope_params = self.config.rope_parameters[layer_type]
            # if rope_params is None:
            #     continue

            # self.rope_type[layer_type] = rope_params["rope_type"]

            # === MODIFICATION START: Handle dictionary fallback for rope_scaling ===
            # self.rope_type = self.config.rope_parameters["rope_type"]
            # Modified because 'Olmo2Config' object has no attribute 'rope_parameters'
            if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
                self.rope_type[layer_type] = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                continue
            # === MODIFICATION END ===

            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]

            # === MODIFICATION START: Safely handle layer_type argument ===
            # curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, device, layer_type=layer_type)
            sig = inspect.signature(rope_init_fn)
            
            if 'layer_type' in sig.parameters:
                # Use it for Gemma 3 specific RoPE functions
                curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, device, layer_type=layer_type)
            else:
                # Fallback for standard HF RoPE functions (like "linear")
                curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, device)
            # === MODIFICATION END ===
            
            self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: Gemma3TextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    # === MODIFICATION START: Remove Dynamic Rope ===
    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    # === MODIFICATION END ===
    def forward(self, x, position_ids, layer_type=None, **kwargs):
        
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        # === MODIFICATION START: Autocast Fix ===
        # replace maybe_autocast with torch.autocast
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        # === MODIFICATION END ===
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# === MODIFICATION START: Remove kernelized decorator ===
# @use_kernelized_func(rotary_pos_emb) # remove because it cannot be imported
# === MODIFICATION END ===
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    # === MODIFICATION START: Graph Extraction Arguments ===
    extract_intermediate: bool = False,
    custom_mask = None,
    unit_test:bool = False,
    # === MODIFICATION END ===
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # === MODIFICATION START: Scale before Matmul to avoid Nan in float16 ===
    # attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    scaled_query = query * scaling
    attn_weights = torch.matmul(scaled_query, key_states.transpose(2, 3))
    if unit_test and attn_weights.dtype==torch.float32:
        attn_weights_original = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        test_reconstruction(attn_weights_original, attn_weights, '[MODELING_QWEN2][DECOMPOSE][ATT_WEIGTH]', 'strict')
    # === MODIFICATION END ===
    
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # === MODIFICATION START: Post-Softmax Attention Masking ===
    # Masking after softmax, zero out masked positions
    if custom_mask:
        reverse_custom_mask = (custom_mask == 0).to(attn_weights.dtype)  # 0 means not masked (kept)
        # Reshape from (Seq_Len,) to (1, 1, Seq_Len, 1)
        mask_aligned = reverse_custom_mask.view(1, 1, -1, 1)
        attn_weights = attn_weights * mask_aligned.to(attn_weights.device)
    # === MODIFICATION END ===

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    # === MODIFICATION START: Decompose Attention Output ===
    outputs = (attn_output, attn_weights,)

    """ Decompose the attention output into per-head and per-token output """
    # attn_weights: (batch, number of heads, tokens_y, tokens_x)
    # value_states: (batch, number of heads, tokens_x, head dimension)
    # decomposed_output: (batch, number of heads, tokens_y, tokens_x, head dimension)
    if extract_intermediate:
        batch_size, num_heads, tokens_y, tokens_x = attn_weights.shape
        _, _, _, head_dim = value_states.shape
        
        all_head_outputs = []
        
        for y_start in range(0, tokens_y, DECOMPOSE_BATCH_SIZE):
            y_end = min(y_start + DECOMPOSE_BATCH_SIZE, tokens_y)
            attn_weights_y = attn_weights[:, :, y_start:y_end, :]

            head_outputs_y = []
            for x_start in range(0, tokens_x, DECOMPOSE_BATCH_SIZE):
                x_end = min(x_start + DECOMPOSE_BATCH_SIZE, tokens_x)
                
                # Extract chunk and cast to float32 only for the active block
                attn_weights_chunk = attn_weights_y[:, :, :, x_start:x_end].to(torch.float32)
                value_chunk = value_states[:, :, x_start:x_end, :].to(torch.float32)
                
                # Perform batched computation
                # shape: (batch, number of heads, y_chunk, x_chunk, head dimension)
                chunk_output = torch.einsum(
                    'bhyx,bhxd->bhyxd', 
                    attn_weights_chunk, 
                    value_chunk
                )
                
                # Move to the CPU immediately to free up GPU VRAM
                head_outputs_y.append(chunk_output.cpu())

            # Concatenate all x batches along the x dimension (dim=3)
            head_outputs_y_cat = torch.cat(head_outputs_y, dim=3)
            all_head_outputs.append(head_outputs_y_cat)

        # Concatenate all y batches along the y dimension (dim=2)
        decomposed_output = torch.cat(all_head_outputs, dim=2)
    else:
        decomposed_output = None
    
    outputs += (decomposed_output,)
    # === MODIFICATION END ===
    
    return outputs



# === MODIFICATION START: Remove kernelized decorator ===
# @use_kernelized_func(apply_rotary_pos_emb)
# === MODIFICATION END ===
class Gemma3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = not self.config.use_bidirectional_attention

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.is_sliding = self.layer_type == "sliding_attention"

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        # === MODIFICATION START: Add Graph and Extraction Args ===
        extract_intermediate: bool = False,
        unit_test: bool = False,
        mask = None,
        # === MODIFICATION END ===
        **kwargs: Unpack[TransformersKwargs],
    # === MODIFICATION START: Return decomposed intermediate dict ===
    ) -> tuple[torch.Tensor, torch.Tensor | None, Optional[dict]]:
    # === MODIFICATION END ===

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # === MODIFICATION START: Force Eager Attention & Pass Masking Args ===
        # Only eager implementation supports extraction
        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights, decomposed_output = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            extract_intermediate=extract_intermediate,
            mask=mask,
            unit_test=unit_test,
            **kwargs,
        )
        # === MODIFICATION END ===

        # === MODIFICATION START: Validation Tests for Decomposed Output ===
        if extract_intermediate:
            if unit_test:
                # decomposed_output: (batch, number of heads, tokens_y, tokens_x, head dimension)
                # attn_output: (batch, tokens_y, number of heads, head dimension)
                reconstructed_output = decomposed_output.sum(-2).transpose(1,2).to(attn_output.dtype)
                test_reconstruction(attn_output.cpu(), reconstructed_output, '[MODELING_QWEN3][DECOMPOSE][EAGER_ATT]', 'strict' if attn_output.dtype == torch.float32 else 'loose')
                # Check that upper triangular part (future tokens) is zero
                seq_len = decomposed_output.size(2)
                for y in range(seq_len):
                    if y + 1 < seq_len:
                        future_components = decomposed_output[:, :, y, y+1:, :]
                        if not torch.all(future_components == 0):
                            raise CausalLeakageError(f"CAUSAL LEAKAGE: Token {y} is attending to future tokens!")

            decomposed_output = decomposed_output.to(attn_output.dtype) # when in float16, this cast may cause wrong approx.
        # === MODIFICATION END ===

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        # === MODIFICATION START: Project Decomposed Output through O_Proj in Batches ===
        if extract_intermediate:
            o_weights = get_real_weight_from_offloaded_module(self.o_proj)
            d_hidden = o_weights.shape[0]
            d_batch, n_heads, d_seq_y, d_seq_x, d_h_head = decomposed_output.shape

            # Reshape weights: [head, hidden_dim, head_dim]
            device = hidden_states.device
            o_weights_reshaped = o_weights.to(device)
            o_weights_reshaped = o_weights_reshaped.reshape(d_hidden, n_heads, d_h_head) 
            o_weights_reshaped = o_weights_reshaped.transpose(0, 1).contiguous() 

            all_projected_y = []
            
            for y_start in range(0, d_seq_y, DECOMPOSE_BATCH_SIZE):
                y_end = min(y_start + DECOMPOSE_BATCH_SIZE, d_seq_y)
                # Slice the y-dimension from the CPU tensor
                chunk_y = decomposed_output[:, :, y_start:y_end, :, :]
                
                projected_x = []
                for x_start in range(0, d_seq_x, DECOMPOSE_BATCH_SIZE):

                    x_end = min(x_start + DECOMPOSE_BATCH_SIZE, d_seq_x)
                    
                    # 1. Slice x-dimension and move ONLY this chunk to the GPU
                    chunk = chunk_y[:, :, :, x_start:x_end, :].to(o_weights_reshaped.device)
                    
                    # 2. Project into the residual stream hidden dimension
                    # chunk shape: (batch, number of heads, tokens_y, tokens_x, head_dim)
                    # output shape: (batch, number of heads, tokens_y, tokens_x, hidden_dim)
                    projected_chunk = torch.einsum(
                        'bnyxd,nhd->bnyxh', 
                        chunk, 
                        o_weights_reshaped
                    )
                    
                    # 3. Move the heavy projected chunk immediately back to CPU
                    projected_x.append(projected_chunk.cpu())
                    
                # Concatenate along the x dimension (dim=3)
                projected_y_cat = torch.cat(projected_x, dim=3)
                all_projected_y.append(projected_y_cat)
                
            # Concatenate all chunks along the y dimension (dim=2)
            decomposed_output = torch.cat(all_projected_y, dim=2)
            
            # Transpose axes on the CPU to achieve final shape: 
            # (batch, tokens_y, tokens_x, number of heads, hidden dimension)
            decomposed_output = decomposed_output.transpose(1, 2).transpose(2, 3)
            
        else:
            decomposed_output = None

        if extract_intermediate and unit_test:
            # decomposed_output: (batch,  tokens_y, tokens_x, number of heads, hidden dimension)
            # attn_output: (batch, tokens_y, hidden dimension)
            reconstructed_output = decomposed_output.sum(2).sum(2).to(attn_output.dtype)
            test_reconstruction(attn_output.cpu(), reconstructed_output.cpu(), '[MODELING_QWEN3][DECOMPOSE][OUT_ATT]', atol=1e-4 if attn_output.dtype == torch.float32 else 0.15)

        return attn_output, attn_weights, decomposed_output
        # === MODIFICATION END ===


class Gemma3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        # === MODIFICATION START: Add Extraction/Graph Masking Args to Forward ===
        extract_intermediate: bool = False,
        unit_test: bool = False,
        graph_mask: dict | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        
        intermediate_representations = {}
        
        residual = hidden_states

        """ BEFORE ATTENTION """
        if extract_intermediate:                
            intermediate_representations['before_attn'] = hidden_states.detach().cpu().clone()

        """ HEAD_OUT and ATTN_OUT"""
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _, decomposed_attn = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            extract_intermediate=extract_intermediate,
            unit_test=unit_test,
            mask=graph_mask['attention'] if graph_mask is not None else None,
            **kwargs,
        )

        if extract_intermediate:
            # TODO
            inv_std, weight = linearize_rms_norm(self.post_attention_layernorm, hidden_states) 
            b, y, _ = inv_std.shape
            inv_std_broadcast = inv_std.view(b, y, 1, 1, 1)
            device = residual.device
            intermediate_representations['head_out'] = apply_linearized_norm(
                decomposed_attn.to(device), 
                inv_std_broadcast.to(device), 
                weight.to(device)
            ).cpu().detach().clone()

        hidden_states = self.post_attention_layernorm(hidden_states)

        if extract_intermediate and unit_test:
            # decomposed_output: (batch,  tokens_y, tokens_x, number of heads, hidden dimension)
            # attn_output: (batch, tokens_y, hidden dimension)
            reconstructed_output = intermediate_representations['head_out'].sum(2).sum(2).to(hidden_states.dtype)
            test_reconstruction(hidden_states.cpu(), reconstructed_output, '[MODELING_GEMMA3][DECOMPOSE][NORM_ATT]', 'strict' if hidden_states.dtype == torch.float32 else 'loose')

        if extract_intermediate:
            intermediate_representations['attn_out'] = hidden_states.detach().cpu().clone()

        # Apply attention residual mask if provided
        if graph_mask is not None:
            residual = torch.einsum('bsd,s->bsd', residual, graph_mask['residual-attention'].to(residual.device).to(residual.dtype))

        hidden_states = residual + hidden_states

        """ BEFORE_MLP """
        if extract_intermediate:
            intermediate_representations['before_mlp'] = hidden_states.detach().cpu().clone()

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states,  mask=graph_mask['mlp'] if graph_mask is not None else None)
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        """ MLP_OUT """
        if extract_intermediate:
            intermediate_representations['mlp_out'] = hidden_states.detach().cpu().clone()
            
        # Apply MLP residual mask if provided
        if graph_mask is not None:
            residual = torch.einsum('bsd,s->bsd', residual, graph_mask['residual-mlp'].to(residual.device).to(residual.dtype))

        hidden_states = residual + hidden_states

        """ AFTER_MLP """
        if extract_intermediate:
            intermediate_representations['after_mlp'] = hidden_states.detach().cpu().clone()
            
        if extract_intermediate and unit_test:
            test_reconstruction(
                intermediate_representations['before_mlp'],
                intermediate_representations['before_attn'] + intermediate_representations['attn_out'],
                '[MODELING_OLMO2][DECODER_LAYER][AFTER ATTENTION]',
                'strict'
            )
            test_reconstruction(
                intermediate_representations['after_mlp'],
                intermediate_representations['before_mlp'] + intermediate_representations['mlp_out'],
                '[MODELING_OLMO2][DECODER_LAYER][AFTER MLP]',
                'strict'
            )
            test_reconstruction(
                intermediate_representations['attn_out'],
                intermediate_representations['head_out'].sum(2).sum(2),
                '[MODELING_OLMO2][DECODER_LAYER][ATTENTION]', 'strict' if hidden_states.dtype == torch.float32 else 'loose')
            
        if extract_intermediate:
            return hidden_states, intermediate_representations
        else:
            return hidden_states
        # === MODIFICATION END ===


# === MODIFICATION START: Comment out missing decorator ===
# @auto_docstring  removed because it was causing an error
# === MODIFICATION END ===
class Gemma3PreTrainedModel(PreTrainedModel):
    config: Gemma3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Gemma3DecoderLayer",
        "SiglipVisionEmbeddings",
        "SiglipEncoderLayer",
        "SiglipMultiheadAttentionPoolingHead",
    ]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Gemma3DecoderLayer,
        "attentions": Gemma3Attention,
    }
    input_modalities = ("image", "text")

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # === MODIFICATION START: Remove bc we are using text only ===
        # if isinstance(module, Gemma3MultiModalProjector):
        #     init.zeros_(module.mm_input_projection_weight)
        # === MODIFICATION END ===
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        if "RMSNorm" in module.__class__.__name__:
            init.zeros_(module.weight)
        elif isinstance(module, Gemma3TextScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        elif isinstance(module, Gemma3RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]

                # === MODIFICATION START: Safely handle layer_type argument ===
                # curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                sig = inspect.signature(rope_init_fn)
                
                if 'layer_type' in sig.parameters:
                    # Use it for Gemma 3 specific RoPE functions
                    curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                else:
                    # Fallback for standard HF RoPE functions (like "linear")
                    curr_inv_freq, _ = rope_init_fn(module.config)
                # === MODIFICATION END ===

                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


def _bidirectional_window_overlay(sliding_window: int) -> Callable[[int, int, int, int], bool]:
    """
    Enables a bidirectional mask within the sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """A token can attend to any other token if their absolute distance is within
        the (exclusive) sliding window size (distance < sliding_window)."""
        return abs(q_idx - kv_idx) < sliding_window

    return inner_mask


# === MODIFICATION START: Comment out missing decorator ===
# @auto_docstring  removed because it was causing an error
# === MODIFICATION END ===
class Gemma3TextModel(Gemma3PreTrainedModel):
    config: Gemma3TextConfig
    input_modalities = ("text",)

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    # === MODIFICATION START: Remove unavailable decorators & update forward signature ===
    # @merge_with_config_defaults removed because impossible to import
    # @capture_outputs removed because impossible to import
    # @auto_docstring removed because it was causing an error
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        build_graph: str | None = None,
        unit_test: bool = False,
        graph_mask: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[BaseModelOutputWithPast, Tuple[BaseModelOutputWithPast, LLM_Graph_NX]]:
        
        """ Prepare the graph """
        extract_intermediate = False
        if build_graph:
            # To build the graph, we need to extract intermediate representations
            extract_intermediate = True
            intermediate_representations = None
            # Initialise the graph
            graph = LLM_Graph_NX(
                architecture_type='sequential',
                n_layers=self.config.num_hidden_layers,
                n_heads=self.config.num_attention_heads,
                input_sentence=input_ids, 
                importance_mode=build_graph)
            # Add input (layer 0) nodes
            # input nodes corresponds to the input embeddings
            for token in range(graph.get_n_tokens()):
                node_idx = graph.node_idx(0, token)
                graph.add_node(node_idx)
                graph.add_input_node(node_idx)
        # === MODIFICATION END ===

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # === MODIFICATION START: Cache Position Fallback ===
        # added because it is necessary for create_causal mask
        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        # === MODIFICATION END ===

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                # === MODIFICATION START: Fix argument name ===
                "input_embeds": inputs_embeds, # inputs_embeds -> input_embeds
                # === MODIFICATION END ===
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                # === MODIFICATION START: Adding cache position ===
                "cache_position": cache_position,
                # === MODIFICATION END ===
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            if self.config.use_bidirectional_attention:
                mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
                sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(self.config.sliding_window)

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }

        # === MODIFICATION START: LLM Strace Only Support BATCH-SIZE=1 ===
        if build_graph:
            assert input_ids.shape[0] == 1, "[LLM_STRACE] Graph building currently only supports batch size 1."
        # === MODIFICATION END ===

        # embed positions
        hidden_states = inputs_embeds
        # === MODIFICATION START: Remove Layer-Type Specific Position Embedding ===
        position_embeddings = {}
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type=layer_type)
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # === MODIFICATION END ===

        # === MODIFICATION START: Iterate Layers and Build Graph Edges ===
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            
            if build_graph:
                # add the previous layer to the graph
                if intermediate_representations:    
                    # take [0] to ignore the batch dim
                    graph.add_edges_from_one_layer(
                        layer_index = i-1,
                        before_attn = intermediate_representations['before_attn'][0],
                        head_out = intermediate_representations['head_out'][0],
                        attn_out = intermediate_representations['attn_out'][0],
                        before_mlp = intermediate_representations['before_mlp'][0],
                        mlp_out = intermediate_representations['mlp_out'][0],
                        after_mlp = intermediate_representations['after_mlp'][0],
                    )
            
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                extract_intermediate=extract_intermediate,
                unit_test=unit_test,
                graph_mask=graph_mask[i] if graph_mask else None, # send the mask for this layer
                **kwargs,
            )
            if extract_intermediate:
                hidden_states, intermediate_representations = hidden_states

        hidden_states = self.norm(hidden_states)

        if build_graph:
            """ We treat the last layer separatly, in order to apply the layer norm """
            inv_std, weight = linearize_rms_norm(self.norm, intermediate_representations['after_mlp'])
            device = hidden_states.device
            for rep in ['before_mlp', 'mlp_out', 'after_mlp']:
                intermediate_representations[rep] = apply_linearized_norm(
                    intermediate_representations[rep].to(device),
                    inv_std.to(device), weight.to(device),
                ).cpu().detach().clone()
                
            if unit_test:
                test_reconstruction(
                    intermediate_representations['after_mlp'],
                    intermediate_representations['before_mlp'] + intermediate_representations['mlp_out'],
                    '[MODELING_QWEN3][DECODER_LAYER][FINAL NORM]', 'strict' if hidden_states.dtype == torch.float32 else 'loose')
                test_reconstruction(
                    hidden_states.cpu(),
                    intermediate_representations['after_mlp'],
                    '[MODELING_QWEN3][DECODER_LAYER][FINAL NORM 2]', 'strict')
            for rep in ['before_mlp', 'mlp_out', 'after_mlp']:
                intermediate_representations[rep] = intermediate_representations[rep].to(hidden_states.dtype)
            
            # take [0] to ignore the batch dim
            graph.add_edges_from_one_layer(
                layer_index = i,
                before_attn = intermediate_representations['before_attn'][0],
                head_out = intermediate_representations['head_out'][0],
                attn_out = intermediate_representations['attn_out'][0],
                before_mlp = intermediate_representations['before_mlp'][0],
                mlp_out = intermediate_representations['mlp_out'][0],
                after_mlp = intermediate_representations['after_mlp'][0],
            )
            # add the final node to the graph root
            graph.set_output_node(graph.node_idx(
                1 + 2 * i + 1, graph.get_n_tokens()-1))

        base_output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

        if build_graph:
            return base_output, graph
        else:
            return base_output
        # === MODIFICATION END ===

# === MODIFICATION START: Comment out missing decorator ===
# @auto_docstring  removed because it was causing an error
# === MODIFICATION END ===
class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    # === MODIFICATION START: Remove decorator and update signature ===
    # @auto_docstring removed because it was causing an error
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        build_graph: str | None = None,
        unit_test: bool = False,
        graph_mask: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[CausalLMOutputWithPast, Tuple[CausalLMOutputWithPast, LLM_Graph_NX]]:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

        >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            build_graph=build_graph,
            unit_test=unit_test,
            graph_mask=graph_mask,
            **kwargs,
        )

        if build_graph:
            outputs, graph = outputs
        # === MODIFICATION END ===

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        # === MODIFICATION START: Repackage Causal Output and return Optional Graph ===
        causal_output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        if build_graph:
            return causal_output, graph
        else:
            return causal_output
        # === MODIFICATION END ===

__all__ = [
    "Gemma3PreTrainedModel",
    "Gemma3TextModel",
    "Gemma3ForCausalLM",
]

# "model.language_model.layers.4.self_attn.q_proj.weight"