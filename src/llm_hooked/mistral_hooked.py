from src.llm_hooked.llm_hooked import LLM_Hooked
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from typing import Callable
import torch
import src.llm_hooked.sanity_checks as sanity_check
from src.llm_hooked.utils import linearize_rms_norm, apply_rotary_pos_emb, eager_attention_forward, MLP_masked, repeat_kv

ATTENTION_MASK_VALUE = -65504

def MistralAttention_masked(
    module,
    hidden_states,
    position_embeddings,
    attention_mask,
    mask,
    mask_before_softmax=False):
    """
    Masked self-attention forward pass for Mistral models.

    Args:
    module: The attention module.
    hidden_states: Input tensor of shape [batch, seq_len, hidden_dim].
    position_embeddings: Tuple of (cos, sin) rotary embeddings.
    attention_mask: Standard attention mask (e.g., causal mask).
    mask: Custom mask for masking specific attention connections.
    mask_before_softmax: If True, apply custom mask before softmax.

    Returns:
    attn_output: Output of the attention layer after masking.
    attn_weights: Attention weights after masking and softmax.
    """
    # Compute input and reshaped hidden shapes
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    # Project hidden states to query, key, and value tensors
    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [batch, num_head, seq_target, head_dim]
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # [batch, num_head, seq_source, head_dim]
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [batch, num_head, seq_source, head_dim]

    # Apply rotary positional embeddings to queries and keys
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Compute attention output and weights with masking
    attn_output, attn_weights = eager_attention_forward(
    module,
    query_states,
    key_states,
    value_states,
    attention_mask,
    scaling=module.scaling,
    custom_mask=mask,
    mask_before_softmax=mask_before_softmax,
    )

    # Reshape attention output back to [batch, seq_len, hidden_dim]
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    # Final linear projection
    attn_output = module.o_proj(attn_output)
    return attn_output, attn_weights

def MistralDecoder_masked(
    module,
    hidden_states,
    attention_mask,
    position_embeddings,
    graph_attn_mask,
    mask_before_softmax,
    graph_mlp_mask,
    attn_residual_mask,
    mlp_residual_mask):
    """
    Forward pass for a masked Mistral decoder block.

    Args:
        module: The decoder layer module.
        hidden_states: Input tensor of shape [batch, seq_len, hidden_dim].
        attention_mask: Standard attention mask (e.g., causal mask).
        output_attentions: If True, return attention weights.
        position_embeddings: Tuple of (cos, sin) rotary embeddings.
        graph_attn_mask: Custom attention mask for masking specific attention connections.
        mask_before_softmax: If True, apply custom mask before softmax.
        graph_mlp_mask: Mask tensor for the MLP.
        attn_residual_mask: Mask for the attention residual connection.
        mlp_residual_mask: Mask for the MLP residual connection.

    Returns:
        outputs: Tuple containing the output hidden states (and optionally attention weights).
    """
    hidden_dtype = hidden_states.dtype
    residual = hidden_states
    # Layer normalization before self-attention
    hidden_states = module.input_layernorm(hidden_states)

    # Self Attention with masking
    hidden_states, self_attn_weights = MistralAttention_masked(
        module.self_attn,
        hidden_states,
        position_embeddings,
        attention_mask,
        graph_attn_mask,
        mask_before_softmax)

    # Apply attention residual mask if provided
    if attn_residual_mask is not None:
        residual = torch.einsum('bsd,s->bsd', residual, attn_residual_mask.to(residual.device))
    # Add residual connection after attention
    hidden_states = residual + hidden_states

    # Layer normalization before MLP
    residual = hidden_states
    # in Mistral, the "post" attention LN is applied AFTER the residual connection
    hidden_states = module.post_attention_layernorm(hidden_states)
    # Masked MLP forward
    hidden_states = MLP_masked(module, hidden_states, graph_mlp_mask)

    # Apply MLP residual mask if provided
    if mlp_residual_mask is not None:
        residual = torch.einsum('bsd,s->bsd', residual, mlp_residual_mask.to(residual.device))
    # Add residual connection after MLP
    hidden_states = residual + hidden_states

    # Restore original dtype
    hidden_states = hidden_states.to(hidden_dtype)

    outputs = hidden_states

    return outputs

class Mistral_Hooked(LLM_Hooked):
    def __init__(self, hf_model_name, half_precision, untrained=False):
        super().__init__(hf_model_name, half_precision, untrained)

    def get_architecture_type(self):
        return 'sequential'
    
    def get_layers(self):
        return self.model.model.layers
    
    def get_attention_dense_layers(self):
        return [layer.self_attn.o_proj for layer in self.get_layers()]

    def get_reshaped_attention_dense(self, layer_i, d_h_head, d_head):
        """
        should be reshaped to [head, hidden_dim, head_dim]
        """
        dense_layer = self.get_attention_dense_layers()[layer_i].weight  # [hidden_dim, hidden_dim]
        Wo_h = [
            dense_layer[:, h_i * d_h_head: (h_i + 1) * d_h_head]  # [hidden_dim, head_dim]
            for h_i in range(d_head)
        ]
        Wo_h = torch.stack(Wo_h, dim=0) # [head, hidden_dim, head_dim]
        return Wo_h

    def get_hidden_size(self):
        return self.config.hidden_size

    def get_head_size(self):
        return self.config.hidden_size // self.config.num_attention_heads
    
    def get_nb_head_groups(self):
        return self.config.num_attention_heads // self.config.num_key_value_heads

    def get_num_key_value_heads(self):
        return self.config.num_key_value_heads

    def get_nb_head(self):
        return self.config.num_attention_heads

    def load_model_from_hf(self):
        # Load the model configuration
        config = AutoConfig.from_pretrained(self.model_name)
        if config.model_type != "mistral":
            AssertionError("This architecture is not supported yet")
        print(config)

        # Load the tokenizer and configure padding
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # For generation

        # Load the model with necessary configurations
        if self.untrained:
            model =  AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                attn_implementation="eager",
                output_hidden_states=True,
                output_attentions=False,
                device_map="auto",
                torch_dtype=torch.float16 if self.half_precision else torch.float32,)
        return config, tokenizer, model

    def register_value_hook(self, layer_i):
        """
        Hook to extract and cache the value tensor (V) from the attention mechanism.
        """
        layers = self.get_layers()
        with torch.no_grad():
            def fn(_, input, output):
                # Reshape the output to separate head dimensions
                input_shape = input[0].shape[:-1] # [batch, seq]
                hidden_shape = (*input_shape, self.get_num_key_value_heads(), self.get_head_size()) # [batch, seq, num_kv_heads, head_dim]
                value = output.view(hidden_shape).transpose(1, 2)  # [batch, num_kv_heads, seq, head_dim]
                # Repeat the value tensor to match the number of query heads
                value = repeat_kv(value, self.get_nb_head_groups())
                # value = value.repeat_interleave(self.get_nb_head_groups(), dim=1)  # [batch, repeat*head, seq, head_dim]
                # Cache the value tensor for the current layer
                self.caches['attention']['value'][layer_i] = value.detach()
        handle = layers[layer_i].self_attn.v_proj.register_forward_hook(fn)
        return handle
    
    def register_attention_hook(self, layer_i):
        """
        Hook to extract the attention map and output from the attention mechanism.
        """
        layers = self.get_layers()
        with torch.no_grad():
            def fn(_, input, output):
                # Unpack the output into attention output and attention weights
                attn_output, attn_weights = output
                # Cache the attention weights and output
                self.caches['attention']['attn_weight'][layer_i] = attn_weights.detach()
                self.caches['attention']['output'][layer_i] = attn_output.detach() # used for sanity check
        handle = layers[layer_i].self_attn.register_forward_hook(fn)
        return handle
    
    def register_mlp_hook(self, layer_i):
        """
        Hook to extract the MLP output, separating the bias contribution.
        """
        layers = self.get_layers()
        def fn(_, input, output):
            # No bias in mistral
            # Cache the outputs
            self.caches['mlp']['output'][layer_i] = output.detach().clone()
        handle = layers[layer_i].mlp.register_forward_hook(fn)
        return handle
    
    def register_residual_hook(self, layer_i):
        """
        Hook to extract the residual stream output at a specific layer.
        Useful for extracting the residual stream before any layer normalization.
        """
        layers = self.get_layers()
        def fn(_, input, output):
            # Detach and clone the residual output to avoid in-place modifications
            residual_output = output[0].detach().clone()
            self.caches['residual']['output'][layer_i] = residual_output.unsqueeze(0)
        handle = layers[layer_i].register_forward_hook(fn)
        return handle

    def linearise_final_norm(self, input_tensor):
        final_rms_norm = self.model.model.norm
        linearized_RMS = linearize_rms_norm(final_rms_norm, input_tensor)
        return linearized_RMS
    
    def get_final_norm(self):
        return self.model.model.norm

    def decoder_masked(self, decoder_input, graph_masks):
        return MistralDecoder_masked(*(decoder_input + graph_masks))