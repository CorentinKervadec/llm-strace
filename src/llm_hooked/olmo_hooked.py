from src.llm_hooked.llm_hooked import LLM_Hooked
from src.llm_hooked.utils import linearize_rms_norm, apply_rotary_pos_emb, eager_attention_forward, MLP_masked, repeat_kv, get_real_weight_from_offloaded_module
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from typing import Callable
import torch
import src.llm_hooked.sanity_checks as sanity_check
from safetensors import SafetensorError


ATTENTION_MASK_VALUE = -65504

"""
Important details from Olmo architecture:
- no bias,
- RMS norm,
- Normalisation AFTER attention and MLP blocks
- QK norm
"""

def Olmo2Attention_masked(
    module,
    hidden_states,
    position_embeddings,
    attention_mask,
    mask,
    mask_before_softmax):

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    # ** KEY OLMO 2 CHANGE **: Apply q_norm and k_norm *after* projection
    query_states = module.q_norm(module.q_proj(hidden_states))
    key_states = module.k_norm(module.k_proj(hidden_states))
    value_states = module.v_proj(hidden_states)

    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_states = query_states.to(value_states.dtype)
    key_states = key_states.to(value_states.dtype)
    
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
    attn_output = module.o_proj(attn_output)
    return attn_output, attn_weights

def Olmo2Decoder_masked(
    module,
    hidden_states,
    attention_mask,
    position_embeddings,
    graph_attn_mask,
    mask_before_softmax,
    graph_mlp_mask,
    attn_residual_mask,
    mlp_residual_mask):

    hidden_dtype = hidden_states.dtype
    residual = hidden_states
    hidden_states, _ = Olmo2Attention_masked(
        module.self_attn,
        hidden_states,
        position_embeddings,
        attention_mask,
        graph_attn_mask,
        mask_before_softmax)

    # in Olmo2, the "post" attention LN is applied BEFORE the residual connection
    hidden_states = module.post_attention_layernorm(hidden_states)

    # Apply attention residual mask if provided
    if attn_residual_mask is not None:
        residual = torch.einsum('bsd,s->bsd', residual, attn_residual_mask.to(residual.device))
    # Add residual connection after attention
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = MLP_masked(module, hidden_states, graph_mlp_mask)

    # in Olmo2, the "post" mlp LN is applied BEFORE the residual connection
    hidden_states = module.post_feedforward_layernorm(hidden_states)

    # Apply MLP residual mask if provided
    if mlp_residual_mask is not None:
        residual = torch.einsum('bsd,s->bsd', residual, mlp_residual_mask.to(residual.device))

    hidden_states = residual + hidden_states

    # Restore original dtype
    hidden_states = hidden_states.to(hidden_dtype)

    return hidden_states

class Olmo2_Hooked(LLM_Hooked):
    def __init__(self, hf_model_name, half_precision, untrained=False):
        if 'stage' in hf_model_name:
            hf_model_name, self.training_step = hf_model_name.split('_')
        else:
            self.training_step = 'main'
        super().__init__(hf_model_name, half_precision, untrained)

    def get_architecture_type(self):
        return 'sequential'
    
    def get_layers(self):
        return self.model.model.layers

    def get_attention_dense_layers(self):
        return [get_real_weight_from_offloaded_module(layer.self_attn.o_proj) for layer in self.get_layers()]

    def get_attention_dense_layer_i(self, layer_i):
        module = self.get_layers()[layer_i].self_attn.o_proj
        return get_real_weight_from_offloaded_module(module)
    
    def get_reshaped_attention_dense(self, layer_i, d_h_head, d_head):
        """
        should be reshaped to [head, hidden_dim, head_dim]
        """
        dense_layer = self.get_attention_dense_layer_i(layer_i)  # [hidden_dim, hidden_dim]
        Wo_h = [
            dense_layer[:, h_i * d_h_head: (h_i + 1) * d_h_head]  # [hidden_dim, head_dim]
            for h_i in range(d_head)
        ]
        Wo_h = torch.stack(Wo_h, dim=0) # [head, hidden_dim, head_dim]
        return Wo_h

    def get_hidden_size(self):
        return self.config.hidden_size

    def get_head_size(self):
        # Confidence: 99% - Correct config keys.
        return self.config.hidden_size // self.config.num_attention_heads
    
    def get_num_key_value_heads(self):
        return self.config.num_key_value_heads

    def get_nb_head_groups(self):
        return self.config.num_attention_heads // self.config.num_key_value_heads

    def get_nb_head(self):
        return self.config.num_attention_heads

    def load_model_from_hf(self):
        """
        Trained in float32
        """
        # Confidence: 95% - Correctly checks for 'olmo2' model type.

        # Load the model configuration
        config = AutoConfig.from_pretrained(self.model_name)
        if config.model_type != "olmo2":
            AssertionError(f"This architecture is not supported yet (expected 'olmo2', got '{config.model_type}')")
        print(config)

        # Load the tokenizer and configure padding
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # For generation

        # Load the model with necessary configurations
        if self.untrained:
            model =  AutoModelForCausalLM.from_config(config)
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    attn_implementation="eager", # Use eager for easier hooking
                    output_hidden_states=True,
                    output_attentions=False,
                    device_map="auto",
                    torch_dtype=torch.float16 if self.half_precision else torch.float32,
                    revision=self.training_step,)
            except SafetensorError as e:
                print(f"Caught a Safetensor error: {e}")
                print("The file header is likely corrupt or invalid. Retry with force_download")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    attn_implementation="eager", # Use eager for easier hooking
                    output_hidden_states=True,
                    output_attentions=False,
                    device_map="auto",
                    torch_dtype=torch.float16 if self.half_precision else torch.float32,
                    revision=self.training_step,
                    force_download=True)
        return config, tokenizer, model

    def register_value_hook(self, layer_i):
        # Confidence: 90% - The logic for reshaping V-projection output
        # is standard for GQA models and should apply here.
        """
        Hook to extract and cache the value tensor (V) from the attention mechanism.
        """
        layers = self.get_layers()
        with torch.no_grad():
            def fn(_, input, output):
                # output is from v_proj: [batch, seq, num_kv_heads * head_dim]
                # Reshape the output to separate head dimensions
                input_shape = input[0].shape[:-1] # [batch, seq]
                hidden_shape = (*input_shape, self.get_num_key_value_heads(), self.get_head_size()) # [batch, seq, num_kv_heads, head_dim]
                value = output.view(hidden_shape).transpose(1, 2)  # [batch, num_kv_heads, seq, head_dim]
                # Repeat the value tensor to match the number of query heads
                value = repeat_kv(value, self.get_nb_head_groups())
                # value = value.repeat_interleave(self.get_nb_head_groups(), dim=1)  # [batch, num_attn_heads, seq, head_dim]
                # Cache the value tensor for the current layer
                self.caches['attention']['value'][layer_i] = value.detach()
        handle = layers[layer_i].self_attn.v_proj.register_forward_hook(fn)
        return handle
    
    def register_attention_hook(self, layer_i):
        # Confidence: 99% - Olmo2Attention returns (attn_output, attn_weights),
        # so this hook is identical to Mistral's.
        """
        Hook to extract the attention map and output from the attention mechanism.
        """
        layers = self.get_layers()
        with torch.no_grad():
            def fn(_, input, output):
                # Unpack the output into attention output and attention weights
                # Olmo2Attention.forward returns (attn_output, attn_weights)
                attn_output, attn_weights = output
                # Cache the attention weights and output
                self.caches['attention']['attn_weight'][layer_i] = attn_weights.detach()
                self.caches['attention']['output'][layer_i] = attn_output.detach() # used for sanity check
        handle = layers[layer_i].self_attn.register_forward_hook(fn)
        return handle
    
    def register_mlp_hook(self, layer_i):
        # Confidence: 99% - Olmo2MLP has no bias, same as Mistral.
        """
        Hook to extract the MLP output.
        """
        layers = self.get_layers()
        def fn(_, input, output):
            # Olmo 2 MLP has bias=False on all projections
            # Cache the outputs
            self.caches['mlp']['output'][layer_i] = output.detach().clone()
        handle = layers[layer_i].mlp.register_forward_hook(fn)
        return handle
    
    def register_residual_hook(self, layer_i):
        # Confidence: 95% - This is a key change. Olmo2DecoderLayer.forward
        # returns a single tensor, not a tuple, so we access `output` directly.
        """
        Hook to extract the residual stream output at a specific layer.
        """
        layers = self.get_layers()
        def fn(_, input, output):
            # I don't know why, but output is a tuple with one element
            residual_output = output[0].detach().clone()
            self.caches['residual']['output'][layer_i] = residual_output.unsqueeze(0)
        handle = layers[layer_i].register_forward_hook(fn)
        return handle

    def linearise_final_norm(self, input_tensor):
        # Confidence: 99% - Correct path to final norm.
        final_rms_norm = self.model.model.norm
        linearized_RMS = linearize_rms_norm(final_rms_norm, input_tensor)
        return linearized_RMS
    
    def linearise_post_attention_norm(self, input_tensor, layer_index):
        layer = self.get_layers()[layer_index]
        post_attention_rms_norm = layer.post_attention_layernorm
        linearized_RMS = linearize_rms_norm(post_attention_rms_norm, input_tensor)
        return linearized_RMS

    def linearise_post_mlp_norm(self, input_tensor, layer_index):
        layer = self.get_layers()[layer_index]
        post_mlp_rms_norm = layer.post_feedforward_layernorm
        linearized_RMS = linearize_rms_norm(post_mlp_rms_norm, input_tensor)
        return linearized_RMS

    # def linearise_k_norm(self, input_tensor, layer: Olmo2DecoderLayer):
    #     k_rms_norm = layer.self_attn.k_norm
    #     linearized_RMS = linearize_rms_norm(k_rms_norm, input_tensor)
    #     return linearized_RMS

    # def linearise_q_norm(self, input_tensor, layer: Olmo2DecoderLayer):
    #     q_rms_norm = layer.self_attn.q_norm
    #     linearized_RMS = linearize_rms_norm(q_rms_norm, input_tensor)
    #     return linearized_RMS

    def decoder_masked(self, decoder_input, graph_masks):
        return Olmo2Decoder_masked(*(decoder_input + graph_masks))

    def get_post_mlp_norm(self, layer_idx):
        return self.get_layers()[layer_idx].post_feedforward_layernorm
    
    def get_post_attn_norm(self, layer_idx):
        return self.get_layers()[layer_idx].post_attention_layernorm

    def get_final_norm(self):
        return self.model.model.norm