from src.llm_hooked.llm_hooked import LLM_Hooked
from src.llm_hooked.utils import linearize_rms_norm, apply_rotary_pos_emb, eager_attention_forward, MLP_masked, repeat_kv
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from typing import Callable
import torch
import src.llm_hooked.sanity_checks as sanity_check


ATTENTION_MASK_VALUE = -65504

"""
Code for masked attention.
TODO: I copy/pasted this function from Qwen3_Hooked, it should be adapted it to Gemma3.
"""
def Gemma3Attention_masked(
    module,
    hidden_states,
    position_embeddings,
    attention_mask,
    mask,
    mask_before_softmax):

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    query_states = module.q_norm(module.q_proj(hidden_states).view(hidden_shape))
    key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape))
    value_states = module.v_proj(hidden_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

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
    attn_output = module.o_proj(attn_output)
    return attn_output, attn_weights

"""
Code for running a decoder block (a layer) with masking.
TODO: I copy/pasted this function from Qwen3_Hooked, it should be adapted it to Gemma3.
"""
def Gemma3Decoder_masked(
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

    # in Qwen3, the attention LN is applied at the input of the attention
    hidden_states = module.input_layernorm(hidden_states)

    hidden_states, _ = Qwen3Attention_masked(
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

    # Fully Connected
    residual = hidden_states

    # in Qwen3, the mlp LN is applied at the input of the MLP
    hidden_states = module.post_attention_layernorm(hidden_states)

    hidden_states = MLP_masked(module, hidden_states, graph_mlp_mask)

    # Apply MLP residual mask if provided
    if mlp_residual_mask is not None:
        residual = torch.einsum('bsd,s->bsd', residual, mlp_residual_mask.to(residual.device))

    hidden_states = residual + hidden_states

    # Restore original dtype
    hidden_states = hidden_states.to(hidden_dtype)

    return hidden_states


"""
The class to handle Gemma's loading and the extraction using hook.
TODO: I copy/pasted this class from Qwen3_Hooked, it should be adapted it to Gemma3.
I'll highlight the function that are more likely to change.

I've seen that Gemma3 is using two types of layers: "sliding_attention" and "full_attention".
I don't know wha't the difference, but we should have a look at it in case it would have an influence
on our extraction code.

"""
class Gemma3_Hooked(LLM_Hooked):
    def __init__(self, hf_model_name, half_precision, untrained=False):
        super().__init__(hf_model_name, half_precision, untrained)

    def get_architecture_type(self):
        return 'sequential'
    
    """
    TODO: do print(model) to check what is the correct name for accessing the layers
    """
    def get_layers(self):
        return self.model.model.layers

    """
    TODO: do print(model) to check what is the correct name for accessing the self-attention
    """
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

    """
    TODO: check in Gemma3's config file if there is an head_dim parameter.
    https://huggingface.co/google/gemma-3-270m/blob/main/config.json
    """
    def get_head_size(self):
        return getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
    
    def get_nb_head_groups(self):
        return self.config.num_attention_heads // self.config.num_key_value_heads

    def get_num_key_value_heads(self):
        return self.config.num_key_value_heads

    def get_nb_head(self):
        return self.config.num_attention_heads

    """
    TODO: model_type: qwen3 -> gemma3_text
    """
    def load_model_from_hf(self):
        # Load the model configuration
        config = AutoConfig.from_pretrained(self.model_name)
        if config.model_type != "qwen3":
            AssertionError(f"This architecture is not supported yet (expected 'qwen3', got '{config.model_type}')")
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
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                attn_implementation="eager", # Use eager for easier hooking
                output_hidden_states=True,
                output_attentions=False,
                device_map="auto",
                torch_dtype=torch.float16 if self.half_precision else torch.float32,)
        return config, tokenizer, model

    """
    TODO: you might have a shape issue here when adapting to Gemma3.
    """
    def register_value_hook(self, layer_i):
        """
        Hook to extract and cache the value tensor (V) from the attention mechanism.
        """
        layers = self.get_layers()
        with torch.no_grad():
            def fn(_, input, output):
                """
                hidden_shape (1, 23, -1, 64)
                output torch.Size([1, 23, 1024])
                value 1 torch.Size([1, 16, 23, 64])
                value 2 torch.Size([1, 32, 23, 64])
                """
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
    
    """
    TODO: should be fine, but have a look.
    """
    def register_attention_hook(self, layer_i):
        """
        Hook to extract the attention map and output from the attention mechanism.
        """
        layers = self.get_layers()
        with torch.no_grad():
            def fn(_, input, output):
                # Unpack the output into attention output and attention weights
                # Qwen3Attention.forward returns (attn_output, attn_weights)
                attn_output, attn_weights = output
                # Cache the attention weights and output
                self.caches['attention']['attn_weight'][layer_i] = attn_weights.detach()
                self.caches['attention']['output'][layer_i] = attn_output.detach() # used for sanity check
        handle = layers[layer_i].self_attn.register_forward_hook(fn)
        return handle
    
    """
    TODO: Check that Gemma3 is not using biases in the mlp. (you should see it when doing print(model))
    """
    def register_mlp_hook(self, layer_i):
        # Confidence: 99% - Qwen3MLP has no bias, same as Mistral.
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

    """
    TODO: Check that Gemma3 is using a final layer norm (after the last layer)
    If it is an RMS norm, then this current function is valid.
    """
    def linearise_final_norm(self, input_tensor):
        # Confidence: 99% - Correct path to final norm.
        final_rms_norm = self.model.model.norm
        linearized_RMS = linearize_rms_norm(final_rms_norm, input_tensor)
        return linearized_RMS
    
    def decoder_masked(self, decoder_input, graph_masks):
        return Gemma3Decoder_masked(*(decoder_input + graph_masks))

    """
    TODO: Does Gemma3 use post-mlp and post-attention normalisation? E.g.:
    x' = x + N(attn(x))
    x'' = x' + N(mlp(x'))   (N is the normalisation)
    If yes, you should uncomment and adapt these two next functions.
    
    Alternatively, the architecture could be something like:
    x' = x + attn(N(x))
    x'' = x' + mlp(N(x'))
    In that case, no need to uncomment these functions.
    """
    # def get_post_mlp_norm(self, layer_idx):
    #     return self.get_layers()[layer_idx].post_feedforward_layernorm
    
    # def get_post_attn_norm(self, layer_idx):
    #     return self.get_layers()[layer_idx].post_attention_layernorm

    def get_final_norm(self):
        return self.model.model.norm