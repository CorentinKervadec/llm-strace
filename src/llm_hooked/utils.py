import torch
from torch import nn

def linearize_rms_norm(rms_norm, input_tensor: torch.Tensor):
    """
    Linearizes a RMS norm operation for a specific input tensor.

    Args:
        rms_norm: The RMS norm module to linearize.
        input_tensor (torch.Tensor): The input tensor to the LayerNorm.

    Returns:
        - L (torch.Tensor): The equivalent affine transformation matrix.
    """
    
    assert input_tensor.dtype == rms_norm.weight.dtype, "Input tensor and LayerNorm weight must have the same dtype"
    
    d_seq = input_tensor.shape[-2]  # Sequence length
    d = input_tensor.shape[-1]      # Hidden size

    eps = rms_norm.variance_epsilon 

    # Compute standard deviation from input
    var = torch.mean(input_tensor.pow(2), dim=-1) #
    # var = torch.var(input_tensor, dim=-1, unbiased=False)
    inv_std = torch.rsqrt(var + eps).to(input_tensor.dtype)
    # Extract gamma (weight) RMSNorm
    weight = rms_norm.weight

    # Compute the affine transformation matrix L
    L = torch.einsum("s,d->sd" ,inv_std, weight)

    return L

"""
Functions copy/pasted from HF's transformers that are used in LLM_Hooked.
I put them here because they are shared accross some LLMs.
"""

def rotate_half(x):
    # copy/pasted from hugging face transormers
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # copy/pasted from hugging face transfomers: mistral, olmo2
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # copy/pasted from hugging face transormers
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(module, query, key, value, attention_mask, scaling, custom_mask, mask_before_softmax, dropout=0.0, softcap=None):
    """
    Forward pass for masked attention in Mistral and Olmo2 (and probably more).

    Args:
        module: The attention module.
        query: Query tensor [batch, num_heads, seq_len, head_dim].
        key: Key tensor [batch, num_heads, seq_len, head_dim].
        value: Value tensor [batch, num_heads, seq_len, head_dim].
        attention_mask: Standard attention mask (e.g., causal mask).
        scaling: Scaling factor for attention logits.
        custom_mask: Custom mask for masking specific attention connections.
        mask_before_softmax: If True, apply custom_mask before softmax.
        dropout: Dropout probability.

    Returns:
        attn_output: Output of the attention layer.
        attn_weights: Attention weights after masking and softmax.
    """

    # Repeat key and value tensors if using multi-query attention
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # Compute raw attention scores
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        # Extract the causal mask for the current sequence length
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # If masking before softmax, add the custom mask to the causal mask
        if mask_before_softmax and (custom_mask is not None):
            causal_mask = causal_mask + custom_mask.to(attn_weights.device) * ATTENTION_MASK_VALUE
            causal_mask = torch.clamp(causal_mask, min=ATTENTION_MASK_VALUE)
        # Add the (possibly combined) mask to the attention weights
        attn_weights = attn_weights + causal_mask

    # Apply softmax to get normalized attention weights
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if mask_before_softmax and (custom_mask is not None):
        # Fix cases where the whole row must be masked (all -inf)
        row_mask = ~(causal_mask.sum(-1) == ATTENTION_MASK_VALUE * causal_mask.shape[-1])
        attn_weights = attn_weights * row_mask.unsqueeze(-1)

    if custom_mask is not None:
        if not mask_before_softmax:
            # If masking after softmax, zero out masked positions
            reverse_custom_mask = (custom_mask == 0).to(attn_weights.dtype)  # 0 means masked
            attn_weights = attn_weights * reverse_custom_mask.to(attn_weights.device)

    # Apply dropout to attention weights
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Compute the weighted sum of value vectors
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def MLP_masked(module, hidden_states, graph_mlp_mask):
    """
    Forward pass for the masked MLP in Olmo2 and Mistral models.

    Args:
        module: The decoder layer module containing the MLP (xxxDecoderLayer).
        hidden_states: Input tensor of shape [batch, seq_len, hidden_dim].
        graph_mlp_mask: Mask tensor of shape [seq_len], where 1 means not masked, 0 means masked.

    Returns:
        hidden_states: Output of the MLP after applying the mask.
    """
    # Ensure the hidden states are in the correct dtype for the MLP
    mlp_dtype = module.mlp.up_proj.weight.dtype
    hidden_states = module.mlp(hidden_states.to(mlp_dtype))
    # Apply the mask if provided: zero out masked positions along the sequence dimension
    if graph_mlp_mask is not None:
        hidden_states = torch.einsum('bsd,s->bsd', hidden_states, graph_mlp_mask.to(hidden_states.device))
    return hidden_states
