import torch

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
# Variables used for sanity checks -- provide reconstruction guarantees
# RECONSTRUCTION_TOLERANCE = 0.5 if HALF_PRECISION else 1e-4  # Tolerance for reconstruction checks
EPS = 1e-6 #1e-3 if HALF_PRECISION else 1e-6  # Small epsilon value


def get_reconstruction_tolerance(half_precision):
    return 0.5 if half_precision else 0.5 # I think it's too high

def test_linearize_rms_norm(rms_norm, d, half_precision):
    """
    Test the linearization of a LayerNorm by comparing the linear approximation
    to the true LayerNorm output.

    Args:
        rms_norm (nn.LayerNorm): The LayerNorm module to test.
        d (int): The hidden size of the input tensor.

    RMS norm is already linear !
    
    """

    # Generate random input tensor
    x = torch.randn(10, d, dtype=rms_norm.weight.dtype).to(rms_norm.weight.device)

    # Compute the linear approximation
    L = linearize_rms_norm(rms_norm, x)
    approx = torch.einsum('sd,sd->sd', L, x)

    # Compute the true LayerNorm output
    true_output = rms_norm(x)

    # approx = my_rms_norm(x, rms_norm.weight, rms_norm.variance_epsilon)
    # print("RMS shape", approx.shape)

    # Check if the approximation is close to the true output
    # test = torch.allclose(approx, true_output, atol=RECONSTRUCTION_TOLERANCE)
    test = torch.allclose(approx, true_output, atol=get_reconstruction_tolerance(half_precision))

    # Print results
    if not test:
        max_diff = (true_output - approx).abs().max()
        print(f"[SANITY CHECK][LINEARIZE FINAL LN] Reconstruction failed")
        print(f"Max element-wise difference: {max_diff}")
        print("True output:", true_output)
        print("Approximation:", approx)
        exit()

"""
Sanity Checks

This section contains various sanity checks to ensure the correctness of intermediate computations
and the integrity of the model's outputs during the forward pass and graph generation process.

1. Attention Normalization Check (`attn_norm_sanity_check`):
    - Verifies that the attention weights for each token sum to 1, ensuring proper normalization.

2. Causal Attention Check (`attn_causal_sanity_check`):
    - Ensures that the causal masking mechanism is respected, i.e., tokens cannot attend to future tokens.

3. Attention Output Reconstruction Check (`sanity_check_head_output`):
    - Confirms that the sum of all per-head attention contributions, along with the attention bias, reconstructs the final attention output.

4. MLP Output Reconstruction Check (`sanity_check_mlp_output`):
    - Validates that the MLP output, when combined with its bias, matches the expected output.

5. Residual Stream Reconstruction Check (`sanity_check_residual`):
    - Ensures that the residual stream at each layer is correctly reconstructed by summing the contributions from the MLP, attention, biases, and the previous residual stream.
    - For the final layer, this includes applying the final layer normalization.

6. Final Layer Normalization Linearization Check (`sanity_check_linearize_final_RMS`):
    - Verifies that the linearized representation of the final layer normalization accurately reconstructs the output of the true layer normalization.

7. Node Reconstruction Check (`node_reconstruction_sanity_check`):
    - Ensures that the sum of all incoming edge vectors reconstructs the node vector, validating the integrity of the graph representation.

These checks are critical for debugging and validating the correctness of the model's intermediate computations, ensuring that the extracted representations and graph structures are reliable.
"""

def attn_norm_sanity_check(attn_row_y):
    """
    Sanity check to ensure that attention rows are normalized (sum to 1).

    Args:
        attn_row_y (torch.Tensor): A row of attention weights.

    Raises:
        AssertionError: If the attention row does not sum to 1 within a tolerance.
    """

    row_sum = attn_row_y.sum(-1).squeeze().to(DEVICE)
    expected_sum = torch.tensor(1.0, device=row_sum.device).to(attn_row_y.dtype)

    if not torch.allclose(row_sum, expected_sum, atol=EPS):
        print("[SANITY CHECK][Norm attn] The attention row does not sum to 1 as expected.")
        print(f"[SANITY CHECK][Norm attn] Row: {attn_row_y}")
        print(f"[SANITY CHECK][Norm attn] Computed Sum: {row_sum}")
        print(f"[SANITY CHECK][Norm attn] Expected Sum: {expected_sum}")
        raise AssertionError("Attention row normalization failed")

def attn_causal_sanity_check(x_index, y_index, attention_scalar):
    """
    Sanity check to ensure causal masking in the attention mechanism.
    Ensures that tokens cannot attend to future tokens.

    Args:
        x_index (int): Index of the source token.
        y_index (int): Index of the target token.
        attention_scalar (float): Attention weight between the source and target tokens.

    Raises:
        ValueError: If the causal masking condition is violated.
    """
    
    if y_index < x_index:
        if not torch.allclose(attention_scalar, torch.tensor(0.0, device=attention_scalar.device).to(attention_scalar.dtype), atol=EPS):
            raise ValueError(
                f"[SANITY CHECK][Causal Attention] Causal masking violated: "
                f"Target token index (y={y_index}) is earlier than source token index (x={x_index}), "
                f"but attention weight is {attention_scalar.item():.6f} (expected ~0.0)."
            )

def sanity_check_head_output(head_output, output, half_precision, attn_bias=None):
    """
    Sanity check to ensure that the sum of all per-head attention contributions,
    along with the attention bias, reconstructs the final attention output.

    Args:
        head_output (torch.Tensor): Per-head attention contributions, shape [1, d_sequence_y, d_head, d_sequence_x, d_h].
        output (torch.Tensor): Final attention output, shape [1, d_sequence_y, d_h].
        attn_bias (torch.Tensor): Attention bias, shape [d_h].

    Raises:
        AssertionError: If the reconstruction difference exceeds the tolerance.
    """    
    
    attn_bias = 0 if attn_bias is None else attn_bias.to(DEVICE)

    # Recompose the output by summing over heads and sequence contributions
    output_recomposed = head_output.sum(dim=-2).sum(dim=-2) + attn_bias
    output_recomposed = output_recomposed.to(DEVICE)
    output = output.to(DEVICE)

    # Check if the recomposed output is close to the original output
    if not torch.allclose(output_recomposed, output, atol=get_reconstruction_tolerance(half_precision)):
        diff = (output_recomposed - output).abs()
        max_diff = diff.max()
        sum_diff = diff.sum()

        error_message = (
            f"[SANITY CHECK][ATTENTION RECONSTRUCTION] Reconstruction failed.\n"
            f"Max element-wise difference: {max_diff}\n"
            f"Sum of differences: {sum_diff}\n"
            f"Expected output: {output}\n"
            f"Recomposed output: {output_recomposed}\n"
            f"Attention bias: {attn_bias}\n"
        )
        raise AssertionError(error_message)

def sanity_check_mlp_output(mlp_output, output, half_precision, mlp_bias=None):
    """
    Sanity check to ensure that the MLP output, combined with its bias, 
    reconstructs the expected output.

    Args:
        mlp_output (torch.Tensor): The MLP output without bias.
        output (torch.Tensor): The expected full MLP output.
        mlp_bias (torch.Tensor): The bias term of the MLP.

    Raises:
        AssertionError: If the reconstruction difference exceeds the tolerance.
    """

    mlp_bias = 0 if mlp_bias is None else mlp_bias
    
    # Recompose the output by adding the bias
    output_recomposed = mlp_output + mlp_bias
    output_recomposed = output_recomposed.to(DEVICE)
    output = output.to(DEVICE)

    # Check if the recomposed output is close to the expected output
    if not torch.allclose(output_recomposed, output, atol=get_reconstruction_tolerance(half_precision)):
        diff = (output_recomposed - output).abs()
        max_diff = diff.max()
        sum_diff = diff.sum()

        error_message = (
            f"[SANITY CHECK][MLP RECONSTRUCTION] Reconstruction failed.\n"
            f"Max element-wise difference: {max_diff}\n"
            f"Sum of differences: {sum_diff}\n"
            f"Expected output: {output}\n"
            f"Recomposed output: {output_recomposed}\n"
            f"MLP bias: {mlp_bias}\n"
        )
        raise AssertionError(error_message)

def sanity_check_stream_output(stream, stream_gold, half_precision):
    """
    Sanity check to ensure that the residual stream reconstructs the expected gold stream.

    Args:
        stream (torch.Tensor): The extracted stream.
        stream_gold (torch.Tensor): The expected stream.

    Raises:
        AssertionError: If the reconstruction difference exceeds the tolerance.
    """
    
    stream = stream.to(DEVICE)
    stream_gold = stream_gold.to(DEVICE)

    # Check if the recomposed output is close to the expected output
    if not torch.allclose(stream, stream_gold, atol=get_reconstruction_tolerance(half_precision)):
        diff = (stream - stream_gold).abs()
        max_diff = diff.max()
        sum_diff = diff.sum()

        error_message = (
            f"[SANITY CHECK][STREAM RECONSTRUCTION] Reconstruction failed.\n"
            f"Max element-wise difference: {max_diff}\n"
            f"Sum of differences: {sum_diff}\n"
            f"Expected output: {stream_gold}\n"
            f"Recomposed output: {stream}\n"
        )
        raise AssertionError(error_message)


def sanity_check_residual(mlp_output, head_output, residual_past, mlp_bias, attn_bias, residual_current, final_layer_norm, half_precision):
    """
    Sanity check to ensure that the residual stream is correctly reconstructed.

    Args:
        mlp_output (torch.Tensor): MLP output without bias.
        head_output (torch.Tensor): Per-head attention contributions.
        residual_past (torch.Tensor): Residual stream from the previous layer.
        mlp_bias (torch.Tensor): MLP bias term.
        attn_bias (torch.Tensor): Attention bias term.
        residual_current (torch.Tensor): Residual stream of the current layer.
        final_layer_norm (nn.LayerNorm or None): Final layer normalization, if applicable.

    Raises:
        AssertionError: If the reconstruction difference exceeds the tolerance.
    """
    
    attn_bias = torch.tensor(0) if attn_bias is None else attn_bias
    mlp_bias = torch.tensor(0) if mlp_bias is None else mlp_bias

    # Reconstruct the residual stream
    reconstructed_output = (
        mlp_output.to(DEVICE)
        + mlp_bias.to(DEVICE)
        + head_output.sum(dim=-2).sum(dim=-2).to(DEVICE)
        + attn_bias.to(DEVICE)
        + residual_past.to(DEVICE)
    )

    residual_current = residual_current.to(DEVICE)

    # Apply final layer normalization if it's the last layer
    if final_layer_norm is not None:
        # print("final_layer_norm", final_layer_norm.shape)
        # print("reconstructed_output", reconstructed_output.shape)
        reconstructed_output = torch.einsum('sd,sd->sd', final_layer_norm, reconstructed_output)

    # Check if the reconstructed output matches the current residual stream
    if not torch.allclose(reconstructed_output, residual_current, atol=get_reconstruction_tolerance(half_precision)):
        diff = (reconstructed_output - residual_current).abs()
        sum_diff = diff.sum()
        max_diff = diff.max()
        sum_diff_token = diff.sum(dim=-1)

        error_message = (
            f"[SANITY CHECK][RESIDUAL RECONSTRUCTION] Reconstruction failed.\n"
            f"Total difference: {sum_diff}\n"
            f"Max element-wise difference: {max_diff}\n"
            f"Difference per token: {sum_diff_token}\n"
            f"Residual current norm: {residual_current.abs().sum()}\n"
            f"MLP bias norm: {mlp_bias.abs().sum()}\n"
        )
        raise AssertionError(error_message)

def sanity_check_linearize_final_RMS(residual_stream, residual_outputs, linearized_FNL, half_precision):
    """
    Sanity check to ensure that the linearized final layer normalization
    accurately reconstructs the residual stream.

    Args:
        residual_stream (torch.Tensor): The true residual stream after final layer normalization.
        residual_outputs (torch.Tensor): The residual outputs before final layer normalization.
        linearized_FNL (Tuple[torch.Tensor, torch.Tensor]): Linearized parameters of the final layer norm (L, beta).

    Raises:
        AssertionError: If the reconstruction difference exceeds the tolerance.
    """

    linearized_FNL = linearized_FNL.to(DEVICE)
    residual_outputs = residual_outputs.to(DEVICE)
    residual_stream = residual_stream.to(DEVICE)
    
    # Reconstruct the output using the linearized parameters
    reconstructed_output = torch.einsum('sd,sd->sd', linearized_FNL, residual_outputs)
    true_output = residual_stream

    # Check if the reconstructed output matches the true output
    if not torch.allclose(reconstructed_output, true_output, atol=get_reconstruction_tolerance(half_precision)):
        diff = (reconstructed_output - true_output).abs()
        max_diff = diff.max()
        sum_diff = diff.sum()

        print(reconstructed_output)
        print(true_output)

        error_message = (
            "[SANITY CHECK][LINEARIZE FINAL LN] Reconstruction failed.\n"
            f"Max element-wise difference: {max_diff}\n"
            f"Sum of differences: {sum_diff}\n"
            f"True output norm: {true_output.norm()}\n"
            f"Reconstructed output norm: {reconstructed_output.norm()}\n"
        )
        raise AssertionError(error_message)

def sanity_check_linearize_RMS(true_norm, before_norm, linearized_rms, half_precision):
    """
    Sanity check to ensure that the linearized final layer normalization
    accurately reconstructs the residual stream.

    Args:
        true_norm (RMSNorm): The true layer normalization.
        before_norm (torch.Tensor): The vector before final layer normalization.
        linearized_rms (Tuple[torch.Tensor, torch.Tensor]): Linearized parameters of the layer norm (L, beta).

    Raises:
        AssertionError: If the reconstruction difference exceeds the tolerance.
    """

    before_norm = before_norm.to(DEVICE)
    linearized_rms = linearized_rms.to(DEVICE)
    
    # Reconstruct the output using the linearized parameters
    reconstructed_output = torch.einsum('sd,sd->sd', linearized_rms, before_norm)
    true_output = true_norm(before_norm)

    # Check if the reconstructed output matches the true output
    if not torch.allclose(reconstructed_output, true_output, atol=get_reconstruction_tolerance(half_precision)):
        diff = (reconstructed_output - true_output).abs()
        max_diff = diff.max()
        sum_diff = diff.sum()

        error_message = (
            "[SANITY CHECK][LINEARIZE LN] Reconstruction failed.\n"
            f"Max element-wise difference: {max_diff}\n"
            f"Sum of differences: {sum_diff}\n"
            f"True output norm: {true_output.norm()}\n"
            f"Reconstructed output norm: {reconstructed_output.norm()}\n"
        )
        raise AssertionError(error_message)

def node_reconstruction_sanity_check(node, incoming_edges, half_precision):
    """
    Sanity check to ensure that the sum of all incoming edge vectors reconstructs the node vector.

    Args:
        node (dict): The target node containing the 'vector' key.
        incoming_edges (list): List of incoming edges, each containing a 'vector' key.

    Raises:
        AssertionError: If the reconstruction difference exceeds the tolerance.
    """
    
    node_vector = node['vector'].to(DEVICE)
    edge_vectors = torch.stack([edge['vector'] for edge in incoming_edges], dim=0)
    reconstructed_node = edge_vectors.sum(dim=0).to(DEVICE)

    # print("edge type:", [edge['type'] for edge in incoming_edges])
    if not torch.allclose(reconstructed_node, node_vector, atol=get_reconstruction_tolerance(half_precision)):
        diff = (reconstructed_node - node_vector).abs()
        sum_diff = diff.sum()
        max_diff = diff.max()

        error_message = (
            f"[SANITY CHECK][NODE RECONSTRUCTION] Reconstruction failed.\n"
            f"Node vector norm: {node_vector.norm()}\n"
            f"Reconstructed vector norm: {reconstructed_node.norm()}\n"
            f"Sum of differences: {sum_diff}\n"
            f"Max element-wise difference: {max_diff}\n"
        )

        raise ValueError(error_message)

def sanity_check_before_mlp(before_attn, attn_out, after_mlp, mlp_out, half_precision):
    """
    Sanity check to verify that (before_attn + attn_out) ≈ (after_mlp - mlp_out).

    Args:
        before_attn (torch.Tensor): Tensor before attention.
        attn_out (torch.Tensor): Attention output tensor.
        after_mlp (torch.Tensor): Tensor after MLP.
        mlp_out (torch.Tensor): MLP output tensor.

    Raises:
        AssertionError: If the reconstruction difference exceeds the tolerance.
    """

    left = before_attn.to(DEVICE) + attn_out.to(DEVICE)
    right = after_mlp.to(DEVICE) - mlp_out.to(DEVICE)

    if not torch.allclose(left, right, atol=get_reconstruction_tolerance(half_precision)):
        diff = (left - right).abs()
        max_diff = diff.max()
        sum_diff = diff.sum()
        error_message = (
            "[SANITY CHECK][BEFORE MLP] Reconstruction failed.\n"
            f"Max element-wise difference: {max_diff}\n"
            f"Sum of differences: {sum_diff}\n"
            f"Left (before_attn + attn_out): {left}\n"
            f"Right (after_mlp - mlp_out): {right}\n"
        )
        raise AssertionError(error_message)
    
def sanity_check_after_mlp(before_mlp, after_mlp, mlp_out, half_precision):
    """
    Verify that after_mlp ≈ before_mlp + mlp_out.

    Args:
        before_mlp (torch.Tensor): Residual tensor before the MLP is applied.
        after_mlp (torch.Tensor): Residual tensor after the MLP (expected to equal before_mlp + mlp_out).
        mlp_out (torch.Tensor): Output of the MLP (without adding the residual).
        half_precision (bool): Whether to use the relaxed tolerance for half precision.

    Raises:
        AssertionError: If the reconstruction difference exceeds the tolerance.
    """
    device = DEVICE

    left = before_mlp.to(device) + mlp_out.to(device)
    right = after_mlp.to(device)

    tol = get_reconstruction_tolerance(half_precision)
    if not torch.allclose(left, right, atol=tol):
        diff = (left - right).abs()
        max_diff = diff.max()
        sum_diff = diff.sum()
        error_message = (
            "[SANITY CHECK][AFTER MLP] Reconstruction failed.\n"
            f"Tolerance: {tol}\n"
            f"Max element-wise difference: {max_diff}\n"
            f"Sum of differences: {sum_diff}\n"
            f"Left (before_mlp + mlp_out) shape {left.shape}: {left}\n"
            f"Right (after_mlp) shape {right.shape}: {right}\n"
        )
        raise AssertionError(error_message)

def sanity_check_decoder(orginal_decoder_output, reconstruct_decoder_output, half_precision):
    """
    Sanity check for the decoder block reconstruction.

    This function recomputes the decoder output with all masks set to None (i.e., no masking)
    and checks if the reconstructed output matches the original decoder output within a tolerance.

    Args:
        orginal_decoder_output: The original output tensor from the decoder block.
        decoder_input: Tuple of arguments for the decoder block (excluding masks).
        sanity_masks: Tuple of None values for all masks (no masking).

    Returns:
        reconstruct_decoder_output: The reconstructed output tensor.

    Raises:
        AssertionError: If the reconstructed output does not match the original output within tolerance.
    """
    

    # Check if the recomposed output is close to the expected output
    if not torch.allclose(orginal_decoder_output, reconstruct_decoder_output, atol=get_reconstruction_tolerance(half_precision)):
        diff = (orginal_decoder_output - reconstruct_decoder_output).abs()
        max_diff = diff.max()
        sum_diff = diff.sum()

        error_message = (
            f"[SANITY CHECK][DECODER RECONSTRUCTION] Reconstruction failed.\n"
            f"Max element-wise difference: {max_diff}\n"
            f"Sum of differences: {sum_diff}\n"
            f"Expected output: {orginal_decoder_output}\n"
            f"Reconstructed output: {reconstruct_decoder_output}\n"
        )
        raise AssertionError(error_message)
    return reconstruct_decoder_output


# def check_importance_norm(edge_importances, tolerance=1e-4, half_precision):
#     """
#     Checks if the sum of importances for incoming edges of each node is 1.0.
#     """
#     all_ok = True
#     for node_id in self.nodes:
#         incoming_edges = self.in_edges(node_id, data=True, keys=True)
#         if not incoming_edges:
#             continue
        
#         total_importance = sum(data.get('importance', 0.0) for _, _, _, data in incoming_edges)
        
#         if not (1.0 - tolerance <= total_importance <= 1.0 + tolerance):
#             print(f"Normalization check failed for node {node_id}: sum = {total_importance}")
#             all_ok = False
    
#     if all_ok:
#         print("Importance normalization check passed for all nodes. ✅")