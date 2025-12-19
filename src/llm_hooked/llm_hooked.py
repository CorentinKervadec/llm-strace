import torch
import src.llm_hooked.sanity_checks as sanity_check
import time
import torch.nn.functional as F

EPS = 1e-7

def surprisal(logits, labels):
    """
    Compute cross-entropy loss, entropy, predicted token id, and ground truth rank.

    Args:
        logits (torch.Tensor): Logits from the model (shape: [batch_size * seq_len, vocab_size]).
        labels (torch.Tensor): Ground truth token ids (shape: [batch_size, seq_len]).

    Returns:
        tuple: (loss, entropy, predicted_token_id, rank)
    """
    # Compute cross-entropy loss, ignoring padding
    loss = None
    if labels is not None:
        loss = F.cross_entropy(logits, labels.view(-1).cpu(), ignore_index=-100).item()

    # Compute probabilities for the last token
    probabilities = torch.softmax(logits[-1].float(), dim=-1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + EPS)).item()

    # Predicted token id (highest probability)
    predicted_token_id = torch.argmax(probabilities).item()

    # Ground truth token id (last label in the sequence)
    ground_truth_token_id = None
    if labels is not None:
        ground_truth_token_id = labels[0, -1].item()

    # Compute rank of the ground truth token
    sorted_indices = torch.argsort(probabilities, descending=True)
    rank = None
    if labels is not None:
        rank = (sorted_indices == ground_truth_token_id).nonzero(as_tuple=True)[0].item() + 1

    return loss, entropy, predicted_token_id, rank

class LLM_Hooked():
    """
    The following functions are used to initialize buffers for caching intermediate outputs,
    prepare the model by registering hooks, and load the model along with its tokenizer and configuration.
    """
    def __init__(self, hf_model_name, half_precision, untrained=False, test_mode=False, track_time=False):
        self.model_name = hf_model_name
        self.config = None
        self.model = None
        self.tokenizer = None
        self.caches = {
            'attention': None,
            'mlp': None,
            'residual': None}
        self.untrained = untrained
        self.half_precision = half_precision
        self.test_mode = test_mode
        self.track_time = track_time
        # save hook handle for later removal
        self.extraction_hook_handles = []
        self.masking_hook_handles = []
        # initialisation
        self.load_and_prepare_model()

    def remove_hooks(self, hook_handles):
        for handle in hook_handles:
            handle.remove()

    def extraction_hook_registred(self):
        return len(self.extraction_hook_handles)

    def remove_extraction_hooks(self):
        self.remove_hooks(self.extraction_hook_handles)
        self.extraction_hook_handles = []

    def remove_masking_hooks(self):
        self.remove_hooks(self.masking_hook_handles)
        self.masking_hook_handles = []

    def turn_time_tracking_on(self):
        self.track_time = True

    def turn_test_mode_on(self):
        self.test_mode = True

    def get_config(self):
        return self.config

    def get_model_device(self):
        return self.model.device

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_architecture_type(self):
        # could be 'sequential' or ?(todo)
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_layers(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_post_mlp_norm(self, layer_idx):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_post_attn_norm(self, layer_idx):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_final_norm(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_attention_dense_layers(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_reshaped_attention_dense(self, layer_i, d_h_head, d_head):
        """
        should be reshaped to [head, hidden_dim, head_dim]
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_hidden_size(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_head_size(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_nb_head(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_nb_head_groups(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_num_key_value_heads(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_n_layers(self):
        return len(self.get_layers())

    def load_model_from_hf(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def register_value_hook(self, layer_i):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def register_attention_hook(self, layer_i):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def register_mlp_hook(self, layer_i):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def register_residual_hook(self, layer_i):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def linearise_final_norm(self, input_tensor):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def linearise_post_attention_norm(self, input_tensor, layer_index):
        """
        Return None by default. Re-write the method in the subclass if needed
        """
        return None

    def linearise_post_mlp_norm(self, input_tensor, layer_index):
        """
        Return None by default. Re-write the method in the subclass if needed
        """
        return None

    def register_extraction_hooks(self):
        """
        Prepares the model by registering forward hooks to extract intermediate outputs.
        """
        # Register hooks for each layer
        for layer_index in range(self.get_n_layers()):
            # Register hooks for attention mechanisms
            value_handle = self.register_value_hook(layer_index)
            attention_handle = self.register_attention_hook(layer_index)
            # Register hooks for MLP outputs
            mlp_handle = self.register_mlp_hook(layer_index)
            # Register hooks for residual stream outputs
            residual_handle = self.register_residual_hook(layer_index)
            # save the hooks handle for later removal
            self.extraction_hook_handles += [value_handle, attention_handle, mlp_handle, residual_handle]
            
    # Initialize buffers for caching intermediate outputs
    def init_buffers(self):
        """
        Initializes buffers to store intermediate outputs for attention, MLP, and residual streams.

        Returns:
            tuple: Three dictionaries for caching attention, MLP, and residual outputs.
        """

        """
        Buffer shape:

        ATTENTION:
        * value = [layers, batch, number of heads, tokens, head dimension]
        * attn_weight = [layers, batch, number of heads, tokens_y, tokens_x]
        * output = [layers, batch, tokens, hidden]
        MLP:
        * output = [layers, batch, tokens, hidden]
        RESIDUAL:
        * output = [layers, batch, tokens, hidden]
        """

        n_layers = self.get_n_layers()
        cache_attn = {
            'value': [torch.empty(0) for _ in range(n_layers)],  # Stores value tensors 
            'attn_weight': [torch.empty(0) for _ in range(n_layers)],  # Stores attention weights
            'output': [torch.empty(0) for _ in range(n_layers)],  # Stores attention outputs for sanity checks
        }
        cache_mlp = {
            'output': [torch.empty(0) for _ in range(n_layers)],  # MLP outputs
        }
        cache_residual = {
            'output': [torch.empty(0) for _ in range(n_layers)],  # Residual stream outputs
        }
        return cache_attn, cache_mlp, cache_residual

    def control_cache_shape(self, tokens):
        """
        input:
        * tokens: size of the input sequence
        """
        layers = self.get_n_layers()
        batch = 1 # because batching is not enable
        nb_heads = self.get_nb_head()
        d_head = self.get_head_size()
        d_hidden = self.get_hidden_size()

        # Check attention cache shapes
        expected_shapes = {
            'attention': {
            'value': (batch, nb_heads, tokens, d_head),
            'attn_weight': (batch, nb_heads, tokens, tokens),
            'output': (batch, tokens, d_hidden)
            },
            'mlp': {
            'output': (batch, tokens, d_hidden)
            },
            'residual': {
            'output': (batch, tokens, d_hidden)
            }
        }

        for i in range(layers):
            for cache_type, cache_dict in expected_shapes.items():
                for key, expected_shape in cache_dict.items():
                    actual_shape = self.caches[cache_type][key][i].shape
                    if actual_shape != expected_shape:
                        raise ValueError(f"Invalid shape for {cache_type} {key} at layer {i}.\n"
                                f"Expected shape: {expected_shape}\n"
                                f"Actual shape: {actual_shape}")

    def reset_buffers(self):
        """
        Resets the buffers for caching intermediate outputs.
        """
        for cache in self.caches.values():
            for key in cache:
                for i in range(len(cache[key])):
                    cache[key][i] = torch.empty(0)
    
    def load_and_prepare_model(self):
        self.config, self.tokenizer, self.model = self.load_model_from_hf()
        self.caches['attention'], self.caches['mlp'], self.caches['residual'] = self.init_buffers()
        self.register_extraction_hooks()

    def decompose_attention_with_batching(self, value, attn_weights, y_batch_size=5, x_batch_size=5):
        """
        I think this operation should be universal across most LLMs, once value and attn_weights are in the
        correct format.
        """
        """
        Decomposes the attention mechanism into per-head and per-token contributions in a memory-efficient manner.

        Args:
            value (torch.Tensor): Value tensor of shape [layer, head, sequence, head_dim].
            attn_weights (torch.Tensor): Attention weights of shape [layer, head, seq_y, seq_x].
            dense (list of nn.Linear): List of Dense layers for projecting attention outputs.
            y_batch_size (int): Batch size for the target sequence dimension (y).
            x_batch_size (int): Batch size for the source sequence dimension (x).
            half_precision (bool): Whether to use half precision for computations.
            cast_output_to_char (bool): Whether to cast the output to CPU for memory efficiency.

        Returns:
            torch.Tensor: Decomposed attention outputs of shape [layer, seq_y, head, seq_x, hidden_dim].
        """
        d_layer, d_head, d_seq_x, d_h_head = value.shape
        _, _, d_seq_y, _ = attn_weights.shape
        value_device = value.device

        # Reshape the dense layer weights for each layer
        Wo = []
        for layer_i in range(d_layer):
            # Extract the weight
            layer_weight = self.get_reshaped_attention_dense(layer_i, d_h_head, d_head)
            Wo.append(layer_weight)

        Wo = torch.stack(Wo, dim=0)  # [layer, head, hidden_dim, head_dim]
        Wo = Wo.to(value_device)

        if self.half_precision:
            # Convert tensors to half precision for faster computation
            Wo = Wo.to(value.dtype)

        # Compute contributions of all source tokens to all target tokens for all heads in batches
        all_head_outputs = []
        for y_start in range(0, d_seq_y, y_batch_size):
            y_end = min(y_start + y_batch_size, d_seq_y)
            attn_weights_batch_y = attn_weights[:, :, y_start:y_end, :]  # [layer, head, batch_y, seq_x]

            head_outputs_y = []
            for x_start in range(0, d_seq_x, x_batch_size):
                x_end = min(x_start + x_batch_size, d_seq_x)
                attn_weights_batch_xy = attn_weights_batch_y[:, :, :, x_start:x_end]  # [layer, head, batch_y, batch_x]
                value_batch = value[:, :, x_start:x_end, :]  # [layer, head, batch_x, head_dim]

                # Perform batched computation
                head_outputs_batch = torch.einsum(
                    'lhyx,lhxe,lhde->lhyxd', attn_weights_batch_xy, value_batch, Wo
                )  # [layer, head, batch_y, batch_x, hidden_dim]

                # Move to the CPU to avoid GPU memory issues
                head_outputs_batch = head_outputs_batch.cpu()

                head_outputs_y.append(head_outputs_batch)

            # Concatenate all x batches along the x dimension
            head_outputs_y = torch.cat(head_outputs_y, dim=3)  # [layer, head, batch_y, seq_x, hidden_dim]
            all_head_outputs.append(head_outputs_y)

        # Concatenate all y batches along the y dimension
        all_head_outputs = torch.cat(all_head_outputs, dim=2)  # [layer, head, seq_y, seq_x, hidden_dim]

        # Reshape to [layer, seq_y, head, seq_x, hidden_dim]
        all_head_outputs = all_head_outputs.transpose(1, 2)

        # delete Wo to avoid memory issue
        del Wo

        return all_head_outputs

    def do_sanity_checks(self, residual_stream, residual_outputs, linearized_norm, head_outputs, outputs_attn, mlp_outputs, post_mlp_norms_linear, post_attn_norms_linear):
        """
        Perform various sanity check to control that the vectors have not been corrupted during their manipulation
        """
        if not self.test_mode:
            return 0
        else:
            # check norm linearisation
            sanity_check.sanity_check_linearize_final_RMS(residual_stream[0, -1], residual_outputs[0, -1], linearized_norm, self.half_precision)
            true_norm =  self.get_final_norm()
            sanity_check.sanity_check_linearize_RMS(true_norm, residual_outputs[0, -1], linearized_norm, self.half_precision)
            # print("[LLM Hooked] Sanity checks final ln passed!")
            # Sanity checks for attention outputs
            for layer in range(self.get_n_layers()):
                sanity_check.sanity_check_head_output(head_outputs[:,layer], outputs_attn[:,layer], self.half_precision, attn_bias=None)
                # print(f"[LLM Hooked][L{layer}/{self.get_n_layers()-1}] Sanity checks head output passed!")
            # Validate MLP output reconstruction
            outputs_mlp = torch.stack(self.caches['mlp']['output'])  # [layers, batch, seq, hidden_dim]
            for layer in range(self.get_n_layers()): # this sanity check is stupid
                sanity_check.sanity_check_mlp_output(mlp_outputs[0][layer], outputs_mlp[layer], self.half_precision, mlp_bias=None)
                # print(f"[LLM Hooked][L{layer}/{self.get_n_layers()-1}] Sanity checks mlp output passed!")
            # Validate residual stream reconstruction
            residual_past = residual_stream[0, :-1]  # [layers, seq, hidden_dim]
            residual_current = residual_stream[0, 1:]  # [layers, seq, hidden_dim]
            for layer in range(self.get_n_layers()-1):# ignore last layer because it has final norm
                for token in range(residual_stream.size(2)):
                    sanity_check.sanity_check_stream_output(residual_outputs[0, layer, token], residual_stream[0, layer+1, token], self.half_precision)
                    # print(f"[LLM Hooked][L{layer}/{self.get_n_layers()-1}][T{token}/{residual_stream.size(2)-1}]  Sanity checks residual passed!")
            for layer in range(self.get_n_layers()):
                # print(f"[LLM Hooked][L{layer}/{self.get_n_layers()-1}]")
                mlp_out = mlp_outputs[0][layer]
                head_out = head_outputs[0, layer]
                attn_out = outputs_attn[0, layer]
                res_in = residual_past[layer]
                res_out = residual_current[layer]
                if not post_mlp_norms_linear[layer] is None:
                    post_mlp_norm_linear = post_mlp_norms_linear[layer] # [seq, hidden_dim]
                    # check norm linearisation
                    true_norm =  self.get_post_mlp_norm(layer)
                    sanity_check.sanity_check_linearize_RMS(true_norm, mlp_out, post_mlp_norm_linear, self.half_precision)
                    # apply normalisation to the mlp output
                    mlp_out = torch.einsum('sd,sd->sd', post_mlp_norm_linear, mlp_out.to(post_mlp_norm_linear.device)).cpu()
                    # print("[LLM Hooked] Sanity checks post mlp ln passed!")
                if not post_attn_norms_linear[layer] is None:    
                    post_attn_norm_linear = post_attn_norms_linear[layer] # [seq, hidden_dim]
                    # check norm linearisation
                    true_norm = self.get_post_attn_norm(layer)
                    sanity_check.sanity_check_linearize_RMS(true_norm, attn_out, post_attn_norm_linear, self.half_precision)
                    # apply normalisation to the mlp output
                    head_out = torch.einsum('sd,sihd->sihd', post_attn_norm_linear, head_out.to(post_attn_norm_linear.device)).cpu()
                    # attn_out = torch.einsum('sd,sd->sd', post_attn_norm_linear, attn_out.to(post_attn_norm_linear.device)).cpu()
                    # assert torch.allclose(head_out.sum(-2).sum(-2), attn_out, atol=0.01)                    
                    # print("[LLM Hooked] Sanity checks post attn ln passed!")
                
                sanity_check.sanity_check_residual(
                    mlp_out,
                    head_out,
                    res_in,
                    None, # mlp bias
                    None, # attn bias
                    res_out,
                    linearized_norm if layer == self.get_n_layers() - 1 else None,
                    self.half_precision
                )
            print("[LLM Hooked] Sanity checks passed succesfully!")
            return 1

    def forward_pass(self, model_input, decompose_attention_batch_size: int, output_pred = False):
        """
        Perform a foward pass once the hook have been properly set, and return
        the attention, mlp and residual intermediate outputs
        """

        if len(self.masking_hook_handles)>0:
            raise ValueError("[LLM Hooked] You forgot to remove the masking hooks while doing extraction! This might cause troubles. Stop here.")

        # useful dimensions
        d_batch = 1
        d_seq = model_input.input_ids.shape[-1]
        d_layer = self.get_n_layers()        
        nb_heads = self.get_nb_head()
        d_head = self.get_head_size()
        d_hidden = self.get_hidden_size()

        with torch.no_grad():
            timings = {}
            # Forward pass
            """
            If the hook are properly set, then the caches are filled with
            intermediate vectors during the foward pass
            """
            start_time = time.time()
            with torch.inference_mode():
                output = self.model(
                    model_input.input_ids.to(self.get_model_device()), 
                    model_input.attention_mask.to(self.get_model_device()))
            timings['forward_pass'] = time.time() - start_time

            # control that the shape of the vectors saved in the cache is correct
            self.control_cache_shape(d_seq)

            # Extract residual stream (includes final layer norm)
            residual_stream = torch.stack(output.hidden_states).detach() # (layer, batch, d_seq, d_hidden)
            residual_stream = residual_stream.permute(1, 0, 2, 3)  # [batch, layer, d_seq, d_hidden]
            assert residual_stream.shape == (d_batch, d_layer+1, d_seq, d_hidden), f"Invalid shape for residual stream" # layer+1 because it includes input embedings

            # Extract residual outputs (before final layer norm)
            residual_outputs = torch.stack(self.caches['residual']['output']) # (layer, batch, d_seq, d_hidden)
            residual_outputs = residual_outputs.permute(1, 0, 2, 3)  # [batch, layer, d_seq, d_hidden]
            
            # Linearize final layer norm
            start_time = time.time()
            linearized_norm = self.linearise_final_norm(residual_outputs[0, -1])
            timings['linearize_final_layer_norm'] = time.time() - start_time

            # Reshape MLP outputs
            mlp_outputs = torch.stack(self.caches['mlp']['output'])
            mlp_outputs = mlp_outputs.permute(1, 0, 2, 3)  # [batch, layer, d_seq, d_hidden]

            # Linearize post mlp norm (None if non-existent)
            post_mlp_norms_linear = [None for l in range(self.get_n_layers())]
            for l in range(self.get_n_layers()):
                post_mlp_norms_linear[l] = self.linearise_post_mlp_norm(mlp_outputs[0,l],l)

            # Prepare attention-related tensors
            values = torch.stack(self.caches['attention']['value'])
            values = values.permute(1, 0, 2, 3, 4)  # [batch, later, head, seq, head_dim]
            attn_weights = torch.stack(self.caches['attention']['attn_weight'])  # [layers, batch, head, seq_y, seq_x]
            attn_weights = attn_weights.permute(1, 0, 2, 3, 4)  # [batch, layer, head, seq_y, seq_x]
            outputs_attn = torch.stack(self.caches['attention']['output'])  # [layers, batch, seq, hidden_dim]
            outputs_attn = outputs_attn.permute(1, 0, 2, 3,)  # [batch, layer, seq, hidden_dim]

            # Linearize post attn norm (None if non-existent)
            post_attn_norms_linear = [None for l in range(self.get_n_layers())]
            for l in range(self.get_n_layers()):
                post_attn_norms_linear[l] = self.linearise_post_attention_norm(outputs_attn[0,l],l)

            # Decompose attention outputs
            start_time = time.time()
            # Perform attention decomposition with batching for memory efficiency
            # The output is moved to the CPU to handle large memory requirements
            head_outputs = self.decompose_attention_with_batching(
                value=values[0],
                attn_weights=attn_weights[0],
                y_batch_size=decompose_attention_batch_size,
                x_batch_size=decompose_attention_batch_size,
            )

            # Add the batch dimension back to the output
            head_outputs = head_outputs.unsqueeze(0)
            # Transpose head outputs 
            # from [batch, layer, seq_out, seq_in, hidden_dim, head] 
            # to [batch, layer, seq_out, seq_in, head, hidden_dim]
            head_outputs = head_outputs.transpose(3, 4)
            assert head_outputs.shape == (d_batch, d_layer, d_seq, d_seq, nb_heads, d_hidden), f"Invalid shape for head outputs"
            # Record the time taken for attention decomposition
            timings['decompose_attention'] = time.time() - start_time

            # Get the next token prediction
            next_token_logits = output.logits[:, -1, :]  # Logits for the last token
            next_token_id = torch.argmax(next_token_logits, dim=-1)  # Predicted token ID

            self.do_sanity_checks(residual_stream, residual_outputs, linearized_norm, head_outputs, outputs_attn, mlp_outputs, post_mlp_norms_linear, post_attn_norms_linear)

            # Print timings
            if self.track_time:
                print("[LLM Hooked] Timings (in seconds):")
                for key, value in timings.items():
                    print(f"[LLM Hooked] {key}: {value:.6f}")

        # remove the batch dimension and go back to CPU
        # NB: residual_stream = WITH final normalisation
        # while residual_outputs = NO final normalisation
        # apart from that, the two representations are equal but extracted using different means
        # -> useful for sanity checks
        residual_stream = residual_stream[0].cpu() # [layer, sequence, hidden_dim]
        head_outputs = head_outputs[0].cpu() # [layer, seq_out, seq_in, head, hidden_dim]
        mlp_outputs = mlp_outputs[0].cpu() # [layer, seq, d_hidden]
        # residual_outputs = residual_outputs[0].cpu() # [layer, seq, d_hidden]

        output_tuple = (residual_stream, mlp_outputs, head_outputs, linearized_norm, post_mlp_norms_linear, post_attn_norms_linear)
        if output_pred:
            output_tuple += (next_token_id, output.logits,)
        return output_tuple


    def decoder_masked(self, decoder_input, graph_masks):
        raise NotImplementedError("Subclasses should implement this method.")

    def mask_decoder_hook(self, graph_attn_mask, mask_before_softmax, graph_mlp_mask, attn_residual_mask, mlp_residual_mask, layer_index):
        """
        Creates a forward hook for a decoder block to apply custom masks during the forward pass.

        Args:
            graph_attn_mask: Mask tensor for attention heads.
            mask_before_softmax: Whether to apply the mask before softmax.
            graph_mlp_mask: Mask tensor for the MLP.
            attn_residual_mask: Mask for the attention residual connection.
            mlp_residual_mask: Mask for the MLP residual connection.

        Returns:
            A function to be used as a forward hook.
        """
        # Note: might be inefficient beacuse it runs the forward 2 times: the normal one plus the masked one.
        with torch.no_grad():
            def fn(module, inputs, kwargs, output):
                # print(kwargs.keys())
                # Unpack the input arguments.
                # Note: With with_kwargs=True, some arguments are in kwargs.
                hidden_states = inputs[0]
                position_embeddings = kwargs['position_embeddings'] if 'position_embeddings' in kwargs else kwargs['position_embeddings_global']
                attention_mask = kwargs['attention_mask']  # shape [1, 1, seq, seq]
                original_decoder_output = output[0]

                # Prepare arguments for the masked decoder forward pass.
                decoder_input = (module, hidden_states, attention_mask, position_embeddings)
                graph_masks = (graph_attn_mask, mask_before_softmax, graph_mlp_mask, attn_residual_mask, mlp_residual_mask)

                # Recompute the decoder output with the mask applied.
                masked_decoder_outputs = self.decoder_masked(decoder_input, graph_masks)

                # Optionally perform a sanity check to verify reconstruction.
                if self.test_mode:
                    # Recompute the decoder output with all masks set to None (no masking)
                    sanity_masks = (None, None, None, None, None)
                    reconstruct_decoder_output = self.decoder_masked(decoder_input, sanity_masks)
                    if isinstance(reconstruct_decoder_output, tuple):
                        reconstruct_decoder_output = reconstruct_decoder_output[0]
                    sanity_check.sanity_check_decoder(original_decoder_output, reconstruct_decoder_output, self.half_precision)
                    print(f"[LLM Hooked][Layer {layer_index}] Decoder output reconstruction sanity check succesfully passed.")

                # Replace the original output with the masked output.
                output = masked_decoder_outputs
                return masked_decoder_outputs
            return fn

    def prepare_attention_mask(self, layer_edges, seq_len):
        """
        Prepare an attention mask tensor for a given layer based on the provided edges.

        Args:
            layer_edges (list): List of edge dictionaries, each with 'source', 'target', and 'name' keys.
            seq_len (int): Sequence length.

        Returns:
            torch.Tensor: Attention mask of shape [num_head, seq_len, seq_len], where 0 means not masked and 1 means masked.
        """
        # Initialize the mask with ones (all masked by default)
        mask = torch.ones((self.get_nb_head(), seq_len, seq_len))  # [head, seq target, seq source]
        # For each edge in the layer, set the corresponding mask position to 0 (not masked)
        for edge in layer_edges:
            source = edge['source']
            target = edge['target']
            head_index = edge['head']
            # Set mask to 0 for this head, target, and source (i.e., allow this connection)
            mask[head_index, target, source] = 0.0  # if the head is in the graph, don't mask it
        return mask
    
    def random_attention_mask(self, p, seq_len):
        """
        Generate a random attention mask for a given probability.

        Args:
            p (float): Probability of masking an attention connection (i.e., setting it to 1).
            num_head (int): Number of attention heads.
            seq_len (int): Sequence length.

        Returns:
            torch.Tensor: Attention mask of shape [num_head, seq_len, seq_len], where
                        0 means not masked (connection kept), 1 means masked (connection dropped).
        """
        # Generate random values for each attention connection
        rand = torch.rand((self.get_nb_head(), seq_len, seq_len))  # [head, seq target, seq source]
        # Mask positions where random value is less than p
        # 0 means not masked (connection kept), 1 means masked (connection dropped)
        mask = (rand < p).float()
        return mask

    def prepare_mlp_mask(self, layer_edges, seq_len):
        """
        Prepare an MLP mask tensor for a given layer based on the provided edges.

        Args:
            layer_edges (list): List of edge dictionaries, each with 'source' key.
            seq_len (int): Sequence length.

        Returns:
            torch.Tensor: MLP mask of shape [seq_len], where 1 means not masked (contribution kept), 0 means masked.
        """
        # Initialize the mask with zeros (all masked by default)
        mask = torch.zeros(seq_len)  # [seq]
        # For each edge in the layer, set the corresponding mask position to 1 (not masked)
        for edge in layer_edges:
            source = edge['source']
            mask[source] = 1.0  # if the node is in the graph, don't mask it
        return mask
    
    def random_mlp_mask(self, p, seq_len):
        """
        Generate a random MLP mask for a given probability.

        Args:
            p (float): Probability of masking an MLP contribution (i.e., setting it to 0).
            seq_len (int): Sequence length.

        Returns:
            torch.Tensor: MLP mask of shape [seq_len], where 1 means not masked (contribution kept), 0 means masked.
        """
        # Generate random values for each position in the sequence
        rand = torch.rand(seq_len)  # [seq]
        # Mask positions where random value is less than or equal to p
        # 1 means not masked (contribution kept), 0 means masked (contribution dropped)
        mask = (rand > p).float()
        return mask

    def prepare_residual_mask(self, layer_edges, seq_len):
        mask = torch.zeros(seq_len) # [head, seq target, seq source]
        # in the mask, 1 means that the input/head contribution is not masked
        for edge in layer_edges:
            source = edge['source']
            mask[source] = 1.0 # if the head is in the graph, don't mask it
        return mask
    
    def random_residual_mask(self, p, seq_len):
        rand = torch.rand(seq_len) # [head, seq]
        # in the mask, 1 means that the input/head contribution is not masked
        mask = (rand > p).float()
        return mask

    def prepare_hooks_mask(self, graph, seq_len, mask_before_softmax, inverse, keep_residual):
        """
        Prepares and registers forward hooks for each decoder layer in the model to apply custom masks
        (attention, MLP, and residual) during the forward pass.

        Args:
            graph: Either a float (probability for random masking) or a graph dict with 'nodes' and 'edges'.
            seq_len: Sequence length.
            mask_before_softmax: Whether to apply the attention mask before softmax.
            inverse: If True, mask the elements in the graph instead of keeping them.
            keep_residual: If True, never mask the residual connections.

        Returns:
            hook_handles: List of hook handles for later removal.
            nb_non_masked_edges: Total number of non-masked edges across all layers.
            total_nb_edges: Total number of possible edges across all layers.
        """

        layers = self.get_layers()
        n_layers = self.get_n_layers()

        if not isinstance(graph, float):
            # Extract edges from the graph if not using random masking
            # group edges by layer (because we'll have one mask per layer and module)
            edges = [{'attention':[], 'residual-attention':[], 'residual-mlp':[], 'mlp':[]} for _ in range(n_layers)] # this is specific for sequential architectures
            for src, trgt, data in graph.edges(data=True):
                source_layer_index, source_token_index = graph.node_position(src)
                target_layer_index, target_token_index = graph.node_position(trgt)
                edge_type = data['name'].split('_')[0]
                if self.get_architecture_type() == 'sequential':
                    layer_index = target_layer_index // 2 - 1# divide by two because mlp and attention were counted as separated layers
                else:
                    layer_index = target_layer_index - 1 
                if edge_type == 'attention':
                    label = data['name'].split('_')[-1]
                    head_index = int(label[1:label.index('t')])
                    edge = {
                        'source': source_token_index,
                        'target': target_token_index,
                        'head': head_index,
                    }
                else:
                    edge = {
                        'source': source_token_index,
                    }
                edges[layer_index][edge_type].append(edge)

        # List to store hook handles for later removal
        hook_handles = []
        nb_non_masked_edges = 0
        total_nb_edges = 0
        for layer_index, layer in enumerate(layers):
            # Prepare attention mask for this layer
            if isinstance(graph, float):
                # Use random mask: graph is the mask probability
                graph_attn_mask = self.random_attention_mask(graph, seq_len)
            else:
                # Use the graph to prepare the mask
                attn_edges = edges[layer_index]['attention']
                graph_attn_mask = self.prepare_attention_mask(attn_edges, seq_len)
            # Prepare MLP mask for this layer
            if isinstance(graph, float):
                graph_mlp_mask = self.random_mlp_mask(graph, seq_len)
            else:
                mlp_edges = edges[layer_index]['mlp']
                graph_mlp_mask = self.prepare_mlp_mask(mlp_edges, seq_len)
            # Prepare residual masks for this layer
            if isinstance(graph, float):
                attn_residual_mask = self.random_residual_mask(graph, seq_len)
                mlp_residual_mask = self.random_residual_mask(graph, seq_len)
            else:
                attn_res_edges = edges[layer_index]['residual-attention']
                attn_residual_mask = self.prepare_residual_mask(attn_res_edges, seq_len)
                mlp_res_edges = edges[layer_index]['residual-mlp']
                mlp_residual_mask  = self.prepare_residual_mask(mlp_res_edges, seq_len)

            # If inverse is True, flip the masks (mask the elements in the graph instead of keeping them)
            if inverse:
                graph_attn_mask = 1 - graph_attn_mask
                graph_mlp_mask = 1 - graph_mlp_mask
                attn_residual_mask = 1 - attn_residual_mask
                mlp_residual_mask = 1 - mlp_residual_mask
            
            # If keep_residual is True, never mask the residual connections
            if keep_residual:
                attn_residual_mask = torch.ones_like(attn_residual_mask)
                mlp_residual_mask = torch.ones_like(mlp_residual_mask)
            # Count the number of non-masked edges for statistics
            nb_attn_mask = (graph_attn_mask == 0).sum()
            nb_mlp_mask = (graph_mlp_mask == 1).sum()
            nb_attn_residual_mask = (attn_residual_mask == 1).sum()
            nb_mlp_residual_mask = (mlp_residual_mask == 1).sum()
            nb_non_masked_edges += (nb_attn_mask + nb_mlp_mask + nb_attn_residual_mask + nb_mlp_residual_mask).item()
            total_nb_edges += len(torch.flatten(graph_attn_mask)) + len(torch.flatten(graph_mlp_mask)) + len(torch.flatten(attn_residual_mask)) + len(torch.flatten(mlp_residual_mask))
            # Prepare hook input arguments
            hook_input = (graph_attn_mask, mask_before_softmax, graph_mlp_mask, attn_residual_mask, mlp_residual_mask)
            # Register the forward hook on the decoder layer
            handle = layer.register_forward_hook(self.mask_decoder_hook(*hook_input, layer_index), with_kwargs=True)
            # Save the hook handle for later removal
            self.masking_hook_handles.append(handle)
        return nb_non_masked_edges, total_nb_edges

    def prepare_input(self, sentence, next_word):
        # Ensure next_word starts with a space
        if not next_word.startswith(' '):
            next_word = ' ' + next_word

        # Tokenize next_word and ensure it's a single token
        tokenized_next_word = self.tokenizer.encode(next_word, add_special_tokens=False)
        if len(tokenized_next_word) != 1:
            tokenized_next_word = [tokenized_next_word[0]]

        # Tokenize the sentence and concatenate with next_word token
        if isinstance(sentence, str):
            sentence_tokens = self.tokenizer(sentence, return_tensors="pt").input_ids
            full_sentence_tokens = torch.cat([sentence_tokens, torch.tensor(tokenized_next_word).unsqueeze(0)], dim=1)
            # Prepare model inputs and labels
            input_ids = full_sentence_tokens[:, :-1]
            attention_mask = torch.ones_like(input_ids)
        elif hasattr(sentence, 'input_ids'):
            input_ids = sentence.input_ids.long()
            full_sentence_tokens = torch.cat([input_ids, torch.tensor(tokenized_next_word).unsqueeze(0)], dim=1)
        
        # send to model's device
        input_ids = input_ids.to(self.model.device)
        attention_mask = sentence.attention_mask.to(self.model.device)

        labels = torch.full_like(input_ids, -100)
        labels[0, -1] = full_sentence_tokens[0, -1]

        # Sanity checks
        assert labels[0, -1] == full_sentence_tokens[0, -1], "Labels should be the last token of the full sentence"
        assert input_ids[0, -1] == full_sentence_tokens[0, -2], "Input IDs should be the penultimate token"

        return input_ids, attention_mask, labels

    def compute_surprisal(self, input_tuple, output_logits=False):
        """
        Compute surprisal metrics for a given sentence and next word.

        Args:
            input_tuple (tuple): (sentence, next_word)
            output_logits (bool): If True, also return raw logits.

        Returns:
            tuple: (loss, entropy, predicted_token_id, rank) [+ logits if output_logits]
        """
        sentence, next_word = input_tuple
        input_ids, attention_mask, labels = self.prepare_input(sentence, next_word)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits.view(-1, self.config.vocab_size).cpu()
            loss, entropy, predicted_token_id, rank = surprisal(logits, labels)

        if output_logits:
            return loss, entropy, predicted_token_id, rank, outputs.logits
        else:
            return loss, entropy, predicted_token_id, rank

    def forward_with_graph(
        self,
        input_tuple,
        graph,
        inverse=False,
        keep_residual=False,
        output_logit=False
    ):
        """
        Runs a forward pass of the model with masking hooks applied according to the provided graph.

        Args:
            input_tuple: Tuple containing (sentence, next_word).
            graph: Either a float (probability for random masking) or a graph (LLM_Graph_NX).
            mask_before_softmax: Whether to apply the attention mask before softmax.
            inverse: If True, mask the elements in the graph instead of keeping them.
            keep_residual: If True, never mask the residual connections.
            output_logit: If True, also return logits.

        Returns:
            Tuple containing (loss, entropy, predicted_token_id, rank, nb_non_masked_edges, total_nb_edges).
        """
        mask_before_softmax = False

        if len(self.extraction_hook_handles)>0:
            raise ValueError("[LLM Hooked] You forgot to remove the extaction hooks while doing masking! This might cause troubles. Stop here.")

        input_sentence = input_tuple[0]
        if isinstance(input_sentence, str):
            seq_len = len(self.tokenizer.encode(input_sentence))
        elif hasattr(input_sentence, 'input_ids'):
            seq_len = input_sentence.input_ids.size(-1)
        

        # If a graph is provided, prepare and register masking hooks for the model
        if graph is not None:
            nb_non_masked_edges, total_nb_edges = self.prepare_hooks_mask(
                graph, seq_len, mask_before_softmax, inverse, keep_residual
            )
        # Run the model forward pass and compute surprisal (loss, entropy, etc.)
        result = self.compute_surprisal(input_tuple, output_logit)
        # Remove masking hooks to restore the model to its original state
        self.remove_masking_hooks()
        # Return the result along with the number of non-masked and total edges
        return result + (nb_non_masked_edges, total_nb_edges)


# total add a function to remove extraction hooks