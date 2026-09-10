import torch

def prepare_attention_mask(layer_edges, seq_len, nb_head):
        """
        Prepare an attention mask tensor for a given layer based on the provided edges.

        Args:
            layer_edges (list): List of edge dictionaries, each with 'source', 'target', and 'name' keys.
            seq_len (int): Sequence length.

        Returns:
            torch.Tensor: Attention mask of shape [num_head, seq_len, seq_len], where 0 means not masked and 1 means masked.
        """
        # Initialize the mask with ones (all masked by default)
        mask = torch.ones((nb_head, seq_len, seq_len))  # [head, seq target, seq source]
        # For each edge in the layer, set the corresponding mask position to 0 (not masked)
        for edge in layer_edges:
            source = edge['source']
            target = edge['target']
            head_index = edge['head']
            # Set mask to 0 for this head, target, and source (i.e., allow this connection)
            mask[head_index, target, source] = 0.0  # if the head is in the graph, don't mask it
        return mask
    
def prepare_mlp_mask(layer_edges, seq_len):
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

def prepare_residual_mask(layer_edges, seq_len):
    mask = torch.zeros(seq_len) # [head, seq target, seq source]
    # in the mask, 1 means that the input/head contribution is not masked
    for edge in layer_edges:
        source = edge['source']
        mask[source] = 1.0 # if the head is in the graph, don't mask it
    return mask

def prepare_mask(graph, seq_len, nb_head, n_layers, inverse, keep_residual):
    """
    Prepares and registers forward hooks for each decoder layer in the model to apply custom masks
    (attention, MLP, and residual) during the forward pass.

    Args:
        seq_len: Sequence length.
        inverse: If True, mask the elements in the graph instead of keeping them.
        keep_residual: If True, never mask the residual connections.

    Returns:
        nb_non_masked_edges: Total number of non-masked edges across all layers.
        total_nb_edges: Total number of possible edges across all layers.
    """

    if not isinstance(graph, float):
        # Extract edges from the graph if not using random masking
        # group edges by layer (because we'll have one mask per layer and module)
        edges = [{'attention':[], 'residual-attention':[], 'residual-mlp':[], 'mlp':[]} for _ in range(n_layers)] # this is specific for sequential architectures
        for src, trgt, data in graph.edges(data=True):
            source_layer_index, source_token_index = graph.node_position(src)
            target_layer_index, target_token_index = graph.node_position(trgt)
            edge_type = data['name'].split('_')[0]
            # divide by two because mlp and attention were counted as separated layers
            # OLD: layer_index = target_layer_index // 2 - 1
            layer_index = (target_layer_index - 1) // 2
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

    nb_non_masked_edges = 0
    total_nb_edges = 0
    graph_mask = []
    for layer_index in range(n_layers):
        # Prepare attention mask for this layer
        masks = {}
        # Use the graph to prepare the mask
        attn_edges = edges[layer_index]['attention']
        graph_attn_mask = prepare_attention_mask(attn_edges, seq_len, nb_head)
        # Prepare MLP mask for this layer
        mlp_edges = edges[layer_index]['mlp']
        graph_mlp_mask = prepare_mlp_mask(mlp_edges, seq_len)
        # Prepare residual masks for this layer
        attn_res_edges = edges[layer_index]['residual-attention']
        attn_residual_mask = prepare_residual_mask(attn_res_edges, seq_len)
        mlp_res_edges = edges[layer_index]['residual-mlp']
        mlp_residual_mask  = prepare_residual_mask(mlp_res_edges, seq_len)

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
        # embed everything in a dict
        masks['attention'] = graph_attn_mask
        masks['mlp'] = graph_mlp_mask
        masks['residual-attention'] = attn_residual_mask
        masks['residual-mlp'] = mlp_residual_mask
        graph_mask.append(masks)
        # Count the number of non-masked edges for statistics
        nb_attn_mask = (graph_attn_mask == 0).sum()
        nb_mlp_mask = (graph_mlp_mask == 1).sum()
        nb_attn_residual_mask = (attn_residual_mask == 1).sum()
        nb_mlp_residual_mask = (mlp_residual_mask == 1).sum()
        nb_non_masked_edges += (nb_attn_mask + nb_mlp_mask + nb_attn_residual_mask + nb_mlp_residual_mask).item()
        total_nb_edges += len(torch.flatten(graph_attn_mask)) + len(torch.flatten(graph_mlp_mask)) + len(torch.flatten(attn_residual_mask)) + len(torch.flatten(mlp_residual_mask))
    return graph_mask, nb_non_masked_edges, total_nb_edges


