import torch
import networkx as nx
from typing import Optional, Tuple
from tqdm import tqdm
from src.llm_hooked.llm_hooked import LLM_Hooked
import src.llm_hooked.sanity_checks as sanity_checks
import random
import numpy as np
import math
import pickle
import os
import networkx.readwrite.json_graph as json_graph

HALF_PRECISION = True
EPS = 1e-3 if HALF_PRECISION else 1e-6
EPS_NORM = 1e-10

def get_weight_stats(weights):
    """
    Calculates and returns basic statistics about a list of edge weights.

    Args:
        weights (list): A list of numbers (int or float) representing
                        the edge weights.

    Returns:
        dict: A dictionary containing the statistics, or None if the
              input list is empty.
    """
    if len(weights) == 0:
        return None  # Return None if the list is empty

    sorted_weights = sorted(weights)
    count = len(sorted_weights)

    # --- Basic Stats ---
    zero_weights = sum(1 for w in sorted_weights if w == 0) / count
    min_weight = sorted_weights[0]
    max_weight = sorted_weights[-1]

    # --- Measures of Central Tendency ---
    mean_weight = sum(sorted_weights) / count

    # Median calculation
    mid = count // 2
    if count % 2 == 0:
        # Even number of elements
        median_weight = (sorted_weights[mid - 1] + sorted_weights[mid]) / 2
    else:
        # Odd number of elements
        median_weight = sorted_weights[mid]

    # --- Measures of Dispersion ---
    # Standard Deviation
    sum_sq_diff = sum((w - mean_weight) ** 2 for w in sorted_weights)
    variance = sum_sq_diff / count
    std_dev = math.sqrt(variance)

    # --- Percentiles (Quartiles) ---
    def get_percentile(sorted_data, percentile):
        """Helper to calculate percentile"""
        if not sorted_data:
            return None
        n = len(sorted_data)
        index = (percentile / 100) * (n - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(math.floor(index))
            upper_index = int(math.ceil(index))
            fraction = index - lower_index
            return (sorted_data[lower_index] + 
                    (sorted_data[upper_index] - sorted_data[lower_index]) * fraction)

    percentile_25 = get_percentile(sorted_weights, 25)
    percentile_75 = get_percentile(sorted_weights, 75)
    interquartile_range = percentile_75 - percentile_25

    return {
        "count": count,
        "zero_weights": zero_weights,
        "min": min_weight,
        "max": max_weight,
        "mean": mean_weight,
        "median": median_weight,
        "std_deviation": std_dev,
        "25th_percentile (Q1)": percentile_25,
        "75th_percentile (Q3)": percentile_75,
        "interquartile_range (IQR)": interquartile_range
    }

def compute_leverage_scores(X, k=None):
    # 1. Compute SVD
    # U: Unitary arrays (Direction)
    # S: Singular values (Importance/Variance)
    U, S, Vt = torch.linalg.svd(X.float(), full_matrices=False)
    
    # 2. Select top k components (if k is provided)
    if k is not None:
        U = U[:, :k]
        
    # 3. Leverage scores are simply the squared norms of the rows of U
    scores = torch.sum(U**2, axis=1)
    
    return scores

AVAILABLE_IMPORTANCE_MODES = [None, 'cosim', 'norm', 'random', 'norm_l2', 'sim', 'ifr', 'lev_1', 'lev_2', 'lev_4', 'lev_8', 'lev_16', 'lev_32', 'lev_64', 'lev_128', 'lev_256']

class LLM_Graph_NX(nx.MultiDiGraph):
    """
    Represents the model's computation graph using NetworkX.
    
    Nodes are identified by a tuple: (layer, token_index).
    Node and edge data (vectors, types, importance) are stored as attributes.
    """
    def __init__(
        self,
        llm_hooked: Optional[LLM_Hooked] = None,
        input_sentence: Optional[str] = None,
        importance_mode: Optional[str] = None,
        **attr
    ):
        """
        Initialize the LLM_Graph.
        llm_hooked: (optionnal, could be None, for instance when representing the trace)
        hf_config: Hugging Face config object describing the LLM hyperparameters.
        """
        super().__init__(**attr)
        self.llm_hooked = llm_hooked
        self.input_sentence = input_sentence
        if llm_hooked is not None:
            self.architecture_type = llm_hooked.get_architecture_type()
            self.model_input, n_tokens = self.preprocess_input_sentence()
            self.graph['n_layers'] = llm_hooked.get_n_layers()
            self.graph['n_tokens'] = n_tokens
            self.graph['n_heads'] = llm_hooked.get_nb_head()
        self.graph['output_node_index'] = None
        self.graph['input_node_index'] = []
        if importance_mode not in AVAILABLE_IMPORTANCE_MODES:
            raise ValueError(f"Selected importance mode '{importance_mode}' is not available.")
        self.importance_mode = importance_mode
        # todo add a test_mode that performs sanity checks?

    # def from_graph(self, graph: nx.MultiDiGraph):
    #     # Add nodes
    #     self.add_nodes_from(graph.nodes(data=True))
    #     # Add edges with their attributes
    #     self.add_edges_from(graph.edges(data=True, keys=True))

    def get_graph_n_layers(self):
        if self.architecture_type == 'sequential':
            return 2 * self.graph['n_layers']
        else:
            raise NotImplementedError("Subclasses should implement this method.")

    def get_size(self):
        """
        Counts all edges in the graph
        """
        return self.number_of_edges()

    def remove_disconnected_nodes(self):
        """
        Removes nodes from the graph (self) that cannot reach the output node.
        This method modifies the graph in-place.
        """
        output_node = self.get_output_node()
        
        try:
            # 1. Get all nodes that CAN reach the output node
            ancestor_nodes = set(nx.bfs_tree(self, output_node, reverse=True).nodes())
        except Exception as e:
            # Handle cases where output_node might not be in graph or no path exists
            print(f"Warning: Could not find ancestors ({e}).")
            if output_node in self:
                ancestor_nodes = {output_node}
            else:
                ancestor_nodes = set()
                print("Warning: Output node not found. Graph may be emptied.")

        # 2. Get all nodes currently in the graph
        all_nodes = set(self.nodes())
        
        # 3. Find nodes to remove (all nodes MINUS the ones we want to keep)
        nodes_to_remove = all_nodes - ancestor_nodes
        
        # 4. Modify the graph *in-place*
        self.remove_nodes_from(nodes_to_remove)

    def get_edge_weight_stats(self):
        edges_weight = np.array([
            d.get('weight').half()
            for u, v, k, d in self.edges(keys=True, data=True)
        ])
        stats = get_weight_stats(edges_weight)
        return stats
    
    def preprocess_input_sentence(self):
        if isinstance(self.input_sentence, str):
            model_input = self.llm_hooked.get_tokenizer()([self.input_sentence], padding=True, return_tensors="pt")
        elif isinstance(self.input_sentence, torch.Tensor):
            model_input = self.input_sentence
        else:
            raise ValueError(f"Input sentence {self.input_sentence} has not the correct type. Expecting str or torch.Tensor")
        return model_input, model_input.input_ids.shape[-1]

    def get_output_node(self):
        return self.graph.get('output_node_index')

    def set_output_node(self, node_index: int):
        if node_index not in self.nodes:
            raise ValueError(f"Node {node_index} does not exist in the graph.")
        self.graph['output_node_index'] = node_index

    def get_input_nodes(self):
        return self.graph.get('input_node_index')

    def clear_input_nodes(self):
        self.graph['input_node_index'] = []

    def add_input_node(self, node_index: int):
        if node_index not in self.nodes:
            raise ValueError(f"Node {node_index} does not exist in the graph.")
        self.graph['input_node_index'].append(node_index)

    def update_edge_importance(self):
        #1) compute importance with batching
        #2) update edge importance accordingly
        raise NotImplementedError("Subclasses should implement this method.")

    def populate_graph_with_importance(self, decompose_attention_batch_size: int):
        """
        Populate the graph using a forward pass from one input sentence.
        This needs to be implemented with model-specific hooks.
        """
        output = self.llm_hooked.forward_pass(self.model_input, decompose_attention_batch_size, output_pred=True)
        if self.architecture_type == 'sequential':
            residual_stream, mlp_outputs, head_outputs, linearized_norm, post_mlp_norms_linear, post_attn_norms_linear = output[:6]
            self.populate_with_edge_importance_sequential(residual_stream, head_outputs, mlp_outputs, linearized_norm, post_attn_norms_linear, post_mlp_norms_linear) # this operation is model specific
        else:
            raise NotImplementedError(f"Graph population from transformer {self.architecture_type} is not implemented.")
        output_logits = output[-1]
        self.remove_disconnected_nodes()
        return output_logits # return the logit outputed by the model after the forward pass
        
    def node_idx(self, layer_i: int, token_j: int):
        return layer_i*self.graph['n_tokens'] + token_j

    def node_position(self, node_idx: int):
        layer_i = node_idx // self.graph['n_tokens']
        token_j = node_idx % self.graph['n_tokens']
        return layer_i, token_j


    def compute_edge_importance_softmax(self, node_vector: torch.Tensor, edge_vectors: torch.Tensor, distance_norm = 1):
        """
        Compute the importance score of the incoming edges of a node,
        with temperature scaling using softmax.

        Args:
            node_vector (torch.Tensor): The vector of the target node.
            edge_vectors (torch.Tensor): Tensor of incoming edge vectors.
            temperature (float): Controls the "peakiness" of the distribution.
                T < 1.0 (e.g., 0.1): More peaky (favors high-proximity edges).
                T > 1.0 (e.g., 2.0): Flatter (makes edges more equal).
                T = 1.0: Standard softmax.
            distance_norm (int): The 'p' norm to use for distance calculation.
        """
        if self.importance_mode == 'ifr':
            # Vectorized computation for efficiency
            diff = edge_vectors - node_vector.unsqueeze(0)  # [n_edges, d_h]
            distance = torch.norm(diff, p=distance_norm, dim=1)  # [n_edges]
            node_norm = torch.norm(node_vector, p=distance_norm)  # Scalar
            
            # Original proximity scores (these are our "logits")
            proximity = (node_norm - distance).clamp(0) # [n_edges]
            
            importance = proximity / proximity.sum()
        
        elif self.importance_mode.startswith('lev'):
            k = int(self.importance_mode.split('_')[-1])
            lev_score = compute_leverage_scores(edge_vectors, k)
            importance = lev_score / lev_score.sum()
            
        elif self.importance_mode == 'cosim':
            # Normalize vectors for cosine similarity
            edge_vectors_norm = edge_vectors / edge_vectors.norm(p=2, dim=1, keepdim=True)  # [n_edges, d_h]
            node_vector_norm = node_vector / node_vector.norm(p=2, dim=0, keepdim=True)  # [d_h]
            
            # Cosine similarity: dot product of normalized vectors
            cosine_similarity = torch.einsum('eh, h -> e', edge_vectors_norm, node_vector_norm)
            
            # (0 to 1, where 0 = opposite, 0.5 = orthogonal, 1 = identical)
            importance = (1+cosine_similarity)/2
            
            # importance = proximity / proximity.sum()

        elif self.importance_mode == 'norm':
            edge_norm = torch.norm(edge_vectors, p=distance_norm, dim=1)
            # print('norm', edge_norm)
            importance = edge_norm / edge_norm.sum()
            # print('importance', importance)
        
        elif self.importance_mode == 'norm_l2':
            edge_norm = torch.norm(edge_vectors, p=2, dim=1)
            # print('norm', edge_norm)
            importance = edge_norm / edge_norm.sum()
            # print('importance', importance)

        elif self.importance_mode == 'random':
            n_edges = edge_vectors.shape[0]
            importance = torch.rand(n_edges) # no need to normalise
        
        else:
            raise ValueError(f"Unsupported mode: {self.importance_mode}")

        return importance


    # def compute_edge_importance(self, node_vector: torch.Tensor, edge_vectors: torch.Tensor, distance_norm = 1):
    #     """
    #     Compute the importance score of the incoming edges of a node.
    #     """
    #     if self.importance_mode == 'dist':
    #         # Vectorized computation for efficiency
    #         diff = edge_vectors - node_vector.unsqueeze(0)  # [n_edges, d_h]
    #         distance = torch.norm(diff, p=distance_norm, dim=1)  # [n_edges]
    #         node_norm = torch.norm(node_vector, p=distance_norm)  # Scalar
    #         proximity = (node_norm - distance).clamp(min=EPS)  # [n_edges]
    #         importance = proximity / proximity.sum()  # [n_edges]
    #     elif self.importance_mode == 'random':
    #         n_edges = edge_vectors.shape[0]
    #         importance = torch.rand(n_edges) # no need to normalise
    #     else:
    #         raise ValueError(f"Unsupported mode: {self.importance_mode}")

    #     return importance

    def add_attention_sequential_layer_to_graph(
        self,
        graph_layer: int,
        token: int,
        before_mlp: torch.Tensor,
        before_attn: torch.Tensor,
        head_out: torch.Tensor,
    ):
        """
        Adds attention-related edges for a given token at a specific layer.

        Args:
            graph_layer (int): Index of the current graph layer.
            token (int): Token index for which attention is computed.
            before_mlp (torch.Tensor): Residual stream before MLP (node vector).
            before_attn (torch.Tensor): Residual stream before attention (edge vector).
            head_out (torch.Tensor): Attention head outputs [n_tokens, n_heads, hidden_dim].
        """
        # Node vector: residual stream before MLP
        node_vector = before_mlp

        # Prepare edge labels and vectors
        edge_labels = ['residual']
        edge_vectors = [before_attn.unsqueeze(0)]

        # Add attention edges from all tokens and heads
        for src_token in range(self.graph['n_tokens']):
            if src_token <= token: # we do not had the connections that are masked by causal attention
                for head in range(self.graph['n_heads']):
                    edge_vec = head_out[src_token, head].unsqueeze(0).detach()
                    label = f'h{head}t{src_token}'
                    edge_labels.append(label)
                    edge_vectors.append(edge_vec)
                    # if head_out[src_token, head].sum() == 0:
                    #     print(f"[{token}][{src_token}][{head}]", head_out[src_token, head].sum())

        edge_vectors = torch.cat(edge_vectors, dim=0)

        # Compute importance scores for each edge
        importance = self.compute_edge_importance_softmax(node_vector, edge_vectors)

        # Add node to the graph
        self.add_node(self.node_idx(graph_layer, token))

        # Add weighted edges to the graph
        for i, label in enumerate(edge_labels):
            if label == 'residual':
                # Residual connection from previous layer
                self.add_edge(
                    self.node_idx(graph_layer - 1, token),
                    self.node_idx(graph_layer, token),
                    weight=importance[i],
                    name='residual-attention'
                )
            else:
                # Attention connection from source token and head
                head_idx = int(label[1:label.index('t')])
                src_token = int(label[label.index('t') + 1:])
                self.add_edge(
                    self.node_idx(graph_layer - 1, src_token),
                    self.node_idx(graph_layer, token),
                    weight=importance[i],
                    name=f'attention_{label}'
                    )

    def add_mlp_sequential_layer_to_graph(
        self,
        graph_layer: int,
        token: int,
        after_mlp: torch.Tensor,
        before_mlp: torch.Tensor,
        mlp_out: torch.Tensor,
        final_norm_linear: torch.Tensor,
    ):
        """
        Adds MLP-related edges for a given token at a specific layer.

        Args:
            graph_layer (int): Index of the current graph layer.
            token (int): Token index for which MLP is computed.
            after_mlp (torch.Tensor): Residual stream after MLP (node vector).
            before_mlp (torch.Tensor): Residual stream before MLP (edge vector).
            mlp_out (torch.Tensor): Output of the MLP (edge vector).
            final_norm_linear (torch.Tensor or None): final normalisation).
        """

        if final_norm_linear is not None:
            # after mlp already contain the last normalisation because it is extracted from the hugging face residual stream
            mlp_out = torch.einsum('j,j->j', final_norm_linear, mlp_out.to(final_norm_linear.device)).cpu()
            before_mlp = torch.einsum('j,j->j', final_norm_linear, before_mlp.to(final_norm_linear.device)).cpu()

        if self.llm_hooked.test_mode:
            sanity_checks.sanity_check_after_mlp(before_mlp, after_mlp, mlp_out, self.llm_hooked.half_precision)

        node_vector = after_mlp
        edge_vectors = torch.stack([before_mlp, mlp_out], dim=0)

        # Compute importance scores for residual and MLP edges
        importance = self.compute_edge_importance_softmax(node_vector, edge_vectors)

        # Add residual connection edge
        self.add_edge(
            self.node_idx(graph_layer - 1, token),
            self.node_idx(graph_layer, token),
            weight=importance[0],
            name='residual-mlp'
        )

        # Add MLP connection edge
        self.add_edge(
            self.node_idx(graph_layer - 1, token),
            self.node_idx(graph_layer, token),
            weight=importance[1],
            name='mlp'
        )

    def populate_with_edge_importance_sequential(self, residual_stream, head_outputs, mlp_outputs, final_norm_linear, post_attn_norms_linear, post_mlp_norms_linear):
        """
        Build a sequential transformer computation graph (attention before MLP).
        Each layer is split into two: attention and MLP.
        TODO: could be improve by computing the all importance scores in one step (unsing batching)
        Currently is it done node by node.
        """
        n_layers, n_tokens, _, n_heads, hidden_dim = head_outputs.shape
        # residual_stream = [layer, sequence, hidden_dim]
        # mlp_outputs = [layer, seq, d_hidden]
        # sanity checks -- control that the dimension matchs
        assert n_tokens == self.graph['n_tokens']
        assert n_layers == self.graph['n_layers']
        assert n_heads == self.graph['n_heads']
        assert residual_stream.size(0) == (self.graph['n_layers']+1) # residual stream includes the embedding layer
        assert residual_stream.size(1) == self.graph['n_tokens']
        assert residual_stream.size(2) == hidden_dim
        assert mlp_outputs.size(0) == self.graph['n_layers']
        assert mlp_outputs.size(1) == self.graph['n_tokens']
        assert mlp_outputs.size(2) == hidden_dim

        # Add input (layer 0) nodes
        # input nodes corresponds to the input embeddings
        for token in range(self.graph['n_tokens']):
            self.add_node(self.node_idx(0, token))
            self.add_input_node(self.node_idx(0, token))

        for layer in tqdm(range(self.graph['n_layers']), desc="[LLM Graph] Populate graph:"):                    
            for token in range(self.graph['n_tokens']):
                # update tqdm description instead of printing a new line
                try:
                    if tqdm._instances:
                        next(iter(tqdm._instances)).set_description_str(
                            f"[LLM Graph] Populate graph: token {token} in layer {layer}"
                        )
                except Exception:
                    pass
                """
                # Diagram of one transformer layer (sequential := attention first, then MLP)
                # Not showing the post attn and post mlp normalisations
                #
                # residual_stream[layer][token]  <-- "before_attn"
                #            |
                #            |  (norm + self-attention: keys/queries/values -> per-head outputs)
                #            v
                # head_out[src_token, head]  -- attention head outputs (for each source token and head)
                #            |
                #   (aggregate over src positions and heads)
                #            v
                # attn_out  = summed attention contribution for current token
                #            |
                # before_mlp = before_attn + attn_out   <-- input to the MLP (:= residual connection before the MLP)
                #            |
                #            |  (MLP)
                #            v
                # mlp_out   = MLP output contribution for current token
                #            |
                # after_mlp = before_mlp + mlp_out = residual_stream[layer + 1][token] (:= residual connection after the MLP)
                #
                # Quick reference:
                # - before_attn: residual_stream[layer][token]
                # - head_out:    head_outputs[layer, token] with shape [n_tokens, n_heads, hidden_dim];
                #                head_out[src_token, head] is the per-head vector used as an incoming attention edge
                # - attn_out:    aggregated attention output for the current token (attn contribution)
                # - before_mlp:  before_attn + attn_out (node vector before MLP; do NOT include final layer norm)
                # - mlp_out:     mlp_outputs[layer, token] (MLP contribution edge; do NOT include final layer norm)
                # - after_mlp:   residual_stream[layer + 1][token] (node vector after MLP; include final layer norm)
                """
                before_attn = residual_stream[layer][token].detach()
                head_out = head_outputs[layer, token].detach()
                if all([norm is not None for norm in post_attn_norms_linear]):
                    post_attn_norm = post_attn_norms_linear[layer][token]
                    head_out = torch.einsum('j,thj->thj', post_attn_norm, head_out.to(post_attn_norm.device)).cpu()
                attn_out = head_out.sum(dim=0).sum(dim=0).detach()
                before_mlp = before_attn + attn_out
                mlp_out = mlp_outputs[layer, token].detach()
                if all([norm is not None for norm in post_mlp_norms_linear]):
                    post_mlp_norm = post_mlp_norms_linear[layer][token]
                    mlp_out = torch.einsum('j,j->j', post_mlp_norm, mlp_out.to(post_mlp_norm.device)).cpu()
                after_mlp = residual_stream[layer + 1][token].detach() # len(residual_stream)=n_layers+1 because it includes the input_embeddings

                # if layer==0:
                #     print('head_out.shape', head_out.shape)
                #     if not all(torch.isfinite(attn_out)):
                #         for head_2 in range(n_heads):
                #             print(f'head_out/{head_2}', torch.isnan(head_out[:, head_2]).sum())

                if self.llm_hooked.test_mode:
                    # Sanity check: verify (before_attn + attn_out) ≈ (after_mlp - mlp_out)
                    if layer < self.graph['n_layers']-1:
                        sanity_checks.sanity_check_before_mlp(
                            before_attn, attn_out, after_mlp, mlp_out, self.llm_hooked.half_precision)

                for offset, layer_type in enumerate(['attention', 'mlp']):
                    current_layer = 1 + 2 * layer + offset  # +1 for input, double for attn/mlp split
                    # print(f"[LLM Graph] Layer {layer}; Token {token}; Current Layer {current_layer}")

                    if layer_type == 'attention':
                        self.add_attention_sequential_layer_to_graph(
                            current_layer,
                            token, 
                            before_mlp, 
                            before_attn, 
                            head_out,
                            )
                    elif layer_type == 'mlp':
                        self.add_mlp_sequential_layer_to_graph(
                            current_layer, 
                            token, 
                            after_mlp,
                            before_mlp, 
                            mlp_out,
                            final_norm_linear[token] if layer == self.graph['n_layers']-1 else None, # None if not last layer
                            )
        # If mode=cosim the edge do not sum to 1
        # if self.llm_hooked.test_mode:
        #     # sanity check: Checks that for every node, the sum of incoming edge weights is approximately 1.0.
        #     self.check_edge_weights_sum_to_one()
        #     print("[LLM Graph] Sanity check passed succesfully!")
        # add output node index = last token of last layer
        self.set_output_node(self.node_idx(current_layer, self.graph['n_tokens']-1))

    def check_edge_weights_sum_to_one(self, atol=1e-3):
        """
        Checks that for every node, the sum of incoming edge weights is approximately 1.0.
        Returns a list of nodes where the sum deviates from 1.0 by more than atol.
        """
        invalid_nodes = []
        for node in self.nodes:
            in_edges = self.in_edges(node, data=True)
            total_weight = torch.tensor([edge_data.get('weight') for _, _, edge_data in in_edges]).sum()
            # check if the node is a leaf (input); in that case it does not have incoming edges
            if node in self.graph['input_node_index']:
                expected_sum = torch.zeros_like(total_weight)
            else:
                expected_sum = torch.ones_like(total_weight)
            if not torch.isclose(total_weight, expected_sum, atol=atol):
                invalid_nodes.append(node)
                error_message = (f"[LLM Graph] ERROR: Node {node} incoming edge weights sum mismatch. "
                      f"current={total_weight.item():.6f}, expected={expected_sum.item():.6f}")
                raise AssertionError(error_message)
            

    def _remove_from_frontier(self, frontier_list, frontier_dict, index_to_remove, node_to_remove):
        """
        Helper function for O(1) removal from our frontier data structures.
        Swaps the item-to-remove with the last item, then pops the end of the list.
        """
        # Get the node at the end of the list
        last_node = frontier_list.pop()
        
        # Remove the target node from the dictionary
        # Use .pop() with a default to avoid errors if it's already gone
        frontier_dict.pop(node_to_remove, None) 
        
        # If the node we're removing *wasn't* the last node,
        # we need to move the 'last_node' into its slot.
        if index_to_remove < len(frontier_list):
            frontier_list[index_to_remove] = last_node
            # Update the dictionary to point to the new index
            frontier_dict[last_node] = index_to_remove

    # --- NEW: Fully Optimized Subgraph Generator ---
    
    def get_random_connected_subgraph(self, n_edges):
        """
        Generates a random, connected subgraph of 'n_edges' starting
        from the output_node and traversing backwards.
        
        This version is highly optimized:
        1. Uses an O(1) data structure for the node frontier.
        2. Memoizes and shuffles available edges for each node,
           making edge selection an O(1) 'pop()' operation.
        """
        start_node = self.get_output_node()
        if start_node not in self:
            print(f"Error: Start node {start_node} not in graph.")
            return

        # 1. Initialize
        subgraph_edges = []
        edges_added_set = set()
        nodes_in_subgraph = {start_node}
        
        # --- Frontier Data Structures ---
        # Maps node -> list of its *remaining* available in-edges
        available_edges_map = {} 
        
        # List/dict combo for O(1) random choice/removal of nodes
        # that *still have available edges*.
        frontier_list = [] 
        frontier_dict = {} 

        # --- Seed the frontier ---
        initial_edges = list(self.in_edges(start_node, keys=True))
        if initial_edges:
            random.shuffle(initial_edges)
            available_edges_map[start_node] = initial_edges
            frontier_list.append(start_node)
            frontier_dict[start_node] = 0
        
        # print(f"Generating random subgraph with {n_edges} edges...")

        # 2. Loop until we have enough edges or run out of options
        while len(subgraph_edges) < n_edges and frontier_list:
            
            # --- O(1) random choice from nodes with available edges ---
            rand_index = random.randrange(len(frontier_list))
            current_node = frontier_list[rand_index]
            
            # --- O(1) selection of a random edge ---
            edge_list = available_edges_map[current_node]
            chosen_edge = edge_list.pop()
            
            # 3. Check if this edge was already added
            # (e.g., from another node's out-edges)
            if chosen_edge in edges_added_set:
                # If that was the last available edge for this node, remove it
                if not edge_list:
                    self._remove_from_frontier(frontier_list, frontier_dict, rand_index, current_node)
                continue # Try again
                
            # 4. Add the new edge
            (u, v, k) = chosen_edge
            subgraph_edges.append(chosen_edge)
            edges_added_set.add(chosen_edge)
            
            # 5. If this edge leads to a new node, add it to the frontier
            if u not in nodes_in_subgraph:
                nodes_in_subgraph.add(u)
                
                # This is the only expensive O(degree) part,
                # and it only happens ONCE per *node*, not per edge.
                new_edge_list = list(self.in_edges(u, keys=True))
                
                if new_edge_list:
                    random.shuffle(new_edge_list)
                    available_edges_map[u] = new_edge_list
                    
                    # Add new node to frontier
                    new_index = len(frontier_list)
                    frontier_list.append(u)
                    frontier_dict[u] = new_index

            # 6. If we just popped the last edge, remove node from frontier
            if not edge_list:
                self._remove_from_frontier(frontier_list, frontier_dict, rand_index, current_node)
                        
        if len(subgraph_edges) < n_edges:
            print(f"Warning: Could only find {len(subgraph_edges)} edges (target was {n_edges}).")

        # 7. Create the final subgraph
        final_subgraph = self.edge_subgraph(subgraph_edges).copy()
        
        return final_subgraph
    
    def pre_save(self):
        """
        pre-saves the LLM_Graph_NX to a disctionary.
        
        The graph structure is serialized using networkx.json_graph,
        and other safe attributes are stored in a dictionary.
        The 'llm_hooked' attribute is explicitly NOT saved.
        """
        
        # 1. Serialize the graph data (nodes, edges, self.graph attributes)
        # We pass 'self' because this class IS the graph.
        try:
            graph_data_dict = json_graph.adjacency_data(self)
        except Exception as e:
            print(f"[LLM_Graph_NX] Error serializing graph data: {e}")
            return

        # 2. Create the dictionary with other attributes to save
        data_to_save = {
            'graph_data': graph_data_dict,
            'input_sentence': self.input_sentence,
            'importance_mode': self.importance_mode,
            # Note: All self.graph['...'] attributes are saved
            # as part of the graph_data_dict.
        }
        
        return data_to_save

    def save(self, file_path: str):
        """
        Saves the LLM_Graph_NX to a pickle file.
        
        The graph structure is serialized using networkx.json_graph,
        and other safe attributes are stored in a dictionary.
        The 'llm_hooked' attribute is explicitly NOT saved.
        """
        data_to_save = self.pre_save()
        
        # 3. Save the dictionary using pickle
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"[LLM_Graph_NX] Successfully saved to {file_path}")
        except Exception as e:
            print(f"[LLM_Graph_NX] Error saving pickle file: {e}")



def load_from_dict(data_to_load: dict, llm_hooked: LLM_Hooked):
    """
    Loads graph data from a dict and reconstructs
    the LLM_Graph_NX object.
    
    Requires an active llm_hooked object to be passed in.
    """
    # 2. Re-hydrate the core NetworkX graph from the dictionary
    try:
        saved_graph = json_graph.adjacency_graph(data_to_load['graph_data'])
    except Exception as e:
        print(f"[LLM_Graph_NX] Error re-hydrating graph from JSON: {e}")
        return None
        
    # 3. Create a new LLM_Graph_NX instance.
    # We pass the essential, non-serialized llm_hooked object
    # and the saved input_sentence.
    new_llm_graph = LLM_Graph_NX(
        llm_hooked=llm_hooked,
        input_sentence=data_to_load.get('input_sentence')
    )
    
    # 4. Clear the blank-slate graph data that __init__ created
    new_llm_graph.clear() 

    # 5. Copy the nodes, edges, and graph attributes from the saved graph
    new_llm_graph.add_nodes_from(saved_graph.nodes(data=True))
    new_llm_graph.add_edges_from(saved_graph.edges(data=True, keys=True))
    new_llm_graph.graph.update(saved_graph.graph)
    
    # 6. Manually set the other saved attributes
    new_llm_graph.importance_mode = data_to_load.get('importance_mode', None)
    
    # 7. Re-set the model_input based on the loaded sentence and new hook
    if new_llm_graph.input_sentence and new_llm_graph.llm_hooked:
         new_llm_graph.model_input, _ = new_llm_graph.preprocess_input_sentence()
    else:
         new_llm_graph.model_input = None
 
    return new_llm_graph


def load_from_file(file_path: str, llm_hooked: LLM_Hooked):
    """
    Loads graph data from a pickle file and reconstructs
    the LLM_Graph_NX object.
    
    Requires an active llm_hooked object to be passed in.
    """
    if not os.path.exists(file_path):
        print(f"[LLM_Graph_NX] Error: File not found at {file_path}")
        return None

    # 1. Load the dictionary from pickle
    try:
        with open(file_path, 'rb') as f:
            data_to_load = pickle.load(f)
    except Exception as e:
        print(f"[LLM_Graph_NX] Error loading pickle file: {e}")
        return None

    new_llm_graph = load_from_dict(data_to_load, llm_hooked)
    
    print(f"[LLM_Graph_NX] Successfully loaded from {file_path}")
    return new_llm_graph
    
    # def get_random_connected_subgraph(self, n_edges):
    #     """
    #     Generates a random, connected subgraph of 'n_edges' starting
    #     from 'start_node' by performing a randomized expansion.

    #     The "connected" property means all nodes in the subgraph are
    #     reachable from 'start_node' by following the graph's directed edges.

    #     Args:
    #         n_edges (int): The target number of edges for the subgraph.

    #     Returns:
    #         nx.MultiDiGraph: The randomly generated, connected subgraph.
    #     """

    #     start_node = self.get_output_node()

    #     # 1. Initialize
    #     subgraph_edges = []
        
    #     # active_nodes are nodes in the subgraph that we can expand from
    #     active_nodes = [start_node] 
        
    #     # nodes_in_subgraph tracks all nodes we've added
    #     nodes_in_subgraph = {start_node} 
        
    #     # Use a set for efficient checking of added edges
    #     edges_added_set = set()

    #     # 2. Loop until we have enough edges or run out of options
    #     while len(subgraph_edges) < n_edges and active_nodes:
            
    #         # Pick a random node from our known subgraph to expand from
    #         current_node = random.choice(active_nodes)
            
    #         # Get all its in-edges (because we are doing it in a backward fashion, starting from the end of the network)
    #         # We must use keys=True for MultiDiGraph
    #         in_edges = list(self.in_edges(current_node, keys=True))
            
    #         # Filter in-edges we've already added
    #         available_edges = [e for e in in_edges if e not in edges_added_set]
            
    #         if not available_edges:
    #             # This node is exhausted (no more *new* out-edges)
    #             # Remove it from the active list and try another node
    #             active_nodes.remove(current_node)
    #             continue
                
    #         # 3. Pick a random edge and add it
    #         chosen_edge = random.choice(available_edges)
    #         (u, v, k) = chosen_edge
            
    #         subgraph_edges.append(chosen_edge)
    #         edges_added_set.add(chosen_edge)
            
    #         # 4. If this edge leads to a new node, add it to our lists
    #         if u not in nodes_in_subgraph:
    #             nodes_in_subgraph.add(u)
    #             active_nodes.append(u)
                
    #     # 5. Create the final subgraph from the list of edges
    #     # .edge_subgraph() automatically includes all nodes for those edges
    #     # We use .copy() to make it an independent graph
    #     final_subgraph = self.edge_subgraph(subgraph_edges).copy()
        
    #     return final_subgraph


"""
TODO:
- treewidth
- assortativity by degree
- max clique
- dominating set
- min maximal matching
- centrality (plot a distribution). Per layer centrality? Per token centrality?
- transitivity: inform on the presence of hihly connected node clusters
- communities: girvan_newman, greedy_modularity_communities
- number_strongly_connected_components(
- condensation -> useful for graph visualisation?
"""