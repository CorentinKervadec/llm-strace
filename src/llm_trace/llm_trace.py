from src.llm_graph.llm_graph_nx import LLM_Graph_NX, load_from_dict
from src.llm_hooked.llm_hooked import LLM_Hooked
from src.llm_graph.graph_utils import get_graph_constructor
import networkx as nx
import time
from tqdm import tqdm
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx.readwrite.json_graph as json_graph
import collections
import os

EPS = 1e-6

THRESHOLD_STRACE = {
    'norm_nucleus': [
        1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95,
        .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
    ],
    'norm_l2_nucleus': [
        1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95,
        .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
    ],
    'ifr_nucleus': [
        1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95,
        .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
    ],
    'ifr_threshold': [
        1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
    'sim_nucleus': [
        1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95,
        .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
    ],
    'sim_threshold': [
        1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 1e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
    'norm_threshold': [
        1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
    'norm_l2_threshold': [
        1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
    'cosim_threshold': [
        0.1, 0.2, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.49, 0.5, 0.51, 0.525, 0.55, 0.575, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0
    ],
    'lev_256_threshold':
    [
        1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
    'lev_128_threshold':
    [
        1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
    'lev_64_threshold':
    [
        1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
    'lev_32_threshold':
    [
        1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
    'lev_16_threshold':
    [
        1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
    'lev_8_threshold':
    [
        1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
    'lev_4_threshold':
    [
        1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
    'lev_2_threshold':
    [
        1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.0
    ],
}

class MockTokenized:
    def __init__(self, ids, mask):
        self.input_ids = ids
        self.attention_mask = mask

    def copy(self):
        return MockTokenized(self.input_ids.clone(), self.attention_mask.clone())


def get_total_variation(original_logits, graph_logits):
    """
    Compute the total variation distance between the probability distributions
    of the original and graph logits for the last token.

    Args:
        original_logits: The logits from the original (full) model.
        graph_logits: The logits from the masked/graph model.

    Returns:
        float: The total variation distance.
    """
    # Extract logits for the last token in the batch
    orig_last = np.array(original_logits[0, -1].cpu().float())
    graph_last = np.array(graph_logits[0, -1].cpu().float())

    # Compute probability distributions (softmax)
    orig_probs = np.exp(orig_last - np.max(orig_last))
    orig_probs /= orig_probs.sum()
    graph_probs = np.exp(graph_last - np.max(graph_last))
    graph_probs /= graph_probs.sum()

    # Compute total variation distance
    tv_distance = 0.5 * np.sum(np.abs(orig_probs - graph_probs))
    return tv_distance

def get_intersection_nucleus(original_logits,graph_logits):
    """
    Compute the intersection size of the nucleus (top-p tokens) between original and graph logits.

    Args:
        original_logits: The logits from the original (full) model.
        graph_logits: The logits from the masked/graph model.

    Returns:
        shared_nucleus (int): The highest percentage p (from a set of candidates) for which the top-p nuclei (token indices covering p% of probability mass) are identical between the original and graph logits. Returns 0 if no such p is found.
        nucleus_size (int): The size of the nucleus (number of tokens) at the shared_nucleus percentage. Returns 0 if no shared nucleus is found.
    """
    orig_last = np.array(original_logits[0, -1].cpu().float())
    graph_last = np.array(graph_logits[0, -1].cpu().float())

    # Compute probability distributions (softmax)
    orig_probs = np.exp(orig_last - np.max(orig_last))
    orig_probs /= orig_probs.sum()
    graph_probs = np.exp(graph_last - np.max(graph_last))
    graph_probs /= graph_probs.sum()

    sorted_indices = np.argsort(orig_probs)[::-1]
    cumulative_orig = np.cumsum(orig_probs[sorted_indices])

    sorted_graph_indices = np.argsort(graph_probs)[::-1]
    cumulative_graph = np.cumsum(graph_probs[sorted_graph_indices])
    # print("Get nucleus")
    # print("cumulative_orig", cumulative_orig)
    # print("cumulative_graph", cumulative_graph)

    p_candidates = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    shared_nucleus = 0
    nucleus_size = 0
    for p in p_candidates[::-1]: # reverse, start with highest nucleus
        cutoff = p / 100.0
        # Find indices that cover top-p% of probability mass in original
        orig_top_indices = sorted_indices[:np.searchsorted(cumulative_orig, cutoff, side='left') + 1]
        if len(orig_top_indices) == 0:
            orig_top_indices = sorted_indices[:1]  # at least top-1
        graph_top_indices = sorted_graph_indices[:np.searchsorted(cumulative_graph, cutoff, side='left') + 1]
        if len(graph_top_indices) == 0:
            graph_top_indices = sorted_graph_indices[:1]
        # Check if the top-p nuclei are equal (as list, order matters)
        if list(orig_top_indices) == list(graph_top_indices):
            shared_nucleus = p
            nucleus_size = len(orig_top_indices)
            break
    # print("orig_top_indices", orig_top_indices)
    # print("graph_top_indices", graph_top_indices)
    return shared_nucleus, nucleus_size

def get_nucleus(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Finds the nucleus (top-p) set of indices from a 1D logit tensor.

    The nucleus is defined as the smallest set of items (tokens)
    whose cumulative probability mass is greater than or equal to p/100.

    Args:
        logits (torch.Tensor): A 1D (vocab_size,) tensor of raw logits.
        p (int or float): The probability mass percentile (e.g., 95 for 95%).
                                 
    Returns:
        torch.Tensor: A 1D tensor of the nucleus indices (e.g., [0, 3, 7]).
    """
    
    # --- 1. Input Validation and Setup ---
    if not 0 <= p <= 100:
        raise ValueError(f"p must be between 0 and 100, but got {p}")
    
    if logits.dim() != 1:
        raise ValueError(f"Input tensor must be 1D, but got {logits.dim()} dimensions")
        
    p_threshold = p / 100.0

    # --- 2. Convert Logits to Probabilities ---
    probs = torch.softmax(logits, dim=-1) # Shape: (vocab_size,)
    
    # --- 3. Sort Probabilities ---
    # Sort in descending order to find the items with the most mass
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # --- 4. Get Cumulative Probability ---
    # This gives us a running total of probability mass
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # --- 5. Find the Nucleus Set ---
    # We need to find the items to *remove*.
    # These are any items *after* the cumulative sum has already
    # crossed the threshold.
    # We get the "previous" cumulative sum by rolling the tensor.
    prev_cumulative_probs = torch.roll(cumulative_probs, 1, dims=-1)
    prev_cumulative_probs[0] = 0.0 # Set the first element to 0.0
    
    # Any item whose *predecessor* sum already crossed the threshold
    # should be removed.
    mask_to_remove_sorted = prev_cumulative_probs > p_threshold
    
    # The nucleus set is the opposite of the "remove" mask
    mask_to_keep_sorted = torch.logical_not(mask_to_remove_sorted)
    
    # --- 6. Map Back to Original Indices ---
    # We now have the correct True/False mask, but it's relative
    # to the *sorted* order. We need to "un-sort" it to match
    # the original logit tensor's order.
    
    # Create an empty boolean mask with the same shape as probs
    final_mask = torch.zeros_like(mask_to_keep_sorted)
    
    # Use scatter_ to place the True/False values at their
    # original positions.
    final_mask.scatter_(dim=-1, index=sorted_indices, src=mask_to_keep_sorted)
    
    # --- 7. Format Output ---
    # Return a 1D tensor of indices
    return torch.where(final_mask)[0]

class LLM_STRACE:
    """
    Represents the model's stratified trace, a subgraph of the full computational graph.
    STRACE := Stratified TRACE
    """
    def __init__(self, input_tuple, llm_hooked: LLM_Hooked, track_time=False):
        # we consider three stratification: by size (relatively to the full graph), by reconstruction error, by importance score (tau)
        self.graph = None
        self.strata_rel_size: List[int] = []
        self.strata_raw_size: List[int] = [] 
        self.strata_tau: List[int] = []
        self.strata_index: List[int] = []
        self.strata_connected_to_input: List[bool] = [] # bool that tells if the stratum is connected to the input
        self.strata_reco_tv = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_reco_nu = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_nucleus_60 = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_loss = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_entropy = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.track_time = track_time
        self.input_tuple = input_tuple
        self.original_logits = None # the logit prediction from the full model
        self.llm_hooked = llm_hooked
        self.nb_strata = None

    def reset_strace(self):
        self.strata_rel_size = [] 
        self.strata_raw_size = [] 
        self.strata_tau = []
        self.strata_connected_to_input = []
        self.nb_strata = None
        self.strata_index = []
        self.strata_reco_tv = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_reco_nu = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_nucleus_60 = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_loss = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_entropy = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
    
    def initialize_graph(self, importance_mode):
        self.graph = get_graph_constructor(self.llm_hooked.model_name)(
            self.llm_hooked,
            self.input_tuple[0],
            importance_mode,
        )

    def populate_graph(self, batch_size, print_stats=True, start_token_idx=0):
        original_logits = self.graph.populate_graph_with_importance(batch_size, start_token_idx)
        self.original_logits = original_logits
        if print_stats:
            stats = self.graph.get_edge_weight_stats()
            for (k,s) in stats.items():
                print(f"[STRACE][GRAPH STATS] {k}: {s}")
    
    def update_trace(self, new_last_token_id, batch_size, print_stats=True):
        # update the input // we ignore the next token
        # we assume that the input tuple was tokenized 
        self.input_tuple[0].input_ids = torch.concat([self.input_tuple[0].input_ids, new_last_token_id], dim=-1)
        self.input_tuple[0].attention_mask = torch.concat([self.input_tuple[0].attention_mask, torch.ones_like(new_last_token_id)], dim=-1)
        # reset the trace metrics
        self.reset_strace()
        # update the graph
        self.graph.add_new_last_token(new_last_token_id)
        # repopulate the graph, updating only the last token
        start_token_idx = self.input_tuple[0].input_ids.size(-1) - 1
        self.populate_graph(batch_size, print_stats, start_token_idx)


    def label_graph_with_stratum(self, initial_graph: LLM_Graph_NX, tau: float, stratum_index: int, mode: str):
        """
        Filters initial_graph to create a connected subgraph and labels
        the corresponding edges on `self.graph` with the stratum_index.

        The filtering method is determined by `mode`:
        - 'threshold': (Original) Keeps edges where `weight > tau`.
        - 'nucleus': (New) For each node, keeps the top-N incoming edges
                    whose weights sum to the cumulative mass `tau`.
        
        Returns a connected subgraph of initial graph.
        """

        # Get the output node
        output_node = initial_graph.get_output_node()

        time_stats = {
            'filtering': None,
            'subgraph': None,
            'output connected': None
        }

        if self.track_time:
            start_time = time.time()

        if mode == 'threshold':
            # Original mode: Filter by a minimum weight threshold
            edges_to_keep = [
                (u, v, k)
                for u, v, k, d in initial_graph.edges(keys=True, data=True)
                if d.get('weight') > tau
            ]
            
        elif mode == 'nucleus':
            # --- NUCLEUS MODE (Optimized) ---
            # Start from the output and traverse backwards, applying
            # nucleus filtering as we go. This combines filtering and
            # connectivity checking in one efficient pass.
            
            edges_to_keep = set()
            # Use a deque for an efficient BFS-style queue
            nodes_to_process = collections.deque([output_node])
            nodes_processed = {output_node} # Keep track of visited nodes to avoid cycles/redundancy

            while nodes_to_process:
                current_node = nodes_to_process.popleft()
                
                # --- Caching Optimization ---
                # Check if we have already computed and sorted the in-edges for this node
                sorted_in_edges = initial_graph.nodes[current_node].get('_cached_sorted_in_edges')

                if sorted_in_edges is None:
                    # --- Cache Miss: Compute, Sort, and Store ---
                    # Get all incoming edges for the current node
                    in_edges = initial_graph.in_edges(current_node, data=True, keys=True)
                    
                    # Store as (weight, edge_key) for sorting
                    weighted_edges = []
                    for u, v, k, data in in_edges:
                        weight = data.get('weight', 0)
                        if weight > 0: # Only consider edges with positive weight
                            weighted_edges.append((weight, (u, v, k)))
                    
                    if not weighted_edges:
                        # Cache the empty list to avoid re-computing
                        initial_graph.nodes[current_node]['_cached_sorted_in_edges'] = []
                        continue
                    
                    # Sort by weight, descending
                    weighted_edges.sort(key=lambda x: x[0], reverse=True)
                    
                    # Store in cache
                    initial_graph.nodes[current_node]['_cached_sorted_in_edges'] = weighted_edges
                    sorted_in_edges = weighted_edges
                    # print(f"Cache MISS for node {current_node}") # For debugging
                # else:
                    # print(f"Cache HIT for node {current_node}") # For debugging

                # --- Use the (now populated) cache ---
                if not sorted_in_edges:
                    continue # Nothing to process (from cache or new)
                
                # Add edges until we reach the cumulative 'tau' mass
                cumulative_weight = 0.0
                for weight, edge_key in sorted_in_edges:
                    # Add edge *first*, then check cumulative weight.
                    # This ensures at least one edge is added (if any exist).
                    if cumulative_weight < tau:
                        edges_to_keep.add(edge_key)
                        cumulative_weight += weight
                        
                        # Get the source node of this edge
                        source_node = edge_key[0] 
                        
                        # If we haven't processed this ancestor, add it to the queue
                        if source_node not in nodes_processed:
                            nodes_processed.add(source_node)
                            nodes_to_process.append(source_node)
                    else:
                        break # Stop as soon as we've crossed the threshold  
        else:
            raise ValueError(f"Unknown mode '{mode}'. Must be 'threshold' or 'nucleus'.")

        if self.track_time:
            time_stats['filtering'] = time.time() - start_time

        if self.track_time:
            start_time = time.time()
        # Create initial subgraph with filtered edges
        initial_subgraph = initial_graph.edge_subgraph(edges_to_keep)
        # note: initial_subgraph is the same tyme as full_graph
        if self.track_time:
            time_stats['subgraph'] = time.time() - start_time
        

        if self.track_time:
            start_time = time.time()
        # Get all nodes that have a path to the output node (including the output node)
        if output_node not in initial_subgraph:
            initial_subgraph = initial_graph.subgraph([output_node])
            print("The filtered graph does not include the output node. The threshold might be too high.")

        # Faster than nx.ancestor
        # reverse_tree = nx.bfs_tree(initial_subgraph, output_node, reverse=True)
        # connected_nodes = set(reverse_tree.nodes())
        connected_nodes = {output_node} | nx.ancestors(initial_subgraph, output_node)

        # Create the final subgraph containing only connected nodes
        # note: subgraph is the same tyme as full_graph
        subgraph = initial_subgraph.subgraph(connected_nodes).copy()
        if self.track_time:
            time_stats['output connected'] = time.time() - start_time
                
        subgraph.set_output_node(output_node)

        # Add input nodes to the LLM subgraph
        subgraph.clear_input_nodes() # it is important to clear it because it already contains the input nodes of the full graph
        for input_node in initial_graph.get_input_nodes():
            if input_node in subgraph:
                subgraph.add_input_node(input_node)

        # Check that the LLM subgraph contains at least one input node
        connected_to_input = True
        if len(subgraph.get_input_nodes()) == 0:
            connected_to_input = False
            # print("[STRACE] The subgraph does not contain input nodes. This might indicate a bug.")
        
        # go back to the full graph and label the edges that belongs to the stratum
        # Create a dictionary of all edges in the final subgraph
        # The key is the (u, v, k) tuple, the value is the stratum index
        edges_to_label = {
            (u, v, k): stratum_index 
            for u, v, k in subgraph.edges(keys=True)
        }
        # Apply all attributes to self.graph in one efficient operation
        nx.set_edge_attributes(self.graph, values=edges_to_label, name='stratum')

        
        return subgraph, connected_to_input, time_stats

    
    def extract_strace(self, threshold_values: list[float], mode: str):
        """
        Extract stratified traces at different weight thresholds.
        Args:
            threshold_values: List of tau values (weight thresholds) to filter edges.
                            Higher values create sparser subgraphs.
        For each threshold:
        1. Extracts a subgraph containing only edges with weights > threshold
        2. Stores the subgraph in self.strata
        3. Records the threshold value used
        """

        self.reset_strace()

        # give a default stratum label to each of the graph
        self.nb_strata = len(threshold_values) + 1 # +1 because we also include the full graph
        stratum_index = self.nb_strata
        nx.set_edge_attributes(self.graph, values=stratum_index, name='stratum')
        self.strata_tau = [1.0 if mode=='nucleus' else 0.0]
        self.strata_connected_to_input = [True]
        self.strata_raw_size = [self.graph.get_size()]
        self.strata_rel_size = [1.0]
        self.strata_index = [stratum_index]

        if self.track_time:
            extraction_times = []
            accu_time_stats = {
                'filtering': [],
                'subgraph': [],
                'output connected': []
            }

        if mode=='threshold':
            threshold_values.sort() # sort from low to high
        elif mode=='nucleus':
            threshold_values.sort(reverse=True) # sort from high to low

        try:
            pbar = tqdm(threshold_values, desc=f"[STRACE] Extracting strata | tau={threshold_values[0]:.6g}")
            iterator = pbar
        except Exception:
            pbar = None
            iterator = threshold_values

        initial_graph = self.graph # start with the full graph

        for threshold in iterator:
            stratum_index = stratum_index - 1 # decrease the index

            if self.track_time:
                start_time = time.time()

            # update progress bar description per-iteration (tqdm will advance automatically when used as the iterator)
            if pbar is not None:
                pbar.set_description(f"[STRACE] Extracting strata | tau={threshold:.6g}")

            # Extract subgraph containing edges above threshold
            subgraph, connected_to_input, time_stats = self.label_graph_with_stratum(initial_graph, threshold, stratum_index, mode)
            initial_graph = subgraph # update the initial graph to save time on the next iteration (the initial graph will be smaller)
            # Store both subgraph and its threshold
            self.strata_tau = [threshold] + self.strata_tau
            self.strata_connected_to_input = [connected_to_input] + self.strata_connected_to_input
            # update size
            self.strata_raw_size = [subgraph.get_size()] + self.strata_raw_size
            self.strata_rel_size = [float(self.strata_raw_size[0])/self.strata_raw_size[-1]] + self.strata_rel_size
            # update strata index
            self.strata_index = [stratum_index] + self.strata_index

            if self.track_time:
                end_time = time.time()
                # Store timing info for later statistics
                extraction_times.append(end_time - start_time)
                for key in time_stats:
                    accu_time_stats[key].append(time_stats[key])

        # ensure progress bar is closed if used
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

        if self.track_time:
            avg_time = sum(extraction_times) / len(extraction_times)
            max_time = max(extraction_times)
            min_time = min(extraction_times)
            print(f"[STRACE] Stratum extraction timing stats (seconds):")
            print(f"[STRACE] Average: {avg_time:.3f}, Max: {max_time:.3f}, Min: {min_time:.3f}")
            filtering_avg = sum(accu_time_stats['filtering']) / len(accu_time_stats['filtering'])
            subgraph_avg = sum(accu_time_stats['subgraph']) / len(accu_time_stats['subgraph'])
            output_avg = sum(accu_time_stats['output connected']) / len(accu_time_stats['output connected'])
            print(f"[STRACE] Filtering: {filtering_avg:.3f}, Subgraph: {subgraph_avg:.3f}, Output: {output_avg:.3f} (all average)")


    def auto_extract_strace(self, nb_stratum: int, log_tau: bool, mode: str):
        """
        Automatically extracts stratified traces by finding appropriate thresholds.
        Args:
            nb_stratum: Number of strata to extract between min and max edge weights
            log_tau: If True, thresholds increase logarithmically (evenly spaced in log-space)
        """
        if mode=='threshold':
            # Get all edge weights from the full graph
            output_node = self.graph.get_output_node()
            all_edge_weights = [d['weight'] for _, _, _, d in self.graph.edges(keys=True, data=True)]
            output_edge_weights = [d['weight'] for _, _, _, d in self.graph.in_edges(nbunch=[output_node], keys=True, data=True)]

            max_weight = max(output_edge_weights)  # Empty stratum above this, because we start from the output node
            min_weight = min(all_edge_weights)  # Full graph below this
        elif mode=='nucleus':
            min_weight = 0.01
            max_weight = 0.999 
        else:
            raise ValueError(f"Unknown mode '{mode}'. Must be 'threshold' or 'nucleus'.")
        
        # Compute thresholds either linearly or logarithmically
        if log_tau:
            if mode=='threshold':
                log_min = math.log(min_weight+EPS)
                log_max = math.log(max_weight)
                step = (log_max - log_min) / nb_stratum
                threshold_values = [math.exp(log_min + (i * step)) for i in range(nb_stratum)]
            if mode=='nucleus': # should be inv log
                exp_min = math.exp(min_weight)
                exp_max = math.exp(max_weight)
                step = (exp_max - exp_min) / nb_stratum
                threshold_values = [math.log(exp_min + (i * step)) for i in range(nb_stratum)]
        else:
            # Linear spacing
            weight_range = max_weight - min_weight
            step = weight_range / nb_stratum
            threshold_values = [min_weight + (i * step) for i in range(nb_stratum)]
                   
        # Extract strata using calculated thresholds
        self.extract_strace(threshold_values, mode)

    # def compute_stratum_size(self):
    #     full_size = self.graph.get_size()
    #     self.strata_raw_size = [stratum.get_size() for stratum_index in range(self.nb_strata)]
    #     self.strata_rel_size = [float(size)/float(full_size) for size in self.strata_raw_size]
    #     return self.strata_rel_size

    def print_graph_sizes_and_thresholds(self):
        """
        Prints the size of each stratum alongside its corresponding threshold value.
        """
        print(f"[STRACE] Displaying threshold and graph size:\n{'S':<5}{'c-input?':<10}{'Threshold':<25}{'Size (rel)':<15}{'Size (raw)':<15}{'TV':<15}{'Nucleus':<15}{'N60 (top 5)':<25}{'Inv. Nucleus':<15}{'inv-N60 (top 5)':<25}{'Rand. Nu.':<15}{'Inv. Rand. Nu.':<15}")
        print("-" * 9 * 25)
        for stratum, threshold, rel_size, raw_size, c_in, tv, nu, n60, inv_nu, inv_n60, r_nu, r_inv_nu in zip(
            self.strata_index, self.strata_tau, self.strata_rel_size, self.strata_raw_size, self.strata_connected_to_input, 
            self.strata_reco_tv['trace']['only'], self.strata_reco_nu['trace']['only'], self.strata_nucleus_60['trace']['only'], 
            self.strata_reco_nu['trace']['inverse'], self.strata_nucleus_60['trace']['inverse'],
            self.strata_reco_nu['random']['only'], self.strata_reco_nu['random']['inverse']):
            top5_nucleus_str = repr('|'.join(n60[:5]))
            inv_top5_nucleus_str = repr('|'.join(inv_n60[:5]))
            c_in_str = 'yes' if c_in else 'no'
            print(f"{stratum:<5}{c_in_str:<10}{threshold:<25.3e}{rel_size:<15.0%}{raw_size:<15.2e}{tv:<15.2e}{nu:<15}{top5_nucleus_str:<25}{inv_nu:<15}{inv_top5_nucleus_str:<25}{r_nu:<15}{r_inv_nu:<15}")


    def compute_stratum_reconstruction_error(self, do_random=False, do_inverse=False):
        output = self.llm_hooked.forward_with_graph(self.input_tuple, self.graph, inverse=False, keep_residual=False, output_logit=True)  
        full_logits = output[4]
        # full_logits = self.original_logits
        for i, stratum_index in tqdm(enumerate(self.strata_index), desc=f"[STRACE] Evaluating strata"):
            for random in [False, True]:
                if not do_random and random: # skip random
                    continue
                if random:
                    stratum_size = self.strata_raw_size[i]
                    stratum = self.graph.get_random_connected_subgraph(stratum_size)
                else:
                    def filter_edges(u, v, k):  
                        return self.graph[u][v][k].get('stratum') <= stratum_index
                    # 2. Create the subgraph view
                    stratum = nx.subgraph_view(self.graph, filter_edge=filter_edges)

                for inverse in [True, False]:
                    if not do_inverse and inverse: # skip inverse
                        continue
                    keep_residual = True if (inverse or random) else False # In case of inverse pruning, we keep the residual
                    unique_stratum_index = list(set(nx.get_edge_attributes(self.graph, 'stratum').values()))
                    output = self.llm_hooked.forward_with_graph(self.input_tuple, stratum, inverse=inverse, keep_residual=keep_residual, output_logit=True)  
                    loss, entropy, predicted_token_id, rank, graph_logits, nb_non_masked_edges, total_nb_edges = output
                    tv_original_graph = get_total_variation(full_logits, graph_logits)
                    shared_nucleus, shared_nucleus_size = get_intersection_nucleus(full_logits,graph_logits)
                    nucleus_indices = get_nucleus(graph_logits[0, -1].cpu(), 60)

                    key_tuple = ('random' if random else 'trace', 'inverse' if inverse else 'only') 
                    self.strata_reco_tv[key_tuple[0]][key_tuple[1]].append(tv_original_graph)
                    self.strata_reco_nu[key_tuple[0]][key_tuple[1]].append(shared_nucleus)
                    self.strata_nucleus_60[key_tuple[0]][key_tuple[1]].append([self.llm_hooked.tokenizer.decode(token_id) for token_id in nucleus_indices[:5]])
                    self.strata_loss[key_tuple[0]][key_tuple[1]].append(loss)
                    self.strata_entropy[key_tuple[0]][key_tuple[1]].append(entropy)

    # def save(self, file_path: str):
    #     """
    #     Saves a specific subset of the LLM_STRACE attributes to a file
    #     as a dictionary.
    #     """
    #     graph_data_dict = self.graph.pre_save()
    #     # Create the dictionary with only the attributes we want
    #     data_to_save = {
    #         'graph': graph_data_dict,
    #         'strata_rel_size': self.strata_rel_size,
    #         'strata_raw_size': self.strata_raw_size,
    #         'strata_reco_tv': self.strata_reco_tv,
    #         'strata_reco_nu': self.strata_reco_nu,
    #         'strata_tau': self.strata_tau,
    #         'strata_index': self.strata_index,
    #         'strata_connected_to_input': self.strata_connected_to_input,
    #         'strata_loss': self.strata_loss,
    #         'strata_entropy': self.strata_entropy,
    #         'input_tuple': self.input_tuple,
    #         'nb_strata': self.nb_strata,
    #         'nucleus_60': self.strata_nucleus_60,
    #         'nb_tokens': self.graph.graph['n_tokens'],
    #         # Note: self.original_logits is NOT saved
    #         # Note: self.llm_hooked is NOT saved
    #     }
        
    #     # Save the dictionary using pickle
    #     try:
    #         with open(file_path, 'wb') as f:
    #             pickle.dump(data_to_save, f)
    #         print(f"[LLM_STRACE] Successfully saved to {file_path}")
    #     except Exception as e:
    #         print(f"[LLM_STRACE] Error saving file: {e}")


    def save_light(self, file_path: str):
        """
        Saves the LLM_STRACE data to a compressed NumPy (.npz) file.
        The graph is serialized into efficient binary arrays.
        """
        
        # --- 1. Serialize the Graph efficiently ---
        graph = self.graph
        if isinstance(self.input_tuple[0], str):
            input = self.input_tuple[0]
        elif hasattr(self.input_tuple[0], 'input_ids'):
            input = np.array(self.input_tuple[0].input_ids, dtype=np.uint32)
        else:
            raise ValueError(f"[LLM TRACE SAVE] Input in wrong format: {self.input_tuple[0]}")
        next_token = self.input_tuple[1]
        

        # Create a mapping for edge names (e.g., 'mlp' -> 0, 'attn_h1t2' -> 1)
        # This is the single biggest space saver.
        edge_names = list(set(nx.get_edge_attributes(graph, 'name').values()))
        name_to_id = {name: i for i, name in enumerate(edge_names)}
        
        # Store edge data in compact numpy arrays
        num_edges = graph.number_of_edges()
        edge_list_array = np.empty((num_edges, 2), dtype=np.uint32) # (u, v)
        edge_weights_array = np.empty(num_edges, dtype=np.float16) # weights
        edge_stratum_array = np.empty(num_edges, dtype=np.float16) # stratum index
        edge_name_ids_array = np.empty(num_edges, dtype=np.uint16) # name IDs
        
        # Store node data
        node_list_array = np.array(list(graph.nodes()), dtype=np.uint32)
        
        i = 0
        for u, v, data in graph.edges(data=True):
            edge_list_array[i] = (u, v)
            edge_weights_array[i] = data.get('weight')
            edge_name_ids_array[i] = name_to_id.get(data.get('name'))
            edge_stratum_array[i] = data.get('stratum', -1)
            i += 1
            
        # if isinstance(self.input_tuple[0], str):
        #     input_tuple_save = self.input_tuple
        # elif hasattr(self.input_tuple[0], 'input_ids'):
        #     input_tuple_save = (
        #         np.array(self.input_tuple[0].input_ids, dtype=np.uint32),
        #         self.input_tuple[1]
        #     )

        # --- 2. Create the dictionary of all data to save ---
        # We save graph attributes (like n_tokens) as a single dict
        graph_attrs = dict(graph.graph)
        
        # We save all the other LLM_STRACE attributes
        data_to_save = {
            # Graph data
            'nodes': node_list_array,
            'edges': edge_list_array,
            'weights': edge_weights_array,
            'stratum': edge_stratum_array,
            'name_ids': edge_name_ids_array,
            'edge_name_map': np.array(edge_names), # Array of strings
            'graph_attrs': graph_attrs,
            
            # Other LLM_STRACE data
            'strata_rel_size': np.array(self.strata_rel_size, dtype=np.float16),
            'strata_raw_size': np.array(self.strata_raw_size, dtype=np.uint32),
            'strata_reco_tv': self.strata_reco_tv,
            'strata_reco_nu': self.strata_reco_nu,
            'strata_tau': np.array(self.strata_tau, dtype=np.float32),
            'strata_index': np.array(self.strata_index, dtype=np.uint16),
            'strata_connected_to_input': self.strata_connected_to_input,
            'strata_loss': self.strata_loss,
            'strata_entropy': self.strata_entropy,
            'input': input,
            'next_token': next_token,
            'nb_strata': self.nb_strata,
            'nucleus_60': self.strata_nucleus_60,
        }
        
        # --- 3. Save as a compressed .npz file ---
        try:
            # Use np.savez_compressed to save the dictionary
            # This is much more efficient than pickle for this data
            np.savez_compressed(file_path, **data_to_save)
            # print(f"[LLM_STRACE] Successfully saved to {file_path}.npz")
        
        except Exception as e:
            print(f"[LLM_STRACE] Error saving file: {e}")

# def load_from_file(file_path: str, llm_hooked: LLM_Hooked, track_time: bool = False):
#     """
#     Loads the saved strace data from a pickle file and reconstructs
#     the LLM_STRACE object.
    
#     Requires an active llm_hooked object to be passed in / or none if not used.
#     """
#     if not os.path.exists(file_path):
#         print(f"[LLM_STRACE] Error: File not found at {file_path}")
#         return None

#     try:
#         with open(file_path, 'rb') as f:
#             data_to_load = pickle.load(f)
#     except Exception as e:
#         print(f"[LLM_STRACE] Error loading pickle file: {e}")
#         return None

#     # Create a new instance using the saved input_tuple
#     new_strace = LLM_STRACE(data_to_load['input_tuple'], llm_hooked, track_time=track_time)

#     # Re-hydrate the graph
#     try:
#         graph_data_dict = data_to_load['graph']
#         new_strace.graph = load_from_dict(graph_data_dict, llm_hooked)
#     except Exception as e:
#         print(f"[LLM_STRACE] Error re-hydrating graph: {e}")
#         return None

#     # Re-populate other attributes, using .get() for safety
#     new_strace.strata_rel_size = data_to_load.get('strata_rel_size')
#     new_strace.strata_raw_size = data_to_load.get('strata_raw_size')
#     new_strace.strata_reco_tv = data_to_load.get('strata_reco_tv')
#     new_strace.strata_reco_nu = data_to_load.get('strata_reco_nu')
#     new_strace.strata_tau = data_to_load.get('strata_tau')
#     new_strace.strata_index = data_to_load.get('strata_index')
#     new_strace.strata_connected_to_input = data_to_load.get('strata_connected_to_input')
#     new_strace.strata_loss = data_to_load.get('strata_loss')
#     new_strace.strata_entropy = data_to_load.get('strata_entropy')
#     new_strace.nb_strata = data_to_load.get('nb_strata')
#     new_strace.strata_nucleus_60 = data_to_load.get('nucleus_60')
#     # Note: 'nb_tokens' is in graph.graph['n_tokens'] and will be
#     # loaded as part of the graph re-hydration.

#     print(f"[LLM_STRACE] Successfully loaded from {file_path}")
#     return new_strace    

def load_from_file_light(file_path: str, llm_hooked: LLM_Hooked):
    """
    Loads an LLM_STRACE object from a .npz file and re-attaches
    the llm_hooked object.
    """
    
    # Ensure the file path ends with .npz if the save method adds it
    if not file_path.endswith('.npz'):
            # If your save logic saves as 'foo.pickle', but you use .npz
            # you might need to adjust this.
            # Assuming file_path is the *exact* path.
            pass

    if not os.path.exists(file_path):
            # Try adding .npz if it's missing
            if os.path.exists(file_path + '.npz'):
                file_path += '.npz'
            else:
                print(f"[LLM_STRACE] Error: File not found {file_path}")
                return None

    try:
        # Load the compressed .npz file
        # allow_pickle=True is required to load dicts and non-array objects
        data = np.load(file_path, allow_pickle=True)
        if isinstance(data['input'], str):
            input_tuple = (data['input'].item(), data['next_token'].item())
        elif isinstance(data['input'], np.ndarray):
            input_ids = torch.tensor(data['input'])
            mock_tokenized = MockTokenized(input_ids, torch.zeros_like(input_ids))
            input_tuple = (mock_tokenized, data['next_token'].item())
        else:
            raise ValueError(f"[LLM TRACE LOAD] Input in wrong format: {data['input']}")
        # --- 1. Reconstruct the graph ---
        
        # Start with an empty graph and set attributes
        llm_graph = LLM_Graph_NX(
            llm_hooked, 
            input_sentence=input_tuple[0])
        
        llm_graph.clear()
        llm_graph.graph = data['graph_attrs'].item() # .item() extracts the dict
        
        # Re-create the reverse map
        id_to_name = data['edge_name_map']
        
        # Add nodes
        llm_graph.add_nodes_from(data['nodes'])
        
        # Reconstruct edges
        edges = data['edges']
        weights = data['weights']
        name_ids = data['name_ids']
        stratum_indeces = data['stratum']

        edge_data = []
        for i in range(len(edges)):
            u, v = edges[i]
            weight = weights[i]
            name = id_to_name[name_ids[i]]
            stratum = stratum_indeces[i]
            edge_data.append((u, v, {'weight': weight, 'name': name, 'stratum': stratum}))
        
        llm_graph.add_edges_from(edge_data)

        # --- 2. Reconstruct the LLM_STRACE object ---
        
        # Create a dummy object first (won't be used, but needed for cls)
        strace = LLM_STRACE(input_tuple, llm_hooked)
        
        # Overwrite the empty graph with our loaded one
        strace.graph = llm_graph
        
        # --- 3. Load all other attributes ---
        strace.strata_rel_size = list(data['strata_rel_size'])
        strace.strata_raw_size = list(data['strata_raw_size'])
        strace.strata_reco_tv = data['strata_reco_tv'].item()
        strace.strata_reco_nu = data['strata_reco_nu'].item()
        strace.strata_tau = list(data['strata_tau'])
        strace.strata_index = list(data['strata_index'])
        strace.strata_connected_to_input = list(data['strata_connected_to_input'])
        strace.strata_loss = data['strata_loss'].item()
        strace.strata_entropy = data['strata_entropy'].item()
        strace.input_tuple = input_tuple
        strace.nb_strata = data['nb_strata'].item()
        strace.strata_nucleus_60 = data['nucleus_60'].item()
        
        return strace

    except Exception as e:
        print(f"[LLM_STRACE] Error loading file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
