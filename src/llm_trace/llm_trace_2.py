from src.llm_graph.llm_graph_nx_light import LLM_Graph_NX, load_from_dict
from src.llm_trace.mask_utils import prepare_mask
from transformers.modeling_utils import PreTrainedModel
import networkx as nx
import time
from tqdm import tqdm
import math
import torch
import numpy as np
import collections
import os
import torch.nn.functional as F
from src.test.unit_tests import MaskingError
import heapq

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
    # print(f"orig_top_indices [{p}]", orig_top_indices)
    # print(f"graph_top_indices [{p}]", graph_top_indices)
    # print("shared_nucleus", shared_nucleus)
    # print("nucleus_size", nucleus_size)
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
    
    return sorted_indices[mask_to_keep_sorted]


class LLM_STRACE:
    """
    Represents the model's stratified trace, a subgraph of the full computational graph.
    STRACE := Stratified TRACE
    """
    def __init__(self, sentence: str = None, next_word: str | None = None, llm: PreTrainedModel = None, tokenizer=None, track_time=False):
        # we consider three stratification: by size (relatively to the full graph), by reconstruction error, by importance score (tau)
        self.graph = None
        self.strata_rel_size: List[int] = []
        self.strata_raw_size: List[int] = [] 
        self.strata_tau: List[int] = []
        self.strata_index: List[int] = []
        self.strata_connected_to_input: List[bool] = [] # bool that tells if the stratum is connected to the input
        self.strata_reco_tv = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_reco_nu = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_size_nu = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_nucleus_60 = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_nucleus_60_tkn = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_loss = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_entropy = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_logits = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.track_time = track_time
        self.original_logits = None # the logit prediction from the full model
        self.llm = llm
        self.tokenizer = tokenizer
        self.input_prepared = self.prepare_input(sentence, next_word)
        self.nb_strata = None
        self.frozen = None

    def prepare_input(self, sentence, next_word):
        if sentence is None:
            return None
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
            attention_mask = sentence.attention_mask

        labels = torch.full_like(input_ids, -100)
        labels[0, -1] = full_sentence_tokens[0, -1]

        # Sanity checks
        assert labels[0, -1] == full_sentence_tokens[0, -1], "Labels should be the last token of the full sentence"
        assert input_ids[0, -1] == full_sentence_tokens[0, -2], "Input IDs should be the penultimate token"

        return input_ids, attention_mask, labels

    def freeze_strace(self, component_prefix_name):
        self.frozen = component_prefix_name

    def reset_strace(self):
        self.strata_rel_size = [] 
        self.strata_raw_size = [] 
        self.strata_tau = []
        self.strata_connected_to_input = []
        self.nb_strata = None
        self.strata_index = []
        self.reset_evaluation()
    
    def reset_evaluation(self):
        self.strata_reco_tv = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_reco_nu = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_size_nu = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_nucleus_60 = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_nucleus_60_tkn = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_logits = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_loss = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}
        self.strata_entropy = {'random': {'inverse': [], 'only': []}, 'trace': {'inverse': [], 'only': []}}

    def populate_graph(self, importance_mode, print_stats=True, unit_test=False):
        device = 'cuda'
        with torch.no_grad():
            output, graph = self.llm(
                input_ids=self.input_prepared[0].to(device), 
                attention_mask=self.input_prepared[1].to(device),
                build_graph=importance_mode, unit_test=unit_test, attn_implementation="eager")
        self.original_logits = output.logits
        self.graph = graph
        self.graph.remove_disconnected_nodes()
        if print_stats:
            stats = self.graph.get_edge_weight_stats()
            for (k,s) in stats.items():
                print(f"[STRACE][GRAPH STATS] {k}: {s}")
    
    def label_graph_with_stratum(self, initial_graph: LLM_Graph_NX, tau: float, stratum_index: int, mode: str):
        """
        Filters initial_graph to create a connected subgraph and labels
        the corresponding edges on `self.graph` with the stratum_index.

        The filtering method is determined by `mode`:
        - 'threshold': (Original) Keeps edges where `weight > tau`.
        - 'nucleus': (New) For each node, keeps the top-N incoming edges
                    whose weights sum to the cumulative mass `tau`.
        
        'frozen' specifies if we want to focus on one type of component:
        - 'attention': only the attention head can be keep/pruned (e.g. mlp are always included in the graph)
        - 'mlp': only the mlps can be keep/pruned (e.g. attention heads are always included in the graph)
        - 'none': all components can be keep/pruned
    
        Returns a connected subgraph of initial graph.
        """

        # # 1. Grab just ONE edge to see what it actually looks like at this exact moment
        # sample_edge = next(iter(initial_graph.edges(data=True)))
        # print(f"[DEBUG LABELER] Sample edge data: {sample_edge}")

        # # 2. Check the raw max of the key you are about to threshold
        # max_w = max(d.get('weight') for u, v, d in initial_graph.edges(data=True))
        # print(f"[DEBUG LABELER] Max 'weight' found: {max_w}")

        # Get the output node
        output_node = initial_graph.get_output_node()

        time_stats = {
            'filtering': None,
            'subgraph': None,
            'output connected': None
        }

        if self.track_time:
            start_time = time.time()

        def condition(datum):
            if self.frozen == None:
                return datum.get('weight') > tau
            else:
                if datum.get('name').startswith(self.frozen):
                    return True # component specified in 'keep' is always included in the strace
                else:
                    return datum.get('weight') > tau

        if mode == 'threshold':
            # Original mode: Filter by a minimum weight threshold
            edges_to_keep = [
                (u, v, k)
                for u, v, k, d in initial_graph.edges(keys=True, data=True)
                if condition(d)
            ]
            # print(f"[{tau}] Nb edges to keep:", len(edges_to_keep))

        elif mode == 'nucleus':

            if keep != 'none':
                raise NotImplementedError

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

        # FIX 4: Output explicit warning if isolation occurs
        if output_node not in initial_subgraph:
            print(f"[WARNING] Output node lost all incoming connections! Deleting {len(edges_to_keep)} valid upstream edges.")
            initial_subgraph = initial_graph.subgraph([output_node])

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

    def label_graph_with_strata_sizes(self, sigmas: list[float]):
        """
        Filters self.graph to create a connected subgraph and labels
        the corresponding edges on `self.graph` with a stratum index from 0 to N-1.
        
        Returns:
            stratum_sizes (list[int]): The exact number of edges assigned to each stratum label.
            is_connected_list (list[bool]): True if the cumulative subgraph at stratum i is connected to an input node.
            time_stats (dict): Performance timings.
        """
        output_node = self.graph.get_output_node()
        input_nodes = set(self.graph.get_input_nodes()) # Convert to set for O(1) lookups

        time_stats = {
            'filtering': None,
            'output connected': 0.0 # Implicitly solved by backward traversal
        }

        if self.track_time:
            start_time = time.time()

        # 1. Prepare target thresholds
        total_edges = self.graph.number_of_edges()
        sorted_sigmas = sorted(sigmas)
        target_counts = [int(s * total_edges) for s in sorted_sigmas]
        max_strata = len(target_counts)

        edges_to_keep = set()
        nodes_in_subgraph = {output_node}
        edges_to_label = {}
        
        # New tracking variables
        stratum_sizes = [0] * max_strata
        min_stratum_connected = float('inf') 
        
        frontier_pq = []
        edge_counter = 0

        def add_in_edges_to_frontier(node):
            nonlocal edge_counter
            for u, v, k, d in self.graph.in_edges(node, data=True, keys=True):
                weight = d.get('weight', 0.0)
                
                # Infinite priority for frozen components
                if self.frozen is not None and d.get('name', '').startswith(self.frozen):
                    weight = float('inf')
                    
                heapq.heappush(frontier_pq, (-weight, edge_counter, (u, v, k)))
                edge_counter += 1

        if output_node in self.graph:
            add_in_edges_to_frontier(output_node)

        # 2. Single-Pass Traversal
        current_stratum = 0
        
        # Fast-forward if the first stratum size is exactly 0
        while current_stratum < max_strata and target_counts[current_stratum] == 0:
            current_stratum += 1

        while frontier_pq and current_stratum < max_strata:
            neg_weight, _, edge_key = heapq.heappop(frontier_pq)
            
            if edge_key in edges_to_keep:
                continue
                
            edges_to_keep.add(edge_key)
            
            # Label the edge and increment the size counter for this specific stratum
            edges_to_label[edge_key] = current_stratum
            stratum_sizes[current_stratum] += 1
            
            # Check if this edge links back to an input node
            source_node = edge_key[0]
            if source_node in input_nodes:
                if current_stratum < min_stratum_connected:
                    min_stratum_connected = current_stratum
            
            # Check if adding this edge pushed us over the current size threshold
            while current_stratum < max_strata and len(edges_to_keep) >= target_counts[current_stratum]:
                current_stratum += 1 
            
            # Expand frontier
            if source_node not in nodes_in_subgraph:
                nodes_in_subgraph.add(source_node)
                add_in_edges_to_frontier(source_node)

        if self.track_time:
            time_stats['filtering'] = time.time() - start_time

        if len(edges_to_keep) < target_counts[-1]:
            print(f"[WARNING] Exhausted connected edges at {len(edges_to_keep)}. Max target was {target_counts[-1]}.")

        # 3. Build the boolean connection list
        # If min_stratum_connected is 2, then strata 0 and 1 are False, and 2, 3, 4... are True.
        is_connected_list = [(i >= min_stratum_connected) for i in range(max_strata)]

        if self.track_time:
            start_time = time.time()
        
        # 4. Apply Labels directly to the main graph
        nx.set_edge_attributes(self.graph, values=edges_to_label, name='stratum')
        
        if self.track_time:
            time_stats['labeling'] = time.time() - start_time
        
        stratum_sizes = [sum(stratum_sizes[:k+1]) for k in range(len(stratum_sizes))]

        # No subgraph generated, just return the raw stats!
        return stratum_sizes, is_connected_list, time_stats


    def extract_strace(self, threshold_values: list[float]):
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
        self.strata_tau = [1.0]
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
    
        threshold_values.sort() # sort from low to high
    
        stratum_sizes, is_connected_list, time_stats = self.label_graph_with_strata_sizes(sigmas=threshold_values)
        self.strata_tau = threshold_values + self.strata_tau
        self.strata_connected_to_input = is_connected_list + self.strata_connected_to_input
        # update size
        self.strata_raw_size = stratum_sizes + self.strata_raw_size
        self.strata_rel_size = [s/self.strata_raw_size[-1] for s in stratum_sizes] + self.strata_rel_size
        # update strata index
        self.strata_index = list(range(len(threshold_values))) + self.strata_index
        

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

    def print_graph_sizes_and_thresholds(self):
        """
        Prints the size of each stratum alongside its corresponding threshold value.
        Safely handles missing (None) values, missing dictionary keys, and empty lists.
        """
        # Header setup
        header = (
            f"{'S':<5}{'c-input?':<10}{'Threshold':<25}{'Size (rel)':<15}{'Size (raw)':<15}"
            f"{'TV':<15}{'Nucleus':<15}{'N60 (top 5)':<25}{'Inv. Nucleus':<15}"
            f"{'inv-N60 (top 5)':<25}{'Rand. Nu.':<15}{'Inv. Rand. Nu.':<15}"
        )
        print(f"[STRACE] Displaying threshold and graph size:\n{header}")
        print("-" * len(header))
        
        # 1. Determine target length (and handle case where strata_index itself is empty)
        n_strata = len(self.strata_index) if self.strata_index else 0
        if n_strata == 0:
            print("No strata data available to display yet.")
            return

        # 2. Bulletproof helper function to extract and pad lists safely
        def get_safe_list(source, key1=None, key2=None):
            lst = source
            # Safely navigate nested dictionaries
            try:
                if key1 is not None:
                    lst = lst.get(key1) if isinstance(lst, dict) else lst[key1]
                if key2 is not None and lst is not None:
                    lst = lst.get(key2) if isinstance(lst, dict) else lst[key2]
            except (KeyError, TypeError, AttributeError):
                lst = None
                
            # If it's None or completely empty ([]), return a list of Nones
            if not lst: 
                return [None] * n_strata
                
            # If it has data, but is shorter than n_strata, pad the end with Nones
            # (Prevents zip() from truncating valid rows)
            if len(lst) < n_strata:
                lst = list(lst) + [None] * (n_strata - len(lst))
                
            return lst

        # 3. Extract ALL lists safely (even the ones we assume are safe)
        tau_list = get_safe_list(self.strata_tau)
        rel_size_list = get_safe_list(self.strata_rel_size)
        raw_size_list = get_safe_list(self.strata_raw_size)
        c_in_list = get_safe_list(self.strata_connected_to_input)
        
        tv_trace_only = get_safe_list(self.strata_reco_tv, 'trace', 'only')
        nu_trace_only = get_safe_list(self.strata_reco_nu, 'trace', 'only')
        n60_trace_only = get_safe_list(self.strata_nucleus_60, 'trace', 'only')
        
        inv_nu_list = get_safe_list(self.strata_reco_nu, 'trace', 'inverse')
        inv_n60_list = get_safe_list(self.strata_nucleus_60, 'trace', 'inverse')
        r_nu_list = get_safe_list(self.strata_reco_nu, 'random', 'only')
        r_inv_nu_list = get_safe_list(self.strata_reco_nu, 'random', 'inverse')

        # 4. Zip will now guaranteed run exactly n_strata times
        for stratum, threshold, rel_size, raw_size, c_in, tv, nu, n60, inv_nu, inv_n60, r_nu, r_inv_nu in zip(
            self.strata_index, tau_list, rel_size_list, raw_size_list, c_in_list,
            tv_trace_only, nu_trace_only, n60_trace_only,
            inv_nu_list, inv_n60_list, r_nu_list, r_inv_nu_list
        ):
            # 1. Handle booleans
            if c_in is True:
                c_in_str = 'yes'
            elif c_in is False:
                c_in_str = 'no'
            else:
                c_in_str = 'N/A'

            # 2. Handle floats and scientific notation
            threshold_str = f"{threshold:.3e}" if threshold is not None else "N/A"
            rel_size_str = f"{rel_size:.0%}" if rel_size is not None else "N/A"
            raw_size_str = f"{raw_size:.2e}" if raw_size is not None else "N/A"
            tv_str = f"{tv:.2e}" if tv is not None else "N/A"

            # 3. Handle list slicing and string joins
            top5_nucleus_str = repr('|'.join(n60[:5])) if n60 is not None else "N/A"
            inv_top5_nucleus_str = repr('|'.join(inv_n60[:5])) if inv_n60 is not None else "N/A"

            # 4. Handle standard integers/floats that just need string conversion
            nu_str = str(nu) if nu is not None else "N/A"
            inv_nu_str = str(inv_nu) if inv_nu is not None else "N/A"
            r_nu_str = str(r_nu) if r_nu is not None else "N/A"
            r_inv_nu_str = str(r_inv_nu) if r_inv_nu is not None else "N/A"
            stratum_str = str(stratum) if stratum is not None else "N/A"

            # 5. Print the formatted row using string-only padding
            print(
                f"{stratum_str:<5}{c_in_str:<10}{threshold_str:<25}{rel_size_str:<15}{raw_size_str:<15}"
                f"{tv_str:<15}{nu_str:<15}{top5_nucleus_str:<25}{inv_nu_str:<15}"
                f"{inv_top5_nucleus_str:<25}{r_nu_str:<15}{r_inv_nu_str:<15}"
            )

    def print_graph_sizes_and_thresholds_short(self):
        """
        Prints the size of each stratum alongside its corresponding threshold value.
        """
        print(f"[STRACE] Displaying threshold and graph size:\n{'S':<5}{'c-input?':<10}{'Threshold':<25}{'Size (rel)':<15}{'Size (raw)':<15}{'TV':<15}{'Nucleus':<15}{'N60 (top 5)':<25}")
        print("-" * 9 * 25)
        for stratum, threshold, rel_size, raw_size, c_in, tv, nu, n60 in zip(
            self.strata_index, self.strata_tau, self.strata_rel_size, self.strata_raw_size, self.strata_connected_to_input, 
            self.strata_reco_tv['trace']['only'], self.strata_reco_nu['trace']['only'], self.strata_nucleus_60['trace']['only']):
            top5_nucleus_str = repr('|'.join(n60[:5]))
            c_in_str = 'yes' if c_in else 'no'
            print(f"{stratum:<5}{c_in_str:<10}{threshold:<25.3e}{rel_size:<15.0%}{raw_size:<15.2e}{tv:<15.2e}{nu:<15}{top5_nucleus_str:<25}")

    def compute_stratum_reconstruction_error(self, do_random=False, do_inverse=False, save_logit=False):
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

                # check that the stratum size is indeed correct
                stratum_size = stratum.get_size()
                if not self.strata_raw_size[i] == stratum_size:
                    raise MaskingError(f"Stratum {stratum_index} size mismatch: expected {self.strata_raw_size[i]}, got {stratum_size}")

                for inverse in [True, False]:
                    if not do_inverse and inverse: # skip inverse
                        continue

                    graph_mask, nb_non_masked_edges, _ = prepare_mask(
                        graph=stratum, 
                        seq_len=self.graph.graph['n_tokens'],
                        nb_head=self.graph.graph['n_heads'], 
                        n_layers=self.graph.graph['n_layers'], 
                        inverse=inverse, 
                        keep_residual=True if (inverse or random) else False) # In case of inverse pruning, we keep the residual)

                    # check that the stratum size is indeed correct
                    if not inverse and not random:
                        if not self.strata_raw_size[i] == nb_non_masked_edges:
                            raise MaskingError(f"Mask {stratum_index} size mismatch: expected {self.strata_raw_size[i]}, got {nb_non_masked_edges}")
                        
                    device = 'cuda'

                    with torch.no_grad():
                        output = self.llm(
                            input_ids=self.input_prepared[0].to(device),
                            attention_mask=self.input_prepared[1].to(device), 
                            # labels=self.input_prepared[2].to(device),
                            graph_mask=graph_mask, build_graph=None, unit_test=False, attn_implementation="eager")
                    graph_logits = output.logits.view(-1, self.llm.config.vocab_size).cpu()
                    loss, entropy, predicted_token_id, rank = surprisal(graph_logits, self.input_prepared[2])
                    
                    tv_original_graph = get_total_variation(self.original_logits, graph_logits.unsqueeze(0))
                    shared_nucleus, shared_nucleus_size = get_intersection_nucleus(self.original_logits,graph_logits.unsqueeze(0))
                    # print(f"\n[stratum {stratum_index}] shared_nucleus [{shared_nucleus_size}]", shared_nucleus)
                    
                    nucleus_indices = get_nucleus(graph_logits[-1].cpu(), 60)
                    # print("Graph nucleus@60", [self.tokenizer.decode(token_id) for token_id in nucleus_indices[:5]])
                    # print("Original nucleus@60", [self.tokenizer.decode(token_id) for token_id in get_nucleus(self.original_logits[0,-1].cpu(), 60)[:5]])

                    key_tuple = ('random' if random else 'trace', 'inverse' if inverse else 'only') 
                    self.strata_reco_tv[key_tuple[0]][key_tuple[1]].append(tv_original_graph)
                    self.strata_reco_nu[key_tuple[0]][key_tuple[1]].append(shared_nucleus)
                    self.strata_size_nu[key_tuple[0]][key_tuple[1]].append(shared_nucleus_size)
                    self.strata_nucleus_60[key_tuple[0]][key_tuple[1]].append([self.tokenizer.decode(token_id) for token_id in nucleus_indices[:5]])
                    self.strata_nucleus_60_tkn[key_tuple[0]][key_tuple[1]].append([token_id for token_id in nucleus_indices[:5]])
                    self.strata_loss[key_tuple[0]][key_tuple[1]].append(loss)
                    self.strata_entropy[key_tuple[0]][key_tuple[1]].append(entropy)
                    if save_logit:
                        self.strata_logits[key_tuple[0]][key_tuple[1]].append(graph_logits[-1].detach().cpu())


    def save_light(self, file_path: str):
        """
        Saves the LLM_STRACE data to a compressed NumPy (.npz) file.
        The graph is serialized into efficient binary arrays.
        """
        
        # --- 1. Serialize the Graph efficiently ---
        graph = self.graph
        
        input = np.array(self.input_prepared[0].cpu(), dtype=np.uint32)
        next_token = np.array(self.input_prepared[2][0,-1].cpu(), dtype=np.uint32)

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
            'strata_size_nu': self.strata_size_nu,
            'strata_tau': np.array(self.strata_tau, dtype=np.float32),
            'strata_index': np.array(self.strata_index, dtype=np.uint16),
            'strata_connected_to_input': self.strata_connected_to_input,
            'strata_loss': self.strata_loss,
            'strata_entropy': self.strata_entropy,
            'input': input,
            'next_token': next_token,
            'nb_strata': self.nb_strata,
            'nucleus_60': self.strata_nucleus_60,
            'nucleus_60_tkn': self.strata_nucleus_60_tkn,
            'strata_logits': self.strata_logits,
            'original_logits': self.original_logits.cpu(),
        }
        
        # --- 3. Save as a compressed .npz file ---
        try:
            # Use np.savez_compressed to save the dictionary
            # This is much more efficient than pickle for this data
            np.savez_compressed(file_path, **data_to_save)
            # print(f"[LLM_STRACE] Successfully saved to {file_path}.npz")
        
        except Exception as e:
            print(f"[LLM_STRACE] Error saving file: {e}")

def load_from_file_light(file_path: str, llm: PreTrainedModel = None, tokenizer = None):
    """
    Loads an LLM_STRACE object from a .npz file and re-attaches
    the llm.
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
        # print("Keys found in the npz file:", list(data.keys()))

        input_ids = torch.tensor(data['input']).long()
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        labels[0, -1] = data['next_token'].item()
        input_prepared = (input_ids, attention_mask, labels)

        # --- 1. Reconstruct the graph ---
        
        # Start with an empty graph and set attributes
        llm_graph = LLM_Graph_NX()
        
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
        strace = LLM_STRACE(llm=llm, tokenizer=tokenizer)
        
        # Overwrite the empty graph with our loaded one
        strace.graph = llm_graph
        
        # --- 3. Load all other attributes ---
        strace.strata_rel_size = list(data['strata_rel_size'])
        strace.strata_raw_size = list(data['strata_raw_size'])
        strace.strata_reco_tv = data['strata_reco_tv'].item()
        strace.strata_reco_nu = data['strata_reco_nu'].item()
        if 'strata_size_nu' in data:
            strace.strata_size_nu = data['strata_size_nu'].item()
        strace.strata_tau = list(data['strata_tau'])
        strace.strata_index = list(data['strata_index'])
        strace.strata_connected_to_input = list(data['strata_connected_to_input'])
        strace.strata_loss = data['strata_loss'].item()
        strace.strata_entropy = data['strata_entropy'].item()
        strace.input_prepared = input_prepared
        strace.nb_strata = data['nb_strata'].item()
        strace.strata_nucleus_60 = data['nucleus_60'].item()
        if 'strata_logits' in data:
            strace.strata_logits = data['strata_logits'].item()
        strace.original_logits = torch.tensor(data['original_logits'])
        
        return strace

    except Exception as e:
        print(f"[LLM_STRACE] Error loading file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
