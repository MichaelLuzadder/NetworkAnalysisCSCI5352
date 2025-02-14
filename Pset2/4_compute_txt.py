

import networkx as nx
import random
import numpy as np
import os

from tqdm import trange


def load_graph_from_txt(txt_path, delimiter=None, remove_selfloops=True):

    G = nx.Graph()
    
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
            parts = line.split(delimiter)
            if len(parts) < 2:
                continue
            src, tgt = parts[0], parts[1]
            G.add_edge(src, tgt)
    
    if remove_selfloops:
        G.remove_edges_from(nx.selfloop_edges(G))
    
    return G

# Compute clustering coefficient via transitivity
def compute_clustering_coefficient(G):

    return nx.transitivity(G)

# Compute mean path length
def compute_mean_path_length(G):

    if nx.is_connected(G):
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        
        # Sum all shortest path lengths and compute the average
        total_dist = sum(sum(lengths.values()) for lengths in path_lengths.values())
        n = len(G)
        num_pairs = n * (n - 1)
        
        return total_dist / num_pairs
    else:
        # Extract the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc).copy()
        
        return compute_mean_path_length(subG)

# double edge swap function via netx
def randomize_graph_by_double_edge_swaps(G, nswaps_factor=20):

    m = G.number_of_edges()
    n_swaps = nswaps_factor * m
    
    # Use built-in double_edge_swap in NetworkX
    nx.double_edge_swap(G, nswap=n_swaps, max_tries=n_swaps*5)
    
    return G

# generate configuration model samples via double edge swaps 1000 times
def configuration_model_experiment(G, n_samples=1000, nswaps_factor=20):
    C_values = []
    L_values = []
    
    # tqdm progress bar
    for _ in trange(n_samples, desc="Generating configuration-model samples"):
        # Make a copy to randomize
        G_random = G.copy()
        # double-edge swaps
        randomize_graph_by_double_edge_swaps(G_random, nswaps_factor=nswaps_factor)
        
        # Compute stats
        c_val = compute_clustering_coefficient(G_random)
        l_val = compute_mean_path_length(G_random)
        
        C_values.append(c_val)
        L_values.append(l_val)
    
    return np.array(C_values), np.array(L_values)

def analyze_network(txt_path, out_prefix="network", delimiter=None, n_samples=1000, nswaps_factor=20):

    # load graph from .txt
    G = load_graph_from_txt(txt_path, delimiter=delimiter, remove_selfloops=True)
    print(f"Loaded graph from {txt_path}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # compute empirical clustering and mean path length
    C_emp = compute_clustering_coefficient(G)
    L_emp = compute_mean_path_length(G)
    print(f"Empirical C: {C_emp:.4f}, Empirical <l>: {L_emp:.4f}")
    
    # compute configuration model r = 20m
    C_values, L_values = configuration_model_experiment(
        G, 
        n_samples=n_samples, 
        nswaps_factor=nswaps_factor
    )
    
    # 4) Save results
    output_file = f"{out_prefix}_results.npz"
    np.savez(output_file, 
             C_emp=C_emp, 
             L_emp=L_emp, 
             C_values=C_values, 
             L_values=L_values)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python analysis.py <txt_file> <output_prefix>")
        sys.exit(1)
    
    txt_file = sys.argv[1]
    out_prefix = sys.argv[2]
    
    analyze_network(txt_file, out_prefix)