
import networkx as nx
import csv
import random
import numpy as np
import os

from tqdm import trange


def load_graph_from_csv(csv_path, delimiter=","):

    G = nx.Graph()
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            src, tgt = row[0], row[1]
            # Add edge
            G.add_edge(src, tgt)
    
    return G

# Compute clustering coefficient via transitivity
def compute_clustering_coefficient(G):
    """
    Computes the average clustering coefficient for an undirected graph G.
    """
    return nx.transitivity(G)
# Compute mean path length
def compute_mean_path_length(G):

    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        # Extract the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        return nx.average_shortest_path_length(subG)

# Randomize graph byr=20m double edge swaps
def randomize_graph_by_double_edge_swaps(G, nswaps_factor=20):

    m = G.number_of_edges()
    n_swaps = nswaps_factor * m
    
    # NetworkX built-in double_edge_swap
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
        #double-edge swaps
        randomize_graph_by_double_edge_swaps(G_random, nswaps_factor=nswaps_factor)
        
        # Compute clustering coefficient and mean path length
        c_val = compute_clustering_coefficient(G_random)
        l_val = compute_mean_path_length(G_random)
        
        C_values.append(c_val)
        L_values.append(l_val)
    
    return np.array(C_values), np.array(L_values)


########################
#  Main Analysis Flow  #
########################

def analyze_network(csv_path, out_prefix="network", delimiter=",", n_samples=1000, nswaps_factor=20):

    #Load graph
    G = load_graph_from_csv(csv_path, delimiter=delimiter, remove_selfloops=True, make_undirected=True)
    print(f"Loaded graph from {csv_path}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Compute empirical clustering and mean path length
    C_emp = compute_clustering_coefficient(G)
    L_emp = compute_mean_path_length(G)
    print(f"Empirical C: {C_emp:.4f}, Empirical <l>: {L_emp:.4f}")
    
    # Generate r=20m configuration-model 
    C_values, L_values = configuration_model_experiment(
        G, 
        n_samples=n_samples, 
        nswaps_factor=nswaps_factor
    )
    
    # Save results as a .npz file
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
        print("Usage: python analysis.py <csv_file> <output_prefix>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    out_prefix = sys.argv[2]
    
    analyze_network(csv_file, out_prefix)