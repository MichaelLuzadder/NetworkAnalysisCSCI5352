
import networkx as nx
import csv
import numpy as np
from tqdm import trange


# load the adjacency matrix from a CSV file
def load_graph_from_adjacency_csv(csv_path, remove_selfloops=True):
   
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        matrix = [list(map(float, row)) for row in reader]
    
    # give adjacency matrix to networkx
    G = nx.Graph()
    G.add_nodes_from(range(n))  
    
    for i in range(n):
        for j in range(i, n):  # only upper triangle to avoid duplicating edges
            val = matrix[i][j]
            if val != 0:
                if i == j and remove_selfloops:
                    continue  # skip self-loop
                else:
                    G.add_edge(i, j)
    
    return G

# compute the clustering coefficient with transitivity
def compute_clustering_coefficient(G):
    return nx.transitivity(G)

#compute mean APSP length
def compute_mean_path_length(G):
    
    if nx.is_connected(G):
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        
        total_dist = sum(sum(lengths.values()) for lengths in path_lengths.values())
        n = len(G)
        num_pairs = n * (n - 1)
        
        return total_dist / num_pairs
    else:
        # Extract the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc).copy()
        
        return compute_mean_path_length(subG)

# randomize graph by double edge swaps
def randomize_graph_by_double_edge_swaps(G, nswaps_factor=20):

    m = G.number_of_edges()
    n_swaps = nswaps_factor * m
    
    # NetworkX double_edge_swap method
    nx.double_edge_swap(G, nswap=n_swaps, max_tries=n_swaps*5)
    
    return G

# generate configuration model samples via double edge swaps 1000 times
def configuration_model_experiment(G, n_samples=1000, nswaps_factor=20):
    
    C_values = []
    L_values = []
    
    for _ in trange(n_samples, desc="Generating configuration-model samples"):
        # make a copy to randomize
        G_random = G.copy()
        # make double-edge swaps
        randomize_graph_by_double_edge_swaps(G_random, nswaps_factor=nswaps_factor)
        
        # Compute clustering coefficient and mean path length
        c_val = compute_clustering_coefficient(G_random)
        l_val = compute_mean_path_length(G_random)
        
        C_values.append(c_val)
        L_values.append(l_val)
    
    return np.array(C_values), np.array(L_values)

def analyze_network(csv_path, out_prefix="network", remove_selfloops=True, n_samples=1000, nswaps_factor=20):

    # load graph from Aij
    G = load_graph_from_adjacency_csv(csv_path, remove_selfloops=remove_selfloops, is_weighted=is_weighted)
    print(f"Loaded adjacency matrix from {csv_path}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # compute empirical clustering and mean path length
    C_emp = compute_clustering_coefficient(G)
    L_emp = compute_mean_path_length(G)
    print(f"Empirical C: {C_emp:.4f}, Empirical <l>: {L_emp:.4f}")
    
    # generate configuration model samples
    C_values, L_values = configuration_model_experiment(
        G, 
        n_samples=n_samples, 
        nswaps_factor=nswaps_factor
    )
    
    # save results as a .npz file
    output_file = f"{out_prefix}_results.npz"
    np.savez(output_file, 
             C_emp=C_emp, 
             L_emp=L_emp, 
             C_values=C_values, 
             L_values=L_values)

if __name__ == "__main__":

    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python analysis.py <csv_file> <output_prefix>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    out_prefix = sys.argv[2]
    
    analyze_network(csv_file, out_prefix)