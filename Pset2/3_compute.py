import networkx as nx
import numpy as np
import csv
# Load the UC Berkeley FB100 network from the edgelist file
def load_berkeley_graph(path):
   
    G = nx.read_edgelist(path, nodetype=int)
    
    return G
# extract the largest connected component
def largest_component_subgraph(G):
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()

#compute APSP then calculate average path length
def compute_apsp_average_length(G):  
    # Compute APSP 
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    # Sum all shortest path lengths 
    total_dist = sum(sum(lengths.values()) for lengths in path_lengths.values())
    
    # Number of node pairs
    num_pairs = n * (n - 1)

    # return average path length
    return total_dist / num_pairs  

# compute clustering coefficient using transitivity in networkx
def measure_clustering_and_pathlength(G):

    C = nx.transitivity(G)
    L = compute_apsp_average_length(G)  
    return C, L

# Generate log-spaced swap values to be sampled
def generate_log_space_swaps(m, num_points=12, max_factor=20):
    import math
    import numpy as np
    
    r_max = max_factor * m
    log_values = np.logspace(0, math.log10(r_max), num_points)
    swap_candidates = sorted(set(int(round(x)) for x in log_values if 1 <= x <= r_max))
    
    if 0 not in swap_candidates:
        swap_candidates = [0] + swap_candidates
    return swap_candidates

def run_double_edge_swap_experiment(G, swap_values):
    """
    Perform one double edge swap randomization experiment, measuring clustering & path length at each r.
    Returns two lists (clustering_vals, path_length_vals), each length == len(swap_values).
    """
    G_swapped = G.copy()
    clustering_vals = []
    path_length_vals = []

    current_r = 0
    
    # r=0
    C0, L0 = measure_clustering_and_pathlength(G_swapped)
    clustering_vals.append(C0)
    path_length_vals.append(L0)
    
    for r in swap_values[1:]:
        nswaps_to_perform = r - current_r
        nx.double_edge_swap(G_swapped, nswap=nswaps_to_perform, max_tries=5*nswaps_to_perform)
        
        C, L = measure_clustering_and_pathlength(G_swapped)
        clustering_vals.append(C)
        path_length_vals.append(L)
        
        current_r = r
    
    return clustering_vals, path_length_vals

def main():
    berkeley_path = "/Users/michaelluzadder/Desktop/Networks/Pset1/FB100/facebook100txt/Berkeley13.txt"
    G_full = load_berkeley_graph(berkeley_path)
    G = largest_component_subgraph(G_full)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"Largest CC has {n} nodes and {m} edges.")

    swap_values = generate_log_space_swaps(m, num_points=12, max_factor=20)
    print("Swap values:", swap_values)
    
    #Repeat the experiment 5 times
    num_experiments = 5
    output_csv = "berkeley_experiment_results_test.csv"

    # save output for plotting
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        # header
        writer.writerow(["experiment", "r", "clustering", "path_length"])
        
        for exp_idx in range(num_experiments):
            print(f"\n--- Experiment {exp_idx+1}/{num_experiments} ---")
            c_vals, l_vals = run_double_edge_swap_experiment(G, swap_values)
            
            for i, r in enumerate(swap_values):
                writer.writerow([exp_idx+1, r, c_vals[i], l_vals[i]])

if __name__ == "__main__":
    main()