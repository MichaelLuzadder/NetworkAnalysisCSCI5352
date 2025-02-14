import re
import csv
import random
import networkx as nx



##
# load the Florentine families network sample line:  0 Acciaiuoli, 0 1 [ (8, 1) ] 
# put it in network x
def load_florentine_families_adjlist(filename):
    
    #  parse each line, store node_id -> (name, adjacency_list)
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split(None, 4)
            node_id = int(tokens[0])
            family_name = tokens[1].rstrip(',')  # remove trailing comma
           
            adjacency_part = tokens[4].strip()  
            
            inside_brackets = adjacency_part.strip('[] ').strip()
            # inside_brackets might be empty if no neighbors
            edges = re.findall(r"\((\d+),\s*(\d+)\)", inside_brackets)
            
            adj_list = []
            for (nbr_str, w_str) in edges:
                nbr = int(nbr_str)
                weight = int(w_str)
                adj_list.append((nbr, weight))
            
            data[node_id] = {
                'name': family_name,
                'adj': adj_list
            }
    
    #  build the Graph using family names as node labels
    G = nx.Graph()
    
    # Add all family names as nodes
    for node_id, info in data.items():
        family_name = info['name']
        G.add_node(family_name)
    
    # Add edges using family names networkx add edge ignores duplicates
    for node_id, info in data.items():
        family_name_u = info['name']
        for (nbr_id, w) in info['adj']:
            family_name_v = data[nbr_id]['name']
            # add_edge will ignore duplicates 
            G.add_edge(family_name_u, family_name_v, weight=w)
    
    return G

#calculate the normalized harmonic centrality
def normalized_harmonic_centrality(G):

    n = G.number_of_nodes()
    # raw harmonic centrality from NetworkX
    raw_hc = nx.harmonic_centrality(G)
    
    # multiply each value by (1/(n-1)) to bound in [0,1]
    norm_hc = {}
    for node, val in raw_hc.items():
        norm_hc[node] = val / (n - 1)
    return norm_hc

# double edge swap function via netx (r=20m double edge swaps)
def randomize_graph_degree_preserving(G, swaps_multiplier=20):
    # Make a copy of the graph
    G_copy = G.copy()
    num_edges = G_copy.number_of_edges()
    # Attempt 'swaps_multiplier * num_edges' swaps
    nx.double_edge_swap(
        G_copy, 
        nswap=swaps_multiplier * num_edges, 
        max_tries=1000 * num_edges
    )
    return G_copy







def main():
    filename = "data/Medici network/medici_network.txt"  # <- update with your adjacency-list file
    G = load_florentine_families_adjlist(filename)
    
    # normalized harmonic centrality in real network
    real_cents = normalized_harmonic_centrality(G)
    
    # Sort families for consistent ordering
    families = sorted(G.nodes())
    
    # generate 1000 r = 20m double edge swap random graphs 
    num_random_graphs = 1000
    
    # for each replicate in a dictionary: {family: [diffs across replicates]}
    differences_dict = {fam: [] for fam in families}
    
    for _ in range(num_random_graphs):
        G_rand = randomize_graph_degree_preserving(G, swaps_multiplier=20)
        # Compute normalized harmonic centralities fopr the random graph
        rand_cents = normalized_harmonic_centrality(G_rand)
        
        # For each family, store the difference
        for fam in families:
            diff = rand_cents[fam] - real_cents[fam]
            differences_dict[fam].append(diff)
    
    #Write to a CSV
    with open("real_centralities.csv", "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["family", "normalized_harmonic_centrality"])
        for fam in families:
            writer.writerow([fam, real_cents[fam]])
    
    # difference values for all replicates to another CSV
    with open("null_differences.csv", "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["replicate", "family", "null_minus_real"])
        for rep_idx in range(num_random_graphs):
            for fam in families:
                writer.writerow([rep_idx, fam, differences_dict[fam][rep_idx]])


if __name__ == "__main__":
    main()