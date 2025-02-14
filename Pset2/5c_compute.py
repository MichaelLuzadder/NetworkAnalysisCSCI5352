import re
import csv
import random
import networkx as nx

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
    
    # Add edges using family names networkx ignores duplicates
    for node_id, info in data.items():
        family_name_u = info['name']
        for (nbr_id, w) in info['adj']:
            family_name_v = data[nbr_id]['name']
            # add_edge will ignore duplicates 
            G.add_edge(family_name_u, family_name_v, weight=w)
    
    return G
# calculate the normalized harmonic centrality
def normalized_harmonic_centrality(G):
    n = G.number_of_nodes()
    raw_hc = nx.harmonic_centrality(G)
    return {node: val / (n - 1) for node, val in raw_hc.items()}

#generate configuration model with lloops and multiedges with configuration model nerworkx which uses stub matching
def generate_stub_matching_graph(G):
   
    degree_sequence = [deg for _, deg in G.degree()]

    G_multi = nx.configuration_model(degree_sequence, seed=random.randint(0, 10000))
    
    # removing self-loops and collapse multi-edges
    G_simplified = nx.Graph(G_multi)
    G_simplified.remove_edges_from(nx.selfloop_edges(G_simplified))
    
    # Map back node indices to family names 
    name_mapping = {i: name for i, name in enumerate(G.nodes())}
    G_final = nx.relabel_nodes(G_simplified, name_mapping)
    
    return G_final

def main():

    filename = "data/Medici network/medici_network.txt"  
    G = load_florentine_families_adjlist(filename)

    real_cents = normalized_harmonic_centrality(G)

    families = sorted(G.nodes())

    num_random_graphs = 1000
    random.seed(42)  
    
    differences_dict = {fam: [] for fam in families}
    
    for _ in range(num_random_graphs):
        G_rand = generate_stub_matching_graph(G)
        rand_cents = normalized_harmonic_centrality(G_rand)
        
        for fam in families:
            diff = rand_cents.get(fam, 0) - real_cents[fam]
            differences_dict[fam].append(diff)
    
    # real network centralities to CSV
    with open("real_centralities_stub.csv", "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["family", "normalized_harmonic_centrality"])
        for fam in families:
            writer.writerow([fam, real_cents[fam]])
    
    # null-model stub-model differences to CSV
    with open("null_differences_stub.csv", "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["replicate", "family", "null_minus_real"])
        for rep_idx in range(num_random_graphs):
            for fam in families:
                writer.writerow([rep_idx, fam, differences_dict[fam][rep_idx]])

if __name__ == "__main__":
    main()