import re
import networkx as nx
import matplotlib.pyplot as plt
import csv
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

def plot_harmonic_centralities(rank_list):
    families = [x[0] for x in rank_list]
    centralities = [x[1] for x in rank_list]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(families)), centralities, color='green', alpha=0.7, edgecolor='black')
    plt.xticks(range(len(families)), families, rotation=45, ha='right')
    plt.ylabel("Normalized Harmonic Centrality")
    plt.title("Normalized Harmonic Centrality of Florentine Families")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

def main():
    filename = "data/Medici network/medici_network.txt" 
    # load the Florentine families network
    G = load_florentine_families_adjlist(filename)
    
    # Compute harmonic centralities normalize to 0-1
    n = len(G)  # Number of nodes
    harmonic_cents = nx.harmonic_centrality(G)

    # Normalize hc (Divide by (n-1))
    normalized_harmonic_cents = {node: hc / (n - 1) for node, hc in harmonic_cents.items()}
    
    # Sort families by harmonic centrality 
    sorted_by_harmonic = sorted(normalized_harmonic_cents.items(), 
                            key=lambda x: x[1], 
                            reverse=True)

    # Export ranks to CSV
    csv_filename = "florentine_harmonic_ranks_normalized.csv"
    with open(csv_filename, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(["Rank", "Family", "NormalizedHarmonicCentrality"])
        for rank, (family, cval) in enumerate(sorted_by_harmonic, start=1):
            writer.writerow([rank, family, f"{cval:.4f}"])
    
    
    plot_harmonic_centralities(sorted_by_harmonic)

if __name__ == "__main__":
    main()