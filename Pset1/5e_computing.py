import networkx as nx
import networkit as nk  
import numpy as np
import glob
import os
import pandas as pd
from tqdm import tqdm

# Load all FB100 networks (excluding *_attr.txt files)
data_folder = "facebook100txt"
edge_list_files = [f for f in glob.glob(os.path.join(data_folder, "*.txt")) if not f.endswith("_attr.txt")]

def compute_metrics(file):
    """Compute exact diameter and mean geodesic distance"""
    network_name = os.path.basename(file).replace(".txt", "")

    # Load the graph using NetworkX
    G = nx.read_edgelist(file, nodetype=int)

    # Find the largest connected component (LCC) using NetworkX
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc).copy()

    # Convert to Networkit for faster processing
    nkG = nk.nxadapter.nx2nk(G_lcc)

    # Compute network size (number of nodes in LCC) using Networkit
    n_lcc = nkG.numberOfNodes()

    # Compute the diameter 
    
    l_max_values = nk.distance.Diameter(nkG, algo=nk.distance.DiameterAlgo.Exact).run().getDiameter()
    l_max = l_max_values[0] if np.isfinite(l_max_values[0]) else None  # Ensure no infinite values

    # Compute mean geodesic distance using APSP algorithm in Networkit
    apsp = nk.distance.APSP(nkG)  
    apsp.run()
        
    all_distances = np.array(apsp.getDistances())  # Get all shortest paths
    finite_distances = all_distances[np.isfinite(all_distances)]  # Remove infinite values
    mean_l = np.mean(finite_distances)  # Compute mean shortest path length

    return network_name, n_lcc, l_max, mean_l

if __name__ == "__main__":
    print("Running FB100 shortest path calculations")

    results = []
    for file in tqdm(edge_list_files, desc="Finding diameters and mean geodesic distances"):
        results.append(compute_metrics(file))  # Process one at a time (avoid running computer out of memory:))

    # Convert results to DataFrame and save
    df_results = pd.DataFrame(results, columns=["Network", "Size_LCC", "Diameter_lmax", "Mean_Geodesic_Distance"])
    df_results.to_csv("5e/FB100_network_diameters.csv", index=False)

    print("Computation complete. Results saved to 'FB100_network_diameters.csv'.")