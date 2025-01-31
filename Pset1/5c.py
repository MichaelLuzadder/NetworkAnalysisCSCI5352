import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Load all FB100 networks (excluding *_attr.txt files)
data_folder = "facebook100txt"
edge_list_files = [f for f in glob.glob(os.path.join(data_folder, "*.txt")) if not f.endswith("_attr.txt")]

# Dictionary to store results
network_results = {}

for file in edge_list_files:
    # Extract network name 
    network_name = os.path.basename(file).replace(".txt", "")

    # Load the graph
    G = nx.read_edgelist(file, nodetype=int)

    # Compute degree for each node
    degrees = dict(G.degree())

    # Mean degree ⟨ku⟩
    mean_degree_ku = np.mean(list(degrees.values()))

    # Compute mean degree of friends ⟨kv⟩
    friend_degrees = []
    for node in G.nodes():
        friends = list(G.neighbors(node))
        friend_degrees.append(np.mean([degrees[f] for f in friends]))
    
    mean_degree_kv = np.mean(friend_degrees) 

    # Compute paradox size ratio
    paradox_size = mean_degree_kv / mean_degree_ku 

    # Store results
    network_results[network_name] = (mean_degree_ku, paradox_size)

# Convert results to arrays for plotting
network_names = list(network_results.keys())
mean_degrees = np.array([network_results[name][0] for name in network_names])
paradox_sizes = np.array([network_results[name][1] for name in network_names])

# Key universities for labeling
key_universities = ["Reed98", "Colgate88", "Mississippi66", "Virginia63", "Berkeley13"]

# Create scatter plot
plt.figure(figsize=(9, 6))
plt.scatter(mean_degrees, paradox_sizes, alpha=0.7, color="black", label="FB100 Networks")
plt.axhline(y=1, color='red', linestyle='--', label="No Paradox Line")
plt.gca().grid(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)

# Overlay highlighted universities with large green dots
for uni in key_universities:
    if uni in network_results:
        idx = network_names.index(uni)
        plt.scatter(mean_degrees[idx], paradox_sizes[idx], color='green', s=110, edgecolors='black', linewidth=1.5, label=uni)

# Label highlighted networks
for uni in key_universities:
    if uni in network_results:
        idx = network_names.index(uni)
        plt.annotate(uni, (mean_degrees[idx], paradox_sizes[idx]), textcoords="offset points", alpha=1,  color="green", xytext=(5, 5), ha='left', fontweight='bold')

# Set labels and title
plt.xlabel("Mean Degree ⟨ku⟩", fontweight="bold")
plt.ylabel("Paradox Size ⟨kv⟩ / ⟨ku⟩", fontweight="bold")
plt.title("Friendship Paradox Across FB100 Networks", fontweight="bold")
plt.legend()

# Show the plot
plt.show()