import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#Watts-Strogatz model parameters
N = 100  # Number of nodes
K = 4    # Each node is connected to K nearest neighbors in ring 
rewire_steps = np.unique(np.concatenate(([0], np.logspace(0, np.log10(N*K//2), num=12, dtype=int)))) 
num_experiments = 100

# Store results across multiple experiments
all_betweenness = np.zeros((num_experiments, len(rewire_steps)))

# generate Watts-Strogatz graph to start with
G_original = nx.watts_strogatz_graph(N, K, 0) #p=0 means no rewiring

# rewire edges using NetworkX's double-edge swap method
## Note: This also could be done by generating new graphs with incemental p values 
## and then computing betweenness centrality for each graph but this seems more efficient
def double_edge_swap_nx(G, num_swaps):
    G_copy = G.copy()
    nx.double_edge_swap(G_copy, nswap=num_swaps, max_tries=num_swaps * 10)
    return G_copy

# to do multiple experiments
for exp in range(num_experiments):
    for idx, r in enumerate(rewire_steps):
        if r == 0:
            G_rewired = G_original.copy()
        else:
            G_rewired = double_edge_swap_nx(G_original, r)
        betweenness = nx.betweenness_centrality(G_rewired).values()
        all_betweenness[exp, idx] = np.mean(list(betweenness))

# Compute mean and standard deviation
mean_betweenness = np.mean(all_betweenness, axis=0)
std_betweenness = np.std(all_betweenness, axis=0)

# Plot mean betweenness centrality with standard deviations
plt.figure(figsize=(10, 6))
plt.errorbar(rewire_steps, mean_betweenness, yerr=std_betweenness, color='green', fmt='o-', markersize=8, capsize=5, label='Mean ± Std Betweenness Centrality')

plt.xlabel('Number of double edge swaps (r)')
plt.ylabel('Betweenness Centrality')
plt.title('Betweenness Centrality vs # of Double Edge Swaps in Watts-Strogatz Model')
plt.xscale('symlog', linthresh=5) # Truncate space between 0 and 1
plt.xlim(left=0)  
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.legend()
plt.show()