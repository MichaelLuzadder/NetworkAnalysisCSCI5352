
import graph_tool.all as gt
import pandas as pd
import numpy as np
from pathlib import Path

graph_dir = Path("data/graphs/gt_final")

results = []

for gp in sorted(graph_dir.glob("*_graph.gt")):
    neuropil = gp.stem.replace("_graph", "")

    g = gt.load_graph(str(gp))

    num_nodes = g.num_vertices()
    num_edges = g.num_edges()

    # reciprocity
    reciprocity = gt.edge_reciprocity(g)

    # connection density
    if num_nodes > 1:
        connection_density = num_edges / (num_nodes * (num_nodes - 1))
    else:
        connection_density = np.nan

    # average weight
    if "weight" in g.ep.keys():
        weights = np.array([g.ep.weight[e] for e in g.edges()])
        average_edge_strength = weights.mean() if len(weights) > 0 else np.nan
    else:
        average_edge_strength = np.nan

    # global clustering coefficient
   
    clustering = gt.global_clustering(g)[0] 

    results.append({
        "neuropil": neuropil,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "reciprocity": reciprocity,
        "connection_density": connection_density,
        "average_edge_strength": average_edge_strength,
        "global_clustering": clustering
    })

df = pd.DataFrame(results)
df.to_csv("Neuron_based_final/network_statistics_proofread_neuropils.csv", index=False)
