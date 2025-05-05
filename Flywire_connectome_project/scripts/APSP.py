
import graph_tool.all as gt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

graphs_dir = Path("/scratch/Users/milu3967/Networks/gt_final")  # adjust if needed
output_dir = Path("/scratch/Users/milu3967/Networks/apsp_results_final")

neuropil = sys.argv[1]
graph_path = graphs_dir / f"{neuropil}_graph.gt"

if not graph_path.exists():
    print(f"Graph file not found: {graph_path}")
    sys.exit(1)

print(f"  Processing {neuropil}...")

g = gt.load_graph(str(graph_path))

comp, hist = gt.label_components(g, directed=True)
largest_label = np.argmax(hist)
v_mask = comp.a == largest_label
g_lcc = gt.GraphView(g, vfilt=v_mask)

dist = gt.shortest_distance(g_lcc, directed=True)

dist_arr = np.array([dist[v].a for v in g_lcc.vertices()])
dist_flat = np.concatenate(dist_arr)

finite_dists = dist_flat[np.isfinite(dist_flat) & (dist_flat > 0)]

diameter = finite_dists.max()
mean_path_length = finite_dists.mean()

outpath = output_dir / f"{neuropil}_apsp_stats.csv"
df = pd.DataFrame([{
    "neuropil": neuropil,
    "diameter": diameter,
    "mean_shortest_path_length": mean_path_length,
    "n_nodes_lcc": g_lcc.num_vertices(),
    "n_edges_lcc": g_lcc.num_edges()
}])
df.to_csv(outpath)

