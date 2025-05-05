
import graph_tool.all as gt
import pandas as pd
from pathlib import Path

graph_dir = Path("data/graphs/gt_final")

records = []

for gp in sorted(graph_dir.glob("*_graph.gt")):
    neuropil = gp.stem.replace("_graph", "")
    if neuropil == "None":
        continue

    g = gt.load_graph(str(gp))

    in_deg = g.get_in_degrees(g.get_vertices())
    out_deg = g.get_out_degrees(g.get_vertices())

    for v, indegree, outdegree in zip(g.vertices(), in_deg, out_deg):
        records.append({
            "neuropil": neuropil,
            "node_id": int(v),  
            "in_degree": indegree,
            "out_degree": outdegree
        })

df = pd.DataFrame(records)
df.to_csv("Neuron_based_final/degrees_by_neuropil.csv")
