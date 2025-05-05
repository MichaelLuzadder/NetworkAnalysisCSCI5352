import pandas as pd
import graph_tool.all as gt
from pathlib import Path
import gc

# === Config ===
INPUT_FEATHER = "data/proofread_connections_783.feather"
OUTPUT_GRAPH = "data/graphs/central_complex_graph.gt"
CENTRAL_COMPLEX_NEUROPILS = {"NO", "PB", "EB", "FB", "SAD", "PRW", "GNG", "OCG"}

df = pd.read_feather(INPUT_FEATHER)

df_cc = df[df["neuropil"].isin(CENTRAL_COMPLEX_NEUROPILS)].copy()


g = gt.Graph(directed=True)

v_id = g.new_vertex_property("string")
v_neuropil = g.new_vertex_property("string")
g.vertex_properties["pt_root_id"] = v_id
g.vertex_properties["neuropil"] = v_neuropil

e_weight = g.new_edge_property("int")
e_neuropil = g.new_edge_property("string")
g.edge_properties["weight"] = e_weight
g.edge_properties["neuropil"] = e_neuropil

id_to_vertex = {}

def get_vertex(nid, neuropil):
    if nid not in id_to_vertex:
        v = g.add_vertex()
        id_to_vertex[nid] = v
        v_id[v] = str(nid)
        v_neuropil[v] = neuropil  # assign neuropil from first encounter
    return id_to_vertex[nid]

for _, row in df_cc.iterrows():
    src = get_vertex(row["pre_pt_root_id"], row["neuropil"])
    tgt = get_vertex(row["post_pt_root_id"], row["neuropil"])
    e = g.add_edge(src, tgt)
    e_weight[e] = int(row["syn_count"])
    e_neuropil[e] = row["neuropil"]

Path(OUTPUT_GRAPH).parent.mkdir(parents=True, exist_ok=True)
g.save(OUTPUT_GRAPH)
gc.collect()