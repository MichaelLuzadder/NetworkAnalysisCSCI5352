
import pandas as pd
import graph_tool.all as gt
from collections import defaultdict
from pathlib import Path
import gc

proofread_conn_file = "data/proofread_connections_783.feather"
graph_output_dir = Path("data/graphs/gt_final")

df = pd.read_feather(proofread_conn_file)

df = df.dropna(subset=["neuropil"])

neuropils = sorted(df["neuropil"].unique())

for neuropil in neuropils:
    df_np = df[df["neuropil"] == neuropil]
    #weights
    edge_df = df_np[["pre_pt_root_id", "post_pt_root_id", "syn_count"]].copy()
    edge_df.rename(columns={"syn_count": "weight"}, inplace=True)

    g = gt.Graph(directed=True)

    v_prop_id = g.new_vertex_property("string")
    g.vertex_properties["pt_root_id"] = v_prop_id

    id_to_vertex = {}

    def get_vertex(cell_id):
        if cell_id not in id_to_vertex:
            v = g.add_vertex()
            id_to_vertex[cell_id] = v
            v_prop_id[v] = str(cell_id)
        return id_to_vertex[cell_id]

    e_weight = g.new_edge_property("int")
    g.edge_properties["weight"] = e_weight

    for _, row in edge_df.iterrows():
        src = get_vertex(row["pre_pt_root_id"])
        tgt = get_vertex(row["post_pt_root_id"])
        e = g.add_edge(src, tgt)
        e_weight[e] = int(row["weight"])

    output_path = graph_output_dir / f"{neuropil}_graph.gt"
    g.save(str(output_path))
    print(f"{output_path.name} ({g.num_vertices()} neurons, {g.num_edges()} edges)")

    del df_np, edge_df, g, id_to_vertex
    gc.collect()
