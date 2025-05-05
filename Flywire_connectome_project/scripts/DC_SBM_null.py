
import graph_tool.all as gt
import numpy as np
import pandas as pd
from pathlib import Path
import sys, os

GRAPHS_DIR = Path("/scratch/Users/milu3967/Networks/gt_final")
OUT_DIR = Path("/scratch/Users/milu3967/Networks/null_model_results")
N_SAMPLES = 300

neuropil = sys.argv[1]
g_path = GRAPHS_DIR / f"{neuropil}_graph.gt"

if not g_path.exists():
    print(f"Graph not found: {g_path}")
    sys.exit(1)

print(f"  [{neuropil}] fitting DC-SBM and generating {N_SAMPLES} null graphs…")

g_emp = gt.load_graph(str(g_path))
use_weights = "weight" in g_emp.ep.keys()

state = gt.minimize_blockmodel_dl(g_emp, state_args=dict(deg_corr=True))

rows = []

for i in range(N_SAMPLES):
 
    state.mcmc_sweep(niter=20)
    gt.seed_rng(np.random.randint(0, 1e9))

    g_null = state.sample_graph()
    clust, _ = gt.global_clustering(g_null)
    n_v = g_null.num_vertices()
    n_e = g_null.num_edges()
    density = n_e / (n_v * (n_v - 1)) if n_v > 1 else np.nan
    reciprocity = gt.edge_reciprocity(g_null)

    if use_weights and "weight" in g_null.ep:
        avg_strength = np.mean([g_null.ep.weight[e] for e in g_null.edges()]) # did not work