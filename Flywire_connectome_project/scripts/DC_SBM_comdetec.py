
from graph_tool.all import load_graph, minimize_blockmodel_dl
from pathlib import Path
import sys

GRAPH_PATH = Path("/scratch/Users/milu3967/Networks/central_complex_graph.gt")  
SAVE_STATE_PATH = Path("/scratch/Users/milu3967/Networks/central_complex_dcsbm_state.gt")  

g = load_graph(str(GRAPH_PATH))

weight = g.ep["weight"] if "weight" in g.ep else None

print("  Fitting degree-corrected SBM...")
state = minimize_blockmodel_dl(
    g,
    state_args=dict(deg_corr=True),
    multilevel_mcmc_args=dict(niter=10)   # optional but often improves inference
)

from graph_tool.inference import save as save_state
save_state(state, str(SAVE_STATE_PATH))
print("the end")

