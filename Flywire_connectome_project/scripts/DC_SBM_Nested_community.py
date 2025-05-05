
import gzip, pickle
from pathlib import Path
from graph_tool.all import load_graph
from graph_tool.inference import minimize_nested_blockmodel_dl

GRAPH_PATH          = Path("/scratch/Users/milu3967/Networks/central_complex_graph.gt")
GRAPH_OUT           = Path("/scratch/Users/milu3967/Networks/central_complex_graph_with_blocks.gt")
STATE_META_PICKLE   = Path("/scratch/Users/milu3967/Networks/central_complex_dcsbm_nested_state.pkl.gz")
N_MCMC              = 10 

t0 = time.time()

g = load_graph(str(GRAPH_PATH))

if "weight" in g.ep: t_fit = time.time()

nested_state = minimize_nested_blockmodel_dl(
     g,
state_args=dict(deg_corr=True),
    multilevel_mcmc_args=dict(niter=N_MCMC)
    )

state = nested_state.levels[0]

block_prop = g.new_vertex_property("int")
blocks     = state.get_blocks()
for v in g.vertices():
    block_prop[v] = int(blocks[v])
g.vp["block_id"] = block_prop
g.save(str(GRAPH_OUT))

meta = {
    "levels"    : len(nested_state.levels),
    "B_fine"    : int(state.get_B()),
    "deg_corr"  : True,
    "entropy"   : float(state.entropy()),
    }

with gzip.open(STATE_META_PICKLE, "wb") as fh:
    pickle.dump(meta, fh)

