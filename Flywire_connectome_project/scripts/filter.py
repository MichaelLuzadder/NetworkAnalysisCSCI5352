import pandas as pd
import pyarrow.feather as feather
from pathlib import Path

input_file = Path("data/proofread_connections_783.feather")
output_file = Path("Neuron_based_final/filtered_synapses_proofread_only.feather")

print(f"loading {input_file}")
df = feather.read_feather(input_file)

#columns_to_keep = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"]
#df_filtered = df[columns_to_keep]

df_filtered.reset_index(drop=True).to_feather(output_file)
