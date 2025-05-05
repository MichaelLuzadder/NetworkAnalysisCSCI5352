import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

if len(sys.argv) != 2:
    print("Usage: python pca_neuropil_networks.py <CSV_FILE>")
    sys.exit(1)

csv_path = sys.argv[1]
if not os.path.isfile(csv_path):
    print(f"CSV file not found: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

# ------------------------------------------------------------
# 2. Choose feature columns  (add new ones later as you compute them)
# ------------------------------------------------------------
feature_cols = [
    "num_nodes",
    "num_edges",
    "reciprocity",
    "connection_density",
    "average_edge_strength",
    "global_clustering",
    "diameter",                        
    "mean_shortest_path_length",      
    "n_nodes_lcc", "n_edges_lcc"      
]

X = df[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=123)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
df.to_csv("pca_coordinates.csv")

sns.set(style="white", context="talk")

plt.figure(figsize=(10, 8))
ax = sns.scatterplot(
    x="PC1",
    y="PC2",
    data=df,
    s=80,
    edgecolor="k",
    linewidth=0.5
)

for _, row in df.iterrows():
    ax.text(row["PC1"], row["PC2"], row["neuropil"],
            ha="left", va="center", fontsize=8)

var_exp = pca.explained_variance_ratio_ * 100
plt.xlabel(f"PC1 ({var_exp[0]:.1f}% var.)")
plt.ylabel(f"PC2 ({var_exp[1]:.1f}% var.)")
plt.title("PCA of Neuropil Network Metrics")
sns.despine()
plt.tight_layout()
plt.savefig("pca_scatter.png", dpi=300)
plt.show()
