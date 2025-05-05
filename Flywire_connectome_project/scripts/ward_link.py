
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

df = pd.read_csv("Neuron_based_final/stats/pca_coordinates.csv") 
X = df[["PC1", "PC2"]].values

k = 42 # Number of clusters 
kmeans = KMeans(n_clusters=k, random_state=1234)
df["kmeans_cluster"] = kmeans.fit_predict(X)

linkage_matrix = linkage(X, method="ward")
sns.set(style="white", context="talk", font_scale=1.2)

plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, labels=df["neuropil"].values, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram ")
plt.tight_layout()
plt.savefig("dendrogram_neuropils.png", dpi=300)
plt.show()


plt.figure(figsize=(10, 8))
palette = sns.color_palette("tab10", n_colors=k)
sns.scatterplot(x="PC1", y="PC2", hue="kmeans_cluster", data=df, palette=palette, s=100, legend="full")

for i in range(df.shape[0]):
    plt.text(df["PC1"][i]+0.03, df["PC2"][i]+0.03, df["neuropil"][i], fontsize=8)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA of Neuropils with K-means Clusters (k={k})")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("pca_clusters_neuropils.png", dpi=300)
plt.show()