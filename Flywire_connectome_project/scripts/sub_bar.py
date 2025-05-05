import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

df = pd.read_csv("Neuron_based_final/network_statistics_proofread_neuropils.csv")


brain_system_order = [
    "Optic Lobe",
    "Lateral Complex",
    "Lateral Horn",
    "Ventrolateral Neuropils",
    "Superior Neuropils",
    "Inferior Neuropils",
    "Ventromedial Neuropils",
    "Mushroom Body",
    "Periesophageal Neuropils",
    "Gnathal Ganglia",
    "Central Complex",
    "Ocelli"
]

neuropil_groups = {
     "EB": "Central Complex", "ME_L": "Optic Lobe", "ME_R": "Optic Lobe",
    "PVLP_L": "Mushroom Body", "PVLP_R": "Mushroom Body",
}

ordered_neuropils = []
for system in brain_system_order:
    candidates = [n for n in df["neuropil"].values if neuropil_groups.get(n, None) == system]
    candidates_sorted = sorted(candidates)
    ordered_neuropils.extend(candidates_sorted)

bar_width = 0.46
x_positions = []
bar_heights = []
bar_colors = []
neuropil_labels = []

i = 0
while i < len(ordered_neuropils):
    n = ordered_neuropils[i]
    region = n.replace("_L", "").replace("_R", "")

    # left hem
    if f"{region}_L" in df["neuropil"].values:
        val = df.loc[df["neuropil"] == f"{region}_L", "global_clustering"].values[0]
        x_positions.append(len(neuropil_labels) - 0.5 * bar_width)
        bar_heights.append(val)
        bar_colors.append("skyblue")

    #right hem
    if f"{region}_R" in df["neuropil"].values:
        val = df.loc[df["neuropil"] == f"{region}_R", "global_clustering"].values[0]
        x_positions.append(len(neuropil_labels) + 0.5 * bar_width)
        bar_heights.append(val)
        bar_colors.append("green")

    # cx
    if region in df["neuropil"].values:
        val = df.loc[df["neuropil"] == region, "global_clustering"].values[0]
        x_positions.append(len(neuropil_labels))
        bar_heights.append(val)
        bar_colors.append("mediumpurple")
        

    neuropil_labels.append(region)

    while i+1 < len(ordered_neuropils) and ordered_neuropils[i+1].replace("_L", "").replace("_R", "") == region:
        i += 1
    i += 1

sns.set(style="white", context="talk")
fig, ax = plt.subplots(figsize=(6,6))

ax.bar(x_positions, bar_heights, width=[bar_width if c != "mediumpurple" else bar_width*2 for c in bar_colors], color=bar_colors)
handles = [
    Patch(color="skyblue", label="Left Hemisphere"),
    Patch(color="green", label="Right Hemisphere"),
    Patch(color="mediumpurple", label="Central Neuropil")
]
ax.legend(handles=handles, loc="upper left")
ax.set_xticks(range(len(neuropil_labels)))
ax.set_xticklabels(neuropil_labels, rotation=90)
ax.set_ylabel("Global Clustering Coefficient")
ax.set_title("Global Clustering Coefficient by Neuropil")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("Number_neurons_by_neuropil_grouped_clean.png", dpi=300)
plt.show()