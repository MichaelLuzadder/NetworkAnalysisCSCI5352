import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import os

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
    "NO": "Central Complex", "PB": "Central Complex", "EB": "Central Complex", "FB": "Central Complex",
    "SAD": "Periesophageal Neuropils", "PRW": "Periesophageal Neuropils",
    "GNG": "Gnathal Ganglia", "OCG": "Ocelli",
    "AME_L": "Optic Lobe", "AME_R": "Optic Lobe", "LA_L": "Optic Lobe", "LA_R": "Optic Lobe",
    "LO_L": "Optic Lobe", "LO_R": "Optic Lobe", "LOP_L": "Optic Lobe", "LOP_R": "Optic Lobe",
    "ME_L": "Optic Lobe", "ME_R": "Optic Lobe",
    "BU_L": "Lateral Complex", "BU_R": "Lateral Complex",
    "LAL_L": "Lateral Complex", "LAL_R": "Lateral Complex",
    "GA_L": "Lateral Complex", "GA_R": "Lateral Complex",
    "LH_L": "Lateral Horn", "LH_R": "Lateral Horn",
    "CAN_L": "Periesophageal Neuropils", "CAN_R": "Periesophageal Neuropils",
    "AMMC_L": "Periesophageal Neuropils", "AMMC_R": "Periesophageal Neuropils",
    "FLA_L": "Periesophageal Neuropils", "FLA_R": "Periesophageal Neuropils",
    "ICL_L": "Inferior Neuropils", "ICL_R": "Inferior Neuropils",
    "IB_L": "Inferior Neuropils", "IB_R": "Inferior Neuropils",
    "ATL_L": "Inferior Neuropils", "ATL_R": "Inferior Neuropils",
    "CRE_L": "Inferior Neuropils", "CRE_R": "Inferior Neuropils",
    "SCL_L": "Inferior Neuropils", "SCL_R": "Inferior Neuropils",
    "VES_L": "Ventromedial Neuropils", "VES_R": "Ventromedial Neuropils",
    "GOR_L": "Ventromedial Neuropils", "GOR_R": "Ventromedial Neuropils",
    "SPS_L": "Ventromedial Neuropils", "SPS_R": "Ventromedial Neuropils",
    "IPS_L": "Ventromedial Neuropils", "IPS_R": "Ventromedial Neuropils",
    "EPA_L": "Ventromedial Neuropils", "EPA_R": "Ventromedial Neuropils",
    "MB_PED_L": "Mushroom Body", "MB_PED_R": "Mushroom Body",
    "MB_VL_L": "Mushroom Body", "MB_VL_R": "Mushroom Body",
    "MB_ML_L": "Mushroom Body", "MB_ML_R": "Mushroom Body",
    "MB_CA_L": "Mushroom Body", "MB_CA_R": "Mushroom Body",
    "AL_L": "Antennal Lobe", "AL_R": "Antennal Lobe",
    "SLP_L": "Superior Neuropils", "SLP_R": "Superior Neuropils",
    "SIP_L": "Superior Neuropils", "SIP_R": "Superior Neuropils",
    "SMP_L": "Superior Neuropils", "SMP_R": "Superior Neuropils",
    "AVLP_L": "Ventrolateral Neuropils", "AVLP_R": "Ventrolateral Neuropils",
    "PVLP_L": "Ventrolateral Neuropils", "PVLP_R": "Ventrolateral Neuropils",
    "WED_L": "Ventrolateral Neuropils", "WED_R": "Ventrolateral Neuropils",
    "PLP_L": "Ventrolateral Neuropils", "PLP_R": "Ventrolateral Neuropils",
    "AOTU_L": "Ventrolateral Neuropils", "AOTU_R": "Ventrolateral Neuropils"
}

ordered_neuropils = []
for system in brain_system_order:
    candidates = [n for n in df["neuropil"].values if neuropil_groups.get(n, None) == system]
    ordered_neuropils.extend(sorted(candidates))

metric_cols = df.columns.drop("neuropil")

for metric in metric_cols:
    bar_width = 0.46
    x_positions = []
    bar_heights = []
    bar_colors = []
    neuropil_labels = []

    i = 0
    while i < len(ordered_neuropils):
        n = ordered_neuropils[i]
        region = n.replace("_L", "").replace("_R", "")

        # Left hemisphere
        if f"{region}_L" in df["neuropil"].values:
            val = df.loc[df["neuropil"] == f"{region}_L", metric].values[0]
            x_positions.append(len(neuropil_labels) - 0.5 * bar_width)
            bar_heights.append(val)
            bar_colors.append("skyblue")

        # Right hemisphere 
        if f"{region}_R" in df["neuropil"].values:
            val = df.loc[df["neuropil"] == f"{region}_R", metric].values[0]
            x_positions.append(len(neuropil_labels) + 0.5 * bar_width)
            bar_heights.append(val)
            bar_colors.append("green")

        # Central complex
        if region in df["neuropil"].values:
            val = df.loc[df["neuropil"] == region, metric].values[0]
            x_positions.append(len(neuropil_labels))
            bar_heights.append(val)
            bar_colors.append("mediumpurple")

        neuropil_labels.append(region)

        while i + 1 < len(ordered_neuropils) and ordered_neuropils[i + 1].replace("_L", "").replace("_R", "") == region:
            i += 1
        i += 1


    sns.set(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x_positions,
        bar_heights,
        width=[bar_width if c != "mediumpurple" else bar_width * 2 for c in bar_colors],
        color=bar_colors
    )

    handles = [
        Patch(color="skyblue", label="Left Hemisphere"),
        Patch(color="green", label="Right Hemisphere"),
        Patch(color="mediumpurple", label="Central Neuropil")
    ]
    ax.legend(handles=handles, loc="upper right")

    ax.set_xticks(range(len(neuropil_labels)))
    ax.set_xticklabels(neuropil_labels, rotation=90)
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.set_title(f"{metric.replace('_', ' ').capitalize()} by Neuropil")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    filename = f"Neuron_based_final/plots/{metric}_by_neuropil_grouped.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()