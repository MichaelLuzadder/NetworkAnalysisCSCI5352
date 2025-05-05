import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import os

# Load empirical data
df = pd.read_csv("Neuron_based_final/network_statistics_proofread_neuropils.csv")

# Load null summary statistics
null_df = pd.read_csv("Neuron_based_final/null_model_summary.csv")

# Set metric columns
metric_cols = ["global_clustering", "reciprocity"]

# === Brain system order (most distal → most central) ===
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

# === Neuropil -> Brain system mapping ===
neuropil_groups = {
    "NO": "Central Complex", "PB": "Central Complex", "EB": "Central Complex", "FB": "Central Complex",
    "SAD": "Periesophageal Neuropils", "PRW": "Periesophageal Neuropils",
    "GNG": "Gnathal Ganglia", "OCG": "Ocelli",
    
}

# === Build ordered neuropils list ===
ordered_neuropils = []
for system in brain_system_order:
    candidates = [n for n in df["neuropil"].values if neuropil_groups.get(n, None) == system]
    ordered_neuropils.extend(sorted(candidates))

# === Output directory ===
os.makedirs("Neuron_based_final/plots", exist_ok=True)

# === Loop through each metric and generate plot ===
for metric in metric_cols:
    fig, ax = plt.subplots(figsize=(14, 6))
    x_positions = []
    bar_heights = []
    bar_colors = []
    error_bars = []
    neuropil_labels = []

    xpos = 0
    for region in ordered_neuropils:
        base_region = region.replace("_L", "").replace("_R", "")

        # Plot empirical bars (no error)
        if region in df["neuropil"].values:
            val = df.loc[df["neuropil"] == region, metric].values[0]
            color = (
                "skyblue" if region.endswith("_L") else
                "green" if region.endswith("_R") else
                "mediumpurple"
            )
            x_positions.append(xpos)
            bar_heights.append(val)
            bar_colors.append(color)
            error_bars.append(None)
            xpos += 1

        # Plot null bars (with error)
        if region in null_df["neuropil"].values:
            mean = null_df.loc[null_df["neuropil"] == region, f"{metric}_mean"].values[0]
            low = null_df.loc[null_df["neuropil"] == region, f"{metric}_q025"].values[0]
            high = null_df.loc[null_df["neuropil"] == region, f"{metric}_q975"].values[0]
            ci_lower = mean - low
            ci_upper = high - mean

            x_positions.append(xpos)
            bar_heights.append(mean)
            bar_colors.append("gray")
            error_bars.append([ci_lower, ci_upper])
            xpos += 1

        # Add base label once per region
        if base_region not in neuropil_labels:
            neuropil_labels.append(base_region)

    # Plot bars
    for x, h, c, e in zip(x_positions, bar_heights, bar_colors, error_bars):
        if e is None:
            ax.bar(x, h, color=c, width=0.4)
        else:
            ax.bar(x, h, color=c, width=0.4, yerr=[[e[0]], [e[1]]], capsize=4)

    # Add legend
    handles = [
        Patch(color="skyblue", label="Left Hemisphere"),
        Patch(color="green", label="Right Hemisphere"),
        Patch(color="mediumpurple", label="Central Neuropil"),
        Patch(color="gray", label="Null Model (95% CI)")
    ]
    ax.legend(handles=handles, loc="upper right")

    region_to_positions = {}
    for region, xpos in zip(neuropil_labels, x_positions):
        region_to_positions.setdefault(region, []).append(xpos)

    tick_positions = [sum(pos_list)/len(pos_list) for pos_list in region_to_positions.values()]
    tick_labels = list(region_to_positions.keys())

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, ha="right")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    filename = f"Neuron_based_final/plots/{metric}_with_nulls.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved: {filename}")
    plt.close()