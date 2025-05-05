
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("Neuron_based_final/degrees_by_neuropil.csv")

sns.set(style="white", context="talk")

neuropils = sorted(df["neuropil"].unique())
n_neuropils = len(neuropils)

n_cols = 6
n_rows = int(np.ceil(n_neuropils / n_cols))
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*n_cols, 3*n_rows))
axes = axes.flatten()

for idx, neuropil in enumerate(neuropils):
    ax = axes[idx]

    sub = df[df["neuropil"] == neuropil]

    # In-degree CCDF
    degrees_in = sub["in_degree"].values
    degrees_in_sorted = np.sort(degrees_in)
    ccdf_in = 1.0 - np.arange(1, len(degrees_in_sorted)+1) / len(degrees_in_sorted)

    # Out-degree CCDF
    degrees_out = sub["out_degree"].values
    degrees_out_sorted = np.sort(degrees_out)
    ccdf_out = 1.0 - np.arange(1, len(degrees_out_sorted)+1) / len(degrees_out_sorted)

    ax.plot(degrees_in_sorted, ccdf_in, label="In-degree", color="skyblue")
    ax.plot(degrees_out_sorted, ccdf_out, label="Out-degree", color="green")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1)
    ax.set_ylim(top=1)

    ax.set_title(neuropil, fontsize=12)
    ax.grid(False)

    if idx % n_cols == 0:
        ax.set_ylabel("CCDF")
    else:
        ax.set_ylabel("")
    ax.set_xlabel("Degree" if idx >= len(neuropils)-n_cols else "")

handles = [
    plt.Line2D([], [], color="skyblue", label="In-degree"),
    plt.Line2D([], [], color="green", label="Out-degree")
]
fig.legend(handles=handles, loc="upper right", fontsize=12)

plt.suptitle("Degree CCDFs by Neuropil", fontsize=18)
plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # leave space for suptitle and legend
plt.savefig("Neuron_based_final/plots/ccdf_degrees_by_neuropil.png", dpi=300)
plt.show()