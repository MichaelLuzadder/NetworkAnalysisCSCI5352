
import pandas as pd

df = pd.read_csv("Neuron_based_final/network_statistics_proofread_neuropils.csv")  # <-- replace with your actual file path

neuropils = df["neuropil"]
lefts = [n for n in neuropils if n.endswith("_L")]
pairs = [(l, l.replace("_L", "_R")) for l in lefts if l.replace("_L", "_R") in neuropils.values]


diffs = []

for left, right in pairs:
    row_l = df[df["neuropil"] == left].iloc[0]
    row_r = df[df["neuropil"] == right].iloc[0]
    

    for col in df.columns:
        if col == "neuropil":
            continue
        diff = abs(row_l[col] - row_r[col])
        diffs.append((col, diff))

diff_df = pd.DataFrame(diffs, columns=["metric", "difference"])
avg_diff = diff_df.groupby("metric").mean().sort_values("difference", ascending=False)

print("Average absolute differences between L and R neuropils:")
print(avg_diff)
