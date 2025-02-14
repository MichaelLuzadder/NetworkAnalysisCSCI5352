import csv
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    real_cents = {}
    with open("real_centralities_stub.csv", "r") as f:
        next(f)  # Skip header
        for line in f:
            family, val_str = line.strip().split(",")
            real_cents[family] = float(val_str)

    differences_dict = defaultdict(list)
    with open("null_differences_stub.csv", "r") as f:
        next(f)  
        for line in f:
            rep_idx_str, family, diff_str = line.strip().split(",")
            differences_dict[family].append(float(diff_str))
    
    # Sort families by real centrality 
    families_sorted = sorted(real_cents.keys(), key=lambda x: real_cents[x], reverse=True)
    data_for_plot = [differences_dict[fam] for fam in families_sorted]

    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data_for_plot, vert=True, patch_artist=True)
    
    for patch in box['boxes']:
        patch.set_facecolor('green')
    for whisker in box['whiskers']:
        whisker.set_color('black')
    
    plt.axhline(y=0, color='grey', linestyle='--')
    plt.xticks(range(1, len(families_sorted) + 1), families_sorted, rotation=45, ha='right')
    plt.ylabel("C^null - C^real (Normalized Harmonic Centrality)")
    plt.title("Stub-Matching Null Model: C(null) - C(real) for 1000 randomizations")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.savefig("stub_matching_boxplot.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()