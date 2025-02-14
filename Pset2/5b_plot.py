import csv
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    # Read the real centralities from 'real_centralities.csv'
    real_cents = {}
    with open("real_centralities.csv", "r") as f:
        next(f)
        for line in f:
            family, val_str = line.strip().split(",")
            real_cents[family] = float(val_str)
    
    # Read the null differences from 'null_differences.csv'
    differences_dict = defaultdict(list)
    with open("null_differences.csv", "r") as f:
        next(f)
        for line in f:
            rep_idx_str, family, diff_str = line.strip().split(",")
            diff_val = float(diff_str)
            differences_dict[family].append(diff_val)
    

    families_sorted = sorted(real_cents.keys(), key=lambda x: real_cents[x], reverse=True)
    
    data_for_plot = [differences_dict[fam] for fam in families_sorted]
    
    # Create a box plot of (C^null - C^real)
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data_for_plot, vert=True, patch_artist=True)
    for patch in box['boxes']:
        patch.set_facecolor('green')
    for whisker in box['whiskers']: 
        whisker.set_color('black')
    for cap in box['caps']:
        cap.set_color('green')
    for median in box['medians']:
        median.set_color('black')
    
    # 6.reference line at 0
    plt.axhline(y=0, color='grey', linestyle='--')

    plt.xticks(range(1, len(families_sorted) + 1), 
               families_sorted, 
               rotation=45, ha='right')
    plt.ylabel("C^null - C^real (Normalized Harmonic Centrality)")
    plt.title("C(null) - C(real) for 1000 Configuration Model Randomizations of Each Family")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig("null_differences_boxplot.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()