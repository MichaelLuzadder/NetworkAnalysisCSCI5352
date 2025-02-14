
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_null_distribution(stat_values, empirical_value, stat_name="Clustering", out_path="null_distribution.png", bins=30):
    plt.figure(figsize=(7,5))
    plt.hist(stat_values, bins=bins, color='green', edgecolor='black', alpha=0.7)
    plt.axvline(empirical_value, color='red', linestyle='--', linewidth=2, label=f"Empirical {stat_name} = {empirical_value:.4f}")
    plt.title(f"Null Distribution of {stat_name}")
    plt.xlabel(f"{stat_name}")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(out_path)
    plt.close()

# make plot_results_from_npz function
def plot_results_from_npz(npz_file, prefix="network"):
    data = np.load(npz_file)
    C_emp = data["C_emp"]
    L_emp = data["L_emp"]
    C_vals = data["C_values"]
    L_vals = data["L_values"]
    
    # Plot for clustering
    out_path_c = f"{prefix}_C_null_dist.png"
    plot_null_distribution(C_vals, C_emp, stat_name="Clustering Coefficient", out_path=out_path_c)
    
    # Plot for mean path length
    out_path_l = f"{prefix}_L_null_dist.png"
    plot_null_distribution(L_vals, L_emp, 
                           stat_name="Mean Path Length", 
                           out_path=out_path_l)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        sys.exit(1)
    
    npz_file = sys.argv[1]
    prefix = sys.argv[2]
    
    plot_results_from_npz(npz_file, prefix)