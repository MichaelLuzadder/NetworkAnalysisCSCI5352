import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
   #read csvs from computation into a dictionary 
    input_csv = "outputs/berkeley_experiment_results_test.csv"

    data_by_r = defaultdict(lambda: {"clustering": [], "path_length": []})

    with open(input_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r_val = int(row["r"])
            clustering = float(row["clustering"])
            path_length = float(row["path_length"])
            data_by_r[r_val]["clustering"].append(clustering)
            data_by_r[r_val]["path_length"].append(path_length)

    swap_values = sorted(data_by_r.keys())

    # Compute mean and std for each r between experiments
    mean_clustering = []
    std_clustering = []
    mean_path_length = []
    std_path_length = []

    for r in swap_values:
        c_arr = np.array(data_by_r[r]["clustering"])
        p_arr = np.array(data_by_r[r]["path_length"])
        mean_clustering.append(c_arr.mean())
        std_clustering.append(c_arr.std())
        mean_path_length.append(p_arr.mean())
        std_path_length.append(p_arr.std())

    # Store values for r=20m
    final_C = mean_clustering[-1]
    final_L = mean_path_length[-1]
    final_r = swap_values[-1]

    # Plotting
    fig_log, (ax1_log, ax2_log) = plt.subplots(1, 2, figsize=(12, 5))

    # -- Clustering (Log X)
    ax1_log.errorbar(
        swap_values, mean_clustering, yerr=std_clustering,
        marker='o', markerfacecolor='green', markeredgecolor='green',
        color='black', ecolor='black', capsize=4, linewidth=1.5,
        label='Mean ± Std'
    )
    ax1_log.axhline(
        y=final_C, color='r', linestyle='--',
        label=f'r = 20m (r={final_r}) = {final_C:.4f}'
    )
    ax1_log.set_xlabel('Number of double-edge swaps (r) (semilog scale)')
    ax1_log.set_ylabel('Clustering coefficient')
    ax1_log.set_title('Clustering coefficient vs. Number of double edge swaps')
    ax1_log.legend()
    ax1_log.spines['top'].set_visible(False)
    ax1_log.spines['right'].set_visible(False)

    # Set x-axis to symlog to include r=0
    ax1_log.set_xscale('symlog', linthresh=1)
    # Optionally, we can ensure it doesn't go negative:
    ax1_log.set_xlim(left=0)

    # Path length
    ax2_log.errorbar(
        swap_values, mean_path_length, yerr=std_path_length,
        marker='o', markerfacecolor='green', markeredgecolor='green',
        color='black', ecolor='black', capsize=4, linewidth=1.5,
        label='Mean ± Std'
    )
    ax2_log.axhline(
        y=final_L, color='r', linestyle='--',
        label=f'r = 20m (r={final_r}) = {final_L:.4f}'
    )
    ax2_log.set_xlabel('Number of double-edge swaps (r) (semilog scale)')
    ax2_log.set_ylabel('Mean shortest path length')
    ax2_log.set_title('Mean shortest path length vs. Number of double edge swaps')
    ax2_log.legend()
    ax2_log.spines['top'].set_visible(False)
    ax2_log.spines['right'].set_visible(False)
    ax2_log.set_xscale('symlog', linthresh=1)
    ax2_log.set_xlim(left=0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()