import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the FB100 files
data_dir = "facebook100txt"  
output_file = "mean_degrees.csv"  

# Initialize a list to store file names and mean degrees
results = []

# Loop through all files in the directory
for filename in os.listdir(data_dir):
    # Skip attribute files (_attr.txt)
    if filename.endswith("_attr.txt"):
        continue

    file_path = os.path.join(data_dir, filename)

    try:
        # Read the edge list
        edge_list = pd.read_csv(file_path, sep='\s+', header=None, usecols=[0, 1])
        
        # Combine both columns into a single series
        all_nodes = pd.concat([edge_list[0], edge_list[1]])

        # Find the number of unique nodes
        num_nodes = all_nodes.nunique() 
        
        # Determine the number of edges
        num_edges = (0.5 * len(edge_list)) #multiply by 0.5 to avoid double counting
        
        # Calculate the mean degree
        mean_degree = (2 * num_edges) / num_nodes
        
        # Append the result (file name and mean degree) to the list
        results.append([filename, mean_degree, num_nodes, num_edges])

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Convert results to a DataFrame
results_df = pd.DataFrame(results, columns=["File Name", "Mean Degree", "Num Nodes", "Num Edges"])

# Write results to a CSV file
results_df.to_csv(output_file, index=False)
print(f"Results have been written to {output_file}.")

# Plot the histogram of mean degree
mean_degrees = results_df["Mean Degree"]
bin_width = 4  # Adjust bin width for better granularity
bins = np.arange(0, max(mean_degrees) + bin_width, bin_width)

plt.figure(figsize=(10, 6))
plt.hist(mean_degrees, bins=bins, color="green", edgecolor="black", alpha=0.75)

# Title and labels with reduced font size and bold style
plt.title("Histogram of Mean Degree for FB100 Networks", fontsize=14, fontweight="bold")
plt.xlabel("Mean Degree", fontsize=12, fontweight="bold")
plt.ylabel("Frequency", fontsize=12, fontweight="bold")

# Remove gridlines and unnecessary spines
plt.gca().grid(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)

