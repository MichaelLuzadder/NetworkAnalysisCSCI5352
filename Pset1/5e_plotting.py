import pandas as pd
import matplotlib.pyplot as plt

# Load the output file from 5e_computing.py
df = pd.read_csv("5e/FB100_network_diameters_accurate.csv")

# Plot l_max vs network size
plt.figure(figsize=(6, 6))
plt.scatter(df["Size_LCC"], df["Diameter_lmax"], alpha=1, color="green", linewidths=1, edgecolors="black")
plt.xlabel("Network size (n)", fontweight="bold")
plt.ylabel("Diameter (l_max)", fontweight="bold")
plt.title("Network Diameter vs. Network Size", fontweight="bold")
plt.grid(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.show()


# Plot mean geodesic distance vs LCC size
plt.figure(figsize=(6, 6))
plt.scatter(df["Size_LCC"], df["Mean_Geodesic_Distance"], alpha=1, color="green", linewidths=1, edgecolors="black")
plt.xlabel("Size of LCC (n)", fontweight="bold")
plt.ylabel("Mean Geodesic Distance (⟨l⟩)", fontweight="bold")
plt.title("Mean Shortest Path Length vs. Network Size", fontweight="bold")
plt.grid(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.show()