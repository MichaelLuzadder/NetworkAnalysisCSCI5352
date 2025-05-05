import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set(style="white", context="talk")

null_dir = Path("Neuron_based_final/null_model_results")
empirical_path = Path("Neuron_based_final/network_statistics_proofread_neuropils.csv")
output_dir = Path("Neuron_based_final/plots/sbm_null_distributions")


neuropils = ["AL_L", "AL_R", "AME_L", "AME_R", "AMMC_L", "AMMC_R", "AOTU_L", "AOTU_R",
    "ATL_L", "ATL_R", "AVLP_L", "AVLP_R", "BU_L", "BU_R", "CAN_L", "CAN_R",
    "CRE_L", "CRE_R", "EB", "EPA_L", "EPA_R", "FB", "FLA_L", "FLA_R", "GA_L",
    "GA_R", "GNG", "GOR_L", "GOR_R", "IB_L", "IB_R", "ICL_L", "ICL_R", "IPS_L",
    "IPS_R", "LA_L", "LA_R", "LAL_L", "LAL_R", "LH_L", "LH_R", "LO_L", "LO_R",
    "LOP_L", "LOP_R", "MB_CA_L", "MB_CA_R", "MB_ML_L", "MB_ML_R", "MB_PED_L",
    "MB_PED_R", "MB_VL_L", "MB_VL_R", "ME_L", "ME_R", "NO", "OCG", "PB", "PLP_L",
    "PLP_R", "PRW", "PVLP_L", "PVLP_R", "SAD", "SCL_L", "SCL_R", "SIP_L", "SIP_R",
    "SLP_L", "SLP_R", "SMP_L", "SMP_R", "VES_L", "VES_R", "SPS_L", "SPS_R",
    "WED_L", "WED_R"]
metrics = ["reciprocity", "global_clustering"]

emp_df = pd.read_csv(empirical_path)

for neuropil in neuropils:
    null_file = null_dir / f"{neuropil}_null_samples.csv"
    if not null_file.exists():
        print(f"⚠️ Warning: Missing null model file for {neuropil}")
        continue

    null_df = pd.read_csv(null_file)
    emp_row = emp_df[emp_df["neuropil"] == neuropil]

    for metric in metrics:
        emp_val = emp_row[metric].values[0]

        plt.figure(figsize=(4, 4))
        sns.histplot(null_df[metric], bins=10, kde=False, color="mediumpurple", edgecolor="black")
        plt.axvline(emp_val, color="red", linestyle="--", label="Empirical")
        plt.title(f"{metric.replace('_', ' ').capitalize()} distribution for {neuropil}")
        plt.xlabel(metric.replace("_", " ").capitalize())
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()

        fname = output_dir / f"{neuropil}_{metric}_sbm_null_hist.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"{fname}")