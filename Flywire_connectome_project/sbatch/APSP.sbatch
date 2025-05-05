#!/bin/bash
#SBATCH -p highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512gb
#SBATCH --time=8:00:00
#SBATCH --job-name=apsp_neuropils
#SBATCH --output=logs/neuropil_%A_%a.out
#SBATCH --error=logs/neuropil_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=milu3967@colorado.edu
#SBATCH --array=0-3

source ~/.bashrc
eval "$(mamba shell hook --shell bash)"
mamba activate Networks  # Keeping the same environment for dependencies

NEUROPILS=(
VES_L VES_R WED_L WED_R
)

NP=${NEUROPILS[$SLURM_ARRAY_TASK_ID]}

python /scratch/Users/milu3967/Networks/gt_APSP_LCC.py $NP





