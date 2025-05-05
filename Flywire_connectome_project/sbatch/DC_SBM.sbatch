#!/bin/bash
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --job-name=DC_SBM_centcomplex
#SBATCH --output=logs/DCSBM_cp_%A_%a.out
#SBATCH --error=logs/DCSBM_cp_%A_%a.err
#SBATCH --mail-type=ALL

source ~/.bashrc
eval "$(mamba shell hook --shell bash)"
mamba activate Networks  # Keeping the same environment for dependencies

chmod +x /scratch/Users/milu3967/Networks/parallel_DC_SBM.py

python /scratch/Users/milu3967/Networks/parallel_DC_SBM.py




