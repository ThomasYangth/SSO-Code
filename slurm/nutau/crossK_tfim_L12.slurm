#!/bin/bash
#SBATCH --job-name=CrossK_TFIM
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/crossK_tfim_L12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# K = 0 <-> K = 1 arc for the TFIM (hx=2.0, hz=0.0), L = 12 (tfim_frontier_L12.slurm first).
# (original: slurm/run_crossK_arcs_tfim_L12.slurm)
python $D/crossing_sectors.py --model=Ising --J=1.0 --hx=2.0 --hz=0.0 \
    --Lmin=12 --Lmax=12 --Ka=0 --Kb=1 --lam=0.0 --pows_max=3 --k_save=2 --usegpu=True
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
