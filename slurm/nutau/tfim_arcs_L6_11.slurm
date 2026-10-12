#!/bin/bash
#SBATCH --job-name=TFIM_arcs
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch (after
# tfim_frontier_L6_11.slurm):
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/tfim_arcs_L6_11.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# TFIM (hx=2.0, hz=0.0), L = 6..11 (figure_D (b)):
#  * K = 0 <-> K = 1 superposition arcs (one W_0 - W_1 sign change per L, theta 1.3..2.5);
#  * the within-K0 crossing, on the same bracket theta in [0.3981, 0.5012] at every L
#    (largest tau jump of each 51-pt grid; L = 12 is within_tfim_L12.slurm).
#   -> Ising_J1.0X2.0Z0.0_DSsoftKsec_Data/crossings/{crossK_L*_K0K1_*,cross_L*_K0_l0.0k2o4_*}.npz
M="--model=Ising --J=1.0 --hx=2.0 --hz=0.0 --lam=0.0 --pows_max=3 --k_save=2 --usegpu=True"
python $D/crossing_sectors.py $M --Lmin=6 --Lmax=11 --Ka=0 --Kb=1
for L in 6 7 8 9 10 11; do
    python $D/crossing_within.py $M --L=$L --K=0 --k_ref=8 --theta_bracket=0.3981,0.5012 \
        || echo "WARNING: within-K0 arc L=$L failed"
done
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
