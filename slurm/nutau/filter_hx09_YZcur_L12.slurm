#!/bin/bash
#SBATCH --job-name=Filt_hx09
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/filter_hx09_YZcur_L12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# Gaussian / hard / time-average filters of the current sum(Y_x Z_x+1 - Z_x Y_x+1)
# in the eigenbasis of H(hx=0.9, hz), L = 12, K = 0, span{I..H^3} removed,
# 60 widths each.  A few minutes per hz on MIG.
#   -> Ising_J1.0X0.9Z{hz}_StaticFilter_Data/filt_L12_K0_YZcur_l0.0o4.npz
# (original: slurm/run_nutau_filter_YZcur_L12_hx0.9.slurm, NuTauFilterStatic.py)
for HZ in 0.005 0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2; do
    python $D/filter_static.py --model=Ising --J=1.0 --hx=0.9 --hz=$HZ --L=12 --K=0 \
        --init_op=YZcur --pows_max=3 --n_width=60 --usegpu=True || echo "WARNING: hz=$HZ failed"
done
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
