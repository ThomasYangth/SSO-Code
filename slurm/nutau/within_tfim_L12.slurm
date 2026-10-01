#!/bin/bash
#SBATCH --job-name=Within_TFIM
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/within_tfim_L12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# Within-K0 crossing of the TFIM (hx=2.0, hz=0.0), L = 12, on the bracket
# theta in [0.3981, 0.5012] of the 51-pt grid (nu 0.2478 -> 0.1223).
#   -> crossings/cross_L12_K0_l0.0k2o4_theta*.npz
# (original: slurm/run_nutau_crossing_ksec_L12.slurm, NuTau_soft_crossing.py
#  --locate_mode=nu, which wrote cross_L12_l0.0k2o4_theta*.npz)
python $D/crossing_within.py --model=Ising --J=1.0 --hx=2.0 --hz=0.0 --L=12 --K=0 \
    --lam=0.0 --pows_max=3 --k_save=2 --k_ref=8 --theta_bracket=0.3981,0.5012 --usegpu=True
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
