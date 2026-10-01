#!/bin/bash
#SBATCH --job-name=NuTauTFIM_L12
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/tfim_frontier_L12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# TFIM (J=1, hx=2.0, hz=0.0), L = 12: K = 0 (o4) on the 51-pt grid plus a
# 21-pt fill of the within-K0 crossing bracket [0.3981, 0.5012], and K = 1 (o0).
#   -> Ising_J1.0X2.0Z0.0_DSsoftKsec_Data/
# (originals: hz = 0.0 entries of run_nutau_ksec_hz_scan_L12.slurm and
#  run_nutau_ksec_hz_fill_bracket.slurm; run_nutau_ds_soft_ksec_tfim_L12_K1.slurm)
COMMON="--model=Ising --J=1.0 --hx=2.0 --hz=0.0 --Lmin=12 --Lmax=12 --pows_max=3 --k_save=2 --lam=0.0 --usegpu=True"
python $D/frontier.py $COMMON --K=0 --theta_min=0.1 --theta_max=10000.0 --theta_num=51
python $D/frontier.py $COMMON --K=0 --theta_min=0.3981 --theta_max=0.5012 --theta_num=21
python $D/frontier.py $COMMON --K=1 --theta_min=0.1 --theta_max=10000.0 --theta_num=51
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
