#!/bin/bash
#SBATCH --job-name=NuTauFloq_L6_12
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/floquet_frontier_L6_12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# Kicked-Ising frontier, K = 0 (o1) and K = 1 for L = 6..12, theta in [0.1, 3000]
# (37 pts) + theta = inf.  -> <SSO_OUTPUT>/KickedIsing_g0.9J1.0h0.809_DSFloquet_Data/
# (original: slurm/run_nutau_floquet_ksec_L6_12.slurm, NuTau_ds_Floquet.py)
python $D/frontier.py --model=KickedIsing --g=0.9 --J=1.0 --h=0.809 \
    --Lmin=6 --Lmax=12 --K=0,1 --k_save=2 --lam=0.0 \
    --theta_min=0.1 --theta_max=3000.0 --theta_num=37 --include_infinity=True --usegpu=True
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
