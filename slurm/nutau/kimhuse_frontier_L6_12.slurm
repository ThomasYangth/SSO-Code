#!/bin/bash
#SBATCH --job-name=NuTauKH_L6_12
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/kimhuse_frontier_L6_12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# Kim-Huse (J=1, hx=0.905, hz=0.809) sector frontier, K = 0 (o4), 1, 2 (o0),
# L = 6..12, theta in [0.1, 1e4] (51 pts).  -> Ising_J1.0X0.905Z0.809_DSsoftKsec_Data/
# (original: slurm/run_nutau_ds_soft_ksec_ising_prod.slurm, NuTau_ds_soft_ksec.py)
python $D/frontier.py --model=Ising --J=1.0 --hx=0.905 --hz=0.809 \
    --Lmin=6 --Lmax=12 --K=0,1,2 --pows_max=3 --k_save=2 --lam=0.0 \
    --theta_min=0.1 --theta_max=10000.0 --theta_num=51 --usegpu=True
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
