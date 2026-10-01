#!/bin/bash
#SBATCH --job-name=NuTauKH_L6dense
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/kimhuse_frontier_L6_dense.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# Kim-Huse L = 6, all K, theta in [0.1, 1e5] (61 pts).  Together with the 51-pt
# production grid this gives the 62-point L = 6 K = 0 / K = 1 grids of
# kimhuse_three_panel (a).  (original: slurm/run_nutau_ds_soft_ksec_ising_L6.slurm)
python $D/frontier.py --model=Ising --J=1.0 --hx=0.905 --hz=0.809 \
    --Lmin=6 --Lmax=6 --K=all --pows_max=3 --k_save=2 --lam=0.0 \
    --theta_min=0.1 --theta_max=100000.0 --theta_num=61 --usegpu=True
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
