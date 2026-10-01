#!/bin/bash
#SBATCH --job-name=NuTauHx09_L12
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/hx09_frontier_L12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# hx = 0.9 hz scan, L = 12, K = 0 (o4), theta in [0.1, 1e4] (51 pts),
# eigenvectors not stored (save_v2=False: ~5 KB instead of 512 MiB per file).
#   -> Ising_J1.0X0.9Z{hz}_DSsoftKsec_Data/   (~1 h per hz on MIG)
# (original: slurm/run_nutau_ksec_hz_scan_L12_hx0.9.slurm)
for HZ in 0.0 0.005 0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2; do
    python $D/frontier.py --model=Ising --J=1.0 --hx=0.9 --hz=$HZ --Lmin=12 --Lmax=12 \
        --K=0 --pows_max=3 --k_save=2 --lam=0.0 \
        --theta_min=0.1 --theta_max=10000.0 --theta_num=51 --save_v2=False --usegpu=True \
        || echo "WARNING: hz=$HZ failed"
done
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
