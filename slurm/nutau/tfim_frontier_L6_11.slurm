#!/bin/bash
#SBATCH --job-name=TFIM_front
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch; LS = sizes:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch LS="6 7 8 9" sbatch --export=ALL slurm/nutau/tfim_frontier_L6_11.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
: "${LS:?export LS (space-separated sizes) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# TFIM (J=1, hx=2.0, hz=0.0) sector frontier for figure_D (b), the analogue of
# kimhuse_frontier_L6_12.slurm: K = 0 (o4) and K = 1 (o0), theta in [0.1, 1e4] (51 pts).
# L = 12 is the archived tfim_frontier_L12.slurm output.
#   -> Ising_J1.0X2.0Z0.0_DSsoftKsec_Data/L{L}_K{0,1}_theta*l0.0k2o{4,0}.npz
for L in $LS; do
    python $D/frontier.py --model=Ising --J=1.0 --hx=2.0 --hz=0.0 --Lmin=$L --Lmax=$L --K=0,1 \
        --pows_max=3 --k_save=2 --lam=0.0 --theta_min=0.1 --theta_max=10000.0 --theta_num=51 \
        --usegpu=True || echo "WARNING: L=$L failed"
done
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
