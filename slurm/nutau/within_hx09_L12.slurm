#!/bin/bash
#SBATCH --job-name=Within_hx09
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/within_hx09_L12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# Within-K0 crossing arcs of the hx = 0.9 frontiers (hx09_frontier_L12.slurm
# first), bracket = largest tau jump of each stored grid; one process per hz
# so the GPU memory pool starts empty each time.  ~45 min per hz on MIG.
# (original: slurm/run_fill_crossing_hx0.9.slurm, fill_crossing_ksec_novec.py)
for HZ in 0.005 0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2; do
    python $D/crossing_within.py --model=Ising --J=1.0 --hx=0.9 --hz=$HZ --L=12 --K=0 \
        --lam=0.0 --pows_max=3 --k_save=2 --k_ref=8 --usegpu=True || echo "WARNING: hz=$HZ failed"
done
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
