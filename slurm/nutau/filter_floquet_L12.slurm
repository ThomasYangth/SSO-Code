#!/bin/bash
#SBATCH --job-name=Filt_Floquet
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/filter_floquet_L12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# Kicked Ising, L = 12: filters of Sum Z_x, Sum X_x (K = 0) and Z_0 (all momenta).
#   gauss (50 widths), hard (45; 50 for Sum X_x), N-period average (Knone,
#   N <= 1e6, 70 samples).  ~1-2 min per run on MIG.
#   -> KickedIsing_g0.9J1.0h0.809_{FloquetHardCut,FloquetTrajectory}_Data/
# (originals: slurm/run_nutau_hardcut_floquet.slurm with FILTER=hard|gauss,
#  slurm/run_nutau_floquet_trajectory*.slurm)
M="--model=KickedIsing --g=0.9 --J=1.0 --h=0.809 --L=12 --usegpu=True"
python $D/filter_floquet.py $M --K=0 --init_op=Zsum --filters=gauss --n_width=50
python $D/filter_floquet.py $M --K=0 --init_op=Xsum --filters=gauss,hard --n_width=50
python $D/filter_floquet.py $M --K=none --init_op=Z0 --filters=gauss --n_width=50
python $D/filter_floquet.py $M --K=0 --init_op=Zsum --filters=hard --n_width=45
python $D/filter_floquet.py $M --K=none --init_op=Z0 --filters=hard --n_width=45
for OP in Zsum Z0 Xsum; do
    python $D/filter_floquet.py $M --K=none --init_op=$OP --filters=avg --N_max=1e6 --n_sample=70
done
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
