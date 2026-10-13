#!/bin/bash
#SBATCH --job-name=Filt_KH
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/filter_kimhuse_L12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# Kim-Huse (J=1, hx=0.905, hz=0.809), L = 12: Sum Z_x, Sum X_x (K = 0) and Z_0 (all
# momenta, --K=none), projected off span{I, H, H^2, H^3} (o4), under the Lorentzian,
# Gaussian, hard-cutoff and time-average filters, 60 widths (figure_G (a), the
# Hamiltonian counterpart of figure_B (a)).
#   -> Ising_J1.0X0.905Z0.809_StaticFilter_Data/filt_L12_K{0,none}_{op}_l0.0o4.npz
M="--model=Ising --J=1.0 --hx=0.905 --hz=0.809 --L=12 --pows_max=3 --n_width=60 --filters=avg,hard,gauss,lorentz --usegpu=True"
python $D/filter_static.py $M --K=0 --init_op=Zsum
python $D/filter_static.py $M --K=0 --init_op=Xsum
python $D/filter_static.py $M --K=none --init_op=Z0
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
