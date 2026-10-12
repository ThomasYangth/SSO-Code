#!/bin/bash
#SBATCH --job-name=TFIM_tauinf
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=40G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/tfim_tauinf_L6_12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# tau = inf optimum nu_o(inf), o = 1..5, and the H^n baseline for the TFIM (hx=2, hz=0),
# L = 6..12 (figure_D (b) inset; capped at L = 12 to stay on MIG).
#   -> Ising_J1.0X2.0Z0.0_TauInf_Data/{tauinf,baseline}_L{L}.csv
python $D/tauinf.py --model=Ising --J=1.0 --hx=2.0 --hz=0.0 --Ls=6,7,8,9,10,11,12 --usegpu=True
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
