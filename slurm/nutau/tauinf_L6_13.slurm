#!/bin/bash
#SBATCH --job-name=TauInf_L6_13
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=64G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/tauinf_L6_13.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# tau = inf optimum nu_o(inf), o = 1..5, and the H^n baseline for Kim-Huse,
# L = 6..13 (full-space spectrum).  L = 13 needs ~40 GB host RAM.
#   -> Ising_J1.0X0.905Z0.809_TauInf_Data/{tauinf,baseline}_L{L}.csv
# (originals: tools/benchmark_nutau_ds_tauinf.py + tools/compute_baseline_hn_nu.py,
#  slurm/run_tauinf_L12.slurm, run_tauinf_L13.slurm)
python $D/tauinf.py --model=Ising --J=1.0 --hx=0.905 --hz=0.809 --Ls=6,7,8,9,10,11,12,13 --usegpu=True
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
