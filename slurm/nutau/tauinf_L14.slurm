#!/bin/bash
#SBATCH --job-name=TauInf_L14
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu40
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=128G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/tauinf_L14.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# L = 14 tau = inf scan.  Needs a full 40 GB A100: n^2 = 2.7e8, so every E-basis
# buffer and Pauli-basis vector is 4.3 GB (V, two buffers, the scatter vector
# and the Pauli transform temporaries).  (original: slurm/run_tauinf_L14.slurm)
python $D/tauinf.py --model=Ising --J=1.0 --hx=0.905 --hz=0.809 --Ls=14 --usegpu=True
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
