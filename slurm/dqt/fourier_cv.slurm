#!/bin/bash
# One seed of the Fourier-correlator DQT estimator (Model B ring, n=1) on one GPU.
# Parameters via --export: L, SEED, NHAAR, NPROD, SMAX [, ENGINE=chebyshev, DT=0.5, NSIGMA=6].
# Output: $SSO_OUTPUT/dqt/modhydro_fourier_cv_L{L}_n1_g-1.05_h+0.5_seed{SEED}.npz
# GPU tier / walltime are chosen per campaign on the sbatch line (see campaigns.sh).
# Submit from the repository root.
#SBATCH --job-name=dqt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

module purge
module load anaconda3/2024.2
conda activate cuspin
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python scripts/compute/dqt/fourier_cv.py --gpu --L="${L}" --seed="${SEED}" \
    --nhaar="${NHAAR}" --nprod="${NPROD}" --smax="${SMAX}" \
    --engine="${ENGINE:-chebyshev}" --dt="${DT:-0.5}" --nsigma="${NSIGMA:-6}"
