#!/bin/bash
#SBATCH --job-name=tiHamP
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
# q = 0 static frontier with the energy density projected out (matrix-free LOBPCG).
#   sbatch --partition=mig slurm/ti/ham_q0_projected_gpu.sh MSPEC [TOL] [THETA_MAXLOG] [NTHETA] [g] [h] [SIG]
# MIG fits M ≤ 11; M = 12, 13 need a full A100 (L1 for M = 13 ≈ 11.7 GB):
#   sbatch --constraint=gpu40 --cpus-per-task=8 --mem=96G slurm/ti/ham_q0_projected_gpu.sh 13
# Output: $SSO_OUTPUT/ti/nutau_M{M}_gpures[_SIG].dat
source slurm/ti/env.sh
ti_log "tiHamP_M$1_${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
julia -t ${SLURM_CPUS_PER_TASK:-1} julia/TIAnsatz/scripts/ham_q0_projected_gpu.jl "$@"
