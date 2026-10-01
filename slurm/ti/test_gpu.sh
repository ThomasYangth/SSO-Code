#!/bin/bash
#SBATCH --job-name=tiTestGPU
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
# Full test suite including the GPU kernel/solver tests, on one MIG slice.
#   sbatch slurm/ti/test_gpu.sh      (from the repository root)
source slurm/ti/env.sh
ti_log "tiTestGPU_${SLURM_JOB_ID}"
julia julia/TIAnsatz/test/runtests.jl
