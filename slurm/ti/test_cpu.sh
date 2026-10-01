#!/bin/bash
#SBATCH --job-name=tiTest
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
# CPU test suite of julia/TIAnsatz (GPU tests are skipped without CUDA).
#   sbatch slurm/ti/test_cpu.sh      (from the repository root)
source slurm/ti/env.sh
ti_log "tiTest_${SLURM_JOB_ID}"
julia -t ${SLURM_CPUS_PER_TASK:-1} julia/TIAnsatz/test/runtests.jl
