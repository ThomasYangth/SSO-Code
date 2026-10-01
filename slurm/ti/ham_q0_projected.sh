#!/bin/bash
#SBATCH --job-name=tiHamExact
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
# q = 0 energy-projected frontier + τ_max ceiling by direct eigensolves (CPU).
#   sbatch slurm/ti/ham_q0_projected.sh [Mmax] [Mmin] [J] [g] [h] [sparse]
# dense: M ≤ 7 in minutes. sparse: M = 8 ~10 min / 20 GB; M = 9 ~5 h / ~300 GB
# (raise --mem accordingly).  Output: $SSO_OUTPUT/ti/nutau_M{M}.dat
source slurm/ti/env.sh
ti_log "tiHamExact_${SLURM_JOB_ID}"
julia -t ${SLURM_CPUS_PER_TASK:-1} julia/TIAnsatz/scripts/ham_q0_projected.jl "$@"
