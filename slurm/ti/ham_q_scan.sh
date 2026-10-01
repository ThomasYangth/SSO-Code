#!/bin/bash
#SBATCH --job-name=tiHamQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null
# Static ν(τ) frontier at momentum q (no energy-density projection), one array task
# per q. Submit from the repository root:
#   sbatch --partition=mig --array=0-<nq-1> slurm/ti/ham_q_scan.sh MSPEC QLIST [TOL] [THETA_MAXLOG] [NTHETA] [g] [h] [SIG]
# MSPEC "11" or "8-10" (the task loops over M); QLIST comma-separated.
# (No partition is fixed here: the GPU tier is chosen on the command line.)
# Output: $SSO_OUTPUT/ti/nutau_M{M}_q{q}.dat.  See slurm/ti/reproduce_fig3c.sh.
MSPEC=$1
QLIST=$2
TOL=${3:-1e-3}
TMAXLOG=${4:-3.0}
NTHETA=${5:-20}
GFIELD=${6:-0.905}
HFIELD=${7:-0.809}
SIG=${8:-}
IFS=',' read -r -a QS <<< "$QLIST"
Q=${QS[${SLURM_ARRAY_TASK_ID:-0}]}

source slurm/ti/env.sh
ti_log "tiHamQ_M${MSPEC}_q${Q}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
julia -t ${SLURM_CPUS_PER_TASK:-1} julia/TIAnsatz/scripts/ham_q_scan.jl \
    "$MSPEC" "$Q" "$TOL" "$TMAXLOG" "$NTHETA" "$GFIELD" "$HFIELD" "$SIG"
