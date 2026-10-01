#!/bin/bash
#SBATCH --job-name=tiFloq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null
# Kicked-Ising Floquet ν(τ) frontier (q = 0), one array task per M. Submit from the
# repository root:
#   sbatch --partition=mig --array=0-<nM-1> slurm/ti/floquet.sh MLIST [tauF] [TOL] [g] [h] [SIG]
# MLIST comma-separated spans, e.g. "6,7,8,9,10,11,12". M = 12 needs ~5.1 GB of GPU
# memory (fits a 10 GB MIG slice, ~40 min); M = 13 (~20 GB canvas + ~13 GB work)
# needs a full A100: --constraint=gpu80 instead of --partition=mig.
# (No partition is fixed here: the GPU tier is chosen on the command line.)
# Output: $SSO_OUTPUT/ti/nutau_floquet_M{M}[_SIG]_gpures.dat
MLIST=$1
TAUF=${2:-0.8}
TOL=${3:-1e-3}
GFIELD=${4:-0.905}
HFIELD=${5:-0.809}
SIG=${6:-}
IFS=',' read -r -a MS <<< "$MLIST"
M=${MS[${SLURM_ARRAY_TASK_ID:-0}]}

source slurm/ti/env.sh
ti_log "tiFloq_M${M}_${SIG}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
julia -t ${SLURM_CPUS_PER_TASK:-1} julia/TIAnsatz/scripts/floquet.jl \
    "$M" "$TAUF" "$TOL" "$GFIELD" "$HFIELD" "$SIG"
