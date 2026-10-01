#!/bin/bash
#SBATCH --job-name=dmrg_ladder_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
# ε-ladder on CPU (16 BLAS threads). Used only for the L=12 χ=128 "repchi128"
# ladder (original ed_L12_single.slurm, jobs 13102819/13177208/13199108), whose
# step-8 state seeds the L=12 χ=256 and χ=512 points. Resubmit to resume.
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
LOGDIR=${SLOWOP_LOGDIR:-$REPO/slurmlogs/$(date +%Y-%m-%d)}
mkdir -p "$LOGDIR"; exec > "$LOGDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>&1
export SLOWOP_THREADS=${SLURM_CPUS_PER_TASK}
source "$REPO/slurm/dmrg/env.sh"
echo "[cfg] L=$SLOWOP_L CHI=$SLOWOP_CHI REP=$SLOWOP_REP MAXSTEPS=${SLOWOP_MAXSTEPS:-16} ANNEAL=${SLOWOP_ANNEAL:-1}"
cd "$PROJ"
stdbuf -oL -eL julia --project=. --threads=1 scripts/eps_ladder.jl
rc=$?; [ "$rc" -ne 0 ] && echo "JULIA FAILED rc=$rc"
exit $rc
