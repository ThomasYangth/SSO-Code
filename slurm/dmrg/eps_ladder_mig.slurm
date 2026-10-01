#!/bin/bash
#SBATCH --job-name=dmrg_ladder
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
# ε-ladder (julia/SlowopDMRG/scripts/eps_ladder.jl) on one A100 MIG slice.
# Configure with --export (see slurm/dmrg/reproduce_plotted_series.sh):
#   SLOWOP_L, SLOWOP_CHI, SLOWOP_REP, SLOWOP_MAXSTEPS, SLOWOP_ANNEAL,
#   SLOWOP_STOP_AT_WALL, SSO_OUTPUT (checkpoints go to $SSO_OUTPUT/dmrg/...)
# Auto-resumes from the last done_step / rolling_step: resubmit the same line.
# Replaces the original ed_gpu128_anneal.slurm, ed_bigL_chi128.slurm,
# ed_L18_chi256.slurm and ed_cont_chi256.slurm (same settings, MIG, -c 1).
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
LOGDIR=${SLOWOP_LOGDIR:-$REPO/slurmlogs/$(date +%Y-%m-%d)}
mkdir -p "$LOGDIR"; exec > "$LOGDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>&1
export USE_CUDA=1
source "$REPO/slurm/dmrg/env.sh"
echo "[cfg] L=$SLOWOP_L CHI=$SLOWOP_CHI REP=$SLOWOP_REP MAXSTEPS=${SLOWOP_MAXSTEPS:-16} ANNEAL=${SLOWOP_ANNEAL:-1} SSO_OUTPUT=${SSO_OUTPUT:-<repo>/output}"
cd "$PROJ"
stdbuf -oL -eL julia --project=. --threads=1 scripts/eps_ladder.jl
rc=$?; [ "$rc" -ne 0 ] && echo "JULIA FAILED rc=$rc"
exit $rc
