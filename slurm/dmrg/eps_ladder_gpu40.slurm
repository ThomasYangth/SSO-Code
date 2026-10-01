#!/bin/bash
#SBATCH --job-name=dmrg_ladder40
#SBATCH --constraint=gpu40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
# ε-ladder on a full 40 GB A100. Only the L=12 χ=512 series used this tier in
# the original campaign (job 13325835), for throughput at χ=512. Note the CUDA.jl
# memory pool grows to fill whatever device it gets (9.5 GiB on MIG, 39.5 GiB
# here), so the logged peak usage is not a working-set measurement.
# Configure with --export (see slurm/dmrg/reproduce_plotted_series.sh):
#   SLOWOP_L, SLOWOP_CHI, SLOWOP_REP, SLOWOP_MAXSTEPS, SLOWOP_ANNEAL,
#   SLOWOP_STOP_AT_WALL, SSO_OUTPUT (checkpoints go to $SSO_OUTPUT/dmrg/...)
# Auto-resumes from the last done_step / rolling_step: resubmit the same line.
# Replaces the original ed_gpu512_step9.slurm.
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
LOGDIR=${SLOWOP_LOGDIR:-$REPO/slurmlogs/$(date +%Y-%m-%d)}
mkdir -p "$LOGDIR"; exec > "$LOGDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>&1
export USE_CUDA=1 SLOWOP_THREADS=4
source "$REPO/slurm/dmrg/env.sh"
echo "[cfg] L=$SLOWOP_L CHI=$SLOWOP_CHI REP=$SLOWOP_REP MAXSTEPS=${SLOWOP_MAXSTEPS:-16} ANNEAL=${SLOWOP_ANNEAL:-1} SSO_OUTPUT=${SSO_OUTPUT:-<repo>/output}"
cd "$PROJ"
stdbuf -oL -eL julia --project=. --threads=1 scripts/eps_ladder.jl
rc=$?; [ "$rc" -ne 0 ] && echo "JULIA FAILED rc=$rc"
exit $rc
