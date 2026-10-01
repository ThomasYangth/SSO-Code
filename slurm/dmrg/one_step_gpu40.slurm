#!/bin/bash
#SBATCH --job-name=dmrg_onestep40
#SBATCH --constraint=gpu40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
# One ε step from a saved MPS (julia/SlowopDMRG/scripts/one_step.jl) on a full 40 GB A100
# (only the L=12 χ=512 step-8 refinement used this tier).
# Configure with --export: SLOWOP_MODE (refine|next), SLOWOP_SRCREP+SLOWOP_STEP
# or SLOWOP_SRC, SLOWOP_CHI, SLOWOP_SWEEPS, SLOWOP_ANNEAL, SLOWOP_TAG, SSO_OUTPUT.
# Replaces the original ed_step8_chi512_gpu40.slurm.
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
LOGDIR=${SLOWOP_LOGDIR:-$REPO/slurmlogs/$(date +%Y-%m-%d)}
mkdir -p "$LOGDIR"; exec > "$LOGDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>&1
export USE_CUDA=1 SLOWOP_THREADS=4
source "$REPO/slurm/dmrg/env.sh"
cd "$PROJ"
stdbuf -oL -eL julia --project=. --threads=1 scripts/one_step.jl
rc=$?; [ "$rc" -ne 0 ] && echo "JULIA FAILED rc=$rc"
exit $rc
