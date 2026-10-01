#!/bin/bash
# Exact submission sequence that regenerates every DMRG point in
# data/attempt_dmrg/dmrg_points.csv (see docs/dmrg_attempt.md for the
# series -> original job table and costs). NOT meant to be run blindly
# end-to-end: stages depend on earlier stages' checkpoints and most ladders
# need resubmission past the 24 h limit (auto-resume makes that safe).
# Run from the repo root; set SSO_OUTPUT to scratch storage first, e.g.
#   export SSO_OUTPUT=/scratch/gpfs/<group>/<user>/SSO-Code-output
set -euo pipefail
: "${SSO_OUTPUT:?set SSO_OUTPUT to a scratch directory}"
S=slurm/dmrg
X="--export=ALL,SSO_OUTPUT=$SSO_OUTPUT"
stage=${1:-help}

case $stage in
A)  # L=12 χ=128 plotted curve: one quench (anneal=0) ladder, steps 1-10 (~6.5 h MIG)
    sbatch $X,SLOWOP_L=12,SLOWOP_CHI=128,SLOWOP_REP=gpu128_a0,SLOWOP_MAXSTEPS=10,SLOWOP_ANNEAL=0 $S/eps_ladder_mig.slurm ;;
B)  # L=12 χ=128 seed ladder for the χ=256/512 points (original: CPU, pre-annealing
    # code => anneal=0). Steps 1-8; ~30 h on 16 cores, resubmit until done_step08 exists.
    sbatch $X,SLOWOP_L=12,SLOWOP_CHI=128,SLOWOP_REP=chi128,SLOWOP_MAXSTEPS=8,SLOWOP_ANNEAL=0 $S/eps_ladder_cpu.slurm ;;
C)  # needs B. χ refinements at the step-8 target and the χ=256 step 9.
    sbatch $X,SLOWOP_MODE=refine,SLOWOP_SRCREP=chi128,SLOWOP_STEP=8,SLOWOP_CHI=256,SLOWOP_SWEEPS=4,SLOWOP_TAG=refine_step8_chi256 $S/one_step_mig.slurm
    sbatch $X,SLOWOP_MODE=refine,SLOWOP_SRCREP=chi128,SLOWOP_STEP=8,SLOWOP_CHI=512,SLOWOP_SWEEPS=6,SLOWOP_TAG=refine_step8_chi512 $S/one_step_gpu40.slurm
    sbatch $X,SLOWOP_MODE=next,SLOWOP_SRCREP=chi128,SLOWOP_STEP=8,SLOWOP_CHI=256,SLOWOP_SWEEPS=6,SLOWOP_ANNEAL=1,SLOWOP_TAG=step9_anneal1_chi256 $S/one_step_mig.slurm ;;
D)  # needs B. L=12 χ=512 ladder step 9 from the χ=128 step-8 state (~9 h A100).
    bash $S/seed_continuation.sh 12 chi128 8 gpu512
    sbatch $X,SLOWOP_L=12,SLOWOP_CHI=512,SLOWOP_REP=gpu512,SLOWOP_MAXSTEPS=9,SLOWOP_ANNEAL=1 $S/eps_ladder_gpu40.slurm ;;
E)  # L=18, 24, 30 χ=128 ladders (MIG; ~16 h / ~31 h / ~43 h: resubmit until walled)
    for L in 18 24 30; do
        sbatch $X,SLOWOP_L=$L,SLOWOP_CHI=128,SLOWOP_REP=bigL$L,SLOWOP_MAXSTEPS=16,SLOWOP_ANNEAL=1 $S/eps_ladder_mig.slurm
    done ;;
F)  # needs E. χ=256 continuations from χ=128 step 8 / 7 / 5 (L=18 / 24 / 30).
    # Each step costs 7-28 h at L=24-30 on MIG; use -t 3-00:00:00 and resubmit.
    bash $S/seed_continuation.sh 18 bigL18 8 bigL18_chi256
    bash $S/seed_continuation.sh 24 bigL24 7 bigL24_chi256
    bash $S/seed_continuation.sh 30 bigL30 5 bigL30_chi256
    for L in 18 24 30; do
        sbatch -t 3-00:00:00 $X,SLOWOP_L=$L,SLOWOP_CHI=256,SLOWOP_REP=bigL${L}_chi256,SLOWOP_MAXSTEPS=16,SLOWOP_ANNEAL=1 $S/eps_ladder_mig.slurm
    done ;;
*)  sed -n 2,12p "$0"; echo "usage: bash $0 {A|B|C|D|E|F}"; exit 1 ;;
esac
