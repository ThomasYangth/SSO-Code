#!/bin/bash
#SBATCH --job-name=slowop_tests
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
# SlowopDMRG CPU test suite: julia/SlowopDMRG/test/runtests.jl, then the
# rolling-resume regression (test/rolling_resume_smoke.sh).
#   sbatch slurm/dmrg/run_tests_cpu.slurm        (from the repo root)
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
LOGDIR=${SLOWOP_LOGDIR:-$REPO/slurmlogs/$(date +%Y-%m-%d)}
mkdir -p "$LOGDIR"; exec > "$LOGDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>&1
export SLOWOP_THREADS=${SLURM_CPUS_PER_TASK:-4}
source "$REPO/slurm/dmrg/env.sh"
cd "$PROJ"
julia --project=. test/runtests.jl; rc1=$?
bash test/rolling_resume_smoke.sh; rc2=$?
echo "runtests rc=$rc1  rolling_resume rc=$rc2"
exit $(( rc1 || rc2 ))
