#!/bin/bash
# Regression test for partial-step (rolling) resume in scripts/eps_ladder.jl.
#
# Scenario: a resubmit is interrupted DURING a step, leaving that step's per-sweep
# rolling_step<NN>.h5 but no done_step<NN>.h5. The resume must warm-start step NN
# from the rolling checkpoint, not silently redo it from step NN-1.
#
# With SLOWOP_ROLLING_RESUME=1 the resume must announce the partial warm-start;
# with =0 it must not (negative control = the behaviour before the fix).
# Compute node only, e.g. via slurm/dmrg/run_tests_cpu.slurm, or
#   srun -p cpu -c 4 --mem-per-cpu=4G -t 02:00:00 bash julia/SlowopDMRG/test/rolling_resume_smoke.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../../../slurm/dmrg/env.sh"
cd "$PROJ"

# Isolated data dir (never the production one); logs next to it.
export SLOWOP_DATA_DIR="${SLOWOP_TEST_DATA_DIR:-${SSO_OUTPUT:-$REPO/output}/dmrg_test}/rolling_resume"
REP="roltest"
ENVc="SLOWOP_L=7 SLOWOP_CHI=32 SLOWOP_REP=$REP SLOWOP_HX=-1.05 SLOWOP_HZ=0.5 SLOWOP_J=1.0"
DDIR="$SLOWOP_DATA_DIR/hk_scan_cmp/L7_lobpcg_sinmc_rep${REP}"
LOGS="$SLOWOP_DATA_DIR/logs"
echo "checkpoint dir: $DDIR"
rm -rf "$DDIR"; mkdir -p "$LOGS"

fail() { echo "FAIL: $1"; exit 1; }
run() { env $ENVc "$@" julia --project=. scripts/eps_ladder.jl; }

echo "=== 1. fresh run, 3 steps ==="
run SLOWOP_MAXSTEPS=3 SLOWOP_FRESH=1 > "$LOGS/fresh.log" 2>&1 \
  || { tail -20 "$LOGS/fresh.log"; fail "fresh run errored"; }
ls "$DDIR"
[ -f "$DDIR/done_step03.h5" ]    || fail "setup: fresh run did not complete step 3 (walled? check log)"
[ -f "$DDIR/rolling_step03.h5" ] || fail "setup: no rolling_step03 to resume from"

echo "=== 2. simulate interruption of step 3: drop done_step03, keep rolling_step03 ==="
rm -f "$DDIR/done_step03.h5"

echo "=== 3. resume WITH rolling resume ==="
run SLOWOP_MAXSTEPS=3 SLOWOP_ROLLING_RESUME=1 > "$LOGS/fix.log" 2>&1 \
  || { tail -20 "$LOGS/fix.log"; fail "resume(fix) errored"; }
grep -qE "partial rolling checkpoint for step 3 found.*warm-starting step 3" "$LOGS/fix.log" \
  || { grep -E "\[resume\]" "$LOGS/fix.log"; fail "resume did NOT warm-start from rolling checkpoint"; }
[ -f "$DDIR/done_step03.h5" ] || fail "resume did not re-complete step 3"
echo "PASS: warm-started step 3 from its partial rolling checkpoint and completed it"

echo "=== 4. negative control: rolling resume OFF ==="
rm -f "$DDIR/done_step03.h5"
run SLOWOP_MAXSTEPS=3 SLOWOP_ROLLING_RESUME=0 > "$LOGS/off.log" 2>&1 \
  || { tail -20 "$LOGS/off.log"; fail "resume(off) errored"; }
if grep -qE "partial rolling checkpoint for step 3 found" "$LOGS/off.log"; then
  fail "control: rolling resume fired even with SLOWOP_ROLLING_RESUME=0"
fi
grep -qE "found completed step 2" "$LOGS/off.log" || fail "control: expected fallback to done_step02"
echo "PASS: with the feature off, resume falls back to redoing step 3 from step 2"

rm -rf "$DDIR"
echo "ALL PASS"
