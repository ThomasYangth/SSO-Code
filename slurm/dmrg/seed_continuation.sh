#!/bin/bash
# Seed a higher-χ ε-ladder continuation: copy a completed χ=128 ladder state
# done_step<NN>.h5 into a fresh checkpoint directory, so that eps_ladder.jl
# (auto-resume) continues the ladder from step NN+1 at the new bond dimension.
# This is how the original χ=256 / χ=512 continuations were started (the copied
# file keeps its χ=128 'variant'; the extraction drops it as a seed).
#
#   bash slurm/dmrg/seed_continuation.sh L SRC_REP STEP DST_REP
#   e.g. bash slurm/dmrg/seed_continuation.sh 18 bigL18 8 bigL18_chi256
# Data root: ${SLOWOP_DATA_DIR:-${SSO_OUTPUT:-<repo>/output}/dmrg} (login node is fine: a file copy).
set -euo pipefail
[ $# -eq 4 ] || { sed -n 2,12p "$0"; exit 1; }
L=$1; SRC=$2; STEP=$(printf %02d "$3"); DST=$4
REPO=$(cd "$(dirname "$0")/../.." && pwd)
D=${SLOWOP_DATA_DIR:-${SSO_OUTPUT:-$REPO/output}/dmrg}/hk_scan_cmp
src="$D/L${L}_lobpcg_sinmc_rep${SRC}/done_step${STEP}.h5"
dst="$D/L${L}_lobpcg_sinmc_rep${DST}"
[ -f "$src" ] || { echo "missing $src"; exit 1; }
mkdir -p "$dst"
[ -e "$dst/done_step${STEP}.h5" ] && { echo "$dst/done_step${STEP}.h5 exists, not overwriting"; exit 1; }
cp "$src" "$dst/"
echo "seeded $dst from $src"
