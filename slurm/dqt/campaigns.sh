#!/bin/bash
# Production seed families behind fig3panel (a,b): Model B ring J=1, g=-1.05, h=0.5, n=1,
# dt=0.5, nsigma=6, Chebyshev engine, cupy.  167 seed files, ~620 GPU-hours in total.
# s grid of a family = production_svals(SMAX) = 0.7,1.4,2,3,4,6,8,10,12,14,16,18,...,SMAX;
# each family's grid is a prefix of the next, as required by sso.dqt.combine.
#
# Usage (from the repo root):   bash slurm/dqt/campaigns.sh          # print sbatch lines
#                               SUBMIT=1 bash slurm/dqt/campaigns.sh # submit them
#                               ONLY_L=20 SUBMIT=1 bash slurm/dqt/campaigns.sh
# Then:  python scripts/compute/dqt/combine.py --L=16,17,18,19,20,21,22,23
#        python scripts/extract/dqt_fig3panel.py   # TI tables from $SSO_OUTPUT/ti
#
# Columns: L  seeds  SMAX  NHAAR  NPROD  tier  walltime     [original hardware, elapsed/job]
# Tiers: smallest that fits.  MIG holds L<=22 (H, H_k CSR ~2 GB each at L=22; 32 GB host RAM);
# L=23 needs a 40 GB A100.  Families originally run on full A100s are rescheduled on MIG
# here with ~8x their A100 walltime (the MIG slice is ~7x slower for this workload).
# Seeds 0..11 of every L come from an earlier script revision that did not store the raw
# curves (C_samples, m_samples); re-running them with the current code produces a superset.
CAMPAIGNS="
16 0-1     20  12 700 mig  02:00:00   # MIG, 53 min
16 400     54   4 400 mig  03:00:00   # MIG, 1.3 h
17 0-3     20   6 350 mig  02:00:00   # MIG, 55 min
17 400-401 62   4 200 mig  03:00:00   # MIG, 1.6 h
18 0-1     22  12 160 mig  03:00:00   # MIG, 1.3 h
18 300-301 68   4 200 mig  08:00:00   # A100, 45 min
19 0-3     24   6  80 mig  03:00:00   # MIG, 1.5 h
19 200-203 38   3  63 mig  03:00:00   # MIG, 1.8 h
19 300-303 76   4 100 mig  08:00:00   # A100, 44 min
20 0-7     24   3  40 mig  03:00:00   # MIG, 1.6 h
20 200-207 42   3  31 mig  04:00:00   # MIG, 2.2 h
20 300-307 84   4  50 mig  10:00:00   # A100, 55 min
21 0-11    26   3  27 mig  04:00:00   # MIG, 2.6 h
21 100-103 46   3  63 mig  14:00:00   # A100, 1.4 h
21 200-217 88   2  22 mig  10:00:00   # MIG, 7.0 h
22 0-3     26   6  80 mig  20:00:00   # A100 80GB, 2.2 h
22 100-107 50   2  31 mig  16:00:00   # A100, 1.6 h
22 200-219 96   2  10 mig  12:00:00   # MIG, 8.4 h
23 0-7     28   3  40 gpu40 04:00:00  # A100 80GB, 2.5 h
23 100-115 56   2  16 gpu40 04:00:00  # A100, 2.1-2.3 h
23 201,202,213,214,215 104 2 25 gpu40 09:00:00   # A100 40/80GB, 5.7-6.5 h
23 300-322 104  1  12 gpu40 09:00:00  # A100 40/80GB, 2.7-5.7 h
"

expand() {   # "0-3" -> 0 1 2 3 ; "201,202" -> 201 202
    local part
    for part in ${1//,/ }; do
        if [[ $part == *-* ]]; then seq "${part%-*}" "${part#*-}"; else echo "$part"; fi
    done
}

echo "$CAMPAIGNS" | sed 's/#.*//' | while read -r L SEEDS SMAX NHAAR NPROD TIER WALL; do
    [ -z "$L" ] && continue
    [ -n "$ONLY_L" ] && [ "$ONLY_L" != "$L" ] && continue
    case $TIER in
        mig)   TIERFLAGS="--partition=mig" ;;
        gpu40) TIERFLAGS="--constraint=gpu40" ;;
    esac
    for S in $(expand "$SEEDS"); do
        cmd="sbatch --job-name=dqt_L${L}_s${S} ${TIERFLAGS} --time=${WALL} \
--export=ALL,L=${L},SEED=${S},NHAAR=${NHAAR},NPROD=${NPROD},SMAX=${SMAX} slurm/dqt/fourier_cv.slurm"
        if [ "${SUBMIT:-0}" = 1 ]; then eval "$cmd"; else echo "$cmd"; fi
    done
done
