#!/bin/bash
# Resubmit the production runs behind fig3panel panel (c): Kim–Huse Ising
# (J, g, h) = (1, 0.905, 0.809), static, momentum-q frontiers for M = 8..11.
# Run from the repository root:  bash slurm/ti/reproduce_fig3c.sh
# Each family maps onto one (or two) original Della job(s); all on MIG 1g.10gb, TOL = 1e-3.
set -e
QSMALL=0.002,0.004,0.006,0.008
QLARGE=0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10
QALL=$QSMALL,$QLARGE

# (1) M = 8–10, small q, 20 θ to 10^3                      [orig. 13437392; ~10 min/task]
sbatch --partition=mig --array=0-3 --mem=32G --time=01:30:00 slurm/ti/ham_q_scan.sh 8-10 $QSMALL 1e-3 3.0 20
# (2) M = 11, all 14 q, 30 θ to 10^4                        [orig. 13443841; ~13 min/task]
sbatch --partition=mig --array=0-13 --mem=32G --time=01:30:00 slurm/ti/ham_q_scan.sh 11 $QALL 1e-3 4 30
# (3) M = 8, 9, 10, q = 0.01–0.10, 30 θ to 10^4            [orig. 13620982/3/4; 1–5 min/task]
for M in 8 9 10; do
    sbatch --partition=mig --array=0-9 --mem=24G --time=01:30:00 slurm/ti/ham_q_scan.sh $M $QLARGE 1e-3 4.0 30
done
# (4) q = 0 baselines (unprojected), 80 θ to 10^4           [orig. interactive srun 13444112, 13443806]
sbatch --partition=mig --mem=32G --time=01:30:00 slurm/ti/ham_q_scan.sh 8-10 0.0 1e-3 4 80
sbatch --partition=mig --mem=32G --time=01:30:00 slurm/ti/ham_q_scan.sh 11 0.0 1e-3 4 80
