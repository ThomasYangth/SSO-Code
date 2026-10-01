#!/bin/bash
# Resubmit the production runs behind floquet_two_panel panel (b), dashed "TI ansatz"
# curves: kicked Ising τF = 1, J = 1, g = 0.9, h = 0.809, M = 6..12, 24 θ to 10^3,
# TOL = 1e-3, MIG 1g.10gb [orig. array 13659320 (tasks 13659330–35 + 13659320), 2026-09-09;
# M = 12 took 2421 s of sweep].
# Run from the repository root:  bash slurm/ti/reproduce_floquet.sh
sbatch --partition=mig --array=0-6 --mem=24G --time=01:30:00 slurm/ti/floquet.sh \
    6,7,8,9,10,11,12 1.0 1e-3 0.9 0.809 g0p9tF1p0
