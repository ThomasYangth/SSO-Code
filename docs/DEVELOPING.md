# Developer conventions

These rules keep the sub-codebases consistent. Read them before adding code.

## Layout

```
sso/                    general Python library (import as `sso`)
  backend.py            numpy/cupy selection: xp(usegpu), LinearOperator(usegpu), eigsh(usegpu), to_cpu
  config.py             output_dir(...) [$SSO_OUTPUT, raw results], data_dir(...) [$SSO_DATA, shipped
                        figure data], figure_path(name) [$SSO_FIGURES]
  cli.py                parse_argv / find_value / fprint
  operators.py          Operator term maps, ising(J, hx, hz)
  nutau/                exact nu(tau) frontier machinery (static + Floquet)
  dqt/                  Chebyshev/DQT modulated-hydrodynamics machinery
  plotting/             shared plot style + figure-level helpers
scripts/compute/<area>/ thin CLI drivers that call the library and write to output_dir()
scripts/extract/        build the slim figure data in data/ from raw output
scripts/figures/        one script per figure; ONLY loads data/ and calls sso.*; writes figures/
slurm/                  sbatch launchers reproducing every production run
julia/TIAnsatz/         thermodynamic-limit (translation-invariant ansatz) LOBPCG solver
julia/SlowopDMRG/       DMRG attempt (documented as an unsuccessful approach)
data/<figure>/          slim figure data (small; version-controlled)
tests/                  pytest (Python) — CPU-only, small L, fast
docs/                   method notes, figure provenance
```

## Rules

1. **No absolute paths** in library or scripts. Use `sso.config`.
2. **Figure scripts never compute physics**: they load `data/` and call `sso.*`
   helpers. Anything reusable (loaders, frontier assembly, tau conventions) lives in `sso/`.
3. **Backends**: every numerical routine accepts `usegpu=False`; CPU must work
   (slowly) everywhere so tests run without a GPU.
4. **Imports**: scripts do `import sso...`; run from the repo root or after
   `pip install -e .`. No `sys.path` hacks except a single repo-root insert at the
   top of scripts (`sys.path.insert(0, <repo root>)`) so they work uninstalled.
5. **Docstrings** state the physics: what operator / quantity, conventions
   (e.g. static tau = 1/sqrt(A), Floquet chord tau = 1/(2 arcsin(sqrt(A)/2))).
6. **No dead code**: no commented-out blocks, debug prints, trial flags, or
   branches that no figure or retained feature uses.
7. Raw NPZ key names and filename patterns of the original production runs are
   kept where possible, so outputs can be compared with archived data.
8. Tests: `pytest -q tests/` must pass on CPU in a few minutes.

## Cluster

Production runs used Princeton Della: conda env with Python 3.12, numpy 2.2,
scipy 1.16, quspin 0.3.7, cupy 13.4, matplotlib 3.10; Julia 1.10. GPU runs on
A100 MIG slices unless stated otherwise in the slurm launcher.
