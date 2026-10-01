# TIAnsatz.jl — thermodynamic-limit ν(τ) frontier (translation-invariant ansatz)

Matrix-free CPU/GPU solver for the slow-operator frontier of an infinite spin
chain, restricted to translation-invariant (momentum-q) operators supported on M
consecutive sites. Method, provenance and caveats: [`docs/tiansatz.md`](../../docs/tiansatz.md).

```
julia/TIAnsatz/
  Project.toml, Manifest.toml   pinned environment (Julia 1.10.2, CUDA.jl 6.3.1)
  LocalPreferences.toml         CUDA runtime 12.9 from the JLL artifact
  ThermodynamicPauli/           local package: Pauli algebra in the thermodynamic limit
  src/                          TIAnsatz module
    basis.jl     canonical basis (3·4^(M-1) stripped strings), R = diag(3^-wt), H, c_H
    kernels.jl   K (dense/sparse), L1 / L1_q, Floquet K_F (assembled + matrix-free CPU)
    exact.jl     direct q = 0 energy-projected solvers (dense / sparse Cholesky)
    lobpcg.jl    LOBPCG, Chebyshev-in-B preconditioner, Lanczos error bar (CPU or GPU arrays)
    gpu.jl       CUDA kernels: KHGpu, KHGpuC (cuSPARSE), DenseKFGpu (gate kernels)
    sweep.jl     θ sweep, .dat writer, backend/seed/output-dir helpers
  scripts/
    ham_q_scan.jl            static frontier at momentum q (no projection)   → nutau_M{M}_q{q}.dat
    floquet.jl               kicked-Ising Floquet frontier                    → nutau_floquet_M{M}[_SIG]_gpures.dat
    ham_q0_projected_gpu.jl  q = 0, energy density projected, matrix-free     → nutau_M{M}_gpures[_SIG].dat
    ham_q0_projected.jl      q = 0, energy density projected, dense/sparse    → nutau_M{M}.dat
  test/runtests.jl           CPU tests (GPU tests auto-skipped without CUDA)
```

## Install

On Della (as in production; other machines: any Julia 1.10, CUDA optional):

```bash
module load anaconda3/2024.2 julia/1.10.2 && conda activate julia && unset LD_LIBRARY_PATH
# from the repository root, on a compute node (first precompile of CUDA.jl takes minutes):
julia --project=julia/TIAnsatz -e 'using Pkg; Pkg.instantiate()'
```

`ThermodynamicPauli` is a path dependency (`julia/TIAnsatz/ThermodynamicPauli`)
recorded in the Manifest, so `instantiate` needs no extra step. CUDA.jl is always
loaded; on a machine without a GPU `CUDA.functional()` is false and every driver
falls back to the host path.

## Run

From the repository root (`slurm/ti/env.sh` performs the module/conda activation
and sets `JULIA_PROJECT`):

```bash
julia --project=julia/TIAnsatz julia/TIAnsatz/scripts/ham_q_scan.jl 8 0.002,0.004 1e-3 3.0 20
julia --project=julia/TIAnsatz julia/TIAnsatz/scripts/floquet.jl 8 1.0 1e-3 0.9 0.809 g0p9tF1p0
julia --project=julia/TIAnsatz julia/TIAnsatz/scripts/ham_q0_projected_gpu.jl 10 1e-3 3.0 20
julia --project=julia/TIAnsatz julia/TIAnsatz/scripts/ham_q0_projected.jl 7 1          # dense
julia --project=julia/TIAnsatz julia/TIAnsatz/test/runtests.jl
```

Positional arguments (all optional after the first, defaults in brackets):

| script | arguments |
|---|---|
| `ham_q_scan.jl` | `M` ("8" or "8-11"), `q[,q…]`, `TOL` [1e-3], `THETA_MAXLOG` [3.0], `NTHETA` [20], `g` [0.905], `h` [0.809], `SIG` [""] |
| `floquet.jl` | `M`, `tauF` [0.8], `TOL` [1e-3], `g` [0.905], `h` [0.809], `SIG` [""] (θ grid fixed: 24 points, 10^-1.5..10^3) |
| `ham_q0_projected_gpu.jl` | `M`, `TOL` [1e-3], `THETA_MAXLOG` [3.0], `NTHETA` [20], `g`, `h`, `SIG` |
| `ham_q0_projected.jl` | `Mmax` [4], `Mmin` [1], `J` [1], `g`, `h`, literal `sparse` to use the sparse solver |

θ grids are `10 .^ range(-1.5, THETA_MAXLOG; length=NTHETA)`.

Environment variables:

| variable | effect |
|---|---|
| `SSO_OUTPUT` | output root; files go to `$SSO_OUTPUT/ti/` (default `<repo>/output/ti/`) |
| `SSO_TI_BACKEND` | `auto` (GPU iff CUDA functional), `gpu` (error without GPU), `cpu` |
| `SSO_SEED` | integer seed for the random start vectors (host + CUDA RNG); unset = unseeded, as in production |

Slurm launchers reproducing the production runs: `slurm/ti/` (see
`reproduce_fig3c.sh`, `reproduce_floquet.sh`).

## Output format

One whitespace table per (M, q) or Floquet run, read by `sso.tiansatz.read_ti_table`:

```
# model=Ising J=1 g=0.905 h=0.809 M=8 dim=49152 q=0.002 solver=mf-L1qL1q-gpures
# theta  nu  sigma2  tau  nu_bar  accepted
0.0316227766  0.3292229222  1.244143006  0.8965300455  7.731e-07  1
...
```

`theta` Lagrange multiplier; `nu = c'Rc/c'c` RPS weight; `sigma2 = c'Kc/c'c`
(static: per-site ‖[H,A]‖²/‖A‖²; Floquet: per-site ‖U_F A U_F† − A‖²/‖A‖²);
`tau = 1/√sigma2`; `nu_bar` the weak-duality error bar on ν; `accepted` 1 if the
Lanczos gate certifies the top eigenpair. Rows: `%.10g` (ν_bar `%.4g`). The
`solver=` tag ends in `-gpures` for the GPU backend and `-cpu` for the host path.
`ham_q0_projected.jl` writes the older 4-column table `theta nu sigma2 tau` with a
final `Inf` row (ν_end, λ_TI, τ_max) and `lambda_TI`, `tau_max` in the header.

## Tests

`test/runtests.jl` (≈1 min on CPU after load): the ThermodynamicPauli suite;
basis/R invariants; static (q = 0 and q ≠ 0) and Floquet kernels against an
independent finite-ring exact-diagonalization oracle; `K = L1†L1`, `L1_q(0) = L1`,
matrix-free applies vs assembled matrices; sparse vs dense direct solvers; the
LOBPCG frontier sweep (q ≠ 0, q = 0 unprojected and projected, Floquet) vs dense
generalized eigensolves at M = 5; with a GPU also the CUDA kernels vs the host
kernels and a GPU sweep.
