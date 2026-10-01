# SlowopDMRG.jl — ε-constrained DMRG for simple slow operators (unsuccessful attempt)

Two-site DMRG over operator-space MPS (Pauli-string basis) that walks the
constraint ⟨C_H²⟩ ≤ ε downward and maximizes the simplicity ν = ⟨M⟩, with
O ⊥ {I, H, H², H³}. **This approach was abandoned**: it reproduces the exact
ν(τ) frontier only up to τ ≈ 14 at L = 12 and hits a bond-dimension wall at
τ ≈ 1.2–1.6 L. Physics, algorithm, results, provenance of every plotted point,
and costs: [`docs/dmrg_attempt.md`](../../docs/dmrg_attempt.md).

## Layout

```
Project.toml / Manifest.toml   pinned environment (Julia 1.10.2; ITensors 0.7.13,
                               ITensorMPS 0.3.6, NDTensors 0.3.74, KrylovKit 0.8.3,
                               IterativeSolvers 0.9.4, HDF5 0.17.2, ...)
LocalPreferences.toml          CUDA_Runtime_jll = 12.9 (GPU runs)
src/SlowopDMRG.jl              module, runtime configuration, device helpers
src/debug_logger.jl            optional per-block debug logs (SLOWOP_DEBUG=1)
src/core/dmrg_sweeper.jl       dmrg_sweeps!: two-site sweeps, factored C†C envs
src/core/local_qcqp.jl         ε-constrained block solve: θ² secant/bisection,
                               :lobpcg (Chebyshev-in-B), :geneig, :eigsolve inner solvers
src/core/local_geneig.jl       fixed-θ generalized eigen primitive, apply_M_local
src/core/persistence.jl        HDF5 MPS files from metadata, tracking CSVs
src/models/                    Pauli site type, C_H / Floquet superoperator MPOs,
                               Krylov tower, run_dmrg_for_size, observables,
                               exact_solver.jl (L ≤ 6 reference)
ext/SlowopDMRGCUDAExt.jl       GPU backend (loaded when `using CUDA` comes first)
scripts/eps_ladder.jl          production ε ladder (orig. ed_compare_L12_lobpcg.jl)
scripts/one_step.jl            one χ-refine / annealed step from a saved MPS
                               (orig. ed_chi_refine.jl, ed_step9_anneal.jl, ed_onestep_anneal.jl)
test/runtests.jl               CPU test suite; test/rolling_resume_smoke.sh
```

## Install

On Della the packages are already in the shared depot; the launchers in
`slurm/dmrg/` source `slurm/dmrg/env.sh`, which loads `julia/1.10.2`, stacks a
private overlay depot on top of `/scratch/gpfs/DABANIN/ty1475/julia_depot/main`
and sets `JULIA_PKG_OFFLINE=true`. Elsewhere:

```bash
cd julia/SlowopDMRG
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**GPU.** CUDA.jl is a weak dependency (`[weakdeps]`, compat `~5.8.3`) and is not in
the project Manifest. Load it *before* SlowopDMRG (`USE_CUDA=1` makes the scripts do
`using CUDA`). On Della it resolves through the stacked global environment `@v1.10`
of the shared depot, which provides CUDA.jl 5.8.3 — exactly how the production GPU
runs loaded it. Elsewhere add CUDA 5.8.x to your global environment
(`julia -e 'using Pkg; Pkg.add(name="CUDA", version="5.8.3")'`). With the extension
active only the tensor contractions run on the device; LOBPCG/Chebyshev/bisection
bookkeeping stays on the host.

## Configuration (environment variables)

| variable | default | |
|---|---|---|
| `SLOWOP_DATA_DIR` | `$SSO_OUTPUT/dmrg`, else `<repo>/output/dmrg` | checkpoints (`hk_scan_cmp/...`), MPS files, tracking CSVs |
| `SLOWOP_USE_GPU` | `1` | use the CUDA backend if loaded |
| `SLOWOP_DEBUG` / `SLOWOP_DEBUG_LOG_DIR` | `0` / `<data dir>/debug_logs` | per-block debug logs |

Script parameters (`SLOWOP_L`, `SLOWOP_CHI`, `SLOWOP_REP`, `SLOWOP_MAXSTEPS`,
`SLOWOP_ANNEAL`, ...) are documented in the header of each script.

## Run

Never on the login node. Examples (from the repo root):

```bash
# CPU tests (~30 min)
sbatch slurm/dmrg/run_tests_cpu.slurm
# L=12 χ=128 ladder, as the plotted curve (MIG, ~6.5 h)
sbatch --export=ALL,SSO_OUTPUT=$SCRATCH_OUT,SLOWOP_L=12,SLOWOP_CHI=128,SLOWOP_REP=gpu128_a0,SLOWOP_MAXSTEPS=10,SLOWOP_ANNEAL=0 \
       slurm/dmrg/eps_ladder_mig.slurm
# every plotted series, stage by stage
bash slurm/dmrg/reproduce_plotted_series.sh {A|B|C|D|E|F}
```

Interactively:

```julia
using SlowopDMRG
H = Dict("ZZ" => 1.0, "X" => 0.905, "Z" => 0.809)
psi = run_dmrg_for_size(8, H; scheduling = :eps_constrained, local_solver = :lobpcg,
                        epsilon_loc = 0.05, num_ortho = 3, maxdim_cap = 64,
                        sweeps_per_iteration = 6, max_iter = 1, save_mps = false)
```

(`L ≤ 6` is routed to the exact solver.)

## Validation

See `docs/dmrg_attempt.md` §7. Short reproduction of the plotted L = 12 χ = 128
ladder (anneal=0, MIG), first three ε steps, against the archived log of job
13325137_0:

| step | ε target | ⟨C_H²⟩ archived | ⟨C_H²⟩ port | rel. diff | ⟨M⟩ archived | ⟨M⟩ port | rel. diff | wall archived / port |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.230261 | 0.229322 | 0.229326 | 1.7e-5 | 0.304205 | 0.304205 | < 3e-6 | 461 s / 396 s |
| 2 | 0.132941 | 0.132458 | 0.132398 | 4.5e-4 | 0.295025 | 0.295018 | 2.4e-5 | 979 s / 909 s |
| 3 | 0.0767536 | 0.0769477 | 0.0768636 | 1.1e-3 | 0.286904 | 0.286889 | 5.2e-5 | 2106 s / 1149 s |

(validation job 14827165, MIG slice, 43 min including precompilation; initial state
identical: ε₀ = 0.398823, ⟨M⟩₀ = 0.250091.) ⟨M⟩ agrees to ≤ 5e-5. ⟨C_H²⟩ is pinned to
the target only to the QCQP tolerances (bisect 5e-3, warm-accept window 0.1 → 5e-3),
and the per-block noise (`randn`, unseeded, as in production) makes runs differ
at that level; the archived run itself misses its targets by up to 2.5e-3. Both
points lie on the same frontier: moving along it by the observed Δ⟨C_H²⟩ with slope
dν/dε = θ² ≈ 0.67 accounts for the step-3 ν difference to 4e-5.
