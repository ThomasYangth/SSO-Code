# SSO-Code

Code and figure data for

> **Simple Slow Operators and Quantum Thermalization**  
> Tian-Hua Yang, Sarang Gopalakrishnan, and Dmitry A. Abanin  
> [arXiv:2604.13172](https://arxiv.org/abs/2604.13172)

The paper asks how *simple* (low Pauli weight, measured by the random-product-state
weight ν) an operator can be while still being *slow* (relaxing on a time scale τ).
The repository computes the optimal trade-off ν(τ) — the *simple-slow-operator
frontier* — exactly on finite chains, in the thermodynamic limit, and through
explicit constructions (filtered local operators, modulated hydrodynamic modes).

Everything needed to regenerate the paper figures from the shipped data is in
this repository. The production computations that created the data can also be
rerun with the launchers in `slurm/` (they need GPUs; costs are listed in `docs/`).

## Quick start: reproduce the figures

```bash
conda env create -f environment.yml     # Python 3.12, numpy 2, scipy, matplotlib, quspin
conda activate sso
for f in scripts/figures/*.py; do python "$f"; done     # -> figures/*.pdf, *.png
pytest -q tests/                        # CPU-only tests, a few minutes
```

Figure scripts read only the slim data in `data/` (about 1 MB in total) and call the
library `sso`; they run in seconds on a laptop.

## Figures → code → data

| Figure (`figures/`) | Panels | Library | Compute drivers → launchers | Details |
|---|---|---|---|---|
| `fig3panel` | (a) ν vs τ/L² of the windowed modulated energy density H_k(s), MFIM ring L = 16–23; (b) autocorrelator C(t) and RPS norm √R(t) vs t/L², inset two-time RPS correlator at L = 20; (c) Δν = ν(0) − ν(q) vs τq² for Kim–Huse in the thermodynamic limit (support M = 11) | `sso.dqt`, `sso.tiansatz`; `julia/TIAnsatz` | `scripts/compute/dqt/{fourier_cv,combine}.py` → `slurm/dqt/campaigns.sh`; panel (c): `julia/TIAnsatz/scripts/ham_q_scan.jl` → `slurm/ti/reproduce_fig3c.sh` | [`docs/dqt.md`](docs/dqt.md), [`docs/tiansatz.md`](docs/tiansatz.md) |
| `floquet_two_panel` | (a) kicked Ising L = 12: exact frontier vs Gaussian / hard-cutoff / N-period-average filtered Σ Z, Z₁, Σ X; (b) exact frontier L = 6–12 vs thermodynamic-limit TI ansatz M = 6–12 | `sso.nutau`, `sso.tiansatz`; `julia/TIAnsatz` | `scripts/compute/nutau/{frontier,filter_floquet}.py` → `slurm/nutau/{floquet_frontier_L6_12,filter_floquet_L12}.slurm`; (b) dashed: `julia/TIAnsatz/scripts/floquet.jl` → `slurm/ti/reproduce_floquet.sh` | [`docs/nutau.md`](docs/nutau.md) |
| `sector_L12_kimhuse_and_tfim` | Momentum-resolved frontier (K = 0 vs K = 1 winner, crossing arcs) for Kim–Huse and the TFIM, L = 12 | `sso.nutau`, `sso.plotting` | `frontier.py`, `crossing_sectors.py`, `crossing_within.py` → `slurm/nutau/{kimhuse_frontier_L6_12,tfim_frontier_L12,crossK_*,within_tfim_L12}.slurm` | [`docs/nutau.md`](docs/nutau.md) |
| `nutau_hx0.9_twopanel` | (left) frontier for hx = 0.9 and a scan of hz, inset collapse vs τ·hz²; (right) hz = 0.03 frontier vs filtered spin current | `sso.nutau` | `frontier.py`, `crossing_within.py`, `filter_static.py` → `slurm/nutau/{hx09_frontier_L12,within_hx09_L12,filter_hx09_YZcur_L12}.slurm` | [`docs/nutau.md`](docs/nutau.md) |
| `kimhuse_three_panel` | (a) global frontier L = 6–12; (b) ν₄(∞) vs L; (c) ν_n(∞) vs n against the Hⁿ baseline | `sso.nutau` | `frontier.py`, `crossing_sectors.py`, `tauinf.py` → `slurm/nutau/{kimhuse_frontier_*,crossK_kimhuse,tauinf_*}.slurm` | [`docs/nutau.md`](docs/nutau.md) |
| `attempt_dmrg_*` *(not in the paper)* | DMRG/MPS variational frontier, L = 12–30, χ = 128–512, vs the exact L = 12 front — an **unsuccessful** approach (it stalls at a bond-dimension “χ wall”) | `julia/SlowopDMRG` | `julia/SlowopDMRG/scripts/{eps_ladder,one_step}.jl` → `slurm/dmrg/reproduce_plotted_series.sh` | [`docs/dmrg_attempt.md`](docs/dmrg_attempt.md) |

Each figure has a matching `scripts/extract/*.py` that rebuilds its `data/<figure>/`
from raw compute output (`$SSO_OUTPUT`), and a `scripts/figures/*.py` that draws it.

## Sub-codebases

| Path | Language | What it is |
|---|---|---|
| `sso/nutau/` | Python (numpy / optional cupy) | Exact ν(τ) frontier on finite chains in the doubled (operator) Hilbert space: momentum-resolved diagonalisation of static H or Floquet U, Lorentzian-penalised eigenproblem M(θ) = Π L_θ B L_θ Π, θ = ∞ commutant, crossing arcs within/across momentum sectors, Gaussian/hard/time-average filters, τ conventions. |
| `sso/dqt/` | Python (numpy / optional cupy) | Dynamical typicality with Chebyshev (or Krylov) propagation for large rings (L ≤ 23): windowed norms N₁, N₂ of H_k(s), RPS norm with an analytic control variate, autocorrelators; exact-diagonalisation and Chebyshev-moment references used as validation. |
| `sso/tiansatz.py` | Python | Readers / analysis of the thermodynamic-limit tables written by `julia/TIAnsatz`. |
| `sso/plotting/` | Python | Frontier assembly (sector winner + arcs) and shared figure style. |
| `julia/TIAnsatz/` | Julia (CUDA optional) | Thermodynamic-limit frontier with a translation-invariant operator ansatz of support M (static with momentum q, and Floquet); matrix-free LOBPCG with a Chebyshev preconditioner. Includes the `ThermodynamicPauli` package (Pauli-string algebra on the infinite chain). |
| `julia/SlowopDMRG/` | Julia (ITensors; CUDA optional) | The DMRG attempt: ε-constrained local QCQP sweeps for an MPS operator. Documented as unsuccessful. |

Shared infrastructure: `sso/backend.py` (numpy/cupy), `sso/config.py` (paths),
`sso/cli.py`, `sso/operators.py`. Conventions for contributors: [`docs/DEVELOPING.md`](docs/DEVELOPING.md).

## Conventions in one place

* ν = ⟨O|B|O⟩/⟨O|O⟩ with the RPS kernel B = 3^(−#non-identity sites) (diagonal in the Pauli basis).
* σ² = ⟨O|Δ²|O⟩/⟨O|O⟩; static Δ = E_m − E_n, τ = 1/σ; Floquet Δ² = 4 sin²((φ_m − φ_n)/2), chord τ = 1/(2 arcsin(σ/2)).
* Kim–Huse Ising: H = Σ Z Z + 0.905 Σ X + 0.809 Σ Z. Kicked Ising: U = e^{−i 0.9 Σ X} e^{−i(Σ Z Z + 0.809 Σ Z)}. Model B (fig3panel a, b): mixed-field Ising ring Σ (Z Z + g X + h Z) with J = 1, g = −1.05, h = 0.5; TFIM: hx = 2, hz = 0.
* `o` = number of conserved powers {I, H, …, H^{o−1}} projected out (`pows_max = o − 1`). Note that kimhuse_three_panel (b) uses o = 5.

## Paths and environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SSO_OUTPUT` | `<repo>/output` | where compute drivers write raw results (large; point at scratch storage) |
| `SSO_DATA` | `<repo>/data` | slim figure data read by the figure scripts |
| `SSO_FIGURES` | `<repo>/figures` | rendered figures |

The Julia projects have their own `Project.toml`/`Manifest.toml`; see
[`julia/TIAnsatz/README.md`](julia/TIAnsatz/README.md) and
[`julia/SlowopDMRG/README.md`](julia/SlowopDMRG/README.md).

## Hardware

Production runs used Princeton's Della cluster (NVIDIA A100, mostly 10 GB MIG
slices). Every Python routine also runs on the CPU (`--usegpu=False`, the
default), which is practical for L ≲ 10–14 depending on the routine. The slurm
launchers are written for Slurm with the Della partition names; adapt the
`#SBATCH` headers to your cluster.

## Citation

If you use this code, please cite

```bibtex
@article{yang2026simple,
  title         = {Simple slow operators and quantum thermalization},
  author        = {Yang, Tian-Hua and Gopalakrishnan, Sarang and Abanin, Dmitry A.},
  year          = {2026},
  eprint        = {2604.13172},
  archivePrefix = {arXiv}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
