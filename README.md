# SSO-Code

> [!NOTE]
> The codebase is created with AI coding agents. The physical correctness of the code and figures are verified through extensive tests; however, the exact code and the comments in the files have not been totally proofread by human. In case any description conflicts with the paper, refer to the paper as source of truth. In case any error is found, please raise an issue or contact the author (yangth@princeton.edu).

Code and figure data for

> **Simple slow operators and quantum thermalization**  
> Tian-Hua Yang, Sarang Gopalakrishnan, and Dmitry A. Abanin  
> [arXiv:2604.13172](https://arxiv.org/abs/2604.13172)

An operator A (orthogonal to powers of H) is characterised by its *simplicity*
ν_A = ‖A‖²_RPS — the squared random-product-state (RPS) norm, an ensemble variance
norm that is dominated by small Pauli strings — and by the timescale on which it
evolves, τ_A = 1/‖[H,A]‖ (A of unit Frobenius norm). The repository computes the
ν(τ) frontier, the maximal simplicity at given τ obtained by optimisation, exactly on
finite chains and in the thermodynamic limit, and compares it with the ν–τ curves
traced by explicit operator families (filtered local operators, modulated
hydrodynamic modes).

Everything needed to regenerate Figures A–H from the shipped data is in this
repository. The production computations that created the data can also be rerun
with the launchers in `slurm/`; they need GPUs, and their costs are listed in
`docs/`.

## Quick start: reproduce the figures

```bash
conda env create -f environment.yml     # Python 3.12, numpy 2, scipy, matplotlib, quspin
conda activate sso
for f in scripts/figures/*.py; do python "$f"; done     # -> figures/*.pdf, *.png
pytest -q tests/                        # CPU-only tests, a few minutes
```

Figure scripts read only the slim data in `data/` (a few MB) and call the library `sso`.
They run in seconds on a laptop.

## Figures → code → data

`scripts/figures/figure_{A..H}.py` writes `figures/figure_{A..H}.{pdf,png}`. Layout:
* Each panel, including its colourbar or secondary axis, takes one third of a 7.0 in
  double column.
* Plot widths are equalised within each figure (`sso.plotting.style.layout_row`).

Filters: every filtered operator uses the Lorentzian f(ω) = m²/(ω² + m²) = 1/(1 + θω²),
the μ = ∞ frontier operator of the Krylov chain. The exception is an explicit filter
comparison, which is labelled (E (b, c)).

[`docs/figure_catalog.md`](docs/figure_catalog.md) gives each panel's data file,
extractor, filter and provenance, and records the computations added for D and G.

| Figure | Panels | Compute drivers → launchers | Details |
|---|---|---|---|
| **A** | (a) Kim–Huse L = 12 frontier resolved by momentum sector (k = 0, k = 1, cross-sector mix; right axis: effective size). (b) Kim–Huse frontier L = 6–12, inset ν(∞) vs L = 6–14 with an exponential fit. | `scripts/compute/nutau/{frontier,crossing_sectors,tauinf}.py` → `slurm/nutau/{kimhuse_frontier_L6_12,kimhuse_frontier_L6_dense,crossK_kimhuse,tauinf_L6_13,tauinf_L14}.slurm` | [`docs/nutau.md`](docs/nutau.md) |
| **B** | (a) Kicked Ising L = 12: frontier vs Lorentzian-filtered Σ Z, Z₁, Σ X. (b) h_x = 0.9 chain, frontier for h_z = 0.005–0.2; inset h_z = 0.03 frontier with the Lorentzian-filtered energy current Q. (c) DQT energy-density mode H_q of the mixed-field ring, ν vs τ/L² for L = 16–23 under the Lorentzian (stored curves re-filtered); inset ν vs τ. | `frontier.py`, `crossing_within.py`, `filter_{floquet,static}.py` → `slurm/nutau/{floquet_frontier_L6_12,hx09_frontier_L12,within_hx09_L12,filter_lorentz_L12}.slurm`. DQT: `scripts/compute/dqt/{fourier_cv,combine,refilter}.py` → `slurm/dqt/campaigns.sh` | [`docs/nutau.md`](docs/nutau.md), [`docs/dqt.md`](docs/dqt.md) |
| **C** | Kim–Huse ν_n(∞) vs the number n of projected powers of H, even L = 6–14, with the Hⁿ baseline and the thermodynamic-limit reference r^d. | `tauinf.py` → `slurm/nutau/tauinf_*.slurm` | [`docs/nutau.md`](docs/nutau.md) |
| **D** | TFIM h_x = 2, h_z = 0. (a) L = 12 frontier resolved by sector. (b) Frontier L = 6–12, inset ν(∞) vs L, which saturates near 0.120. | `frontier.py`, `crossing_{sectors,within}.py`, `tauinf.py` → `slurm/nutau/{tfim_frontier_L12,within_tfim_L12,crossK_tfim_L12,tfim_frontier_L6_11,tfim_arcs_L6_11,tfim_tauinf_L6_12}.slurm` | [`docs/nutau.md`](docs/nutau.md) |
| **E** | h_x = 0.9 chain. (a) Collapse Δν = 1/9 − ν vs τh_z². (b) h_z = 0.03 frontier vs the current Q under Lorentzian, Gaussian, hard-cutoff and time-average filters (filter comparison). (c) ν_frontier − ν_filter at equal τ, log scale. | `frontier.py`, `crossing_within.py`, `filter_static.py` → `slurm/nutau/{hx09_frontier_L12,within_hx09_L12,filter_hx09_YZcur_L12,filter_lorentz_L12}.slurm` | [`docs/nutau.md`](docs/nutau.md) |
| **F** | (a) DQT: bare autocorrelator C(t) and RPS norm of H_q(t) vs t/L², L = 16–23; inset two-time RPS correlator at L = 20. (b) Δν vs τq² for Kim–Huse in the thermodynamic limit (TI ansatz, M = 11). | `scripts/compute/dqt/fourier_cv.py` → `slurm/dqt/campaigns.sh`; (b) `julia/TIAnsatz/scripts/ham_q_scan.jl` → `slurm/ti/reproduce_fig3c.sh` | [`docs/dqt.md`](docs/dqt.md), [`docs/tiansatz.md`](docs/tiansatz.md) |
| **G** | (a) Kim–Huse L = 12 frontier, coloured by sector, against two Lorentzian-filtered operators: the single-site Pauli sum Σ_x n̂·σ_x with the largest ν(∞), and H_q with q = 2π/L. (b) Kicked Ising: ED frontier L = 6–12 vs the TI ansatz M = 6–12. (c) Kim–Huse: ED frontier L = 6–12 vs the static TI ansatz M = 6–12, both with I…H³ removed (o4). | `filter_static.py`, `optimal_onesite.py` → `slurm/nutau/{filter_kimhuse_L12,filter_kimhuse_onesite_Hq_L12}.slurm`; (b) `julia/TIAnsatz/scripts/floquet.jl` → `slurm/ti/reproduce_floquet.sh`; (c) TI tables from the TI workstream (`ham_mf_driver_rtp.jl`, order 4) | [`docs/nutau.md`](docs/nutau.md), [`docs/tiansatz.md`](docs/tiansatz.md) |
| **H** | DMRG attempt (unsuccessful): (a) L = 12, χ = 128/256/512 vs the exact frontier; (b) L = 12–30 × χ vs τ/L². | `julia/SlowopDMRG/scripts/{eps_ladder,one_step}.jl` → `slurm/dmrg/reproduce_plotted_series.sh` | [`docs/dmrg_attempt.md`](docs/dmrg_attempt.md) |

**Data.** The slim data lives in `data/<dir>/`. Each directory is rebuilt from the raw
output (`$SSO_OUTPUT`) by an extractor in `scripts/extract/`. The directory names come
from the original paper figures. The table lists which figures read each one.

| `data/` | figures | extractor(s) |
|---|---|---|
| `sector_L12_kimhuse_and_tfim/` | A (a), D (a) | `nutau_sector_L12.py` |
| `kimhuse_three_panel/` | A (b), C, G (a, c) | `nutau_kimhuse_three_panel.py`, `nutau_kimhuse_filters.py`, `ti_kimhuse_o4.py` |
| `tfim_frontier/` | D (b) | `nutau_tfim_frontier.py` |
| `nutau_hx0.9_twopanel/` | B (b), E | `nutau_hx09_twopanel.py` |
| `floquet_two_panel/` | B (a), G (b) | `nutau_floquet_two_panel.py` |
| `fig3panel/` | B (c), F | `dqt_fig3panel.py`, `dqt_lorentz_fronts.py` |
| `attempt_dmrg/` | H | `dmrg_points.py` |

The earlier paper figures (`fig3panel`, `floquet_two_panel`, `sector_L12_kimhuse_and_tfim`,
`nutau_hx0.9_twopanel`, `kimhuse_three_panel`, `attempt_dmrg_*`) are deprecated. Their
scripts were removed, and every panel they had is either in A–H or listed under "Not
shown" in the catalog. They remain in the git history up to commit `29d2621`.

## Sub-codebases

| Path | Language | What it is |
|---|---|---|
| `sso/nutau/` | Python (numpy / optional cupy) | Exact ν(τ) frontier on finite chains in the doubled (operator) Hilbert space. Covers: momentum-resolved diagonalisation of a static H or Floquet U; the Lorentzian-penalised eigenproblem M(θ) = Π L_θ B L_θ Π; the θ = ∞ commutant; crossing arcs within and across momentum sectors; explicit filters (Lorentzian, plus squared-Lorentzian, Gaussian, hard and time-average for comparisons); τ conventions. |
| `sso/dqt/` | Python (numpy / optional cupy) | Dynamical typicality with Chebyshev (or Krylov) propagation for large rings (L ≤ 23). Covers: windowed norms N₁, N₂ of H_k(s); the RPS norm with an analytic control variate; autocorrelators; re-filtering of the stored curves with other filters, including the Lorentzian (`sso.dqt.refilter`); exact-diagonalisation and Chebyshev-moment references, used for validation. |
| `sso/tiansatz.py` | Python | Readers / analysis of the thermodynamic-limit tables written by `julia/TIAnsatz`. |
| `sso/plotting/` | Python | Frontier assembly (sector winner plus arcs), the shared panel helpers (`sector_panel`, `hx09`, `floquet`) and figure layout (`style.layout_row`). |
| `julia/TIAnsatz/` | Julia (CUDA optional) | Thermodynamic-limit frontier with a translation-invariant operator ansatz of support M (static with momentum q, and Floquet); matrix-free LOBPCG with a Chebyshev preconditioner. Includes the `ThermodynamicPauli` package (Pauli-string algebra on the infinite chain). |
| `julia/SlowopDMRG/` | Julia (ITensors; CUDA optional) | The DMRG attempt: ε-constrained local QCQP sweeps for an MPS operator. Documented as unsuccessful. |
| `KrylovChain/` | Python (numpy) | Self-contained: ν(τ) frontier of the su(1,1) Krylov-chain model (b_n = √(n(n+1)), M = e^{−μ(n+1)}), with (1+θL²)⁻¹ evaluated exactly by ω-space integrals. See [`KrylovChain/README.md`](KrylovChain/README.md). |

Shared infrastructure: `sso/backend.py` (numpy/cupy), `sso/config.py` (paths),
`sso/cli.py`, `sso/operators.py`. Conventions for contributors: [`docs/DEVELOPING.md`](docs/DEVELOPING.md).

## Conventions in one place

**Simplicity, slowness and filters**
* ν = ‖O‖²_RPS/‖O‖² = ⟨O|B|O⟩/⟨O|O⟩. B is the kernel of the RPS norm: it is diagonal in
  the Pauli-string basis, with entry (d+1)^(−|P|) = 3^(−|P|) for a string P of weight |P|.
* σ² = ⟨O|Δ²|O⟩/⟨O|O⟩.
  * Static: Δ = E_m − E_n and τ = 1/σ.
  * Floquet: Δ² = 4 sin²((φ_m − φ_n)/2), with the chord τ = 1/(2 arcsin(σ/2)).
* `o` is the number of conserved powers {I, H, …, H^{o−1}} projected out
  (`pows_max = o − 1`). The ν(∞) insets of A and D use o = 5, i.e. ν₄(∞).
* Filters multiply the eigenbasis elements of an operator by f(ω). The default is the
  Lorentzian f = m²/(ω² + m²), whose time window is (m/2)e^{−m|t|}.

**Models.** All static chains are mixed-field Ising rings H = Σ_x (J Z_x Z_{x+1} + h_x X_x + h_z Z_x) with J = 1 and periodic boundary conditions.

| model | parameters | figures |
|---|---|---|
| Kim–Huse (chaotic) | h_x = 0.905, h_z = 0.809 | A, C, F (b), G (a, c); H uses OBC |
| TFIM, integrable | h_x = 2, h_z = 0 | D |
| TFIM with a weak longitudinal field (near-integrable) | h_x = 0.9, h_z = 0.005–0.2 | B (b), E |
| Mixed-field ring ("model B", DQT) | g = h_x = −1.05, h = h_z = 0.5 | B (c), F (a) |
| Kicked Ising (Floquet) | U = e^{−i 0.9 Σ X} e^{−i(Σ Z Z + 0.809 Σ Z)} | B (a), G (b) |
| Krylov chain | b_n = √(n(n+1)), μ-kernel e^{−μ(n+1)} | `KrylovChain/` |

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
