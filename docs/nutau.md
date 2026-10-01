# sso.nutau — the exact ν(τ) frontier (static and Floquet)

This part of the code computes, by exact diagonalization, the most local
operator at a given slowness ("simple slow operators"), for the mixed-field
Ising chain and the kicked Ising chain, and the explicit filtered
constructions it is compared against.  It produces the data of four figures:
`floquet_two_panel`, `sector_L12_kimhuse_and_tfim`, `nutau_hx0.9_twopanel`,
`kimhuse_three_panel`.

## Definitions

* Operators are vectors in the doubled space.  In the eigenbasis {|m⟩} of H
  (or of the Floquet unitary U) an operator is its matrix M_mn = ⟨m|O|n⟩.
* **Locality** ν = ⟨v|B|v⟩/⟨v|v⟩ with the RPS kernel, diagonal in the Pauli-string
  basis, B_σ = 3^{-|σ|} (|σ| = number of non-identity sites).  NPZ key `Bs`.
* **Slowness** σ² = ⟨v|Δ²|v⟩/⟨v|v⟩, NPZ key `As`, with Δ diagonal on (m, n):
  * static: Δ_mn = E_m − E_n − λ, τ = 1/√σ²;
  * Floquet: Δ²_mn = 4 sin²((φ_m − φ_n − λ)/2), chord τ = 1/(2 arcsin(√σ²/2))
    (τ = 1/|ω| for a single quasi-energy difference ω; τ ≥ 1/π).

  `sso.nutau.conventions.tau_static / tau_floquet / tau_of`.  All production
  data use λ = 0.
* **Soft frontier.**  L_θ = 1/√(1 + θ²Δ²) and M(θ) = Π L_θ B L_θ Π, with Π the
  projector off the trivially slow operators: static span{I, H, …, H^{o−1}};
  Floquet {I} only ("o1").  The top eigenvector w of M (eigenvalue W, NPZ key
  `w`) gives the operator v = L_θ w.  W is comparable across momentum sectors,
  ν is not the right criterion for which sector wins.  The stored `v2` is the
  pre-filter w in the Pauli basis (not v).
* **Sectors.**  With momentum-resolved eigenvectors (k_m), M is block diagonal in
  the doubled momentum K = (k_m − k_n) mod L, so each K is solved on its own
  d_K ≈ 4^L/L dimensional index set ("compressed").  K ≠ 0 forces o = 0
  (powers of H live in K = 0).  K and L−K give identical curves.
* **θ = ∞.**  L_θ → 1 on the exact commutant (pairs inside degenerate
  multiplets, Δ² ≤ (10⁻¹⁰)²) and 0 elsewhere: M = Π B Π there, σ² = 0.
* **o vs pows_max.**  File names carry `o` = number of projected-out operators;
  static drivers take `--pows_max` = o − 1.  `k2o4` = k_save 2, o = 4
  (I, H, H², H³).  Floquet is always o1 (K = 0) / o0 (K ≠ 0).
  **Kim-Huse panel (b) is ν₄(∞) = the o = 5 commutant** (I..H⁴ removed); panel
  (c) plots ν at n = o − 1.
* **Arcs.**  Where two branches exchange the maximum of M at θ*, the two
  degenerate eigenvectors w_a, w_b give the continuous family
  u = cos α w_a + sin α w_b with ν(α) = W/N(α), σ²(α) = D(α)/N(α),
  N_ij = ⟨w_i|L²|w_j⟩, D_ij = ⟨w_i|L²Δ²|w_j⟩ (`sso.nutau.crossing`).
  Across sectors (`crossK_*`) θ* is the sign change of W_Ka − W_Kb (false
  position) and the cross terms vanish; within one sector (`cross_*`) θ* is
  located by bisecting s(θ) = ν₀(θ) − (ν_lo + ν_hi)/2 with the top pair only.
* **Filters** (`sso.nutau.filters`): a fixed base operator multiplied
  element-wise by f(ω): Gaussian exp(−ω²/2s²), hard 1[|ω| ≤ w], time average
  (static (1/T)∫₀ᵀ e^{iωt}dt; Floquet (1/N)Σ_{t<N} e^{iωt} in closed form —
  the "trajectory" needs no time stepping).

## Module map

| module | content |
|---|---|
| `conventions` | τ(σ²) for both kinds |
| `pauli` | Pauli ↔ computational basis, `rps_kernel`, `EnergyBasisB` (B acting on E-basis matrices, preallocated buffers; `nu`, `to_pauli`, `from_pauli`) |
| `models` | QuSpin matrices of `sso.operators.Operator`, `kicked_ising` layers, base operators Zsum/Xsum/Z0/YZcur, `modulated_density` (H_q), `mixture_matrix`, `model_from_args` |
| `spectrum` | `Spectrum`: static `eigh` or Floquet Schur, momentum-resolved (QuSpin k-blocks) or full space (any boundary, incl. OBC); detuning, frequencies, `diag_ortho`, sector/resonant index sets, multiplets |
| `frontier` | `Support` (index set + projector + Δ² of one sector problem, finite θ or commutant), `solve`, `top_eigenpairs`, `frontier_scan` (θ grid with warm start, caching, NPZ output) |
| `crossing` | bracket detection (`sign_change_brackets`, `largest_tau_jump`, `detect_crossings` by stored top-multiplet overlap), `cross_sector_arc`, `within_sector_arc` (locate mode `nu` or `overlap`, any sector or the full space), `multiplet_tau_nu` (symmetry-partner diagnostic), `degenerate_pair_arc` |
| `filters` | filter functions, width grids (incl. `keep_slowest`), `filter_sweep`, `sample_periods` |
| `tauinf` | `tauinf_nu` (θ = ∞, full space), `power_baseline_nu`, `ising_r` |
| `io` | raw grid readers (o matched explicitly), slim-data loaders |

`sso.plotting.frontier` holds the frontier assembly used by the figures:
`grid_curve`, `break_gaps`, and `sector_frontier` (winner by W with tie → K=0,
hand-over test for cross-sector arcs, bridging to neighbouring winners,
clipping at τ_max, within-sector arcs placed in the branch holding θ*).  Both
`sector_L12_kimhuse_and_tfim` (coloured segments) and `kimhuse_three_panel`
(the `union` line) use it.

Unification relative to the original MPS-Concentration scripts: one
diagonalization routine for static/Floquet/full/momentum; one sector solver
for static and Floquet, finite θ and θ = ∞, compressed sector or full space
(the static θ = ∞ multiplet-packed solver of the τ = ∞ benchmark is the K = None
commutant of the same solver); one arc formula; one filter sweep (the Floquet
N-period trajectory is the discrete-average filter); one τ-convention module;
one B operator.

## Retained features not used by the four figures

Kept on request, validated against the original code at small L (below):

* **Full-space soft solver and OBC** — `frontier.py --K=none [--obc=True]`
  (original `NuTau_ds_soft.py`).
* **Modulated energy density filter** — `filter_static.py --init_op=Hdens --K=<K>`
  filters H_q = Σ_x e^{−iqx} h_x, q = 2πK/L (original `NuTauFilterStatic.py`).
  No projector (K ≠ 0); the operator's actual sector may be L−K.  At K = 1 it
  has an exactly conserved component only at **odd L** (degenerate k ↔ −k pairs
  in sector 2k = 1 mod L): then all filters converge to that projection; at even
  L there is none and the sweep's narrow end keeps exactly the slowest occupied
  element (`keep_slowest`), checked by τ ≤ 1/|ω|_min and monotonic τ.
  `--mix=Zsum:c1,Xsum:c2 [--mix_tag=MIX]` filters a real mixture of base operators.
* **Overlap locate mode and overlap detection** — `crossing_within.py
  --detect=overlap --mode=overlap --k_calc=8`: brackets from drops (< 0.9) of
  the stored top-multiplet subspace overlap between adjacent θ; at each trial
  θ, k_calc eigenpairs are grouped into multiplets and labelled A/B by overlap
  with the bracket-end references (the stored top multiplets) and g = W_A − W_B
  is bisected (original `NuTau_soft_crossing.py --locate_mode=overlap`).  k_calc
  must hold both branches at every trial θ.
* **Full-space crossings** — `crossing_within.py --K=none` works on the
  full-space grid (`<model>_DSsoft_Data`, files `cross_L{L}_l…`).  For each
  degenerate endpoint multiplet the per-member (τ, ν) is printed
  (`multiplet_tau_nu`): identical = symmetry partners (momentum ±k, no arc),
  distinct = the crossing lies inside the multiplet.

## Drivers, launchers and figure provenance

All drivers live in `scripts/compute/nutau/` and write to `$SSO_OUTPUT`
(`sso.config.output_dir`); the launchers in `slurm/nutau/` reproduce the
production parameters exactly (submit from the repo root with `SSO_OUTPUT`
set).  `scripts/extract/nutau_*.py` turn raw output into `data/<figure>/`
(default source `$SSO_OUTPUT`, or `--src=`; read-only), and
`scripts/figures/<figure>.py` draws from `data/` only.

| figure / panel | raw data | driver | launcher |
|---|---|---|---|
| floquet_two_panel (a) frontier, (b) ED L = 6..12 | `KickedIsing_g0.9J1.0h0.809_DSFloquet_Data/L{L}_K0_theta*l0.0k2o1.npz` | `frontier.py --model=KickedIsing` | `floquet_frontier_L6_12.slurm` |
| floquet_two_panel (a) filters | `..._FloquetHardCut_Data/{gausscut,hardcut}_L12_{K0_Zsum,K0_Xsum,Knone_Z0}_l0.0.npz`, `..._FloquetTrajectory_Data/ftraj_L12_Knone_{Zsum,Z0,Xsum}_N1e+06l0.0.npz` | `filter_floquet.py` | `filter_floquet_L12.slurm` |
| floquet_two_panel (b) TI ansatz | `nutau_floquet_M{6..12}_g0p9tF1p0_gpures.dat` (Julia TIAnsatz, copied into `data/`) | `julia/TIAnsatz` | (TI workstream) |
| sector_L12 Kim-Huse | `Ising_J1.0X0.905Z0.809_DSsoftKsec_Data/L12_K0_*o4`, `L12_K1_*o0`, `crossings/crossK_L12_K0K1_*` | `frontier.py`, `crossing_sectors.py` | `kimhuse_frontier_L6_12.slurm`, `crossK_kimhuse.slurm` |
| sector_L12 TFIM | `Ising_J1.0X2.0Z0.0_DSsoftKsec_Data/L12_K0_*o4` (51 + 21 fill θ), `L12_K1_*o0`, `crossings/crossK_L12_K0K1_*`, `crossings/cross_L12[_K0]_l0.0k2o4_*` | `frontier.py`, `crossing_sectors.py`, `crossing_within.py` | `tfim_frontier_L12.slurm`, `crossK_tfim_L12.slurm`, `within_tfim_L12.slurm` |
| nutau_hx0.9 left/right frontier + arcs | `Ising_J1.0X0.9Z{hz}_DSsoftKsec_Data/L12_K0_*o4` (save_v2=False), `crossings/cross_L12_K0_*` | `frontier.py`, `crossing_within.py` | `hx09_frontier_L12.slurm`, `within_hx09_L12.slurm` |
| nutau_hx0.9 right filters | `Ising_J1.0X0.9Z0.03_StaticFilter_Data/filt_L12_K0_YZcur_l0.0o4.npz` | `filter_static.py` | `filter_hx09_YZcur_L12.slurm` |
| kimhuse_three_panel (a) | `Ising_J1.0X0.905Z0.809_DSsoftKsec_Data/L{6..12}_K{0,1}_*`, `crossK_L{L}_*` | `frontier.py`, `crossing_sectors.py` | `kimhuse_frontier_L6_12.slurm`, `kimhuse_frontier_L6_dense.slurm` (L=6 has the 51- and 61-point grids, 62 θ), `crossK_kimhuse.slurm` |
| kimhuse_three_panel (b), (c) | `Ising_J1.0X0.905Z0.809_TauInf_Data/{tauinf,baseline}_L{6..14}.csv` (originally `tools/benchmark_nutau_ds_tauinf.csv`, `tools/baseline_hn_nu.csv`) | `tauinf.py` | `tauinf_L6_13.slurm`, `tauinf_L14.slurm` |

Archived grids may contain extra θ from earlier probe runs beyond the
launcher grid; the extractors keep every stored point.

The shipped `data/` was extracted from the archived production output
(`--src=/scratch/gpfs/DABANIN/ty1475/MPS-Concentration`, TI tables from
`OGH/data`, τ = ∞ tables from the original CSVs via
`--tauinf_csv/--baseline_csv`).  Some archived Floquet grids contain extra θ
from earlier probe runs (L = 6: 41, L = 7: 40, L = 10: 46 files instead of
38); the figure uses every stored point, so a fresh run of
`floquet_frontier_L6_12.slurm` gives a curve on the 38-point grid only
(indistinguishable at plotting resolution but not bit-identical data).

## Hardware and run times

Production: Princeton Della, A100 MIG slices (10 GB) unless noted.

| run | cost |
|---|---|
| static / Floquet sector solve, L = 12, K = 0 | ~70 s per θ from a cold start on MIG, ~1 h per 51-point grid |
| cross-sector arc | 10–12 eigensolves (k = 1): seconds at L ≤ 8, ~15 min at L = 12 |
| within-sector arc L = 12 | ~35 solves (k = 2) + 2 × (k_ref = 8): ~45 min |
| static filters L = 12 (3 × 60 widths) | a few minutes per hz |
| Floquet filter / trajectory L = 12 | 1–2 min each |
| τ = ∞ scan | L ≤ 13 MIG; L = 14 needs a 40 GB A100 (every n² buffer is 4.3 GB) |
| L ≤ 8 anything | seconds to minutes on CPU (`--usegpu=False`, the default) |

## Validation against the archived data

The new code was run independently (outputs under a separate `SSO_OUTPUT`)
and compared with the archived production files (max relative deviation):

| quantity | cases | max rel. dev. |
|---|---|---|
| static sector grid As, Bs, W | Kim-Huse L = 6, 7, 8, K = 0 (o4) and K = 1, all 51 θ + ∞ (CPU) | 8e-13 |
| Floquet sector grid As, Bs, W | kicked Ising L = 6, 7, 8, K = 0 (o1) and K = 1, 37 θ + ∞ | 9e-15 |
| L = 12 grid points (MIG) | static K0 θ = 0.1, 10, 1000, K1 θ = 100; Floquet K0 θ = 0.1, 96.5, ∞, K1 θ = 17.3 | 8e-14 |
| full-space static solver | L = 6 PBC λ = 0 and λ = 0.5, OBC o5 (NuTau_ds_soft output) | 1.5e-9 (As at θ = 1e5, As ~ 1e-20), else 1e-12 |
| cross-sector arcs | Kim-Huse L = 6, 7, 8, 12; TFIM L = 12: θ*, arc ν(α), τ(α) | 5e-14 |
| within-sector arcs | TFIM L = 12 (θ* = 0.43624775), hx = 0.9 hz = 0.03 L = 12 (θ* = 2.0895184): θ*, arc | 1.5e-14 |
| τ = ∞ ν_o(∞), o = 1..5, and H^n baseline | L = 6, 7, 8 vs the CSVs (10 significant digits) | 0 at CSV precision |
| Floquet filters / trajectory | gauss/hard Σ Z (K0) L = 6, 8, 12; gauss Z0 (Knone) L = 6; hard Z0 L = 12; time average Σ Z, Z0 L = 6, 8, Z0 L = 12 | ≤ 1e-15 (As_avg 2e-11 at the N = 1e6 end, As ~ 1e-10) |
| static filters (YZcur, hx = 0.9, hz = 0.03, L = 12) | gauss/hard/avg on the plotted range (τ ≤ 3e4, As > 1e-14) | ν 1e-9, As 1.5e-7 |
| retained: H_q filter vs original run (copy) | Kim-Huse K = 1, L = 7 (conserved part) and L = 8 (keep_slowest); mix Zsum/Xsum L = 8 K = 0 | 1.3e-11 (all keys, As > 1e-14) |
| retained: overlap detect + overlap locate vs original run | full space Kim-Huse L = 6 (θ* = 11.42494912, dim S = 3); TFIM K = 0 L = 8 (θ* = 0.4370349505) | θ* identical, arc 2.5e-14 |

The static-filter deviations come from the width grid: its narrow end is set by
the smallest occupied |ω| = 3.4e-7, a near-degenerate splitting known only to
~2e-8 relative accuracy from `eigh`, and the whole grid scales with it.  The
four figures rebuilt from the extracted `data/` are pixel-identical to the
originals (200 dpi rasterisation, zero differing pixels).

Validation runs: CPU (4 cores) L ≤ 8 everything in 23 min; on one MIG slice
the L = 12 points + two cross-sector arcs + filters took 41 min, each L = 12
within-sector arc 44 min.

## Caveats

* The archived TFIM within-K0 arc was produced by `NuTau_soft_crossing.py`
  (file `cross_L12_l0.0k2o4_theta*.npz`, references from stored eigenvectors);
  `crossing_within.py` recomputes the references and writes
  `cross_L12_K0_l0.0k2o4_theta*.npz`.  The extractor accepts both names.
* Static full-space θ = ∞ files are now stored like every other θ (v2 in the
  Pauli basis, `d_res`) instead of the old multiplet-packed `v2_packed`.
* Floquet finite-θ solves now start from a seeded random vector (the original
  let ARPACK pick one); static and Floquet θ = ∞ use seed 999 + K.
* At narrow filter widths only exactly degenerate pairs survive and As drops
  to machine precision (τ ~ 10¹⁷): the figures cut Floquet filter points at
  τ ≥ 10⁸ and static ones at As ≤ 10⁻¹⁴.
* The large-θ end of every frontier (τ ≳ 10⁵) is eigensolver resolution, not
  physics; the figures stop at τ_max = 10⁵ (3·10⁴ for hx = 0.9).
* `crossing_within.py` handles one λ per call (the original `--lams=all` loop
  is a shell loop now).
