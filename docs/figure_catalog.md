# Figure catalog (overhauled figure set, Oct 2026)

Each figure is `scripts/figures/figure_X.py` → `figures/figure_X.{pdf,png}` and reads `data/` only.

**Conventions**
- Each panel occupies one third of a 7.0 in double column, including its colourbar or secondary axis.
- Plot boxes are 1.36 in high. Their widths are equalised within each figure by `sso.plotting.style.layout_row`.
- Filters are the Lorentzian m²/(ω² + m²) = 1/(1 + θω²) (KrylovChain convention) unless the panel is an explicit filter comparison, which the table states.

Status: ✅ generated (all figures A–H).

| Fig. | Panel | Content | Filter | Data (`data/…`) | Source of the panel | Status |
|---|---|---|---|---|---|---|
| A | (a) | Kim–Huse L = 12 sector-resolved frontier (k = 0 / k = 1 / mix), effective-size axis | none | `sector_L12_kimhuse_and_tfim/` | left panel of `sector_L12_kimhuse_and_tfim` | ✅ |
| A | (b) | Kim–Huse frontier L = 6–12; inset ν(∞) = ν₄(∞) vs L = 6–14 with exponential fit | none | `kimhuse_three_panel/` | (a) + (b) of `kimhuse_three_panel` (power-law inset dropped) | ✅ |
| B | (a) | kicked Ising L = 12 frontier + Lorentzian ΣZ, Z₁, ΣX | Lorentzian | `floquet_two_panel/lorentzcut_*` | (a) of `floquet_two_panel` | ✅ |
| B | (b) | h_x = 0.9 frontier vs h_z; inset h_z = 0.03 frontier + Lorentzian-filtered current Q, bare-Q star | Lorentzian | `nutau_hx0.9_twopanel/` (`filt_hz0.03_YZcur_lorentz.npz`) | left panel of `nutau_hx0.9_twopanel`; inset after its right panel | ✅ |
| B | (c) | DQT energy-density mode H_q, ν vs τ/L², L = 16–23; inset ν vs τ | Lorentzian (stored curves re-filtered, τ/L² ≤ 0.15–0.17) | `fig3panel/modhydro_lorentz_*_REFILTER.npz` | layout of `fig3panel` (a), which was Gaussian | ✅ |
| C | — | Kim–Huse ν_n(∞) vs n, even L = 6–14, Hⁿ baseline, r^d reference | none | `kimhuse_three_panel/{tauinf,baseline}.csv` | (c) of `kimhuse_three_panel` (item 3) | ✅ |
| D | (a) | TFIM h_x = 2, h_z = 0, L = 12 sector-resolved frontier | none | `sector_L12_kimhuse_and_tfim/` | right panel of `sector_L12_kimhuse_and_tfim` | ✅ |
| D | (b) | TFIM frontier L = 6–12 (sector union, cross-sector and within-K0 arcs); inset ν(∞) = ν₄(∞) vs L = 6–12 (saturates at ≈ 0.120) | none | `tfim_frontier/` (new: L = 6–11 grids, arcs, θ = ∞; `slurm/nutau/tfim_*`) | new, analogue of A(b) | ✅ |
| E | (a) | h_x = 0.9 collapse Δν = 1/9 − ν vs τh_z² | none | `nutau_hx0.9_twopanel/` | collapse inset of `nutau_hx0.9_twopanel` (item 4) | ✅ |
| E | (b) | h_z = 0.03 frontier + Q under Lorentzian, Gaussian, hard, sinc; bare-Q star | **filter comparison** | `nutau_hx0.9_twopanel/` | right panel of `nutau_hx0.9_twopanel` + Lorentzian (item 5) | ✅ |
| E | (c) | ν_frontier − ν_filter at equal τ (log y) for the four filters | **filter comparison** | same | new | ✅ |
| F | (a) | DQT C(t)/C(0) and ‖H_q(t)‖_RPS vs t/L², L = 16–23; inset \|G_RPS(t₁,t₂)\| at L = 20 | none (bare dynamics) | `fig3panel/correlators_*`, `grps_L20.npz` | (b) of `fig3panel` (item 6) | ✅ |
| F | (b) | Δν vs τq², Kim–Huse TI ansatz M = 11 | none | `fig3panel/ti_q_scan/` | (c) of `fig3panel` (item 7) | ✅ |
| G | (a) | Kim–Huse L = 12 frontier coloured by sector + Lorentzian-filtered optimal Σ n̂·σ (n̂ = (0.71, 0, −0.70), maximises ν(∞)) and H_q (q = 2π/L) | Lorentzian | `kimhuse_three_panel/filt_L12_{K0_OPT1S,K1_Hdens}`, `onesite_opt_L12_o4` (`slurm/nutau/filter_kimhuse_onesite_Hq_L12.slurm`) | new | ✅ |
| G | (b) | kicked Ising ED frontier L = 6–12 vs TI ansatz M = 6–12 | none | `floquet_two_panel/` | (b) of `floquet_two_panel` (item 10) | ✅ |
| G | (c) | Kim–Huse ED K = 0 o4 frontier L = 6–12 vs static TI ansatz o4 (H, H², H³ removed as restricted true powers), M = 6–12; TI 2–5 % above ED at small τ | none | `kimhuse_three_panel/grid_L*_K0`, `ti_rtp_o4/` | new | ✅ |
| H | (a) | DMRG attempt L = 12, χ = 128/256/512 vs exact frontier | none | `attempt_dmrg/` | `attempt_dmrg_L12` | ✅ |
| H | (b) | DMRG attempt L = 12–30 × χ vs bare τ, exact L = 12 frontier | none | `attempt_dmrg/` | `attempt_dmrg_multiL` (was τ/L²) | ✅ |

In Figure H, the two L = 12, step-8 points at χ = 256 and 512 have no saved MPS; their values come from the logs of completed (converged) χ-refinement runs. They are drawn like all other converged points, with no special legend entry.

## New computations (Oct 2026)

* **D:** TFIM L = 6–11 sector grids, plus all arcs, plus θ = ∞ for L = 6–12. All on MIG, about 40 min in total.
  - Launchers: `tfim_frontier_L6_11.slurm`, `tfim_arcs_L6_11.slurm`, `tfim_tauinf_L6_12.slurm`.
  - Arc positions: the within-K0 crossing θ* goes from 0.4386 (L = 6) to 0.4364 (L = 11), approaching 0.4362 at L = 12. The cross-sector θ* goes from 1.54 to 2.53.
* **G(a):** Kim–Huse L = 12. About 20 min on MIG.
  - `filter_kimhuse_L12.slurm` filters ΣZ, ΣX and Z₁ with all filters. Z₁ uses the new full-space `--K=none` path in `filter_static.py`; ΣZ and ΣX agree with the archive to about 1e-4.
  - `filter_kimhuse_onesite_Hq_L12.slurm` runs the optimal-direction search and the Lorentzian for the optimal Σ n̂·σ and for H_q.
  - Optimal direction by L: n̂ = (−0.33, 0, 0.95) at L = 8, (−0.60, 0, 0.80) at L = 10, (0.71, 0, −0.70) at L = 12. It tends toward perpendicular to the field direction.
  - ΣY has no conserved part. A single-site n̂·σ₁ has a lower ν(∞): 0.0142 at L = 12, against 0.0209 for the sum.

## Not shown in any figure

From the old production scripts:
* `kimhuse_three_panel` (b): the power-law inset.
* `floquet_two_panel` (a): the Gaussian, hard and sinc curves. G (a) shows only Lorentzian curves.
* `fig3panel` (a): the Gaussian-window version. B (c) shows the Lorentzian.

The scripts and rendered files of these deprecated figures were removed from `main`;
they remain in the git history up to commit `29d2621`. Their data directories are
kept, because the new figures read them.
