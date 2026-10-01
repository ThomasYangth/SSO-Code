# DQT / Chebyshev machinery for the modulated energy density (`sso.dqt`) and fig3panel

This note covers the method behind `figures/fig3panel.pdf`, the code map, how each
panel was produced, cost and hardware, and known caveats.

## 1. Model and target quantities

**Model B** is the mixed-field Ising ring with periodic boundary conditions, written
with a site-centred energy density:

    h_x = (J/2)(Z_{x-1} Z_x + Z_x Z_{x+1}) + g X_x + h Z_x ,   H = sum_x h_x ,
    J = 1,  g = -1.05 (transverse),  h = 0.5 (longitudinal).

The energy-density Fourier mode is `H_k = sum_x e^{ikx} h_x` with `k = 2 pi n / L`.
The figure uses `n = 1` (`q = k` in the paper notation).
`sso.dqt.model.mfim_hk` builds it as a sparse matrix in the kron basis. That matrix
agrees element by element (to 1e-15) with the QuSpin `spin_basis_1d(pauli=1)` matrices
of the original code, and `tests/dqt` checks this.

We use the normalized Hilbert–Schmidt product `<<A|B>> = 2^{-L} tr A^dag B`, which
equals `E_phi <phi|A^dag B|phi>` over Haar states `phi`. The Gaussian frequency filter
with width `s` is

    H_k(s) = exp(-s^2 ad_H^2 / 4) H_k = int dt g_s(t) e^{iHt} H_k e^{-iHt},
    g_s(t) = e^{-t^2/s^2} / (sqrt(pi) s)          (std s/sqrt2; FT e^{-s^2 Omega^2/4})

The three functionals are

    N1(s) = <<H_k(s)|H_k(s)>>,   N2(s) = <<H_k(s)|ad_H^2|H_k(s)>>,
    M(s)  = E_psi |<psi|H_k(s)|psi>|^2   (random product states psi: the RPS norm),

which give the frontier pair **tau_obs = sqrt(N1/N2)** and **nu = M/N1**.

## 2. Fourier-correlator estimator (production; `sso.dqt.fourier`)

* **N1, N2.** Two Gaussian windows convolve to a Gaussian of variance `s^2`, so
  `N1 = int G_{s^2}(tau) C(tau) dtau` and `N2 = int (-G''_{s^2})(tau) C(tau) dtau`.
  Here `G_v(tau) = e^{-tau^2/2v}/sqrt(2 pi v)` and `-G_v'' = G_v (v - tau^2)/v^2`
  (`sso.dqt.kernels`).
  The correlator `C(tau) = <<H_k|H_k(tau)>>` is estimated per normalized Haar state
  `phi` by `C(tau) = <gamma(tau)|H_k|alpha(tau)>`, where `alpha = e^{-iH tau} phi` and
  `gamma = e^{-iH tau} H_k phi`. It is computed on `tau = 0, dt, ..., T_c` with
  `T_c = nsigma * s_max`, and `C(-tau) = C(tau)^*`. The trapezoid rule on the
  symmetric grid gives one `N1`, `N2` per Haar sample, and the error is the SEM over
  samples.
* **M.** For each product state, the trajectory `m_psi(t) = <psi(t)|H_k|psi(t)>` is
  computed on `t in [-T_p, T_p]` with `T_p = nsigma * s_max / sqrt2`. The window
  integral `a_s(psi) = int g_s m_psi` equals `<psi|H_k(s)|psi>`.
  States have uniform Bloch vectors on every site.
* **Control variate** (not described in the older KrylovSize tex note).
  1. Project `H_k(s)` onto `span{H_k, ad_H^2 H_k}` in the HS metric:
     `c(s) = G_HS^{-1} (V0, V2)`, where `V0 = <<H_k|H_k(s)>> = int G_{s^2/2} C` and
     `V2 = <<ad^2 H_k|H_k(s)>> = int (-G''_{s^2/2}) C`. Both come from the same Haar
     curves. The odd rung `ad_H H_k` has zero overlap by time-reversal parity.
  2. The projection's expectation `p(psi) = c0 <psi|H_k|psi> + c2 <psi|ad^2 H_k|psi>`
     is computed exactly per state with sparse mat-vecs.
  3. Its RPS norm `E|p|^2 = c^T R c` is analytic. The 2x2 Grams `R` (RPS) and `G_HS`
     come from exact symbolic Pauli algebra for any `L` (`sso.dqt.pauli`).
  4. Then `M = mean(|a_s|^2 - |p|^2) + c^T R c`. This is unbiased for any `c`, and
     its variance is much lower at large `s`. The plain mean is kept as `M_direct`
     for diagnostics.
* **Propagator** (`sso.dqt.propagators`). The production engine is a Chebyshev
  expansion: `e^{-iHt} = e^{-ibt} sum_n (2-delta_n0)(-i)^n J_n(at) T_n((H-b)/a)`.
  It uses a padded Gershgorin bracket (`model.spectral_bracket`, pad 2 %). The order
  is set by the Bessel tail at tolerance 1e-14 (60 at dt = 0.5, L = 23). Coefficients
  are computed once for the fixed step `dt`.
  An adaptive Krylov/Arnoldi engine is kept as an alternative (`engine="krylov"`).
* **Seeds and pooling** (`sso.dqt.combine`).
  * Every seed draws its states from `RandomState(seed)`: all Haar states first,
    then all product states, exactly as in the original script. A given seed
    therefore reproduces the archived samples.
  * `combine_fourier` pools seeds whose s-grids are prefixes of the longest one. It
    rebuilds `c(s)` from all pooled Haar samples and re-applies the control variate.
    Errors are `tau_err = tau/2 sqrt((dN1/N1)^2 + (dN2/N2)^2)` and
    `nu_err = nu sqrt((dM/M)^2 + (dN1/N1)^2)`.
  * `pooled_correlators` gives `C(t)` and `R(t) = ||H_k(t)||^2_RPS = E|m_psi(t)|^2`,
    with `R` symmetrized over ±t.
  * `g_rps_window` gives `|G_RPS(t1,t2)| = |E m(t1)^* m(t2)|`, averaged per entry
    over the states that cover both times.

Cross-check tools, kept for validation only:

* `sso.dqt.exact`: full-ED `N1, N2, M`, with M from the exact 4^L Pauli transform,
  plus the analytic bare `||H_k||^2_RPS`.
* `sso.dqt.chebmoment`: the Chebyshev-moment estimator, which reconstructs all `s`
  from s-independent moment blocks and has no time grid.

## 3. Module map

| file | content |
|---|---|
| `sso/dqt/model.py` | `mfim_hk`, `spectral_bracket`, `haar_state`, `random_product_state`, `to_backend` |
| `sso/dqt/propagators.py` | `chebyshev_coeffs`, `chebyshev_expm_multiply`, `krylov_expm_multiply`, `adaptive_krylov_expm_multiply`, `Propagator` |
| `sso/dqt/kernels.py` | `gauss`, `neg_gpp`, `gwin` |
| `sso/dqt/pauli.py` | Pauli algebra, `rps_gram`, `hs_gram` |
| `sso/dqt/fourier.py` | `production_svals`, `time_grids`, `run_fourier_cv` (production estimator), `cv_estimate`, `seed_filename` |
| `sso/dqt/combine.py` | `seed_files`, `combine_fourier`, `pooled_correlators`, `g_rps_window` |
| `sso/dqt/exact.py` | `exact_reference`, `rps_norm2`, `to_pauli_coeffs`, `hk_rps_analytic` |
| `sso/dqt/chebmoment.py` | `cheb_vectors`, `moment_K/P`, `cheb_coeff_matrix`, `kernel_matrix`, `make_tau_grid`, `cheb_order_for`, `chebmoment_estimate` |
| `sso/dqt/figdata.py` | loaders for `data/fig3panel` |
| `sso/tiansatz.py` (appended) | q-scan analysis for panel (c): `discover_qs`, `load_q_curve`, `dnu_curve`, `mstab_family`, `log_ramp`, `dnu_tq2` |
| `scripts/compute/dqt/fourier_cv.py` | one seed → seed NPZ (`output_dir("dqt")`) |
| `scripts/compute/dqt/combine.py` | seeds → `*_COMBINED.npz` |
| `scripts/compute/dqt/validate_ed.py` | small-L DQT vs ED check (both engines, optional Chebyshev-moment) |
| `scripts/extract/dqt_fig3panel.py` | raw seeds + TI tables → `data/fig3panel/` (checks against archived COMBINED) |
| `scripts/figures/fig3panel.py` | the figure |
| `slurm/dqt/fourier_cv.slurm`, `slurm/dqt/campaigns.sh` | launcher + the full production campaign table |

Raw file names and NPZ keys are those of the original production runs:
`modhydro_fourier_cv_L{L}_n1_g-1.05_h+0.5_seed{S}.npz` and
`modhydro_fourier_L{L}_n1_g-1.05_h+0.5_COMBINED.npz`.

## 4. Provenance of fig3panel

* **(a)** `nu` vs `tau_obs/L^2` for L = 16..23, with an inset vs `tau_obs`. The
  fronts are pooled from every seed of the campaigns in `slurm/dqt/campaigns.sh`:
  167 seed files, 85 MB. The archive is
  `/scratch/gpfs/DABANIN/ty1475/KrylovSize/data_modhydro` on Della.
  All runs used J=1, dt=0.5, nsigma=6, the Chebyshev engine and cupy.
  The final s_max per L = 16..23 is 54, 62, 68, 76, 84, 88, 96, 104, with grid
  lengths 30, 34, 37, 41, 45, 47, 51, 55.
* **(b)** `C(t)/C(0)` and `sqrt(R(t)/R(0))` vs `t/L^2`. These use only the seeds that
  store raw curves (seeds ≥ 100; curve counts per L, product/Haar: 400/4, 400/8,
  400/8, 652/28, 648/56, 648/48, 448/56, 657/65). The inset is `|G_RPS|` at L = 20
  from seeds 200–207 and 300–307 (648 states), on |t| ≤ 0.15 L², normalized to its
  maximum.
* **(c)** `Delta nu = nu0 - nu(q)` vs `tau q^2` at span M = 11. This panel uses a
  **different model and geometry**: the thermodynamic-limit translation-invariant
  ansatz for the Kim–Huse chain (J=1, g=0.905, h=0.809, infinite chain), with q in
  rad/site. Panels (a,b) use the finite Model B ring.
  * Tables `ti_q_scan/nutau_M{8..11}_q{q}.dat`: q = 0 and the 14 values
    0.002…0.1. They come from the `julia/TIAnsatz` q-scan driver
    (`ham_mf_driver_q_gpures.jl`), which writes them to `$SSO_OUTPUT/ti/`
    (`sso.config.output_dir("ti")`). The originals are archived in `OGH/data`.
  * `nu0` is the same-M q=0 table interpolated with PCHIP in log tau.
  * The opacity is the M-stability ramp on `r = |dnu_11 - dnu_10|/|dnu_11|`:
    ramp 0.05–0.40, alpha_min 0.08, requiring ≥3 of M = 8..11 to cover the point.
  * No hard truncations are applied (the original "notrunc" mode).
  * The q = 0 baselines are not energy-projected: plateau nu_pl = 0.243518, against
    the analytic value 0.243492.

Regenerate:

```
bash slurm/dqt/campaigns.sh                       # (SUBMIT=1 to submit)  -> $SSO_OUTPUT/dqt
python scripts/compute/dqt/combine.py --L=16,17,18,19,20,21,22,23
python scripts/extract/dqt_fig3panel.py   # --raw defaults to $SSO_OUTPUT/dqt, --ti to $SSO_OUTPUT/ti
python scripts/figures/fig3panel.py
```

## 5. Cost and hardware

* Total ≈ 620 GPU-hours. Per-sample cost grows as ~2^L · s_max: at L = 23 and
  s_max = 104 it is ~1800 s per Haar sample and ~1550 s per product state on an A100.
* Memory is dominated by the two CSR operators: (L+1) 2^L entries each, ~4 GB at
  L = 23 including indices. A 10 GB MIG slice holds L ≤ 22; L = 23 ran on 40/80 GB
  A100s. Several families with L = 18–23 originally ran on full A100s for speed.
  `campaigns.sh` reschedules them on MIG where they fit (~7–8× the A100 walltime).
* CPU (numpy) works for every routine. It is practical up to L ≈ 14.

## 6. Validation

* **Reproduction.** A seeded rerun of L = 16, seed 400 (4 Haar, 8 product states,
  s_max = 54) on a MIG slice took 14.3 s/Haar and 12.2 s/state, against the archived
  13.7 / 11.8 s. It reproduces the archived per-sample arrays (`C_samples`,
  `m_samples`, `N1/N2/V0/V2/a/b0/b2_samples`) to ≤ 3e-14 relative.
* **Pooling.** Re-pooling the 167 archived seeds with `combine_fourier` matches the
  archived COMBINED files to ≤ 4e-15 relative, for all keys and all L.
* **Figure.** `fig3panel.pdf` rasterized at 300 dpi is pixel-identical to the
  original `KrylovSize/Plots/modhydro/fig3panel.pdf`.
* **Physics vs ED** (`scripts/compute/dqt/validate_ed.py`, L = 10, 40 Haar and 400
  product states, s = 0.7, 1.4, 2, 3, 4, 6, 8; Chebyshev engine).
  * Every N1, N2, M and nu agrees with ED within |z| ≤ 1.1 SEM, at both dt = 0.5 and
    dt = 0.1. Relative deviations are ≤ 1.1 %, consistent with the 2^{-L/2} DQT
    noise.
  * The control variate reduces the variance of M by factors of 760, 64, 26, 13, 9,
    6 and 5 (smallest to largest s).
  * The Chebyshev-moment estimator agrees with ED within |z| ≤ 1.5.
  * Inside the full pipeline (L = 10, dt = 0.5, same seed), the Chebyshev and Krylov
    engines give identical correlators and trajectories to 4e-14. Against `expm`
    they agree to < 1e-10 (`tests/dqt`, L = 7). The Krylov engine is >20x slower
    on CPU.
* `tests/dqt` (pytest, CPU, ~40 s; pytest comes with the `environment.yml` env) covers:
  * the model against QuSpin;
  * both engines against `expm`, unitarity and step composition;
  * Pauli Grams against dense 4^L transforms;
  * the analytic bare RPS norm;
  * Fourier and Chebyshev-moment estimators against ED at L = 8;
  * combine/pooling regressions;
  * the production s-grid.

## 7. Caveats

* **dt vs s_min.** The `N2` kernel `-G''_{s^2}` has width ~s, so the dt-grid
  quadrature fails once s ≲ dt. The deterministic quadrature bias was measured by
  applying the production grid to the exact L = 8 correlator, at dt = 0.5:

  | s | 0.3 | 0.5 | 0.7 (production s_min) | ≥ 1 |
  |---|---|---|---|---|
  | N2 relative bias | +63 % | +4e-4 | +3e-5 | ≤ 1e-6 |
  | N1 relative bias | 2e-3 | — | ≤ 1e-6 | ≤ 1e-6 |

  At dt = 0.25 the bias is < 1e-6 for all of these s. The production fronts
  (s ≥ 0.7) are therefore unaffected. Do not extend the s-grid below ~dt without
  reducing dt; use ED or the Chebyshev-moment path there instead.
* **Mixed revisions.** Seeds 0–11 of each L were produced by an earlier script
  revision, without `m_samples/C_samples/tau_h/tau_p`. The estimator was otherwise
  identical, and the current code produces a superset of their keys. They enter
  panel (a) only.
* **Unpickling.** The archived NPZ `parameters` entry is a pickled dict that needs
  numpy ≥ 2 (`allow_pickle=True`).
* **Finite-size floor.** `R(t)` saturates at an ETH floor ~N1/2^L, and `C(t)` at a
  DQT noise floor. Both floors are visible at late `t/L^2` in panel (b).
* **Gram validity.** `pauli.rps_gram` and `pauli.hs_gram` assume the range of
  ad^2 h_x (7 sites) is shorter than the ring. All production L qualify.
