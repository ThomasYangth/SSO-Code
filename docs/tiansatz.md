# Thermodynamic-limit frontier: the translation-invariant (TI) ansatz

Code: `julia/TIAnsatz/` (Julia 1.10, CUDA.jl 6.3.1). Launchers: `slurm/ti/`.
Reader: `sso.tiansatz.read_ti_table`.

## 1. Ansatz

On an infinite chain we optimize over momentum-q operators built from one local
operator supported on M consecutive sites,

    A = Σ_j e^{iqj} τ_j(O_M),     O_M = Σ_a c_a P_a ,

where the P_a run over the **distinct stripped Pauli strings of span ≤ M**
(no leading/trailing identities; first and last site non-identity). There are
`3·4^(M−1)` of them (M = 8: 49 152; M = 11: 3 145 728; M = 12: 12 582 912). In this
basis different translates are never double counted, so all quadratic forms are
per-site and the Hilbert–Schmidt Gram matrix is the identity:

| quantity | form | meaning |
|---|---|---|
| ‖A‖² per site | `c†c` | HS norm |
| ν | `c†Rc / c†c`, `R = diag(3^{−wt(P_a)})` | RPS weight (infinite-temperature random-product-state weight; wt = number of non-identity sites) |
| σ² (static) | `c†K_q c / c†c` | per-site ‖[H, A]‖² / ‖A‖² |
| σ² (Floquet) | `c†K_F c / c†c` | per-site ‖U_F A U_F† − A‖² / ‖A‖² |

`tau = 1/√σ²` is written to the tables; the paper's Floquet time convention
(chord τ = 1/(2 arcsin(√σ²/2))) is applied on the Python side.

The Pauli algebra and the translation-folded commutator rule are in the local
package `ThermodynamicPauli` (stripped 2-bit-per-site `UInt128` keys; `Liouvillian(H)`
is the update rule of `[H,·]`; `apply_rule` applies it at every window position and
re-strips, so the coefficients it returns are already translation sums).

### Static kernel K_q

H = Σ_i (J Z_iZ_{i+1} + g X_i + h Z_i); Kim–Huse point (J, g, h) = (1, 0.905, 0.809).
`L1_q : span ≤ M → span ≤ M+1` is the single-commutator map with the translation
offset δ of each output term phased in,

    L1_q[c, b] = Σ_δ ([H, P_b] at offset δ)[P_c] · e^{−iqδ} ,

(δ = leftmost non-identity site of the output minus that of P_b; the commutator is
taken on an explicit width-(M+4) canvas so δ survives), and

    K_q = L1_q† L1_q ,      σ²(q) = ‖L1_q c‖² / ‖c‖² .

At q = 0, `L1_q = L1` (`build_L1`, purely imaginary, via `apply_rule`) and
`K = L1†L1` equals the double-commutator kernel `K[a,b] = ([H,[H,P_b]])[P_a]`
(`build_K`). These identities and the momentum-q convention are pinned by tests
against a finite periodic ring (2^L exact diagonalization, L = 2M + 4, q = 2πk/L).
K_q is never assembled: one apply is two sparse mat-vecs with L1_q (13–18 nnz per
column; M = 11: 55 M nnz, 1.0 GB complex on the GPU).

### Floquet kernel K_F

Kicked Ising U_F = e^{−iτ_F H_x} e^{−iτ_F H_z}, H_x = Σ g X_i,
H_z = Σ (h Z_i + Z_iZ_{i+1}) (coupling J = 1). Production point of the paper:
τ_F = 1, g = 0.9, h = 0.809 (file signature `g0p9tF1p0`).

    K_F = 2I − T − Tᵀ ,     T[a,b] = (U_F P_b U_F†)[P_a] .

K_F is real symmetric PSD. It is applied matrix-free on the dense canvas of all 4^W
Hermitian Paulis of W = M + 2 sites (the one-period light cone has radius 1): lift
c to the canvas, conjugate by the 3W − 1 commuting one- and two-site gates of U_F
(each an exact rotation of an anticommuting Pauli pair, `cos2α ψ[j] − sin2α ε ψ[j ⊻ mask]`),
fold back onto the span-≤M basis, and the same with U_F† for Tᵀ. On the GPU each
gate is one CUDA kernel and the fold an atomic scatter-add. Canvas memory
≈ 20·4^(M+2) bytes: M = 12 ≈ 5.1 GB (fits a 10 GB MIG slice); M = 13 ≈ 20 GB (full
A100); the Int32 fold map caps M ≤ 13. The matrix-free apply is tested against the
assembled `build_KF_sparse` (exact Heisenberg conjugation of each P_b), which is
itself tested against finite-ring ED.

## 2. Frontier problem and solver

Maximizing ν at fixed σ² with Lagrange multiplier θ gives

    R c = μ (I + θK) c      (largest μ),

and sweeping θ traces ν(σ²). Each θ is solved as the smallest eigenpair of the
Hermitian pencil (−R, I + θK):

- **LOBPCG**, block size 1, all full-length vectors on the device (only the ≤3×3
  Rayleigh–Ritz problem on the host). Stop when ‖Ax − λBx‖ ≤ TOL·(|λ|‖Bx‖ + ‖Ax‖),
  TOL = 1e-3, maxiter 500; if the Gram matrix of [X, W, P] becomes singular the
  conjugate direction P is dropped for that step.
- **Preconditioner**: degree-k Chebyshev semi-iteration for (I + θK)⁻¹ on [1, β],
  β = 1 + 1.2 θ λ_max(K) (λ_max from 40 power iterations), k = clamp(⌈2 + log10 β⌉, 2, 8).
- **Warm start**: θ is swept upward and each solve starts from the previous θ's
  eigenvector; the first start is a random Gaussian vector (see caveats).
- **θ grid**: `10 .^ range(−1.5, θmax_log; length=nθ)`.
- **Error bar** (column `nu_bar`) and gate (column `accepted`): with λ̄ = ν/(1+θσ²),
  μ̃ = λ̄θ and M = R − μ̃K, the generalized residual is an ordinary residual of M,
  ε = ‖Mv − λ̄v‖. Weak duality gives 0 ≤ a(σ²) − ν ≤ λ_max(M) − ρ (ρ = v†Mv, a the
  exact frontier at the attained σ²), and Kato–Temple bounds that by
  `nu_bar = min(ε, ε²/δ)`, δ = λ_max(M) − λ₂(M). The point is `accepted` iff
  ρ > λ₂(M) (v lies in the top eigenpair's basin); λ₂(M) comes from a 15-step
  Lanczos with full reorthogonalization from a random start. All 2108 archived
  points behind the two figures (70 tables) are accepted, with nu_bar ≤ 4.1e-5.

The same solver runs on host arrays (`SSO_TI_BACKEND=cpu`; used by the tests and
as a fallback) — the code path is identical, only the array type changes.

### Energy density at q = 0

At q = 0 the energy density c_H ∝ (J·ZZ + g·X + h·Z) is an exact zero mode of K
(the conserved H). Two q = 0 variants exist:

1. **Unprojected** (`ham_q_scan.jl` with q = 0): the baseline of the q scan. Its θ→∞
   end converges to c_H itself, whose RPS weight is
   ν(c_H) = (h²/3 + g²/3 + J²/9)/(h² + g² + J²) = 0.243492 at the Kim–Huse point.
   The archived M = 11 plateau at θ = 10⁴ is ν = 0.243518 (σ² = 5.3e-9): the
   2.6e-5 excess is the finite-θ admixture of faster, heavier-weight operators, not
   a solver error.
2. **Projected** (`ham_q0_projected_gpu.jl`, `ham_q0_projected.jl`): c_H is
   projected out (P = I − ĉĉᵀ on A, B and the preconditioner, B extended by the
   identity on ĉ) — the standard PRE15 q = 0 curve, whose θ→∞ end is
   τ_max(M) = 1/√λ_TI(M), λ_TI = min σ² ⟂ c_H (M = 5..9: λ_TI = 0.0582, 0.0309,
   0.0193, 0.01296, 0.00917, matching the PRE15 appendix to ≤ 2 %). The direct
   variant uses dense LAPACK (M ≤ 7) or sparse Cholesky + power iteration with an
   exact bordered constraint solve (M ≤ 9; M = 9 needed ~300 GB / 5 h) and also
   writes the λ_TI endpoint. Not used by the paper's figures; retained.

## 3. Provenance of figure data

All production runs: Princeton Della, MIG 1g.10gb slices (`-c 1`), Julia 1.10.2,
CUDA.jl 6.3.1, CUDA runtime 12.9, TOL = 1e-3. Original code: `~/OGH/nutau/lobpcg/`
(`ham_mf_driver_q_gpures.jl`, `floquet_mf_driver_gpures.jl`), archived tables in
`~/OGH/data/`. File names are unchanged.

**fig3panel (c)** — Kim–Huse, static, `nutau_M{M}_q{q}.dat`, M = 8..11,
q ∈ {0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, …, 0.10}
(`slurm/ti/reproduce_fig3c.sh`):

| family | θ grid | original job | wall (per task) |
|---|---|---|---|
| M = 8–10, q = 0.002–0.008 | 20 pts, 10^−1.5..10^3 | 13437392 (array 0–3; M = 11 of that job later overwritten) | M = 8: 9 s, 9: 13 s, 10: 63 s sweep |
| M = 11, all 14 q ≠ 0 | 30 pts, ..10^4 | 13443841 (array 0–13) | ≈ 13 min (L1 build 80 s, sweep ≈ 10 min) |
| M = 8, 9, 10, q = 0.01–0.10 | 30 pts, ..10^4 | 13620982 / 13620983 / 13620984 | 1–2 min (M = 10: sweep 114 s) |
| q = 0 baselines (unprojected), M = 8–10 / M = 11 | 80 pts, ..10^4 | interactive srun 13444112 / 13443806 (no logs) | 6 min / 26 min |

**floquet_two_panel (b)**, dashed "TI ansatz" — `nutau_floquet_M{M}_g0p9tF1p0_gpures.dat`,
M = 6..12, τ_F = 1, g = 0.9, h = 0.809, 24 θ to 10³ (`slurm/ti/reproduce_floquet.sh`;
original array 13659320, 2026-09-09). Sweep wall time: M ≤ 8 ≈ 10 s, M = 9: 31 s,
10: 122 s, 11: 538 s, 12: 2421 s (+ 100 s canvas build).

## 4. Validation of this port (2026-10-01)

The cleaned code was rerun with the launchers in `slurm/ti/` (MIG 1g.10gb unless
noted) and compared row by row (same θ grid) with the archived tables:

| run | rows | max \|Δν\|/ν | max \|Δσ²\|/σ² | max \|Δν\|/ν, θ ≥ 3rd point | accepted |
|---|---|---|---|---|---|
| Kim–Huse M = 8, q = 0 (80 θ) | 80 | 7.4e-5 | 1.9e-3 | 5.0e-6 | 80/80 |
| M = 8, q = 0.002 (20 θ) | 20 | 1.7e-4 | 4.5e-3 | 6.1e-8 | 20/20 |
| M = 8, q = 0.05 (30 θ) | 30 | 4.8e-5 | 1.2e-3 | 7.2e-8 | 30/30 |
| M = 8, q = 0.10 (30 θ) | 30 | 1.1e-5 | 3.3e-4 | 1.7e-7 | 30/30 |
| Floquet M = 6 | 24 | 1.3e-5 | 2.6e-3 | 1.3e-5 | 24/24 |
| Floquet M = 7 | 24 | 4.8e-5 | 1.2e-3 | 4.8e-5 | 24/24 |
| Floquet M = 8 | 24 | 1.1e-5 | 1.2e-3 | 1.1e-5 | 24/24 |
| Floquet M = 6, CPU backend | 24 | 1.9e-6 | 2.8e-3 | 1.1e-6 | 24/24 |
| projected q = 0, M = 10 (`nutau_M10_gpures.dat`) | 20 | 3.3e-6 | 3.0e-5 | 2.8e-7 | 20/20 |
| projected q = 0 dense, M = 1, 4, 5, 6 (CPU) | 41 | 0 | 3e-12 | 0 | — |

The large first-row deviations are at θ = 10^−1.5, where the frontier is flat:
from a cold random start LOBPCG stops (TOL = 1e-3) at a slightly different point
*along the same curve*. Correcting for the frontier slope
(Δν − μ̃Δσ², μ̃ = θν/(1+θσ²)) the residual mismatch is ≤ 5e-6 relative for all
static runs and ≤ 9e-5 for Floquet, i.e. of the order of the archived and new
`nu_bar` combined (the archived Floquet M = 6, 7 tables carry nu_bar up to
1.7e-5 at small θ; the new values are higher, as a better-converged lower bound
should be). The test suite passes on CPU (300 tests, `slurm/ti/test_cpu.sh`) and
on a MIG slice including the GPU tests (311 tests, `slurm/ti/test_gpu.sh`).
Not rerun: M = 9–11 static and Floquet M = 9–12 (same code path; only larger).

## 5. Caveats

- **Random starts.** The first LOBPCG start, the power-iteration start and each
  Lanczos start are drawn from the unseeded (CUDA) RNG, as in production, so
  reruns reproduce the archive only to solver tolerance, not bitwise. Set
  `SSO_SEED=<int>` for run-to-run reproducibility with a given device/library stack.
- **σ² at small θ** is determined only to ~TOL (flat frontier); ν is accurate there.
- **q = 0 baseline is unprojected** (Sec. 2): its large-θ end is the conserved
  energy density, not a slow non-conserved operator.
- **Finite M.** All quantities are exact in the thermodynamic limit *for the
  span-M ansatz*; M-convergence (M = 8..11 static, 6..12 Floquet) is part of the
  figure, not of the solver.
- **GPU memory.** Static: L1_q complex CSC ≈ 20 bytes × nnz (M = 11: 1.0 GB;
  M = 13 real projected: 11.7 GB → gpu40). Floquet: Sec. 1.
- **Precompilation.** With Julia 1.10.2 + CUDA.jl 6.3.1 the CUDA subpackages hit a
  circular-dependency precompile skip; they compile at load (~1.5 min per job
  start). This matches the production environment and does not affect results.
