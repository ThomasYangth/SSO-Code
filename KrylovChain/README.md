# KrylovChain — ν(τ) frontier of the su(1,1) Krylov-chain model

A self-contained linear-algebra computation (numpy/scipy; mpmath only for a test).
It is not imported by the `sso` package. The notation follows the paper appendix,
"Solution of the Krylov chain model".

## Model

* **Chain.** A Krylov chain O₀, O₁, … with 𝓛 = [H,·] acting as
  𝓛_{n−1,n} = 𝓛_{n,n−1} = b_n = √(n(n+1)).
* **Polynomials and density.** q_k are monic of degree k, with q₀ = 1 and
  q_{k+1} = ω q_k − b_k² q_{k−1}. B_k = ∏_{l≤k} b_l. The spectral density is
  ρ(ω) = ω/(2 sinh(πω/2)).
* **Norms.** For O = Σ c_k O_k:
  * ‖O‖² = cᵀc
  * ‖[H,O]‖² = cᵀ𝓛²c
  * ‖O‖²_μ = Σ e^{−μ(k+1)} c_k², i.e. 𝓜_μ = diag(e^{−μ(k+1)})
* **Kernels.** μ = log 3 is the EVN/RPS kernel. μ = ∞ is the limit of e^{μ}𝓜_μ, so
  𝓜_∞ = |0⟩⟨0|.
* **Frontier.** The frontier operator solves 𝓜_μ c = λ(1 + θ𝓛²)c.

## Method (`krylovchain.py`)

1. **Reduced eigenproblem.** With 𝓡 = (1+θ𝓛²)⁻¹ and c̃ = √𝓜_μ c/λ, solve
   √𝓜 𝓡 √𝓜 c̃ = λ c̃ in the first N sites, giving c̃_N (`solve_frontier`).
2. **Matrix elements by ω-integrals.** (g(𝓛))_mn = (B_m B_n)⁻¹ ∫ q_m q_n g ρ dω for
   g = 1/(1+θω²) (𝓡), 1/(1+θω²)² (𝓡²) and ω²/(1+θω²)² (𝓡²𝓛²). These are
   `matrix_elements`. 𝓡 is never obtained by inverting a truncated 𝓛. Gauss
   quadrature from a truncated Jacobi matrix is avoided too, since it equals that
   inverse.
3. **Recovering the operator.** c = 𝓡√𝓜 c̃_N on all sites. Then:
   * ‖O‖² = c̃_Nᵀ√𝓜 𝓡² √𝓜 c̃_N
   * ‖[H,O]‖² = c̃_Nᵀ√𝓜 𝓡²𝓛² √𝓜 c̃_N
   * ‖O‖²_μ ≈ λ² c̃_Nᵀc̃_N

   This is the default, `mu_norm="truncated"`, the notes' 𝓡 → 𝓡_N in the 𝓡𝓜𝓡 form.
   Then ν = ‖O‖²_μ/‖O‖² and τ = ‖O‖/‖[H,O]‖.
4. **Exact μ-norm option.** `mu_norm="exact"` sums ‖O‖²_μ = Σ_{k<K} e^{−μ(k+1)} c_k² to
   K = N + ⌈45/μ⌉, using the K×N block of 𝓡. Then (τ, ν) is exact for the operator c.
   * The truncated form drops a positive tail of relative size at most about
     0.4·3^{−N}, for any θ, so it always underestimates ν.
   * At large τ this is the dominant truncation error.
   * At N = 60 the two forms agree to 1.1e-15.
5. **μ = ∞.** The reduced problem is 1×1 for every N: c̃ = e₀, c = 𝓡e₀, the Lorentzian
   1/(1+θω²) on O₀. The result is independent of N and equals `closed_form_mu_inf`:
   * ν_θ = [∫ρ/(1+θω²)]²/∫ρ/(1+θω²)²
   * τ_θ = (∫ρ/(1+θω²)² / ∫ω²ρ/(1+θω²)²)^{1/2}

**Quadrature.** The integrals use composite Gauss–Legendre, with panels graded at the
pole scale 1/√θ, on [0, 3K + 60]. Parity (−1)^k is used. √ρ·q_k/B_k is generated
directly by the recursion, so nothing overflows.

## Files

```
krylovchain.py          b, rho, sqrt_rho_p, M_diag, quadrature_grid, function_of_L,
                        matrix_elements, solve_frontier, closed_form_mu_inf, lorentzian_filter
run_frontier.py         theta scan (0, 1e-3..1e8): data/frontier_mulog3_N60.npz,
                        data/closed_form_muinf.npz
plot_frontier_evn.py    plot 1 -> figures/frontier_mulog3.{pdf,png}
plot_truncation.py      plot 2 -> figures/truncation_mulog3.{pdf,png} (+ data/truncation_mulog3.npz)
test_krylovchain.py     checks
```

```
python test_krylovchain.py      # ~3 s
python run_frontier.py          # ~5 s
python plot_frontier_evn.py
python plot_truncation.py       # ~15 s (root-finds theta per (tau, N))
```

## Checks (`test_krylovchain.py`)

| check | result |
|---|---|
| ∫p_m p_n ρ = δ and ∫ω p_m p_n ρ = 𝓛_mn, K = 60 and 200 | ≤ 1e-11, ≤ 1e-9 |
| θ = 0: λ = ν = 1/3, τ = 1/√2 (both μ-norm forms) | exact |
| `closed_form_mu_inf` vs mpmath (30 digits), θ = 0.1 … 1e6; `solve_frontier(∞, θ, N)` for N = 1, 2, 5, 20 | 1e-11; 1e-12 |
| Gauss order 24 → 48 and margin 60 → 120 | ≤ 1e-12 |
| N = 40 vs 60, both forms; dropped μ-norm tail 0 < tail < 0.4·3^{−N} | ≤ 1e-12 |
| exact-form truncated solutions (N = 1, 3, 8) lie on or below the converged frontier at equal τ | holds |
| ν τ → 2 artanh(e^{−μ}): ln 2 at μ = log 3, 2 at μ = ∞ | 2e-6 at θ = 1e12 |

## Results

* **Plot 1** (`figures/frontier_mulog3.png`): the μ = log 3 frontier at N = 60. ν falls
  from 1/3 at τ = 1/√2 to ln 2/τ at large τ.
* **Plot 2** (`figures/truncation_mulog3.png`), using the notes' method:
  * The relative error |ν_N/ν₆₀ − 1| at fixed τ decays as about 3^{−N}.
  * Pairs N, N + 1 coincide, because 𝓡 only couples sites of equal parity.
  * The rate is the same for every τ, and the error reaches the 1e-15 floor by
    N ≈ 27. For τ ≥ 10 the error is nearly τ-independent; τ = 1 converges faster.
  * At N = 1 (the Lorentzian on O₀ with the truncated μ-norm) the error is 4e-3 at
    τ = 1 and 3.8e-2 for τ ≥ 100.
* **The explicit Lorentzian.** 1/(1+θω²) on O₀ with the full μ = log 3 norm
  (`lorentzian_filter`) is within 1.5e-4 of the frontier at equal τ.
