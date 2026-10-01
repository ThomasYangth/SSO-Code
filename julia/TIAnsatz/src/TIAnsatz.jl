"""
    TIAnsatz

Thermodynamic-limit ν(τ) frontier from the translation-invariant (TI) finite-support
ansatz (Kim, Bañuls, Cirac, Hastings, Huse, PRE 92, 012128 (2015)).

The variational operator is `A = Σ_j e^{iqj} τ_j(O_M)` on an infinite chain, `O_M`
traceless Hermitian on M consecutive sites, `O_M = Σ_a c_a P_a` over the
`3·4^(M-1)` distinct stripped Pauli strings of span ≤ M (`canonical_basis`). Per site:

  * HS norm²  `c'c`                (Gram matrix = identity in this basis)
  * RPS weight `ν = c'Rc/c'c`,     `R = diag(3^{-wt})` (`build_R`)
  * slowness   `σ² = c'Kc/c'c`,    `K = K_q = L1_q†L1_q` (static, `build_L1_q`) or
                                   `K = K_F = 2I − T − Tᵀ` (kicked-Ising Floquet)

and the frontier at Lagrange multiplier θ is the top eigenvector of
`R c = μ (I + θK) c` (`frontier_sweep`), solved matrix-free by LOBPCG on CPU or GPU.
"""
module TIAnsatz

using ThermodynamicPauli
using LinearAlgebra, SparseArrays, Printf, Random
using CUDA, CUDA.CUSPARSE

export canonical_basis, build_R, kim_huse_H, energy_density_vector, complement_projector,
       build_K, build_K_sparse, build_L1, build_L1_q,
       heisenberg_conjugate_op, build_KF_sparse, DenseKF, apply_KF,
       nutau_sweep, lambda_TI, nutau_sweep_sparse, lambda_TI_sparse,
       lobpcg, power_lmax, chebyshev_precond, chebyshev_degree, error_bar, randn_vec,
       KHCpu, KHCpuC, KHGpu, KHGpuC, DenseKFGpu, apply_K, l1_bytes, gpu_bytes,
       use_gpu, to_device, seed_rng!, output_dir, qtag, theta_grid, parse_Mspec,
       frontier_sweep, DAT_COLUMNS

include("basis.jl")
include("kernels.jl")
include("exact.jl")
include("lobpcg.jl")
include("gpu.jl")
include("sweep.jl")

end # module
