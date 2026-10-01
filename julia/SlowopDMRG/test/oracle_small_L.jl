#=
Small-L oracle test: the ε-constrained two-site sweeper (:eps_constrained ×
:lobpcg, the production combination) against exact diagonalization.

At L = 6 the operator space is 4^6 = 4096-dimensional and a bond dimension of
4^3 = 64 is exact, so the only approximation left in the sweeper is the local
QCQP / LOBPCG tolerance. The oracle is the exact frontier on the
complement of {I, H, H², H³}: for a multiplier t the optimum of
⟨M⟩ − t⟨C_H²⟩ is a point (ε(t), ν(t)) of the exact ν-vs-⟨C_H²⟩ frontier.
We bisect t (dense LAPACK, robust at any t) until ε(t) equals the ε the
sweeper achieved and require the sweeper's ν to match the frontier there;
`exact_ground_state` (src/models/exact_solver.jl, the package's own small-L
solver) is checked against the same frontier point.

`run_dmrg_for_size` itself routes L ≤ 6 to `exact_ground_state`, so this test
calls `dmrg_sweeps!` directly with the same setup `_run_dmrg_core` uses.
=#

using SlowopDMRG
using ITensors
using ITensorMPS
using LinearAlgebra
using Random
using Test

const DT = SlowopDMRG.DTYPE
const NAIVE = Dict(:alg => "naive", :truncate => false)

function measure(psi, C_H, M_mpo)
    nrm   = real(inner(psi, psi))
    CHpsi = apply(C_H, psi; NAIVE...)
    return real(inner(CHpsi, CHpsi)) / nrm,
           real(inner(psi, apply(M_mpo, psi; NAIVE...))) / nrm
end

"""sin-mode warm start Σ_x (sin(πx/L) − c(L)) h_x, as in scripts/eps_ladder.jl."""
function sinmc_state(L, sites, H)
    k = π / L
    plus  = build_operator_mps(L, sites, H; k = DT(k))
    minus = build_operator_mps(L, sites, H; k = DT(-k))
    Hm    = build_operator_mps(L, sites, H; k = DT(0.0))
    psi = (minus - plus) * (1 / (2im)) - (cot(π / (2L)) / L) * Hm
    return complex(normalize(psi))
end

"""Dense column vector of an operator-space MPS (site order `sites`)."""
dense(psi::MPS, sites) = vec(Array(array(contract(psi), sites...)))

"""Dense matrix of an MPO acting on operator space (rows = primed sites)."""
dense(W::MPO, sites) = (n = 4^length(sites);
    reshape(Array(array(contract(W), prime.(sites)..., sites...)), n, n))

"""
Exact frontier from dense linear algebra on the complement of the tower:
`A = Pc' M Pc`, `B = Pc' C_H² Pc` (real symmetric in the Pauli basis). For a
multiplier t the optimum of ⟨M⟩ − t⟨C_H²⟩ is the top eigenvector of A − tB.
"""
struct DenseFrontier
    A::Matrix{Float64}
    B::Matrix{Float64}
end
function DenseFrontier(C_H, M_mpo, orthos, sites)
    C = dense(C_H, sites)
    Mx = dense(M_mpo, sites)
    T = hcat([dense(o, sites) for o in orthos]...)
    Fq = qr(T)
    Qfull = Matrix(Fq.Q * Matrix{ComplexF64}(I, size(T, 1), size(T, 1)))
    Pc = Qfull[:, size(T, 2)+1:end]                       # orthonormal complement
    A = Pc' * Mx * Pc
    B = Pc' * (C' * C) * Pc
    @assert norm(imag(A)) < 1e-10 * norm(A) && norm(imag(B)) < 1e-10 * norm(B)
    # the complement basis can be chosen complex; A, B are Hermitian. Keep the
    # real parts only when they are real (the tower is real in the Pauli basis).
    return DenseFrontier(Matrix(Symmetric(real(A))), Matrix(Symmetric(real(B))))
end
function point(F::DenseFrontier, t)
    E = eigen(Symmetric(F.A .- t .* F.B), size(F.A, 1):size(F.A, 1))
    x = E.vectors[:, 1]
    return dot(x, F.B * x), dot(x, F.A * x)                 # (ε, ν)
end
function nu_at(F::DenseFrontier, ε; lo = 1e-4, hi = 1e6)
    @assert point(F, lo)[1] > ε > point(F, hi)[1]
    t = e = ν = NaN
    for _ in 1:80
        t = sqrt(lo * hi)
        e, ν = point(F, t)
        abs(e - ε) / ε < 1e-9 && break
        e > ε ? (lo = t) : (hi = t)
    end
    return t, e, ν
end

@testset "L=6 :eps_constrained × :lobpcg sweeper vs exact frontier" begin
    Random.seed!(1234)
    L = 6
    H = Dict("ZZ" => 1.0, "X" => 0.905, "Z" => 0.809)
    sites = siteinds("Pauli", L)
    C_H, _, H_L, _ = build_hamiltonian_operators(L, sites, H; pbc = false)
    M_mpo = MPO(sites, "M")
    orthos = build_krylov_basis_raw(H_L, sites, 4)[1:4]          # {I, H, H², H³}
    F = DenseFrontier(C_H, M_mpo, orthos, sites)

    psi = sinmc_state(L, sites, H)
    eps0, nu0 = measure(psi, C_H, M_mpo)
    @info "warm start" eps0 nu0

    tsq_out = Ref{DT}(1.0)
    ema = Ref{DT}(1.0)
    # Same ladder as production, εᵢ = ε₀·(1/√3)^i. Default 6 steps (down to
    # ε₀/27 ≈ 0.04, multiplier ≈ 0.5); SLOWOP_ORACLE_STEPS=12 goes to ε₀/729
    # (multiplier ≈ 20) in ~15 more minutes.
    for step in 1:parse(Int, get(ENV, "SLOWOP_ORACLE_STEPS", "6"))
        ε = eps0 * (1 / sqrt(3))^step
        sw = Sweeps(8)
        setmaxdim!(sw, 64)
        setcutoff!(sw, 0)
        setnoise!(sw, 1e-6, 1e-8, 1e-10, 0.0)
        sweep_time = @elapsed _, psi, _ = dmrg_sweeps!(psi, C_H, M_mpo, ema[], sw;
            dispon = 0,
            orthogonalize_states = orthos,
            subspace_coeffs = (DT(1.0), DT(1.0)),
            eig_kwargs = Dict(:nev => 1, :krylovdim => 30),
            scheduling = :eps_constrained, local_solver = :lobpcg,
            epsilon_loc = DT(ε),
            eps_anneal_sweeps = 1, eps_anneal_start = DT(eps0 * (1 / sqrt(3))^(step - 1)),
            thetasq_out = tsq_out, qcqp_thetasq_ema = ema,
            qcqp_theta_sq_max = DT(1e14), qcqp_bisect_tol = 5e-3,
            qcqp_warm_accept_tol = 1e-1, qcqp_warm_accept_tol_min = 5e-3)

        eps_d, nu_d = measure(psi, C_H, M_mpo)
        ovl = maximum(abs(inner(o, psi)) / norm(o) / norm(psi) for o in orthos)
        t_x, eps_x, nu_x = nu_at(F, eps_d)
        # exact_solver.jl at the same multiplier must land on the same point
        # (it minimizes t·C_H² − M with a KrylovKit eigensolve; t is moderate).
        _, gs_e, conv_e = exact_ground_state(C_H, M_mpo, DT(t_x); orthos = orthos, ortho_weight = 1.0)
        eps_e, nu_e = measure(gs_e, C_H, M_mpo)
        @info "step $step" ε eps_d nu_d t_x eps_x nu_x rel_dnu = (nu_d - nu_x) / nu_x eps_e nu_e tower_overlap = ovl sweep_time
        @test abs(eps_d - ε) / ε < 0.05            # constraint pinned (bisect/warm tol)
        @test ovl < 1e-6                           # stays ⊥ {I, H, H², H³}
        @test nu_d <= nu_x * (1 + 1e-6)            # feasible point cannot beat the optimum
        @test (nu_x - nu_d) / nu_x < 2e-3          # and reaches it at exact χ
        @test conv_e >= 1
        @test isapprox(nu_e, nu_x; rtol = 1e-6) && isapprox(eps_e, eps_x; rtol = 1e-5)
    end
end
