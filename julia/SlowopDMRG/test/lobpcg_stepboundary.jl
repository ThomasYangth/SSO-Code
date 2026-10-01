#=
Regression test for the LOBPCG CholQR `PosDefException` at ε-step boundaries.

This is the failure that took down job 12900383 (variant A, `sinmc`): step 1
pinned ⟨C_H²⟩ to its target at 0.02 %, then the *first block of step 2's first
sweep* threw

    ERROR: PosDefException: matrix is not positive definite;
      Cholesky factorization failed.

from `sygvd!` inside `IterativeSolvers.lobpcg`, which reduces the small pencil
`G_A ξ = μ G_B ξ` via a Cholesky of `G_B = [X R P]' B [X R P]`.

Mechanism: at a step boundary the warm-start `X0` is
the *previous* step's converged Ritz vector, and θ² carries over from the
previous step, so the block operator is essentially unchanged. LOBPCG iteration
1 therefore has

    col 1 (X0): norm 1
    col 2 (R) : norm ~1e-14, pure round-off
    col 3 (P) : exactly 0

and a tight Chebyshev preconditioner maps `R` back toward `X0` rather than into
orthogonal content, so `G_B ≈ diag(1, 1e-28, 0)` and the Cholesky trips.

`test/lobpcg_smoke.jl` cannot catch this: it is a single cold-start run, so the
warm start is never a converged eigenvector and the retry path is never
executed. Hence this separate test.

What is exercised: the two fixes (original repo commit 3bf14eb) — the
3-attempt perturbation retry in `_call_lobpcg_inner`, and
`setmindim!(sweeps, warm_chi)` in `_run_dmrg_core`. Part of test/runtests.jl.

NOTE ON VALIDITY: a regression test only means something if it fails without
the fix. Set `SLOWOP_STEPBOUNDARY_EXPECT_CRASH=1` in the environment and run
this against a tree with the retry hunk reverted; it should then report the
PosDefException rather than passing. If it passes there too, the L=7 block
dimension is too small to reproduce the production mechanism and this test is
not load-bearing — say so rather than trusting it.
=#

using SlowopDMRG
using ITensors
using ITensorMPS
using LinearAlgebra
using Test

const DT = SlowopDMRG.DTYPE

const L        = 7
const H_DICT   = Dict("ZZ" => 1.0, "X" => 0.905, "Z" => 0.809)
const EPS_1    = DT(0.05)
const EPS_2    = DT(0.05 / sqrt(3))     # same 1/√3 ladder as production
const NAIVE    = Dict(:alg => "naive", :truncate => false)

"""Global (⟨C_H²⟩, ⟨M⟩) for `psi`, measured off the full chain."""
function measure(psi, C_H, M_mpo)
    nrm   = real(inner(psi, psi))
    CHpsi = apply(C_H, psi; NAIVE...)
    return real(inner(CHpsi, CHpsi)) / nrm,
           real(inner(psi, apply(M_mpo, psi; NAIVE...))) / nrm
end

@testset "LOBPCG survives an ε-step boundary (CholQR regression)" begin

    # ---- Step 1: cold start. The warm start handed to step 2 must be a
    # genuine near-eigenvector — that is what makes the pencil collapse; a
    # sloppily-converged step 1 would not reproduce the bug. 2 sweeps is
    # enough: at L=7 the smoke test lands ⟨C_H²⟩ within 0.06 % of target in 2
    # sweeps, with per-block LOBPCG residuals at ~1e-14.
    # Do NOT read the runtime of step 1 as a predictor of step 2's: step 2 has
    # tighter ε, hence larger θ² and worse κ(B).
    psi1 = run_dmrg_for_size(L, H_DICT;
                             theta                = DT(1.0),
                             num_ortho            = 1,
                             max_iter             = 1,
                             sweeps_per_iteration = 2,
                             initial_noise        = 1e-6,
                             final_noise          = 1e-10,
                             scheduling           = :eps_constrained,
                             local_solver         = :lobpcg,
                             epsilon_loc          = EPS_1,
                             override             = true)

    sites = siteinds(psi1)
    C_H, _, _, _ = build_hamiltonian_operators(L, sites, H_DICT; pbc = false)
    M_mpo = MPO(sites, "M")

    eps1, nu1 = measure(psi1, C_H, M_mpo)
    @info "step 1" eps_target=EPS_1 eps_achieved=eps1 nu=nu1 chi=maxlinkdim(psi1)
    @test eps1 > 0

    # ---- Step 2: warm-start from psi1. This is the exact production
    # configuration that crashed — psi0_in is a converged state and θ² carries
    # over, so the first block's X0 is a near-exact eigenvector of an almost
    # unchanged operator.
    local psi2
    crashed = false
    try
        psi2 = run_dmrg_for_size(L, H_DICT;
                                 psi0_in              = psi1,
                                 theta                = DT(1.0),
                                 num_ortho            = 1,
                                 max_iter             = 1,
                                 sweeps_per_iteration = 2,
                                 initial_noise        = 1e-6,
                                 final_noise          = 1e-10,
                                 scheduling           = :eps_constrained,
                                 local_solver         = :lobpcg,
                                 epsilon_loc          = EPS_2,
                                 override             = true)
    catch e
        crashed = true
        @error "step 2 threw — the CholQR retry did not recover" exception=(e, catch_backtrace())
        rethrow(e)
    end

    @test !crashed

    eps2, nu2 = measure(psi2, C_H, M_mpo)
    @info "step 2" eps_target=EPS_2 eps_achieved=eps2 nu=nu2 chi=maxlinkdim(psi2)

    # The constraint must actually tighten. Loose bracket here: 4 sweeps at
    # L=7 is a smoke-grade budget, not a converged production run.
    @test eps2 < eps1
    @test 0 < nu2 < 1.0

    # setmindim!(sweeps, warm_chi): step 2 must not throw away the bond
    # dimension step 1 built up. This is the second half of commit 3bf14eb and
    # is otherwise silently untested.
    @test maxlinkdim(psi2) >= maxlinkdim(psi1)
end
