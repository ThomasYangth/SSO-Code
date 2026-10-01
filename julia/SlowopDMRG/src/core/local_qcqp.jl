#=
===============================================================================
LOCAL QCQP SOLVER (ε-constrained scheduling)
===============================================================================

At each two-site DMRG block, solve

    max ⟨x | Ã | x⟩   s.t.   ⟨x | B̃ | x⟩ ≤ ε_loc,   ‖x‖ = 1

where Ã is the M-environment action (the concentration operator) and B̃ is the
factored two-layer C_H² action from the un-squared envs (see
`apply_Heff_local!` in dmrg_sweeper.jl).

Algorithm: dual scan / bisection on the Lagrange multiplier of the QCQP.
The KKT stationary point satisfies (Ã − θ² B̃) x = ν x, and we adjust θ² so
that ⟨x_{θ²} | B̃ | x_{θ²}⟩ = ε_loc whenever the constraint is active.

Naming convention (matches the rest of the codebase): the Lagrange multiplier
here is called `thetasq` throughout — it plays the same role as, and in fact
equals, the `thetasq = θ²` that the `:fixed_theta` driver uses as a fixed
scalar. Under ε-constrained scheduling this θ² "floats along the sweep"
(varies block to block) rather than being pinned globally.

Requires factored envs (LH/RH must be 4-index tensors sandwiching two C_H
layers).
===============================================================================
=#

"""
    apply_CH2_local(psi_vec, psiinds, LH, RH, CH_p, CH_p1, Ct_p, Ct_p1) -> Vector

Bare two-layer factored action of `C†C` on a local two-site block, without
orthogonalization or subspace-coefficient bookkeeping. Used inside the QCQP dual
scan to measure ⟨x | B̃ | x⟩ separately from the combined matvec, and by
`local_step_geneig` for the B branch of its generalized eigenproblem.

`CH_p, CH_p1` are the bottom layer (prime level 0→1) and `Ct_p, Ct_p1` the top
layer (1→2), the latter being the adjoint of the former. On the Hamiltonian path
the caller passes the same tensors twice, since `C_H = H_L − H_R` is Hermitian and
`C_H · C_H = C_H† C_H`. On the Floquet path the top layer must be
`adjoint_mpo(C_U)`, because `C_U² ≠ C_U† C_U` for the non-Hermitian `C_U = 𝒰 − 𝟙`.
"""
function apply_CH2_local(psi_vec, psiinds, LH, RH, CH_p, CH_p1, Ct_p, Ct_p1)
    # `psi_vec` arrives from LOBPCG's CPU workspace; the environment tensors live
    # on whatever device the sweep runs on. `movedevice` puts the block tensor on
    # that device (no-op when GPU is off), so the contraction is single-device,
    # and `cpuarray` brings the result back so LOBPCG's CPU bookkeeping and the
    # raw `dot` measurements stay on the host. Both are true no-ops on the CPU
    # path, so it is byte-identical there.
    psiT = movedevice(ITensor(psi_vec, psiinds...))
    tmp = LH * psiT
    tmp = tmp * CH_p            # bottom layer, site p:  σ_p(0) → σ_p(1)
    tmp = tmp * CH_p1           # bottom layer, site p+1
    tmp = tmp * prime(Ct_p)     # top layer,    site p:  σ_p(1) → σ_p(2)
    tmp = tmp * prime(Ct_p1)    # top layer,    site p+1
    tmp = tmp * RH
    result = replaceprime(tmp, 2 => 0)
    return vec(cpuarray(array(result, psiinds...)))
end

# ==========================================================================
# LOBPCG helpers: matrix-free operator wrapper + Chebyshev-in-B preconditioner
# + power iteration for λ_max(L²).
#
# These support the `:lobpcg` inner primitive in `local_step_qcqp`.
# `IterativeSolvers.lobpcg` accepts A, B, and P via duck-typed multiplication
# and left-division; we build MatFreeOp for the matvec side and ChebyBInv for
# the preconditioner side.
# ==========================================================================

"""
    MatFreeOp{F,T}

Matrix-free linear operator that dispatches `mul!(dest, op, src) = dest .= op.f(src)`.
Duck-typed as a linear map for `IterativeSolvers.lobpcg` (needs `mul!`, `size`,
`eltype`).
"""
struct MatFreeOp{F,T}
    f::F         # v -> A·v
    dim::Int     # both dimensions (square)
end
MatFreeOp{T}(f::F, dim::Int) where {F,T} = MatFreeOp{F,T}(f, dim)

Base.eltype(::MatFreeOp{F,T}) where {F,T} = T
Base.size(op::MatFreeOp, ::Integer) = op.dim
Base.size(op::MatFreeOp) = (op.dim, op.dim)

# `mul!(y, op, x)` — the only mul! form LOBPCG actually calls.
# `f` returns a fresh vector (that's how apply_M_local / apply_CH2_local are
# written); we copy into `y` to preserve the caller's storage.
#
# LOBPCG passes SubArray views of matrix columns for both x and y, but ITensor's
# `ITensor(array, inds...)` constructor rejects SubArrays via its NeverAlias
# machinery. Materialize `x` into a plain Vector before dispatching to `op.f`.
LinearAlgebra.mul!(y::AbstractVector{T}, op::MatFreeOp{F,T},
                    x::AbstractVector) where {F,T} =
    (y .= op.f(Vector{T}(x)); y)
LinearAlgebra.mul!(y::AbstractMatrix{T}, op::MatFreeOp{F,T},
                    x::AbstractMatrix) where {F,T} = begin
    # LOBPCG occasionally hands us block-of-columns matrices; process column-wise.
    for j in axes(x, 2)
        y[:, j] .= op.f(Vector{T}(view(x, :, j)))
    end
    y
end

"""
    ChebyBInv{FB,T}

Degree-`k` shifted-Chebyshev polynomial approximation of `B⁻¹` on the interval
`[α, β] ⊇ spec(B)`. Applied as a preconditioner via `ldiv!(y, P, r)`.

For our `B = I + θ²·L²` we have `α = 1` exactly (since `L²` is PSD) and
`β ≥ 1 + θ²·λ_max(L²)`. Cost per apply: `k` invocations of `B_apply` (each a
factored two-layer C_H² contraction).

Reference: Saad, *Iterative Methods for Sparse Linear Systems*, §12.3.
"""
struct ChebyBInv{FB,T}
    α::T           # λ_min(B) (typically 1 for us)
    β::T           # λ_max(B) (or a conservative upper bound)
    k::Int         # polynomial degree
    B_apply::FB    # v -> B·v
end

Base.eltype(::ChebyBInv{FB,T}) where {FB,T} = T

# `ldiv!(y, P, r)` — approximates `y ≈ B⁻¹ · r` by degree-k Chebyshev iteration
# on `Bx = r`. Standard three-term recurrence (Saad, algorithm 12.1).
function LinearAlgebra.ldiv!(y::AbstractVector, P::ChebyBInv{FB,T},
                              r::AbstractVector) where {FB,T}
    α, β, k, B_apply = P.α, P.β, P.k, P.B_apply
    # Degenerate interval — happens when tsq is at (or near) 0, making
    # B ≈ I. Then B⁻¹ ≈ I and the identity is the right preconditioner;
    # the standard Chebyshev formula divides by δ = (β−α)/2 = 0 and
    # produces NaNs that then contaminate LOBPCG's small-pencil eigsolve.
    if real(β - α) < eps(real(T)) * max(real(α), real(β), one(real(T)))
        y .= r
        return y
    end
    θ = (β + α) / 2
    δ = (β - α) / 2
    σ = θ / δ
    ρ_prev = one(T) / σ

    # x_0 = r/θ ; d_0 = x_0. Materialize into fresh vectors so downstream
    # `B_apply` doesn't receive a view (ITensor's NeverAlias rejects SubArrays).
    y_curr = Vector{T}(r ./ θ)
    d = copy(y_curr)
    resid = similar(y_curr)
    r_full = Vector{T}(r)     # ensure `r_full - B_apply(...)` avoids view arithmetic
    for j in 1:k
        # r_j = r - B·y
        resid .= r_full .- B_apply(y_curr)
        ρ = one(T) / (2σ - ρ_prev)
        # d_j = ρ_j ρ_{j-1} d_{j-1} + (2 ρ_j / δ) r_j
        d .= (ρ * ρ_prev) .* d .+ (2 * ρ / δ) .* resid
        # y_j = y_{j-1} + d_j
        y_curr .+= d
        ρ_prev = ρ
    end
    # Copy back into caller's y (which may be a SubArray).
    y .= y_curr
    return y
end

# Block-column version (LOBPCG applies preconditioners to matrices of search
# directions when running block-size ≥ 1).
function LinearAlgebra.ldiv!(y::AbstractMatrix, P::ChebyBInv,
                              r::AbstractMatrix)
    for j in axes(r, 2)
        ldiv!(view(y, :, j), P, view(r, :, j))
    end
    y
end

"""
    power_iterate_L2(v0, apply_L2; iters=10) -> λ_max_est

Estimate `λ_max` of the block-level factored C_H² operator by `iters` steps of
power iteration starting from `v0`. Returns the Rayleigh quotient
`⟨v|L²|v⟩ / ⟨v|v⟩` of the final iterate. `v0` is not modified.
"""
function power_iterate_L2(v0, apply_L2; iters::Int=10)
    v = copy(v0)
    n = norm(v)
    v = n > 0 ? (v ./ n) : v
    L2v = apply_L2(v)
    for _ in 1:iters
        v .= L2v
        n = norm(v)
        n = n > 0 ? n : one(real(eltype(v)))
        v ./= n
        L2v = apply_L2(v)
    end
    return real(dot(v, L2v))
end

"""
    local_step_qcqp(initvec, psiinds,
                    LM, RM, M_p, M_p1,
                    LH, RH, CH_p, CH_p1, Ct_p, Ct_p1,
                    epsilon_loc, thetasq_ref, Q, subspace_coeffs,
                    dispon, dlog, matvec_counter;
                    eig_kwargs, bisect_tol, bisect_maxiter, bracket_expansions)

Solve the local QCQP by bisecting θ² so that ⟨x_{θ²} | B̃ | x_{θ²}⟩ = ε_loc
(tight constraint), or accept θ² = 0 if the unconstrained optimum is already
feasible.

# Arguments
- `initvec::Vector`: initial guess for the local block (from psi[p]·psi[p+1])
- `psiinds`: index list for reshaping vec ↔ ITensor
- `LM, RM, M_p, M_p1`: M-envs and site tensors (single-layer)
- `LH, RH, CH_p, CH_p1`: factored two-layer C_H² envs and bottom-layer C_H site
  tensors
- `Ct_p, Ct_p1`: top-layer site tensors (adjoint of the bottom layer). Same as
  `CH_p, CH_p1` on the Hamiltonian path; `adjoint_mpo(C_U)` tensors on the
  Floquet path — see `apply_CH2_local`
- `epsilon_loc::DTYPE`: local constraint value (target ⟨x|B̃|x⟩)
- `thetasq_ref::Ref{DTYPE}`: **in-place** θ² multiplier — warm-starts from the
  previous site and is updated to the accepted θ² on return ("the multiplier
  floats along the sweep").
- `Q`: orthogonalization basis (Krylov states projected out), or nothing
- `subspace_coeffs::Tuple{DTYPE,DTYPE}`: (scale, ortho_penalty) for the combined
  matvec (passed straight through to `apply_Heff_local!`)
- `eig_kwargs::Dict`: passed to inner `eigsolve` calls
- `bisect_tol::Real`: relative tolerance on |B_val − ε_loc| / ε_loc
- `bisect_maxiter::Int`: hard cap on bisection iterations
- `bracket_expansions::Int`: max ×10 expansions of the upper bracket

# Returns
- `En::Vector` (single element) — eigenvalue ν at the final θ²; matches the
  layout returned by `eigsolve(...)` in the current DMRG loop
- `gs::Vector` (single element) — corresponding eigenvector (unit-normalized)
- `info::NamedTuple` — has `numops` plus `qcqp_thetasq`, `qcqp_B`,
  `qcqp_expects`, `qcqp_iters`, `qcqp_status` for logging
- Side effect: `thetasq_ref[]` is updated to the accepted θ²
"""
function local_step_qcqp(
    initvec, psiinds,
    LM, RM, M_p, M_p1,
    LH, RH, CH_p, CH_p1, Ct_p, Ct_p1,
    epsilon_loc::DTYPE, thetasq_ref::Ref{DTYPE},
    Q, subspace_coeffs::Tuple{DTYPE,DTYPE},
    dispon::Int,
    dlog::DebugLogger,
    matvec_counter::Union{Ref{Int},Nothing};
    eig_kwargs::Dict = Dict(:krylovdim => 25),
    # Inner numerical primitive used at each θ² step of the bisection.
    # :eigsolve — ordinary eigsolve on (-M + θ²·C_H²) (default; supports dense fallback + retries)
    # :geneig   — generalized eigsolve on (-M , I + θ²·C_H²)
    inner_primitive::Symbol = :eigsolve,
    bisect_tol::Real = 1e-3,
    bisect_maxiter::Int = 20,
    bracket_expansions::Int = 12,
    # Hard ceiling on θ² during bracket-up. If the bracket-up loop tries to go
    # above this, exit with :bracket_failed (treating the block as effectively
    # infeasible for this ε_loc). Default 1e8 keeps the operator diagonal
    # (~θ²·⟨C_H²⟩) below KrylovKit's absolute Hermiticity tolerance √eps ≈
    # 1.5e-8 under GPU ComplexF64 fp noise. Raise/lower via the caller's
    # `qcqp_theta_sq_max` kwarg to run_dmrg_for_size.
    theta_sq_max::DTYPE = DTYPE(1e8),
    verbose::Bool = false,
    call_id::Union{Int,Nothing} = nothing,
    # Ortho-penalty auto-scale: start with a b that pushes the P-subspace
    # eigenvalue above the typical (1-P) spectrum. If Lanczos still returns a
    # state with too much P-component, multiply b by 10 and retry.
    ortho_weight_init::DTYPE = DTYPE(100.0),
    max_ortho_weight::DTYPE = DTYPE(1e12),
    p_tolerance::Real = 1e-6,
    max_ow_retries::Int = 15,
    # Dense-LAPACK fallback: when the two-site block dim is small enough,
    # materialize H_eff and call LinearAlgebra.eigen. Guarantees the true
    # minimum eigenvalue, bypassing KrylovKit's convergence pathologies.
    use_dense_at_small_dim::Bool = true,
    dense_max_dim::Int = 2048,
    # KrylovKit path: check info.converged and Ritz residual. If the eigsolve
    # failed either check, retry with (warm-start + noise) as the initial vector.
    # (`:geneig` uses a different scheme — see `_call_geneig_inner` docstring.)
    kryl_resid_tol::Real = 1e-4,   # relaxed from 1e-6: at χ≥256 CPU, tighter tol drives KrylovKit to maxiter, costing 60+ min per block; 1e-4 is well below QCQP's `infeasibility_rel_tol=1e-2` so it doesn't confuse the bisection
    max_kryl_retries::Int = 4,
    # Bracket-expansion infeasibility detection: if B_hi stops decreasing across
    # bracket-expansion steps, the block cannot achieve ⟨x|B̃|x⟩ ≤ ε_loc no
    # matter how large θ² is. Return the argmin-B̃ state with :infeasible.
    infeasibility_rel_tol::Real = 1e-2,
    # Warm-start acceptance: if warm-start B is within this relative tolerance
    # of ε_loc, skip bisection entirely. Trades constraint-tightness precision
    # for compute time; typically fine when neighboring sites' θ* are similar.
    warm_accept_tol::Real = 5e-2,
    # Optional exponential moving average of the converged θ² across blocks.
    # Caller passes a `Ref{DTYPE}` seeded with the previous DMRG call's EMA
    # (or the outer θ² on the first call); we update it at every exit that
    # sets `thetasq_ref[]`. The caller reads this at the end of a DMRG call
    # and re-uses it as the warm-start seed for the next ε step.
    thetasq_ema::Union{Ref{DTYPE},Nothing} = nothing,
    ema_alpha::Real = 0.1,
    # `:lobpcg` inner primitive: LOBPCG + Chebyshev-in-B preconditioner. The
    # Chebyshev polynomial approximates B⁻¹ on the spectrum interval
    # [1, 1+θ²·λ_max(L²)]. `λ_max(L²)` is estimated once per `local_step_qcqp`
    # invocation by a short power iteration on `apply_CH2_local`.
    # Chebyshev degree. `0` (the default) means ADAPTIVE: the degree is derived
    # from the actual conditioning at this θ², since κ(B) = 1 + θ²·λ_max(C_H²)
    # grows as ε tightens along the scan. A degree-k Chebyshev contracts the
    # B-residual by ((√κ−1)/(√κ+1))^k, so reaching a target contraction η needs
    #     k ≈ √κ · ln(2/η) / 2.
    # A fixed degree is wrong at both ends of an ε ladder: at κ≈61 (step 1 of the
    # L=15 scan) k=15 is ~4× more than needed, while by κ≈10⁶ it contracts by
    # only 0.97 and buys almost nothing. Pass a positive integer to pin the
    # degree instead (reproduces pre-adaptive runs).
    lobpcg_cheby_degree::Int = 0,
    lobpcg_cheby_min::Int = 2,
    lobpcg_cheby_max::Int = 8,    # measured ceiling: cost rises monotonically past ~8 (array 12974859)
    lobpcg_cheby_eta::Real = 0.1,
    # Do not spend preconditioner degree on a block whose θ² search is running
    # away to the ceiling — that result is going to be capped or discarded, so
    # escalating the polynomial is pure waste. Above this fraction of
    # `theta_sq_max`, pin the degree to `lobpcg_cheby_min`.
    lobpcg_cheby_runaway_frac::Real = 0.01,
    # LOBPCG stopping. `lobpcg` takes an ABSOLUTE residual tolerance and
    # B-orthonormalizes X internally, so at θ²≫1 ‖x‖₂ ~ 1/√‖B‖ ≪ 1 and the
    # absolute test is meaningless — it fires after one iteration on garbage.
    # The old workaround pinned tol at machine ε and ran to `maxiter`, which
    # ground every solve to ~1e-14 to serve an outer bisection needing only
    # `bisect_tol` (5e-3). Instead: run LOBPCG in short bursts of
    # `lobpcg_inner_maxiter` and apply our own RELATIVE test between bursts,
    # restarting from the returned block. Total iteration budget is
    # `lobpcg_inner_maxiter × lobpcg_max_restarts`, kept at/below the old 100 so
    # the worst case cannot regress.
    lobpcg_rel_tol::Real = 1e-6,
    lobpcg_inner_maxiter::Int = 12,
    lobpcg_max_restarts::Int = 8,
    lobpcg_maxiter::Int = 100,   # retained for callers that pin it; see above
    lobpcg_power_iters::Int = 10,
    # Persistent secant-slope estimate carried across blocks. Model:
    # ⟨C_H²⟩ ≈ A · (θ²)^(−s). Given two (θ², B) samples we compute the local
    # slope; the caller carries the exponentially-averaged value across blocks
    # so the very first secant step at each block starts from a warm slope
    # instead of the flat s = 1 fallback. If `nothing`, the pre-secant
    # geometric bracket (×2/×10) code path is used.
    slope_ref::Union{Ref{DTYPE}, Nothing} = nothing,
    slope_ema_alpha::Real = 0.3,
    slope_min::Real = 0.1,
    slope_max::Real = 5.0,
    slope_infeasible_floor::Real = 0.05,
    # Consecutive flat-slope samples required before declaring :infeasible, and
    # the minimum |Δ log θ²| for a sample to count. Both exist because a single
    # flat reading is not evidence — see the guards at the slope test below.
    slope_infeasible_streak::Int = 2,
    slope_min_logstep::Real = 0.1,
    secant_maxiter::Int = 5,
    secant_log_step_cap::Real = log(10.0),
)
    # GPU strategy for the :eps_constrained inner solvers (:lobpcg, :geneig,
    # :eigsolve): the small dense work — LOBPCG's B-pencil solve, eigsolve's
    # dense fallback, the Chebyshev recurrence, the λ_max power iteration — runs
    # on the host, and ONLY the matvec contractions in apply_CH2_local /
    # apply_M_local touch the GPU (they move the block tensor to the
    # environments' device and return the result to the host). Force the warm
    # start and the Q-projector to the host here so X0, `proj`, and the power
    # iteration all stay CPU-side; the eigenvector is moved back to the device
    # by the sweeper after the solve. Both calls are no-ops when GPU is off, so
    # the CPU path is byte-identical.
    initvec = cpuarray(initvec)
    Q = Q === nothing ? nothing : cpuarray(Q)

    _trace(msg) = verbose && logwrite("[QCQP call=$(call_id === nothing ? "?" : call_id)] ", msg)

    # Push the final block-converged θ² into the caller's EMA tracker. Skipped
    # on `:unconstrained` exits (final_tsq == 0) since those don't reflect an
    # actual θ*; the constraint was inactive so θ² is unbounded from below.
    _ema_update(final_tsq) = begin
        if thetasq_ema !== nothing && final_tsq > 0
            α = DTYPE(ema_alpha)
            thetasq_ema[] = α * final_tsq + (one(DTYPE) - α) * thetasq_ema[]
        end
    end

    # Cumulative matvec counter (physical eigsolve matvecs; local B expectations
    # count too since they invoke apply_CH2_local which is essentially one MPO
    # application).
    total_numops = 0
    total_expect_calls = 0

    # Per-source contraction accounting. `matvec_counter` used to be bumped ONLY
    # in A_apply/B_apply, so it silently omitted the Chebyshev preconditioner
    # (k applications of apply_CH2_local per ldiv!), the λ_max power iteration,
    # and B_expectation. Since the preconditioner dominates — degree k means k
    # of the ~k+2 contractions per LOBPCG iteration — the reported counts
    # undercounted real work by roughly the Chebyshev degree. Every site that
    # actually calls apply_CH2_local / apply_M_local now bumps both the caller's
    # aggregate and the breakdown below, so "matvec per site" in the sweep
    # summary is a true contraction count and the preconditioner's share is
    # directly visible.
    mv_A    = Ref(0)   # apply_M_local   via A_apply
    mv_B    = Ref(0)   # apply_CH2_local via B_apply
    mv_prec = Ref(0)   # apply_CH2_local via the Chebyshev preconditioner
    mv_aux  = Ref(0)   # apply_CH2_local via power iteration / B_expectation
    _bump_global!() = (matvec_counter !== nothing && (matvec_counter[] += 1); nothing)

    # Track the current ortho-weight (b) across all inner eigsolves inside this
    # local_step_qcqp call. Starts at ortho_weight_init and is doubled/×10-ed on
    # retry if Lanczos returns too much P-component.
    current_ow = Ref(ortho_weight_init)

    # λ_max(L²) estimate: hoisted here so we compute it once per block, reused
    # by every θ² step of the LOBPCG bracket-up. Held in a Ref so it lazily
    # initializes only when the :lobpcg path first needs it. `L²` here is the
    # projected `P̃ · apply_CH2 · P̃` operator; power iteration on the raw
    # apply_CH2_local is fine as an upper bound since the projection can only
    # shrink the spectrum.
    lambda_max_L2 = Ref{DTYPE}(DTYPE(0))
    lambda_max_L2_computed = Ref(false)
    _ensure_lambda_max_L2() = begin
        if !lambda_max_L2_computed[]
            L2_apply = v -> begin
                mv_aux[] += 1; _bump_global!()
                apply_CH2_local(v, psiinds, LH, RH, CH_p, CH_p1, Ct_p, Ct_p1)
            end
            lambda_max_L2[] = DTYPE(
                power_iterate_L2(initvec, L2_apply; iters=lobpcg_power_iters)
            )
            lambda_max_L2_computed[] = true
            debuglog(dlog, "[lobpcg] λ_max(L²) power-iterate ($lobpcg_power_iters iters) = $(round(lambda_max_L2[], sigdigits=4))")
        end
        return lambda_max_L2[]
    end

    # ---- helpers ----

    # Combined matvec at a specific θ² with a specific ortho-weight.
    make_matvec = (tsq, ow) -> ((v) -> apply_Heff_local!(
        v, psiinds,
        LM, RM, M_p, M_p1,
        LH, RH, CH_p, CH_p1, Ct_p, Ct_p1,
        -one(DTYPE), tsq,
        Q, (subspace_coeffs[1], ow), dlog, matvec_counter,
    ))

    # ⟨x | B̃ | x⟩ for a (unit-norm) candidate x
    function B_expectation(x)
        total_expect_calls += 1
        mv_aux[] += 1; _bump_global!()
        Bx = apply_CH2_local(x, psiinds, LH, RH, CH_p, CH_p1, Ct_p, Ct_p1)
        return real(dot(x, Bx))
    end

    # ‖Q†x‖² — the P-subspace weight of x (should be ≈ 0 for a proper (1-P)
    # eigenvector; ≈ 1 if Lanczos accidentally converged onto the P subspace).
    function p_component(x)
        Q === nothing && return 0.0
        return sum(abs2, Q' * x)
    end

    # Densely materialize H_eff (dim × dim) at a given θ² and ortho-weight.
    # Device-agnostic: allocates matching the storage of `initvec` so the whole
    # eigen problem runs on GPU when the DMRG state is GPU-backed.
    function build_Heff(tsq, ow, dim)
        mv = make_matvec(tsq, ow)
        Heff = similar(initvec, dim, dim)
        fill!(Heff, zero(eltype(initvec)))
        for i in 1:dim
            # Build unit vector on CPU then move to matching device (no-op on CPU).
            e_i_cpu = zeros(eltype(initvec), dim)
            e_i_cpu[i] = one(eltype(initvec))
            Heff[:, i] = mv(movedevice(e_i_cpu))
        end
        (Heff + Heff') / 2   # symmetrize away roundoff
    end

    # Geneig inner primitive: solve (-M) v = λ (I + tsq·C_H²) v via
    # KrylovKit.geneigsolve. See `local_step_geneig` for why no ortho_penalty
    # is needed here: the operator structure gives the P subspace a natural
    # eigenvalue of 0 (A_op = 0 on P, B_op = identity on P), while the
    # physical eigenvalue is strictly negative — :SR picks it automatically.
    # `ow` is accepted for signature parity with the eigsolve inner path but
    # is not used.
    #
    # Convergence: gscale = 1/(1+tsq) is applied inside AB_op to keep the
    # operator norms O(1) so KrylovKit's absolute-tolerance Hermiticity check
    # doesn't abort at large tsq (see comment in AB_op below). That same
    # rescaling would fool KrylovKit's convergence check — normres = gscale ·
    # ||Ax − λBx|| gets trivially small for any x at large tsq. To keep the
    # convergence gate physically meaningful we hand KrylovKit a proportionally
    # scaled tolerance: `tol = kryl_resid_tol · gscale`. Its own check then
    # requires physical residual < kryl_resid_tol regardless of tsq.
    # KrylovKit's Krylov-Schur restarts (up to `maxiter`) do the actual
    # incremental convergence work — no external retry loop needed.
    _call_geneig_inner(tsq, v_init, ow) = begin
        scale = subspace_coeffs[1]
        gscale = one(DTYPE) / (one(DTYPE) + tsq)
        AB_op = (v) -> begin
            if matvec_counter !== nothing
                matvec_counter[] += 1
            end
            v_perp = Q === nothing ? v : v .- Q * (Q' * v)
            Mv = apply_M_local(v_perp, psiinds, LM, RM, M_p, M_p1)
            Av_perp = (-scale) .* Mv
            CH2v_perp = apply_CH2_local(v_perp, psiinds, LH, RH, CH_p, CH_p1, Ct_p, Ct_p1)
            Bv_perp = v_perp .+ tsq .* CH2v_perp
            if Q !== nothing
                Av_perp .-= Q * (Q' * Av_perp)
                Bv_perp .-= Q * (Q' * Bv_perp)
            end
            v_par = Q === nothing ? v .* 0 : (v .- v_perp)
            # Scale both operators by 1/(1+tsq) — exact for a generalized
            # eigenproblem (λ and v are invariant under A,B → cA,cB) and needed
            # because this path drives tsq up by bisection. See the same guard
            # in `local_step_geneig` for the failure it prevents: at tsq ≈ 1e9
            # the O(1e-13)-relative roundoff in ⟨v|B|v⟩ becomes O(1e-4)
            # absolute, and KrylovKit's absolute-tolerance hermiticity check
            # aborts the solve.
            return (Av_perp .* gscale, (Bv_perp .+ v_par) .* gscale)
        end

        # Undo the gscale on KrylovKit's tolerance so it stops on physical residual.
        # Floor the tol at machine-precision × dim (Lanczos-B can't beat that in
        # absolute normres), so at huge tsq we don't hand KrylovKit an unreachable
        # target and force it to burn every one of its maxiter restarts.
        # Also cap maxiter explicitly — its default is high enough that at large
        # tsq + large block dim, iterations blow up wall time.
        call_kwargs = Dict{Symbol,Any}(pairs(eig_kwargs))
        tol_scaled = DTYPE(kryl_resid_tol) * gscale
        tol_floor  = DTYPE(1e-14)   # ≈ 100·eps(Float64); anything below is fp noise
        call_kwargs[:tol]     = max(tol_scaled, tol_floor)
        # Match KrylovKit's own default; the tol floor above is what actually
        # ensures we don't sit here at unreachable-tol wasting matvecs.
        call_kwargs[:maxiter] = get(call_kwargs, :maxiter, 100)

        kd_use    = get(call_kwargs, :krylovdim, "?")
        maxit_use = get(call_kwargs, :maxiter, "default")
        debuglog(dlog, "[geneig] call_id=$(call_id) tsq=$(round(tsq, sigdigits=4)) gscale=$(round(gscale, sigdigits=3)) krylovdim=$kd_use maxiter=$maxit_use tol=$(round(call_kwargs[:tol], sigdigits=3)) dim=$(length(v_init))")

        t_geneig = time()
        En, gs, info_ = geneigsolve(
            AB_op, v_init, 1, :SR;
            ishermitian = true, isposdef = true,
            verbosity = max(0, dispon - 2),
            call_kwargs...,
        )
        wall_geneig = time() - t_geneig
        imin = argmin(En)
        x = gs[imin]
        nx = norm(x)
        if nx > 0
            x = x ./ nx
        end

        debuglog(dlog, "[geneig]   geneigsolve done in $(round(wall_geneig, digits=2))s converged=$(info_.converged) numiter=$(info_.numiter) numops=$(info_.numops) En=$(round(real(En[imin]), sigdigits=6))")

        # Physical relative residual as a diagnostic + safety fallback. If
        # KrylovKit exhausted maxiter without converging AND the returned Ritz
        # pair is still visibly bad, one defensive re-solve with a randn kick
        # to the initial vector can escape a pathological invariant subspace.
        Av_check, Bv_check = AB_op(x)
        r = Av_check .- En[imin] .* Bv_check
        denom = max(abs(En[imin]) * norm(Bv_check), norm(Av_check), eps())
        rel_resid = norm(r) / denom

        # Detailed Ritz-ratio breakdown (once per geneigsolve). Reports the
        # scaled (‖A'x‖, ‖B'x‖) that KrylovKit sees plus the physical
        # (‖Ax‖, ‖Bx‖, ⟨C_H²⟩ per site) so you can trace whether tol-vs-tsq
        # numerics or the underlying gap is what's driving iteration count.
        nAv    = norm(Av_check)       # ‖A' x‖ = gscale · ‖A x‖
        nBv    = norm(Bv_check)       # ‖B' x‖ = gscale · ‖B x‖
        xAx    = real(dot(x, Av_check))         # ⟨x|A'|x⟩
        xBx    = real(dot(x, Bv_check))         # ⟨x|B'|x⟩
        # Undo gscale for physical quantities:
        xAx_phys = xAx / gscale                 # ⟨x|A|x⟩ = -scale·⟨M⟩ physically
        xBx_phys = xBx / gscale                 # ⟨x|B|x⟩ = 1 + θ²·⟨C_H²⟩
        # Extract ⟨C_H²⟩ from ⟨B⟩ decomposition (only defined when tsq > 0):
        cH2 = tsq > 0 ? (xBx_phys - 1) / tsq : NaN
        debuglog(dlog, "[geneig]   ⟨A⟩_phys=$(round(xAx_phys, sigdigits=4)) ⟨B⟩_phys=$(round(xBx_phys, sigdigits=4)) ⟨C_H²⟩=$(round(cH2, sigdigits=4)) λ=$(round(real(En[imin]), sigdigits=4))")
        debuglog(dlog, "[geneig]   ‖A'x‖=$(round(nAv, sigdigits=3)) ‖B'x‖=$(round(nBv, sigdigits=3)) |λ|·‖B'x‖=$(round(abs(En[imin])*nBv, sigdigits=3)) ‖r'‖=$(round(norm(r), sigdigits=3))")
        debuglog(dlog, "[geneig]   rel_resid=$(round(rel_resid, sigdigits=3)) (tol=$(round(kryl_resid_tol, sigdigits=3)))")

        # Accept on the PHYSICAL residual, NEVER on KrylovKit's flag.
        #
        # The tolerance handed to geneigsolve is `kryl_resid_tol · gscale`, but
        # geneigsolve (Golub-Ye) measures its residual in the B-metric and
        # ‖B‖ ~ θ², so the two do not correspond. At tsq ≳ 10 the warm start
        # satisfies the test trivially. Measured at L=8: `numiter=1, numops=2,
        # converged=1` on 37 of 72 solves, with `rel_resid` 10-35× OVER
        # `kryl_resid_tol`.
        #
        # The previous guard was `info_.converged < 1 && rel_resid > tol` — an
        # AND — so a falsely-reported convergence bypassed it entirely. The
        # consequence chain: x returned unchanged → ⟨C_H²⟩ bit-identical across
        # a 10× change in θ² → secant slope reads -0.0 → the block declares
        # `:infeasible` on a target that reps 4 and 6 of the same ladder proved
        # reachable. Downstream that froze ⟨C_H²⟩ across ε steps, or landed at
        # the wrong ε with ⟨M⟩ ~15 % below the on-target value (0.221 vs 0.261).
        #
        # This is the same failure mode already documented for `:lobpcg` — an
        # absolute tolerance rendered meaningless by θ²-dependent operator
        # scaling — and is the geneig analogue of the burst loop used there.
        retry_tol = call_kwargs[:tol]
        for kretry in 1:max_kryl_retries
            rel_resid <= DTYPE(kryl_resid_tol) && break
            # Tighten geometrically: re-calling at the SAME tol reproduces the
            # same no-op. Restart from the current x (never worse than the warm
            # start) rather than from a random kick, which discards progress.
            retry_tol = max(retry_tol / DTYPE(10), DTYPE(1e-16))
            call_kwargs[:tol] = retry_tol
            debuglog(dlog, "[geneig]   resid=$(round(rel_resid, sigdigits=3)) > tol=$(kryl_resid_tol) after numiter=$(info_.numiter) numops=$(info_.numops) — retry $kretry at tol=$(round(retry_tol, sigdigits=3))")
            t_fb = time()
            En, gs, info_ = geneigsolve(
                AB_op, x, 1, :SR;
                ishermitian = true, isposdef = true,
                verbosity = max(0, dispon - 2),
                call_kwargs...,
            )
            wall_fb = time() - t_fb
            imin = argmin(En)
            x = gs[imin]
            nx = norm(x)
            if nx > 0
                x = x ./ nx
            end
            Av_check, Bv_check = AB_op(x)
            r = Av_check .- En[imin] .* Bv_check
            denom = max(abs(En[imin]) * norm(Bv_check), norm(Av_check), eps())
            rel_resid = norm(r) / denom
            debuglog(dlog, "[geneig]   retry $kretry done in $(round(wall_fb, digits=2))s numiter=$(info_.numiter) numops=$(info_.numops) rel_resid=$(round(rel_resid, sigdigits=3))")
        end
        # Log an exhausted budget rather than silently accepting a bad solve —
        # this is the signal that the cost/accuracy trade needs attention, and
        # the thing whose absence let the original bug hide.
        if rel_resid > DTYPE(kryl_resid_tol)
            debuglog(dlog, "[geneig]   BUDGET EXHAUSTED: rel_resid=$(round(rel_resid, sigdigits=3)) > tol=$(kryl_resid_tol) after $max_kryl_retries retries (tsq=$tsq)")
        end

        return En[imin], x, info_
    end

    # LOBPCG inner primitive with a Chebyshev-in-B preconditioner. Solves
    # `(-scale·M) v = λ (I + tsq·L²) v` for the smallest λ via
    # `IterativeSolvers.lobpcg`, which is inherently preconditioner-friendly.
    # Chebyshev polynomial p_k(B) approximates B⁻¹ on [1, 1+tsq·λ_max(L²)] —
    # reduces the effective condition number from κ(B) ≈ 10⁶ (at large tsq)
    # to O(1) so LOBPCG converges in a few dozen iterations regardless of tsq.
    #
    # The P-subspace projector (Q) is enforced two ways for defence in depth:
    # (i) as LOBPCG's `C` constraint kwarg so search directions are projected;
    # (ii) inside A_op, B_op, and the Chebyshev preconditioner via
    # `v_perp = v - Q*(Q'*v)`, mirroring `_call_geneig_inner:250-256`.
    _call_lobpcg_inner(tsq, v_init, ow) = begin
        scale = subspace_coeffs[1]

        # Q-projection helper (soft dispatch: no-op if Q === nothing).
        proj = Q === nothing ? identity : (v -> v .- Q * (Q' * v))

        # A, B, and B̃ (for the preconditioner) as matrix-free operators.
        # Track matvec count through the parent counter for accounting parity
        # with the other primitives.
        matvec_bump!() = (matvec_counter !== nothing && (matvec_counter[] += 1); nothing)

        # A_eff = P̃ (−M) P̃    B_eff = I + tsq · P̃ · C_H² · P̃
        # Mirroring `_call_geneig_inner:410-431` verbatim: the C_H² block is
        # sandwiched between two projectors, and the identity on B keeps it
        # strictly SPD with spec(B) ⊆ [1, ∞). Writing B_eff = v + tsq·P̃CH²P̃v
        # instead of P̃(I + tsq·CH²)P̃v guarantees that the Q-piece of B_eff·v
        # is exactly v_par (no CH² leakage), so the eigenvalue equation on Q
        # is 0 = λ · v_par ⇒ q = 0 for any λ ≠ 0. This forces LOBPCG's Ritz
        # vectors to be strictly Q-perp; the earlier "leave B unprojected"
        # formulation allowed a nonzero q to balance CH²'s off-diagonal
        # Q↔Q⊥ couplings in the Ritz residual, letting LOBPCG return
        # Q-contaminated eigenvectors that then produced spurious ⟨C_H²⟩
        # readings downstream. Q ⊂ ker(C_H²) is only a *global* property; at
        # the block level `apply_CH2_local` uses ψ's current environments
        # and does not annihilate Q locally.
        A_apply = v -> begin
            matvec_bump!(); mv_A[] += 1
            v_perp = proj(v)
            Mv = apply_M_local(v_perp, psiinds, LM, RM, M_p, M_p1)
            proj((-scale) .* Mv)
        end
        B_apply = v -> begin
            matvec_bump!(); mv_B[] += 1
            v_perp = proj(v)
            v_par  = Q === nothing ? zero(v) : v .- v_perp
            CH2v_perp = apply_CH2_local(v_perp, psiinds, LH, RH, CH_p, CH_p1, Ct_p, Ct_p1)
            Bv_perp = v_perp .+ tsq .* CH2v_perp
            # Strip any Q-content that CH² leaked back in, then add v_par
            # (identity block on Q). Equivalent form: I + tsq · P̃·CH²·P̃.
            Bv_perp .-= Q === nothing ? zero(Bv_perp) : Q * (Q' * Bv_perp)
            Bv_perp .+ v_par
        end

        n = length(v_init)
        T = eltype(v_init)
        A_op = MatFreeOp{T}(A_apply, n)
        B_op = MatFreeOp{T}(B_apply, n)

        # Preconditioner: Chebyshev of degree `lobpcg_cheby_degree` on
        # [α, β] = [1, 1 + tsq · 1.2·λ_max_est(L²)]. The 1.2× fudge is a
        # conservative upper bound so the polynomial doesn't undershoot at the
        # tail of the spectrum (power iteration is an underestimate in general).
        lam_max = _ensure_lambda_max_L2()
        α_B = one(T)
        β_B = one(T) + tsq * DTYPE(1.2) * lam_max
        # Preconditioner approximates B⁻¹ where B = I + tsq · P̃·CH²·P̃ — same
        # form as B_apply above. λ_max(P̃CH²P̃) ≤ λ_max(CH²), so the interval
        # [1, β_B] built from the unprojected λ_max stays a valid upper bound.
        prec_B_apply = v -> begin
            mv_prec[] += 1; _bump_global!()
            v_perp = proj(v)
            v_par  = Q === nothing ? zero(v) : v .- v_perp
            CH2v_perp = apply_CH2_local(v_perp, psiinds, LH, RH, CH_p, CH_p1, Ct_p, Ct_p1)
            Bv_perp = v_perp .+ tsq .* CH2v_perp
            Bv_perp .-= Q === nothing ? zero(Bv_perp) : Q * (Q' * Bv_perp)
            Bv_perp .+ v_par
        end

        # --- Chebyshev degree: adaptive in κ(B) unless explicitly pinned ---
        # κ = β_B since α_B = 1 exactly (C_H² is PSD). Reaching a residual
        # contraction of η needs k ≈ √κ·ln(2/η)/2. Note this only moves the √κ
        # from LOBPCG's iteration count into the per-iteration cost; it does not
        # remove it. `k_max` is the cost ceiling, and the runaway guard keeps us
        # from spending degree on a block heading for `theta_sq_max`.
        κ_B = real(β_B)
        runaway = theta_sq_max > 0 &&
                  tsq >= DTYPE(lobpcg_cheby_runaway_frac) * theta_sq_max
        cheby_deg = if lobpcg_cheby_degree > 0
            lobpcg_cheby_degree
        elseif runaway
            lobpcg_cheby_min
        else
            # MEASURED, not derived. The √κ theory scaling below is REFUTED for
            # this problem: a degree × ε grid at L=7 (array 12974859, each point
            # warm-started from a shared converged seed so degree is the only
            # variable) found the total-contraction minimum at degree 2–8 across
            # κ ≈ 10²–5×10⁴, with cost rising monotonically above that — e.g. at
            # ε=0.01, degree 8/16/32/64 cost 8.9k/12.2k/20.3k/98.1k contractions
            # for identical ⟨M⟩ and ⟨C_H²⟩. Theory would have asked for ~335 at
            # the top of that κ range, i.e. ~40× too much.
            #
            # Why: the degree needed to genuinely precondition grows like √κ, so
            # past a small degree you are paying linearly for a contraction
            # factor that is still ≈1. It is cheaper to let LOBPCG iterate.
            #
            # Kept as a mild κ-dependence rather than a hard constant only
            # because the grid is L=7/χ=64; `lobpcg_cheby_max` is the real
            # protection. Re-measure before trusting this at production κ.
            k_emp = ceil(Int, 2 + log10(max(κ_B, one(DTYPE))))
            clamp(k_emp, lobpcg_cheby_min, lobpcg_cheby_max)
        end

        P_op = ChebyBInv{typeof(prec_B_apply),T}(α_B, β_B, cheby_deg,
                                                   prec_B_apply)

        # Warm-start matrix (LOBPCG wants a Matrix even for single-vector).
        X0 = reshape(copy(v_init), :, 1)
        # LOBPCG's `tol` is absolute on ‖A x − λ B x‖ AND it B-orthonormalizes
        # X internally: at tsq ≫ 1 that makes the Euclidean norm of x scale as
        # 1/√‖B‖ ≪ 1, so ‖A x − λ B x‖ is trivially small in absolute terms
        # even for a garbage x — the check fires after numiter=1 and returns
        # essentially the initial guess. Pin tol at machine ε to force LOBPCG
        # to actually iterate up to maxiter; we then check the *physical*
        # relative residual ourselves.
        # Kept at machine ε so LOBPCG's own (absolute, hence meaningless here)
        # test never terminates a burst early. The real stopping decision is the
        # relative-residual check between bursts, below.
        tol_abs = DTYPE(1e-14)

        debuglog(dlog, "[lobpcg] call_id=$(call_id) tsq=$(round(tsq, sigdigits=4)) β_B=$(round(β_B, sigdigits=4)) κ_B=$(round(κ_B, sigdigits=4)) cheby_deg=$cheby_deg$(lobpcg_cheby_degree > 0 ? " (pinned)" : (runaway ? " (runaway→min)" : " (adaptive)")) rel_tol=$(lobpcg_rel_tol) burst=$(lobpcg_inner_maxiter)×$(lobpcg_max_restarts) dim=$n")

        # `not_zeros=true`: tell LOBPCG our initial X0 is populated (from
        # warm-start), skipping a safety-check code path that can trigger a
        # premature CholQR PosDefException on our problem shape.
        # We deliberately skip the `C = Q` constraint arg and rely on the
        # manual projection inside A_op/B_op/P_op — LOBPCG's `Constraint`
        # path calls into an extra CholQR that also fails on complex vectors
        # with our conditioning.
        # Retry policy for LOBPCG's internal CholQR PosDefException. This
        # happens when the X'BX gram matrix goes near-singular — search
        # directions have collapsed to a lower-dimensional subspace, usually
        # after the warm-start x is already very close to the true eigenvector.
        # Retry with a small random perturbation on X0: pushes X back to
        # full column rank without moving the physics.
        function _lobpcg_with_retries(X0_in, maxit::Int)
            local res_out
            X0_cur = X0_in
            for attempt in 1:3
                try
                    res_out = lobpcg(A_op, B_op, false, X0_cur;
                                     P = P_op, tol = Float64(tol_abs),
                                     maxiter = maxit, not_zeros = true)
                    return res_out
                catch e
                    if isa(e, LinearAlgebra.PosDefException)
                        noise_amp = DTYPE(1e-6) * DTYPE(10.0)^(attempt - 1)
                        debuglog(dlog, "[lobpcg]   CholQR PosDefException (attempt $attempt) — retrying with noise ε=$(round(noise_amp, sigdigits=3))")
                        X0_cur = X0_in .+ noise_amp .* randn(T, size(X0_in))
                        col = view(X0_cur, :, 1); col ./= norm(col)
                    else
                        rethrow(e)
                    end
                end
            end
            error("LOBPCG hit CholQR PosDefException on 3 successive perturbation retries; " *
                  "tsq=$tsq dim=$n call_id=$call_id")
        end
        t_lobpcg = time()
        # Burst-and-check: LOBPCG's absolute residual test cannot be trusted at
        # large tsq (see `tol_abs` above), so run it in short bursts and apply
        # our own relative test on the returned Ritz pair, restarting from that
        # block if it isn't good enough yet. The two extra operator applies per
        # check are cheap next to a burst of `lobpcg_inner_maxiter` iterations,
        # each of which costs ~cheby_deg contractions.
        #
        # Cost of restarting, stated honestly: `lobpcg(...)` copies X0 and builds
        # a fresh iterator, so the conjugate-direction block P starts at zero
        # each burst, and LOBPCG only uses P from its iteration 3 onward. Each
        # extra burst therefore spends 2 iterations without the "locally
        # optimal" acceleration — at most 16 of the 96-iteration budget if a
        # block needs every restart, and exactly zero in the common case where
        # one burst suffices. (Switching to the `lobpcg!`/LOBPCGIterator API
        # would not avoid this: it resets `iteration[] = 1` too.) The logged
        # `bursts=`/`iters=` pair makes the penalty measurable, so if hard
        # blocks are routinely exhausting all restarts, raise
        # `lobpcg_inner_maxiter` and lower `lobpcg_max_restarts`.
        local res
        X_cur = X0
        n_bursts = 0
        total_iters = 0
        rel_check = DTYPE(NaN)
        for restart in 1:lobpcg_max_restarts
            res = _lobpcg_with_retries(X_cur, lobpcg_inner_maxiter)
            n_bursts += 1
            total_iters += res.iterations
            λ_try = res.λ[1]
            x_try = Vector{eltype(v_init)}(res.X[:, 1])
            Ax_t = A_apply(x_try)
            Bx_t = B_apply(x_try)
            rel_check = norm(Ax_t .- λ_try .* Bx_t) /
                        max(abs(λ_try) * norm(Bx_t), norm(Ax_t), eps(real(DTYPE)))
            # Deliberately NOT `|| res.converged`: LOBPCG's own flag comes from
            # its absolute residual test, which is the untrustworthy quantity
            # this whole loop exists to replace (at tsq≫1 it reports success on
            # a garbage vector). Only our relative test may stop the loop.
            if rel_check <= DTYPE(lobpcg_rel_tol)
                break
            end
            X_cur = reshape(x_try, :, 1)
        end
        wall_lobpcg = time() - t_lobpcg
        debuglog(dlog, "[lobpcg]   bursts=$n_bursts iters=$total_iters rel_resid=$(round(rel_check, sigdigits=3)) (tol $(lobpcg_rel_tol)) | contractions A=$(mv_A[]) B=$(mv_B[]) prec=$(mv_prec[]) aux=$(mv_aux[])")

        λ = res.λ[1]
        x = copy(res.X[:, 1])
        # NOTE: no post-hoc Q-projection here. With B_eff sandwiching C_H²
        # between P̃'s and adding v_par as an identity block on Q, any Ritz
        # pair with λ ≠ 0 is forced to satisfy q = 0 (see the derivation in
        # the comment on B_apply). Post-projecting would only strip
        # roundoff-level Q content, but would break the LOBPCG residual
        # identity (A_apply annihilates Q, B_eff carries v_par through, so
        # r_Q = −λ·v_par: removing v_par from x adds this term back onto x's
        # residual). Just Euclidean-normalize.
        nx = norm(x)
        if nx > 0
            x = x ./ nx
        end
        # `converged` reports OUR relative test only. res.converged is derived
        # from the absolute test and would report success on a garbage vector at
        # large tsq, which is what the burst loop exists to avoid trusting.
        info_out = (numops = matvec_counter === nothing ? 0 : matvec_counter[],
                     converged = rel_check <= DTYPE(lobpcg_rel_tol) ? 1 : 0,
                     numiter = total_iters)
        # Extra diagnostics (4 additional block contractions) only when a debug
        # log is being written; they do not affect the returned pair.
        if dlog isa ActiveDebugLogger
            # LOBPCG's OWN residual on the raw returned block (no projection, no
            # renormalization). If this disagrees with the freshly-recomputed
            # residual below, LOBPCG's cached A_block/B_block has drifted from
            # A·X, B·X during the linear-recombination sweeps.
            lobpcg_own_resid = length(res.residual_norms) >= 1 ? real(res.residual_norms[1]) : NaN
            x_int_raw = Vector{eltype(v_init)}(res.X[:, 1])
            Ax_raw = A_apply(x_int_raw)
            Bx_raw = B_apply(x_int_raw)
            r_raw  = Ax_raw .- λ .* Bx_raw
            raw_resid_norm = norm(r_raw)
            qparallel = Q === nothing ? 0.0 : Float64(norm(Q' * x_int_raw))
            debuglog(dlog, "[lobpcg]   done in $(round(wall_lobpcg, digits=2))s converged=$(info_out.converged) numiter=$(info_out.numiter) λ=$(round(real(λ), sigdigits=6))")
            debuglog(dlog, "[lobpcg]   [raw] lobpcg_resid=$(round(lobpcg_own_resid, sigdigits=3)) my_resid_on_x_int=$(round(raw_resid_norm, sigdigits=3)) ratio=$(round(raw_resid_norm / max(lobpcg_own_resid, eps()), sigdigits=3)) ‖Q'x_int‖=$(round(qparallel, sigdigits=3)) ‖x_int‖=$(round(norm(x_int_raw), sigdigits=4))")

            # Diagnostic: physical residual on the returned Ritz pair (unscaled A,B).
            Ax = A_apply(x); Bx = B_apply(x)
            r = Ax .- λ .* Bx
            denom = max(abs(λ) * norm(Bx), norm(Ax), eps())
            rel_resid = norm(r) / denom
            xAx = real(dot(x, Ax))
            xBx = real(dot(x, Bx))
            cH2 = tsq > 0 ? (xBx - real(dot(x, x))) / tsq : DTYPE(NaN)
            debuglog(dlog, "[lobpcg]   ⟨A⟩=$(round(xAx, sigdigits=4)) ⟨B⟩=$(round(xBx, sigdigits=4)) ⟨C_H²⟩=$(round(cH2, sigdigits=4)) rel_resid=$(round(rel_resid, sigdigits=3))")
        end

        return λ, x, info_out
    end

    # One inner solve at θ², with automatic ortho-weight amplification on
    # retry if the returned x has too much P-component. Under :eigsolve, at
    # small block dim bypasses KrylovKit for dense LAPACK. Under :geneig,
    # single geneigsolve call, no dense fallback / krylov retries.
    function eigsolve_at_thetasq(tsq, v_init)
        local En_min, x, info, B_val, p_wt
        dim = length(v_init)
        used_dense = inner_primitive === :eigsolve && use_dense_at_small_dim && dim <= dense_max_dim
        for retry in 0:max_ow_retries
            ow = current_ow[]
            if inner_primitive === :geneig
                En_val, x_try, info_local = _call_geneig_inner(tsq, v_init, ow)
                total_numops += info_local.numops
            elseif inner_primitive === :lobpcg
                En_val, x_try, info_local = _call_lobpcg_inner(tsq, v_init, ow)
                total_numops += info_local.numops
            elseif used_dense
                Heff = build_Heff(tsq, ow, dim)
                dense_eig = eigen(Hermitian(Heff))
                # Move scalar eigenvalue to CPU (dense_eig.values may be a
                # CuVector; direct getindex would trigger a scalar-indexing
                # error). The eigenvector stays on device for the next matvec.
                En_val = Array(dense_eig.values)[1]
                x_try = dense_eig.vectors[:, 1]
                info_local = (numops = dim, converged = 1)
                total_numops += info_local.numops
            else
                # (A + E) KrylovKit with convergence/residual retry: start from
                # v_init, and if the returned Ritz pair fails the residual or
                # info.converged checks, retry from v_init + noise (increasing).
                mv = make_matvec(tsq, ow)
                v_try = copy(v_init)
                v_try ./= max(norm(v_try), eps())
                # Track the best attempt so far (guaranteed to be updated on
                # every iteration, so !accepted branch has valid values).
                En_val_local = zero(eltype(v_init))
                x_try_local = copy(v_try)
                info_try = (numops = 0, converged = 0)
                accepted = false
                noise_scale = DTYPE(0.0)
                for kryl_retry in 0:max_kryl_retries
                    En, gs, info_try_ = eigsolve(
                        mv, v_try, 1, :SR;
                        verbosity = max(0, dispon - 2), ishermitian = true, eig_kwargs...,
                    )
                    total_numops += info_try_.numops
                    imin = argmin(En)
                    x_cand = gs[imin]
                    nx = norm(x_cand)
                    if nx > 0
                        x_cand = x_cand ./ nx
                    end
                    # Residual check on the returned Ritz pair
                    Hx = mv(copy(x_cand))
                    resid = norm(Hx - En[imin] * x_cand) / max(norm(x_cand), eps())
                    # Always record the latest attempt so !accepted branch has a fallback
                    En_val_local = En[imin]
                    x_try_local = x_cand
                    info_try = info_try_
                    if info_try_.converged >= 1 && resid < kryl_resid_tol
                        accepted = true
                        break
                    end
                    # Failed — retry with warm-start + noise
                    noise_scale = kryl_retry == 0 ? DTYPE(0.05) : min(noise_scale * DTYPE(2.0), DTYPE(1.0))
                    _trace("  kryl_retry $(kryl_retry+1): converged=$(info_try_.converged) resid=$(round(resid, sigdigits=3)) → warm+noise scale=$noise_scale")
                    # v_init is host-side in the QCQP path (coerced at entry), so
                    # keep the noise on the host too — a movedevice here would put
                    # it on the GPU and mismatch v_try in the mix below.
                    noise = randn(eltype(v_init), length(v_init))
                    noise ./= max(norm(noise), eps())
                    v_try = (one(DTYPE) - noise_scale) .* copy(v_init) .+ noise_scale .* noise
                    v_try ./= max(norm(v_try), eps())
                end
                if !accepted
                    _trace("  kryl_retry exhausted at θ²=$tsq ow=$ow — accepting last Ritz pair")
                    En_val = En_val_local
                    x_try = x_try_local
                    info_local = info_try
                else
                    En_val = En_val_local
                    x_try = x_try_local
                    info_local = info_try
                end
            end
            p_wt = p_component(x_try)

            if p_wt <= p_tolerance || ow >= max_ortho_weight
                # Accept: either well within (1-P) subspace or we've maxed out.
                En_min = En_val
                x = x_try
                info = info_local
                B_val = B_expectation(x)
                if p_wt > p_tolerance
                    _trace("  ow_retry maxed out: p_wt=$p_wt at ow=$ow tsq=$tsq")
                end
                break
            else
                # Retry with 10× the ortho-weight
                _trace("  ow_retry: p_wt=$p_wt > $p_tolerance at ow=$ow → ow×10")
                current_ow[] = min(ow * DTYPE(10.0), max_ortho_weight)
                continue
            end
        end

        return En_min, x, info, B_val
    end

    # ---- Step 1: try current θ² (warm-started) ----
    tsq = max(thetasq_ref[], zero(DTYPE))
    _trace("enter dim=$(length(initvec)) ε=$epsilon_loc warm-θ²=$tsq")
    En_now, x_now, _, B_now = eigsolve_at_thetasq(tsq, initvec)
    _trace("step1 warm     θ²=$(tsq) B=$(B_now) En=$(En_now)")

    if dispon >= 2
        println("    QCQP start: θ²=$tsq, ⟨x|B̃|x⟩=$B_now, ε_loc=$epsilon_loc")
    end

    # Case A — warm-start is already essentially on the constraint boundary.
    # A wider tolerance (warm_accept_tol >> bisect_tol) means: if the warm-start
    # value is close enough to ε_loc, accept without a full bisection. Assumes
    # neighboring sites' optimal θ² don't differ much, so warm-start is likely
    # near-optimal.
    if abs(B_now - epsilon_loc) / max(abs(epsilon_loc), eps()) <= warm_accept_tol
        thetasq_ref[] = tsq
        _ema_update(tsq)
        _trace("exit tight_warm θ²=$tsq B=$B_now (|ΔB|/ε=$(round(abs(B_now-epsilon_loc)/epsilon_loc, sigdigits=3)))")
        return [En_now], [x_now], (numops = total_numops, qcqp_thetasq = tsq,
                                    qcqp_B = B_now, qcqp_expects = total_expect_calls,
                                    qcqp_iters = 0, qcqp_status = :tight_warm)
    end

    # Establish a bracket [tsq_lo, tsq_hi] with B_lo > ε > B_hi. Prefer the
    # secant path (log-log Newton on the model ⟨C_H²⟩ ≈ A·(θ²)^(−s)) when a
    # persistent slope is provided by the caller: it warm-starts from the last
    # block's slope and converges superlinearly near the target. On straddle
    # we fall through to the shared bisection loop; on same-side stagnation
    # (slope collapse) we exit :infeasible; on ceiling breach we exit
    # :ceiling_hit. Legacy geometric ×2/×10 is retained as the fallback path
    # for callers that pass `slope_ref = nothing`.

    local tsq_lo, tsq_hi, En_lo, x_lo, B_lo, En_hi, x_hi, B_hi

    if slope_ref !== nothing
        # ---- Secant search (safeguarded Newton in log-log) ----
        # Sign convention: model is B(θ²) = A · (θ²)^(−s) with s > 0. So
        #   log θ²_next = log θ²_curr + (log B_curr − log ε) / s
        # sends B toward ε in one step when the model is exact. Two guards
        # matter most: a per-step log-cap prevents runaway when s is tiny
        # (near-flat B curve), and a slope floor triggers :infeasible when
        # the local ⟨C_H²⟩ minimum sits above ε_loc.

        tsq_prev = tsq
        B_prev = B_now
        En_prev, x_prev = En_now, x_now
        s = clamp(slope_ref[], DTYPE(slope_min), DTYPE(slope_max))
        _trace("secant enter s=$s tsq_prev=$tsq_prev B_prev=$B_prev")

        straddled = false
        # Buffer the last-two samples so we can either update the persistent
        # slope on exit or drop into bisection on a straddle.
        tsq_lo, B_lo, En_lo, x_lo = tsq_prev, B_prev, En_prev, x_prev
        tsq_hi, B_hi, En_hi, x_hi = tsq_prev, B_prev, En_prev, x_prev
        best_s = s
        flat_streak = 0            # consecutive flat-slope samples (see below)

        # sign of B - ε: +1 = above (need higher θ²), -1 = below (need lower θ²).
        sgn_prev = sign(B_prev - epsilon_loc)
        # If B_prev is essentially ε already, tight-warm would have exited. So
        # sgn_prev is nonzero here — but be defensive.
        if iszero(sgn_prev)
            sgn_prev = one(DTYPE)
        end

        # Guard against a zero warm-start θ² when we need to grow it.
        if tsq_prev <= eps()
            tsq_prev = DTYPE(1e-6)
        end

        secant_success = false
        for k in 1:secant_maxiter
            # Step: log θ²_next = log θ²_prev + Δ, Δ clamped to secant_log_step_cap.
            Δ = (log(B_prev) - log(epsilon_loc)) / s
            Δ = clamp(Δ, -DTYPE(secant_log_step_cap), DTYPE(secant_log_step_cap))
            tsq_next = tsq_prev * exp(Δ)

            if tsq_next > theta_sq_max
                _trace("secant#$k tsq_next=$tsq_next > theta_sq_max — ceiling")
                tsq_lo, B_lo, En_lo, x_lo = tsq_prev, B_prev, En_prev, x_prev
                thetasq_ref[] = tsq_prev
                _ema_update(tsq_prev)
                return [En_prev], [x_prev], (numops = total_numops, qcqp_thetasq = tsq_prev,
                                              qcqp_B = B_prev, qcqp_expects = total_expect_calls,
                                              qcqp_iters = k, qcqp_status = :ceiling_hit)
            end
            if tsq_next < eps()
                # Shrinking below machine ε: treat as θ² = 0 candidate.
                En_next, x_next, _, B_next = eigsolve_at_thetasq(zero(DTYPE), x_prev)
                # Log what the θ²=0 solve actually DID, not just its endpoint.
                # At θ²=0 the block maximizes ⟨M⟩ with no constraint, and max ⟨M⟩
                # favours LOW Pauli weight (M's eigenvalue is 3^-weight), i.e.
                # local operators, which have LARGE ⟨C_H²⟩. So B should RISE
                # here and the constraint should re-activate within a few
                # blocks. Observed instead: `unconstrained=13/14` with global
                # ⟨C_H²⟩ stuck ~38 % below target and ⟨M⟩ ~15 % below the
                # on-target value. Either the solve is not moving the state
                # (env/local-optimum trap) or it is not finding the true max.
                # These three deltas discriminate the two.
                _trace("secant#$k θ²≈0 B: $(B_prev) → $(B_next) (ε=$epsilon_loc)  En: $(En_prev) → $(En_next)  |Δx|=$(norm(x_next .- x_prev))")
                if B_next <= epsilon_loc * (one(DTYPE) + DTYPE(bisect_tol))
                    thetasq_ref[] = zero(DTYPE)
                    return [En_next], [x_next], (numops = total_numops, qcqp_thetasq = zero(DTYPE),
                                                  qcqp_B = B_next, qcqp_expects = total_expect_calls,
                                                  qcqp_iters = k, qcqp_status = :unconstrained)
                end
                # Otherwise: B(0) > ε, use it as the "high" bracket and let
                # bisection tighten.
                tsq_lo, B_lo, En_lo, x_lo = zero(DTYPE), B_next, En_next, x_next
                tsq_hi, B_hi, En_hi, x_hi = tsq_prev, B_prev, En_prev, x_prev
                straddled = true
                break
            end

            En_next, x_next, _, B_next = eigsolve_at_thetasq(tsq_next, x_prev)
            _trace("secant#$k θ²=$(tsq_next) B=$(B_next) (Δ=$(round(Δ, sigdigits=3)) s=$(round(s, sigdigits=3)))")

            # Local slope from (tsq_prev, B_prev) → (tsq_next, B_next).
            dlog_tsq = log(tsq_next) - log(tsq_prev)
            dlog_B   = log(B_next)   - log(B_prev)
            # Slope-based infeasibility detection. Three guards, all added after
            # this test was measured to produce FALSE POSITIVES at a ~67 % rate
            # (6-rep ν(τ) ladder, L=8: 4 of 6 runs froze at ⟨C_H²⟩≈0.0506 while
            # the target fell 0.05→0.0289→0.0167, every block reporting
            # `infeasible`; reps 4 and 6 reached the same targets, proving they
            # were feasible all along).
            #
            #   (a) require a MEANINGFUL θ² step before believing a flat slope —
            #       a converging secant takes ever-smaller steps, so dlog_B
            #       shrinks and s_local reads "flat" precisely when the block is
            #       about to succeed;
            #   (b) require CONSECUTIVE flat readings instead of latching
            #       forever on the first one;
            #   (c) require B to still be materially above ε — if we are already
            #       near the target the constraint is plainly satisfiable.
            if abs(dlog_tsq) > eps() && isfinite(dlog_B)
                s_local = -dlog_B / dlog_tsq                  # +ve when B decreases as θ² grows
                step_is_informative = abs(dlog_tsq) > DTYPE(slope_min_logstep)
                far_from_target = (B_next - epsilon_loc) / max(abs(epsilon_loc), eps()) >
                                  DTYPE(infeasibility_rel_tol)
                if step_is_informative && s_local < DTYPE(slope_infeasible_floor) && far_from_target
                    flat_streak += 1
                    _trace("secant#$k slope $s_local < floor $slope_infeasible_floor (flat_streak=$flat_streak)")
                else
                    flat_streak = 0
                end
                if s_local > 0
                    best_s = clamp(DTYPE(s_local), DTYPE(slope_min), DTYPE(slope_max))
                    s = best_s
                end
            end

            # Convergence check (bisect_tol precision).
            if abs(B_next - epsilon_loc) / max(abs(epsilon_loc), eps()) < DTYPE(bisect_tol)
                thetasq_ref[] = tsq_next
                _ema_update(tsq_next)
                slope_ref[] = DTYPE(slope_ema_alpha) * best_s +
                              (one(DTYPE) - DTYPE(slope_ema_alpha)) * slope_ref[]
                slope_ref[] = clamp(slope_ref[], DTYPE(slope_min), DTYPE(slope_max))
                _trace("secant exit converged θ²=$tsq_next B=$B_next s_ema=$(slope_ref[])")
                return [En_next], [x_next], (numops = total_numops, qcqp_thetasq = tsq_next,
                                              qcqp_B = B_next, qcqp_expects = total_expect_calls,
                                              qcqp_iters = k, qcqp_status = :converged)
            end

            # Straddle → hand off to bisection.
            sgn_next = sign(B_next - epsilon_loc)
            if sgn_next != sgn_prev && !iszero(sgn_next)
                if B_prev > epsilon_loc
                    tsq_lo, B_lo, En_lo, x_lo = tsq_prev, B_prev, En_prev, x_prev
                    tsq_hi, B_hi, En_hi, x_hi = tsq_next, B_next, En_next, x_next
                else
                    tsq_lo, B_lo, En_lo, x_lo = tsq_next, B_next, En_next, x_next
                    tsq_hi, B_hi, En_hi, x_hi = tsq_prev, B_prev, En_prev, x_prev
                end
                straddled = true
                secant_success = true
                _trace("secant#$k straddled → bisect [lo=$tsq_lo B_lo=$B_lo | hi=$tsq_hi B_hi=$B_hi]")
                break
            end

            # Same-side stagnation + repeated flat slope → declare infeasibility.
            if flat_streak >= slope_infeasible_streak && k >= 2
                # Exit at the tighter of the two same-side iterates.
                tsq_exit = B_next < B_prev ? tsq_next : tsq_prev
                B_exit   = B_next < B_prev ? B_next   : B_prev
                En_exit  = B_next < B_prev ? En_next  : En_prev
                x_exit   = B_next < B_prev ? x_next   : x_prev
                thetasq_ref[] = tsq_exit
                _ema_update(tsq_exit)
                _trace("secant exit infeasible θ²=$tsq_exit B=$B_exit")
                return [En_exit], [x_exit], (numops = total_numops, qcqp_thetasq = tsq_exit,
                                              qcqp_B = B_exit, qcqp_expects = total_expect_calls,
                                              qcqp_iters = k, qcqp_status = :infeasible)
            end

            tsq_prev, B_prev, En_prev, x_prev = tsq_next, B_next, En_next, x_next
            sgn_prev = sgn_next
        end

        # Fell out of the secant loop without converging or straddling.
        if !straddled
            # Same-sided the whole way. Establish a bracket from whichever side
            # of ε we're stuck on: if B_prev > ε (Case-B-like), we need a lower
            # tsq to bring B below ε; if B_prev < ε (Case-C-like), we need θ²=0
            # as the "high" endpoint. In practice this branch is rare — the
            # secant_maxiter cap is deliberately large enough (~5) that a
            # well-conditioned block converges or straddles inside it.
            if B_prev > epsilon_loc
                # Grow one more time under the cap; that's this block's "hi".
                Δ = DTYPE(secant_log_step_cap)
                tsq_hi = min(tsq_prev * exp(Δ), theta_sq_max)
                En_hi, x_hi, _, B_hi = eigsolve_at_thetasq(tsq_hi, x_prev)
                if B_hi > epsilon_loc
                    # No progress even with a full cap step — infeasibility.
                    thetasq_ref[] = tsq_hi
                    _ema_update(tsq_hi)
                    _trace("secant exit infeasible (post-loop) θ²=$tsq_hi B=$B_hi")
                    return [En_hi], [x_hi], (numops = total_numops, qcqp_thetasq = tsq_hi,
                                              qcqp_B = B_hi, qcqp_expects = total_expect_calls,
                                              qcqp_iters = secant_maxiter, qcqp_status = :infeasible)
                end
                tsq_lo, B_lo, En_lo, x_lo = tsq_prev, B_prev, En_prev, x_prev
                straddled = true
            else
                # Feasible warm-start that never crossed ε: probe θ² = 0.
                En0, x0, _, B0 = eigsolve_at_thetasq(zero(DTYPE), x_prev)
                if B0 <= epsilon_loc * (one(DTYPE) + DTYPE(bisect_tol))
                    thetasq_ref[] = zero(DTYPE)
                    _trace("secant exit unconstrained (post-loop) θ²=0 B=$B0")
                    return [En0], [x0], (numops = total_numops, qcqp_thetasq = zero(DTYPE),
                                          qcqp_B = B0, qcqp_expects = total_expect_calls,
                                          qcqp_iters = secant_maxiter, qcqp_status = :unconstrained)
                end
                tsq_lo, B_lo, En_lo, x_lo = zero(DTYPE), B0, En0, x0
                tsq_hi, B_hi, En_hi, x_hi = tsq_prev, B_prev, En_prev, x_prev
                straddled = true
            end
        end

        # Refresh persistent slope estimate with the best measurement we saw.
        slope_ref[] = DTYPE(slope_ema_alpha) * best_s +
                      (one(DTYPE) - DTYPE(slope_ema_alpha)) * slope_ref[]
        slope_ref[] = clamp(slope_ref[], DTYPE(slope_min), DTYPE(slope_max))

    elseif B_now > epsilon_loc
        # ---- Case B (legacy): bracket UP from warm-start ----
        # Adaptive step: start narrow (×2, tuned for warm-started scenarios
        # where θ* is close to warm) then switch to ×10 for far-away θ*.
        tsq_lo = tsq
        B_lo = B_now
        En_lo, x_lo = En_now, x_now

        tsq_hi = tsq > eps() ? tsq * DTYPE(2.0) : one(DTYPE)
        En_hi, x_hi, B_hi = En_now, x_now, B_now
        local_hi_found = false
        local_infeasible = false
        B_prev = B_now
        local_ceiling_hit = false
        for k in 1:bracket_expansions
            # Refuse to push θ² past theta_sq_max — beyond that the operator
            # diagonal amplifies GPU fp noise past KrylovKit's Hermiticity tol.
            if tsq_hi > theta_sq_max
                _trace("step3 bracket-up: tsq_hi=$tsq_hi > theta_sq_max=$theta_sq_max — bailing")
                local_ceiling_hit = true
                break
            end
            En_hi, x_hi, _, B_hi = eigsolve_at_thetasq(tsq_hi, x_lo)
            _trace("step3 bracket-up#$k θ²=$(tsq_hi) B=$(B_hi)")
            if B_hi <= epsilon_loc
                local_hi_found = true
                break
            end
            # Infeasibility: if B_hi stops decreasing meaningfully, we've hit
            # the block's argmin B̃, which sits above ε_loc.
            if k >= 2 && (B_prev - B_hi) / max(abs(B_prev), eps()) < infeasibility_rel_tol
                _trace("step3 bracket-up: B saturated (prev=$B_prev now=$B_hi rel=$(round((B_prev-B_hi)/B_prev, sigdigits=3))) — locally infeasible")
                local_infeasible = true
                break
            end
            tsq_lo, B_lo, En_lo, x_lo = tsq_hi, B_hi, En_hi, x_hi
            B_prev = B_hi
            # First 3 tries: ×2 growth (narrow). Beyond that: ×10 (rapid).
            tsq_hi *= (k < 3 ? DTYPE(2.0) : DTYPE(10.0))
        end

        if local_ceiling_hit
            thetasq_ref[] = tsq_lo
            _ema_update(tsq_lo)
            _trace("exit ceiling_hit θ²=$tsq_lo B=$B_lo (θ² would exceed theta_sq_max)")
            return [En_lo], [x_lo], (numops = total_numops, qcqp_thetasq = tsq_lo,
                                      qcqp_B = B_lo, qcqp_expects = total_expect_calls,
                                      qcqp_iters = 0, qcqp_status = :ceiling_hit)
        end

        if local_infeasible
            thetasq_ref[] = tsq_hi
            _ema_update(tsq_hi)
            _trace("exit infeasible θ²=$tsq_hi B=$B_hi (block argmin B̃ > ε_loc)")
            return [En_hi], [x_hi], (numops = total_numops, qcqp_thetasq = tsq_hi,
                                      qcqp_B = B_hi, qcqp_expects = total_expect_calls,
                                      qcqp_iters = 0, qcqp_status = :infeasible)
        end

        if !local_hi_found
            @warn "local_step_qcqp: could not bracket upper θ² after $bracket_expansions expansions " *
                  "(final tsq_hi=$tsq_hi, B_hi=$B_hi, ε_loc=$epsilon_loc). Returning largest-θ² solution."
            thetasq_ref[] = tsq_hi
            _ema_update(tsq_hi)
            _trace("exit bracket_failed θ²=$tsq_hi B=$B_hi")
            return [En_hi], [x_hi], (numops = total_numops, qcqp_thetasq = tsq_hi,
                                      qcqp_B = B_hi, qcqp_expects = total_expect_calls,
                                      qcqp_iters = 0, qcqp_status = :bracket_failed)
        end

    else  # B_now < epsilon_loc — Case C (legacy): strictly feasible warm-start.
        if tsq <= zero(DTYPE) + eps()
            thetasq_ref[] = zero(DTYPE)
            _trace("exit unconstrained θ²=0 B=$B_now (warm=0 feasible)")
            return [En_now], [x_now], (numops = total_numops, qcqp_thetasq = zero(DTYPE),
                                        qcqp_B = B_now, qcqp_expects = total_expect_calls,
                                        qcqp_iters = 0, qcqp_status = :unconstrained)
        end

        tsq_hi = tsq
        B_hi = B_now
        En_hi, x_hi = En_now, x_now
        tsq_lo = zero(DTYPE)
        B_lo = zero(DTYPE)
        En_lo, x_lo = En_now, x_now

        tsq_try = tsq / DTYPE(2.0)
        found_narrow = false
        for k in 1:bracket_expansions
            En_try, x_try, _, B_try = eigsolve_at_thetasq(tsq_try, x_hi)
            _trace("step3 bracket-down#$k θ²=$(tsq_try) B=$(B_try)")
            if B_try > epsilon_loc
                tsq_lo, B_lo, En_lo, x_lo = tsq_try, B_try, En_try, x_try
                found_narrow = true
                break
            else
                tsq_hi, B_hi, En_hi, x_hi = tsq_try, B_try, En_try, x_try
                tsq_try = tsq_try * (k < 3 ? DTYPE(0.5) : DTYPE(0.1))
                if tsq_try < eps()
                    break
                end
            end
        end

        if !found_narrow
            En0, x0, _, B0 = eigsolve_at_thetasq(zero(DTYPE), x_hi)
            _trace("step3 bracket-down fallback θ²=0 B=$(B0)")
            if B0 <= epsilon_loc * (1 + bisect_tol)
                thetasq_ref[] = zero(DTYPE)
                _trace("exit unconstrained θ²=0 B=$B0")
                return [En0], [x0], (numops = total_numops, qcqp_thetasq = zero(DTYPE),
                                      qcqp_B = B0, qcqp_expects = total_expect_calls,
                                      qcqp_iters = 0, qcqp_status = :unconstrained)
            end
            tsq_lo = zero(DTYPE)
            B_lo, En_lo, x_lo = B0, En0, x0
        end

        _trace("step3 bracket-down established [lo=$tsq_lo B_lo=$B_lo | hi=$tsq_hi B_hi=$B_hi]")
    end

    # ---- Step 4: bisect [tsq_lo, tsq_hi] with B_lo > ε_loc > B_hi ----
    En_best, x_best, B_best, tsq_best = En_hi, x_hi, B_hi, tsq_hi
    iters_used = 0
    for iter in 1:bisect_maxiter
        iters_used = iter
        tsq_mid = DTYPE(0.5) * (tsq_lo + tsq_hi)
        En_mid, x_mid, _, B_mid = eigsolve_at_thetasq(tsq_mid, x_best)
        _trace("step4 bisect#$iter θ²=$(tsq_mid) B=$(B_mid) [lo=$tsq_lo hi=$tsq_hi]")

        En_best, x_best, B_best, tsq_best = En_mid, x_mid, B_mid, tsq_mid

        if abs(B_mid - epsilon_loc) / epsilon_loc < bisect_tol
            break
        end

        if B_mid > epsilon_loc
            tsq_lo = tsq_mid
        else
            tsq_hi = tsq_mid
        end
    end

    thetasq_ref[] = tsq_best
    _ema_update(tsq_best)
    status = abs(B_best - epsilon_loc) / epsilon_loc < bisect_tol ? :converged : :maxiter
    _trace("exit $status θ²=$tsq_best B=$B_best after $iters_used bisects")
    return [En_best], [x_best], (numops = total_numops, qcqp_thetasq = tsq_best,
                                  qcqp_B = B_best, qcqp_expects = total_expect_calls,
                                  qcqp_iters = iters_used, qcqp_status = status)
end
