# Device-resident block-1 preconditioned LOBPCG, Chebyshev-in-B preconditioner,
# Lanczos weak-duality error bar.
#
# All routines are written with broadcasting and `dot` only, so the same code runs
# on host `Vector`s (CPU tests / fallback) and on `CuVector`s (production: every
# full-dimension vector stays on the GPU; only the ≤3×3 Rayleigh–Ritz problem and
# the Lanczos tridiagonal are solved on the host).

"""
    randn_vec(V, n)

Standard-normal vector of array type `V` (e.g. `Vector{Float64}`,
`CuVector{ComplexF64}`) and length `n`; complex: `randn + i·randn`. The GPU
method lives in `gpu.jl`. Uses the global (resp. CUDA) RNG; see `seed_rng!`.
"""
randn_vec(::Type{Vector{T}}, n::Int) where {T<:Real} = randn(T, n)
randn_vec(::Type{Vector{Complex{T}}}, n::Int) where {T<:Real} = randn(T, n) .+ im .* randn(T, n)

_vnorm(x) = sqrt(real(dot(x, x)))

# smallest-μ solution of the small generalized eigenproblem GA a = μ GB a, or
# `nothing` if GB is numerically singular (⇒ the caller drops a search direction).
function _rr(GA::AbstractMatrix, GB::AbstractMatrix)
    evB = eigen(Hermitian(GB)).values
    (minimum(evB) < 1e-12 * max(maximum(evB), eps())) && return nothing
    E = eigen(Hermitian(GA), Hermitian(GB))
    i = argmin(E.values)
    (E.values[i], E.vectors[:, i])
end

_gram(V, MV) = [dot(V[i], MV[j]) for i in eachindex(V), j in eachindex(MV)]

"""
    lobpcg(applyA, applyB, applyP, x0; tol=1e-4, maxiter=300) -> (x, λ)

Smallest λ of the Hermitian pencil `A x = λ B x` (B ≻ 0) by block-1 LOBPCG with
preconditioner `applyP ≈ B⁻¹`. Stops when `‖Ax − λBx‖ ≤ tol·(|λ|‖Bx‖ + ‖Ax‖)`.
Only `A W`, `B W` are applied fresh per iteration (A, B images of X and P are
updated by linear combination). If the Gram matrix of `[X, W, P]` becomes
singular as the block collapses near convergence, `P` is dropped for that step.
`x` is returned B-normalized.
"""
function lobpcg(applyA, applyB, applyP, x0::AbstractVector; tol::Float64=1e-4, maxiter::Int=300)
    X = copy(x0); AX = applyA(X); BX = applyB(X)
    s = 1 / sqrt(real(dot(X, BX))); X .*= s; AX .*= s; BX .*= s
    λ = real(dot(X, AX))
    P = nothing; AP = nothing; BP = nothing
    for _ in 1:maxiter
        R = AX .- λ .* BX
        scale = abs(λ) * _vnorm(BX) + _vnorm(AX)
        _vnorm(R) <= tol * max(scale, eps()) && break
        W = applyP(R); AW = applyA(W); BW = applyB(W)
        haveP = P !== nothing
        V  = haveP ? (X, W, P)    : (X, W)
        AV = haveP ? (AX, AW, AP) : (AX, AW)
        BV = haveP ? (BX, BW, BP) : (BX, BW)
        rr = _rr(_gram(V, AV), _gram(V, BV))
        if rr === nothing && haveP                     # block collapsed → drop P, retry [X,W]
            AV = (AX, AW); BV = (BX, BW); V = (X, W)
            rr = _rr(_gram(V, AV), _gram(V, BV)); haveP = false
        end
        rr === nothing && break                        # cannot proceed; keep current Ritz pair
        μ, a = rr
        if haveP
            Xn  = a[1].*X  .+ a[2].*W  .+ a[3].*P;  AXn = a[1].*AX .+ a[2].*AW .+ a[3].*AP
            BXn = a[1].*BX .+ a[2].*BW .+ a[3].*BP
            Pn  = a[2].*W  .+ a[3].*P;  APn = a[2].*AW .+ a[3].*AP;  BPn = a[2].*BW .+ a[3].*BP
        else
            Xn  = a[1].*X  .+ a[2].*W;  AXn = a[1].*AX .+ a[2].*AW;  BXn = a[1].*BX .+ a[2].*BW
            Pn  = a[2].*W;              APn = a[2].*AW;              BPn = a[2].*BW
        end
        sx = 1 / sqrt(real(dot(Xn, BXn))); Xn .*= sx; AXn .*= sx; BXn .*= sx
        X, AX, BX, P, AP, BP, λ = Xn, AXn, BXn, Pn, APn, BPn, μ
    end
    return X, λ
end

"""
    power_lmax(applyK, x0; iters=40) -> λ

Power-iteration estimate of λ_max(K) (‖K x‖ for the normalized iterate) from the
start vector `x0`; sets the upper end of the Chebyshev interval.
"""
function power_lmax(applyK, x0::AbstractVector; iters::Int=40)
    x = x0 ./ _vnorm(x0); λ = 0.0
    for _ in 1:iters
        y = applyK(x); λ = _vnorm(y); x = y ./ λ
    end
    λ
end

"""
    chebyshev_precond(applyK, θ, β, k; proj=nothing) -> (r -> y)

Degree-`k` shifted Chebyshev semi-iteration for `(I + θK) y = r` on the spectral
interval `[1, β]` (β ≥ 1 + θ λ_max(K)): `y ≈ (I + θK)⁻¹ r` at the cost of `k`
K-applies. With `proj`, input and output are projected (q = 0 projected path).
"""
function chebyshev_precond(applyK, θ::Float64, β::Float64, k::Int; proj=nothing)
    α = 1.0; θc = (β+α)/2; δc = (β-α)/2; σ = θc/δc
    pr = proj === nothing ? copy : proj
    r0 -> begin
        r = pr(r0); (β-α < 1e-12*β) && return (proj === nothing ? r : proj(r))
        ρp = 1/σ; yc = r ./ θc; d = copy(yc)
        for _ in 1:k
            resid = r .- (yc .+ θ .* applyK(yc))
            ρ = 1/(2σ - ρp); d = (ρ*ρp).*d .+ (2ρ/δc).*resid; yc .+= d; ρp = ρ
        end
        proj === nothing ? yc : proj(yc)
    end
end

"Chebyshev degree used in production: k = clamp(⌈2 + log10 β⌉, 2, 8)."
chebyshev_degree(β) = clamp(ceil(Int, 2 + log10(max(β, 1.0))), 2, 8)

# top-2 eigenvalues of the Hermitian operator Mop via m-step Lanczos from a
# random start, with full reorthogonalization.
function _lanczos_top2(Mop, v::AbstractVector, m::Int)
    n = length(v)
    Q = typeof(v)[]
    q = randn_vec(typeof(v), n); q ./= _vnorm(q); push!(Q, q)
    α = Float64[]; β = Float64[]
    w = Mop(q)
    a = real(dot(q, w)); push!(α, a); w .-= a .* q
    for j in 2:m
        for u in Q; w .-= dot(u, w) .* u; end
        b = _vnorm(w); (b < 1e-14) && break
        push!(β, b); q = w ./ b; push!(Q, q)
        w = Mop(q); a = real(dot(q, w)); push!(α, a); w .-= a .* q .+ β[end] .* Q[end-1]
    end
    ev = sort(eigen(SymTridiagonal(α, β)).values; rev=true)
    (ev[1], length(ev) >= 2 ? ev[2] : ev[1])
end

"""
    error_bar(Rd, applyK, proj, θ, v; m=15) -> (ν, σ2, ε, bar, accepted)

Weak-duality certificate for the converged Ritz vector `v` at multiplier θ.
With `λ̄ = ν/(1+θσ²)`, `μ̃ = λ̄θ` and the shifted operator `M = R − μ̃K`, the exact
frontier value a(σ²) satisfies `0 ≤ a(σ²) − ν ≤ λ_max(M) − ρ`, `ρ = v'Mv`; with the
residual `ε = ‖Mv − λ̄v‖` and gap `δ = λ_max(M) − λ₂(M)` the bound is
`bar = min(ε, ε²/δ)`. `accepted = ρ > λ₂(M)` (v lies in the top eigenpair's
basin). λ₂(M) comes from an `m`-step Lanczos (random start). `proj` is the
identity or the c_H-complement projector.
"""
function error_bar(Rd::AbstractVector{Float64}, applyK, proj, θ::Float64, v::AbstractVector; m::Int=15)
    v = proj(v); v ./= _vnorm(v)
    Kv = applyK(v)
    ν = real(dot(v, Rd .* v)); σ2 = max(real(dot(v, Kv)), 0.0)
    λ̄ = ν / (1 + θ*σ2); μ̃ = λ̄*θ
    Mv = Rd .* v .- μ̃ .* Kv
    r = proj(Mv .- λ̄ .* v); ε = _vnorm(r); ρ = real(dot(v, Mv))
    Mop(x) = proj(Rd .* x .- μ̃ .* applyK(x))
    λmaxM, λ2M = _lanczos_top2(Mop, v, m)
    λmaxM = max(λmaxM, ρ)
    δ = λmaxM - λ2M; bar = δ > 0 ? min(ε, ε^2/δ) : ε
    (ν=ν, σ2=σ2, ε=ε, bar=bar, accepted=(ρ > λ2M))
end
