# Exact / direct q = 0 energy-projected solvers (the PRE15 "README path").
#
# Each θ solves R c = μ (I + θK) c for the LARGEST μ with the conserved energy
# density c_H excluded; ν = c'Rc/c'c, σ² = c'Kc/c'c, τ = 1/√σ². The θ→∞ endpoint
# λ_TI = min σ² over the ansatz gives the ceiling τ_max = 1/√λ_TI.
#   dense  : build_K + nutau_sweep + lambda_TI              (M ≲ 7)
#   sparse : build_K_sparse + nutau_sweep_sparse + lambda_TI_sparse (M ≲ 9, Cholesky fill-in bound)

"""
    nutau_sweep(R, K, Q, thetas) -> NamedTuple

For each θ, solve the reduced generalized eigenproblem `R c = μ (I + θK) c` in
`range(Q)`, take the largest μ, and record ν = c'Rc/c'c, σ² = c'Kc/c'c,
τ = 1/√σ², and the lifted optimizer c_opt.
"""
function nutau_sweep(R::Diagonal, K::Symmetric, Q::AbstractMatrix, thetas::AbstractVector)
    Rf = Matrix(R)
    Kf = Matrix(K)
    Rr = Symmetric(Q' * Rf * Q)
    Kr = Symmetric(Q' * Kf * Q)
    m = size(Q, 2)
    Ir = Matrix{Float64}(I, m, m)

    nθ = length(thetas)
    nu = zeros(nθ); sigma2 = zeros(nθ); tau = zeros(nθ)
    copt = Vector{Vector{Float64}}(undef, nθ)

    for (i, θ) in enumerate(thetas)
        B = Symmetric(Ir + θ * Kr)
        E = eigen(Rr, B)                 # ascending eigenvalues
        c̃ = E.vectors[:, end]            # largest μ
        c = Q * c̃
        nrm = dot(c, c)
        nu[i] = dot(c, Rf * c) / nrm
        s2 = dot(c, Kf * c) / nrm
        sigma2[i] = max(s2, 0.0)
        tau[i] = 1 / sqrt(sigma2[i])
        copt[i] = c
    end
    return (theta=collect(float(thetas)), nu=nu, sigma2=sigma2, tau=tau, c_opt=copt)
end

"""
    lambda_TI(K, Q) -> (λ, τ_max, c)

Endpoint θ→∞: smallest eigenvalue of `K` restricted to `range(Q)` (energy
density removed), giving the ceiling `τ_max = 1/√λ_TI(M)`.
"""
function lambda_TI(K::Symmetric, Q::AbstractMatrix)
    Kf = Matrix(K)
    Kr = Symmetric(Q' * Kf * Q)
    E = eigen(Kr)
    λ = max(E.values[1], 0.0)
    c = Q * E.vectors[:, 1]
    return λ, 1 / sqrt(λ), c
end

# Sparse path: K stored sparse; per θ a sparse Cholesky of the well-conditioned
# B = I + θK and a power iteration c ← (PBP)⁺(PRP)c to the top generalized
# eigenpair, with c ⟂ c_H enforced exactly by a bordered solve (no penalty term).

"Largest eigenpair of a symmetric linear operator `applyOp` via k-step Lanczos
with full reorthogonalization; `proj` keeps vectors in a subspace."
function _top_eig_lanczos(applyOp, n::Int; k::Int=50, proj=identity, seed_vec=nothing)
    k = min(k, n)                             # k>n breeds spurious "ghost" eigenvalues
    # deterministic, generic start (avoids run-to-run RNG variation)
    q0 = seed_vec === nothing ? Float64[sin(0.3 * i + 1.0) for i in 1:n] : copy(seed_vec)
    q = proj(q0)
    q ./= norm(q)
    Q = zeros(Float64, n, k)
    α = zeros(Float64, k); β = zeros(Float64, k - 1)
    Q[:, 1] = q
    kact = k
    for j in 1:k
        w = applyOp(view(Q, :, j))
        α[j] = dot(view(Q, :, j), w)
        for i in 1:j                          # full reorthogonalization
            w .-= dot(view(Q, :, i), w) .* view(Q, :, i)
        end
        w = proj(w)
        if j < k
            β[j] = norm(w)
            if β[j] < 1e-12
                kact = j; break
            end
            Q[:, j + 1] = w ./ β[j]
        end
    end
    Tt = SymTridiagonal(α[1:kact], β[1:kact-1])
    E = eigen(Tt)
    μ = E.values[end]
    v = Q[:, 1:kact] * E.vectors[:, end]
    return μ, v ./ norm(v)
end

"""
    nutau_sweep_sparse(Rdiag, K, c_H, thetas; tol=1e-11, maxit=3000) -> NamedTuple

Sparse/iterative θ-sweep. `Rdiag = diag(R)`, `K` sparse. Returns the same fields
as [`nutau_sweep`](@ref). The conserved `c_H` is excluded *exactly* by the
projector `P = I − ĉ_H ĉ_Hᵀ` (pass a zero vector for M = 1).

For each θ we take a sparse Cholesky of the **clean, well-conditioned**
`B = I + θK` and power-iterate `c ← (PBP)⁺ (PRP) c` to the largest μ. The
constrained solve of `P B y = P r` for `y ⟂ c_H` uses a bordered elimination
(`y = u − (c_Hᵀu / c_Hᵀv) v`, `B u = r`, `B v = c_H`) — no penalty parameter, so
`cond` stays `cond(B)` rather than blowing up.
"""
function nutau_sweep_sparse(Rdiag::Vector{Float64}, K::SparseMatrixCSC,
                            c_H::Vector{Float64}, thetas::AbstractVector;
                            tol::Float64=1e-11, maxit::Int=3000, callback=nothing)
    n = length(Rdiag)
    hasC = norm(c_H) > 1e-13
    cH   = copy(c_H); cHn2 = dot(cH, cH)
    projC(x) = hasC ? (x .- (dot(cH, x) / cHn2) .* cH) : x   # P x
    Ispn = sparse(1.0I, n, n)

    nθ = length(thetas)
    nu = zeros(nθ); sigma2 = zeros(nθ); tau = zeros(nθ)
    copt = Vector{Vector{Float64}}(undef, nθ)

    c = projC(Float64[sin(0.3 * i + 1.0) for i in 1:n]); c ./= norm(c)  # start ⟂ c_H
    for (i, θ) in enumerate(thetas)
        B = Ispn + θ .* K
        F = cholesky(Symmetric(B))
        v = hasC ? (F \ cH) : zeros(n)        # B v = c_H (once per θ)
        den = hasC ? dot(cH, v) : 1.0
        constr_solve(r) = begin               # y ⟂ c_H with P B y = P r
            u = F \ r
            hasC ? (u .- (dot(cH, u) / den) .* v) : u
        end
        μ_prev = 0.0
        for _ in 1:maxit                       # power iteration on (PBP)⁺(PRP)
            y = constr_solve(projC(Rdiag .* c))
            cnew = y ./ norm(y)
            μ = dot(cnew, Rdiag .* cnew) / dot(cnew, B * cnew)
            c = cnew
            abs(μ - μ_prev) <= tol * abs(μ) && break
            μ_prev = μ
        end
        nrm = dot(c, c)
        nu[i] = dot(c, Rdiag .* c) / nrm
        s2 = dot(c, K * c) / nrm
        sigma2[i] = max(s2, 0.0)
        tau[i] = 1 / sqrt(sigma2[i])
        copt[i] = copy(c)                      # c warm-starts next θ
        callback !== nothing && callback(i, float(θ), nu[i], sigma2[i], tau[i])
    end
    return (theta=collect(float(thetas)), nu=nu, sigma2=sigma2, tau=tau, c_opt=copt)
end

"""
    lambda_TI_sparse(K, c_H; α=1e3, k=60) -> (λ, τ_max, c)

Sparse endpoint θ→∞: smallest eigenvalue of `K` with the conserved `c_H` deflated
(`K + α c_H c_Hᵀ`), via Lanczos on `K_def⁻¹` (largest eigenpair ⇒ smallest of
`K_def`). Pass a zero `c_H` for M = 1.
"""
function lambda_TI_sparse(K::SparseMatrixCSC, c_H::Vector{Float64}; α::Float64=1e3, k::Int=60)
    n = size(K, 1)
    Kd = norm(c_H) > 1e-13 ? K + α .* sparse(c_H * c_H') : copy(K)
    F = cholesky(Symmetric(Kd))
    _, v = _top_eig_lanczos(x -> F \ x, n; k)     # top eigvec of Kd⁻¹ = min eigvec of Kd
    λ = dot(v, Kd * v) / dot(v, v)
    λ = max(λ, 0.0)
    return λ, 1 / sqrt(λ), v
end

