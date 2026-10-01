# θ sweep of the frontier problem and the .dat output format.
#
# For each Lagrange multiplier θ, the optimal ansatz operator maximizes
# ν = c'Rc/c'c at fixed σ² = c'Kc/c'c, i.e. it is the top eigenvector of
#     R c = μ (I + θK) c,
# solved as the smallest eigenpair of the pencil (A, B) = (−R, I + θK) by LOBPCG
# with a Chebyshev-in-B preconditioner, warm-started from the previous θ.

# ─────────────────────────── CPU kernel holders ───────────────────────────

"q = 0 static kernel `K = L1ᵀL1` on the host, `L1 = imag(L1ℂ)` (real)."
struct KHCpu
    L1::SparseMatrixCSC{Float64,Int}
end
KHCpu(L1c::SparseMatrixCSC{ComplexF64,Int}) = KHCpu(SparseMatrixCSC{Float64,Int}(imag(L1c)))
apply_K(K::KHCpu, x::Vector{Float64}) = K.L1' * (K.L1 * x)

"Momentum-q static kernel `K_q = L1_q†L1_q` on the host."
struct KHCpuC
    L1::SparseMatrixCSC{ComplexF64,Int}
end
apply_K(K::KHCpuC, x::Vector{ComplexF64}) = K.L1' * (K.L1 * x)

apply_K(K::DenseKF, x::Vector{Float64}) = apply_KF(K, x)

# ─────────────────────────── backend / I/O helpers ───────────────────────────

"""
    use_gpu() -> Bool

GPU backend iff CUDA is functional, unless `SSO_TI_BACKEND=cpu` forces the host
path (`SSO_TI_BACKEND=gpu` makes a missing GPU an error).
"""
function use_gpu()
    b = lowercase(get(ENV, "SSO_TI_BACKEND", "auto"))
    b == "cpu" && return false
    b == "gpu" && (CUDA.functional() || error("SSO_TI_BACKEND=gpu but CUDA is not functional"); return true)
    return CUDA.functional()
end

"Move a host vector to the active backend."
to_device(x::Vector, gpu::Bool) = gpu ? CuArray(x) : x

"""
    seed_rng!()

If `SSO_SEED` is set, seed the host and CUDA RNGs that draw the random LOBPCG /
power-iteration / Lanczos start vectors. Unset (the production runs): unseeded,
so results reproduce only to solver tolerance.
"""
function seed_rng!()
    s = get(ENV, "SSO_SEED", "")
    isempty(s) && return nothing
    seed = parse(Int, s)
    Random.seed!(seed)
    CUDA.functional() && CUDA.seed!(seed)
    return seed
end

"""
    output_dir() -> String

`\$SSO_OUTPUT/ti` (created), default `<repo>/output/ti` — the same convention as
the Python `sso.config.output_dir("ti")`.
"""
function output_dir()
    base = get(ENV, "SSO_OUTPUT", normpath(joinpath(@__DIR__, "..", "..", "..", "output")))
    mkpath(joinpath(base, "ti"))
end

"Filename tag of a momentum: 0.002 → \"0p00200\"."
qtag(q::Real) = replace(@sprintf("%.5f", q), "." => "p")

"Production θ grid: `10 .^ range(-1.5, θmaxlog; length=nθ)`."
theta_grid(θmaxlog::Real, nθ::Int) = exp10.(range(-1.5, θmaxlog; length=nθ))

"Parse an M spec: \"8\" or an inclusive range \"8-11\"."
function parse_Mspec(s::AbstractString)
    occursin('-', s) || return parse(Int, s):parse(Int, s)
    a, b = split(s, '-')
    parse(Int, a):parse(Int, b)
end

const DAT_COLUMNS = "# theta  nu  sigma2  tau  nu_bar  accepted"

# ─────────────────────────── the sweep ───────────────────────────

"""
    frontier_sweep(io, thetas, Rd, applyK, x0; tol=1e-3, proj=nothing, maxiter=500, m=15, λmaxK)

Sweep θ ∈ `thetas`; per θ solve `R c = μ(I + θK)c` (largest μ) by [`lobpcg`](@ref)
on the pencil `(−R, I + θK)` with the degree-`k` Chebyshev preconditioner on
`[1, β]`, `β = 1 + 1.2 θ λmaxK`, warm-starting from the previous θ, then certify
with [`error_bar`](@ref). Writes one row per θ to `io` (flushed):
`theta nu sigma2 tau nu_bar accepted`, with `tau = 1/√σ²`.

`proj === nothing`: unconstrained (q ≠ 0, the q = 0 baseline, Floquet).
`proj = x -> x - ĉ(ĉ·x)`: q = 0 with the energy density ĉ projected out; then A, B
and the preconditioner act on range(proj) and B is extended by the identity on ĉ.
Returns a NamedTuple of the columns.
"""
function frontier_sweep(io::IO, thetas::AbstractVector, Rd::AbstractVector{Float64}, applyK,
                        x0::AbstractVector; λmaxK::Float64, tol::Float64=1e-3, proj=nothing,
                        maxiter::Int=500, m::Int=15, verbose::Bool=true)
    nθ = length(thetas)
    nu = zeros(nθ); sigma2 = zeros(nθ); tau = zeros(nθ); nu_bar = zeros(nθ); accepted = fill(false, nθ)
    projf = proj === nothing ? identity : proj
    X = x0
    for (i, θ) in enumerate(thetas)
        θ = float(θ)
        if proj === nothing
            applyA = x -> -(Rd .* x)
            applyB = x -> x .+ θ .* applyK(x)
        else
            applyA = x -> -proj(Rd .* proj(x))
            applyB = x -> (pv = proj(x); proj(pv .+ θ .* applyK(pv)) .+ (x .- pv))
        end
        β = 1.0 + θ * 1.2 * λmaxK
        applyP = chebyshev_precond(applyK, θ, β, chebyshev_degree(β); proj)
        c, _ = lobpcg(applyA, applyB, applyP, X; tol, maxiter)
        X = c
        eb = error_bar(Rd, applyK, projf, θ, c; m)
        nu[i] = eb.ν; sigma2[i] = eb.σ2; tau[i] = 1/sqrt(eb.σ2); nu_bar[i] = eb.bar; accepted[i] = eb.accepted
        @printf(io, "%.10g  %.10g  %.10g  %.10g  %.4g  %d\n",
                thetas[i], nu[i], sigma2[i], tau[i], nu_bar[i], accepted[i] ? 1 : 0); flush(io)
        verbose && (@printf("θ=%9.3g  ν=%.6f  σ²=%.6f  τ=%.4f  bar=%.2e  %s\n",
                            θ, eb.ν, eb.σ2, tau[i], eb.bar, eb.accepted ? "yes" : "REJECT"); flush(stdout))
    end
    return (theta=collect(float.(thetas)), nu=nu, sigma2=sigma2, tau=tau, nu_bar=nu_bar, accepted=accepted)
end
