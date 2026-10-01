# q = 0 static ν(τ) frontier with the conserved energy density PROJECTED OUT, by
# direct (dense or sparse-Cholesky) eigensolves: the PRE15-style curve plus its
# θ→∞ ceiling τ_max = 1/√λ_TI(M), λ_TI(M) = min σ² over the ansatz ⟂ c_H.
#
# usage: julia --project=julia/TIAnsatz julia/TIAnsatz/scripts/ham_q0_projected.jl \
#            [Mmax=4] [Mmin=1] [J=1] [g=0.905] [h=0.809] [sparse]
#   `sparse` (any position): sparse Cholesky + power iteration (M ≲ 9; M = 9 needs
#   ~300 GB RAM); otherwise dense LAPACK (M ≲ 7).
#   θ grid 10^range(-2, 3; length=40)
#   output $SSO_OUTPUT/ti/nutau_M{M}.dat with columns `theta nu sigma2 tau` and a
#   final `Inf` row holding the endpoint (ν_end, λ_TI, τ_max).
using TIAnsatz, ThermodynamicPauli, LinearAlgebra, Printf

const THETAS = exp10.(range(-2, 3; length=40))

function solve_M(M, H, J, g, h, use_sparse)
    basis, index = canonical_basis(M)
    R = build_R(basis)
    c_H = M >= 2 ? energy_density_vector(basis, index; J, g, h) : zeros(length(basis))
    if use_sparse
        K = build_K_sparse(H, basis, index)
        res = nutau_sweep_sparse(diag(R), K, c_H, THETAS;
                                 callback=(i, θ, ν, s2, τ) -> (@printf("  θ=%9.3g ν=%.6f σ²=%.6g τ=%.4f\n", θ, ν, s2, τ); flush(stdout)))
        λ, τmax, cλ = lambda_TI_sparse(K, c_H)
    else
        K = build_K(H, basis, index)
        Q = M >= 2 ? complement_projector(c_H) : Matrix{Float64}(I, length(basis), length(basis))
        res = nutau_sweep(R, K, Q, THETAS)
        λ, τmax, cλ = lambda_TI(K, Q)
    end
    ν_end = dot(cλ, diag(R) .* cλ) / dot(cλ, cλ)
    cn = cλ / norm(cλ)          # weight of the endpoint operator on the density strings
    edens = sum(haskey(index, pauli_to_int(s)) ? cn[index[pauli_to_int(s)]]^2 : 0.0 for s in ("X", "Z", "ZZ"))
    return (; res, λ, τmax, ν_end, edens, dim=length(basis))
end

function main(args)
    use_sparse = "sparse" in args
    pos = filter(!=("sparse"), args)
    Mmax = length(pos) >= 1 ? parse(Int, pos[1]) : 4
    Mmin = length(pos) >= 2 ? parse(Int, pos[2]) : 1
    J = length(pos) >= 3 ? parse(Float64, pos[3]) : 1.0
    g = length(pos) >= 4 ? parse(Float64, pos[4]) : 0.905
    h = length(pos) >= 5 ? parse(Float64, pos[5]) : 0.809
    H = PauliOp("ZZ" => J, "X" => g, "Z" => h)
    println("Ising H = $J·ZZ + $g·X + $h·Z  (q=0, energy density projected)  solver=$(use_sparse ? "sparse" : "dense")")
    for M in Mmin:Mmax
        t = @elapsed out = solve_M(M, H, J, g, h, use_sparse)
        fn = joinpath(output_dir(), "nutau_M$(M).dat")
        open(fn, "w") do f
            @printf(f, "# model=Ising J=%g g=%g h=%g q=0 M=%d dim=%d lambda_TI=%.12g tau_max=%.12g endpoint_edens_frac=%.6g solver=%s\n",
                    J, g, h, M, out.dim, out.λ, out.τmax, out.edens, use_sparse ? "sparse" : "dense")
            println(f, "# theta  nu  sigma2  tau")
            for i in eachindex(out.res.theta)
                @printf(f, "%.12g  %.12g  %.12g  %.12g\n", out.res.theta[i], out.res.nu[i], out.res.sigma2[i], out.res.tau[i])
            end
            @printf(f, "Inf  %.12g  %.12g  %.12g\n", out.ν_end, out.λ, out.τmax)
        end
        @printf("M=%d dim=%d λ_TI=%.6g τ_max=%.6g edens-frac=%.4g (%.2fs) -> %s\n",
                M, out.dim, out.λ, out.τmax, out.edens, t, fn)
    end
end

main(ARGS)
