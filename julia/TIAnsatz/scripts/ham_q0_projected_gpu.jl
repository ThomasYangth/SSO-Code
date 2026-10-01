# q = 0 static ν(τ) frontier with the conserved energy density PROJECTED OUT, matrix-free
# (K = L1ᵀL1, real) LOBPCG — the large-M (M ≤ 13) version of ham_q0_projected.jl.
# The projector P = I − ĉĉᵀ acts on A = −R, B = I + θK (B extended by the identity on ĉ)
# and the Chebyshev preconditioner; the start vector is deterministic, sin(0.3i + 1).
#
# usage: julia --project=julia/TIAnsatz julia/TIAnsatz/scripts/ham_q0_projected_gpu.jl \
#            M [TOL=1e-3] [THETA_MAXLOG=3.0] [NTHETA=20] [g=0.905] [h=0.809] [SIG=""]
#   output $SSO_OUTPUT/ti/nutau_M{M}_gpures[_SIG].dat   (6-column format)
# env: SSO_TI_BACKEND=auto|gpu|cpu, SSO_SEED, SSO_OUTPUT.
using TIAnsatz, LinearAlgebra, SparseArrays, Printf, CUDA

function run_M(M::Int, tol::Float64, thetas, g::Float64, h::Float64, sig::String, gpu::Bool)
    basis, index = canonical_basis(M); n = length(basis)
    _, index1 = canonical_basis(M + 1)
    Rd = to_device(diag(build_R(basis)), gpu)
    ĉ = to_device(energy_density_vector(basis, index; J=1.0, g, h), gpu)
    tb = @elapsed L1 = build_L1(kim_huse_H(; J=1.0, g, h), basis, index1)
    K = gpu ? KHGpu(L1) : KHCpu(L1)
    applyK = x -> apply_K(K, x)
    proj = x -> x .- ĉ .* dot(ĉ, x)
    @printf("M=%d dim=%d nnz(L1)=%d  build %.1fs  [%s, energy-density projected]\n",
            M, n, nnz(L1), tb, gpu ? @sprintf("GPU, L1 %.2f GB", l1_bytes(K)/2^30) : "CPU"); flush(stdout)
    V = gpu ? CuVector{Float64} : Vector{Float64}
    λmaxK = power_lmax(x -> proj(applyK(proj(x))), randn_vec(V, n))
    x0 = proj(to_device(Float64[sin(0.3i + 1.0) for i in 1:n], gpu)); x0 ./= norm(x0)
    sfx = isempty(sig) ? "" : "_$(sig)"
    datf = joinpath(output_dir(), "nutau_M$(M)_gpures$(sfx).dat")
    t = @elapsed res = open(datf, "w") do io
        @printf(io, "# model=Ising J=1 g=%.4g h=%.4g q=0 proj=energy-density M=%d dim=%d solver=%s\n",
                g, h, M, n, gpu ? "mf-L1L1-gpures" : "mf-L1L1-cpu")
        println(io, DAT_COLUMNS); flush(io)
        frontier_sweep(io, thetas, Rd, applyK, x0; λmaxK, tol, proj)
    end
    @printf("swept %d θ in %.1fs  accepted %d/%d  [M=%d projected]\nSaved %s\n",
            length(thetas), t, count(res.accepted), length(thetas), M, datf)
end

function main(args)
    Mlist = parse_Mspec(length(args) >= 1 ? args[1] : "8")
    tol  = length(args) >= 2 ? parse(Float64, args[2]) : 1e-3
    tmax = length(args) >= 3 ? parse(Float64, args[3]) : 3.0
    nθ   = length(args) >= 4 ? parse(Int, args[4]) : 20
    g    = length(args) >= 5 ? parse(Float64, args[5]) : 0.905
    h    = length(args) >= 6 ? parse(Float64, args[6]) : 0.809
    sig  = length(args) >= 7 ? args[7] : ""
    gpu = use_gpu(); seed = seed_rng!()
    println("backend: ", gpu ? "GPU ($(CUDA.name(CUDA.device())))" : "CPU", "  seed: ", something(seed, "none"))
    for M in Mlist
        println("\n========== Hamiltonian M = $M (q=0, projected) ==========")
        run_M(M, tol, theta_grid(tmax, nθ), g, h, sig, gpu)
    end
end

main(ARGS)
