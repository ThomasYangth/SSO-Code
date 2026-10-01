# Static (Hamiltonian) ν(τ) frontier at momentum q, thermodynamic limit.
#
# H = Σ_i (J Z_iZ_{i+1} + g X_i + h Z_i), J = 1. Kernel K_q = L1_q†L1_q applied
# matrix-free; NO energy-density projection (at q ≠ 0 the density is not conserved;
# at q = 0 this gives the unprojected baseline used for the q-scan, whose θ→∞ limit
# is the conserved density itself).
#
# usage: julia --project=julia/TIAnsatz julia/TIAnsatz/scripts/ham_q_scan.jl \
#            M q[,q,...] [TOL=1e-3] [THETA_MAXLOG=3.0] [NTHETA=20] [g=0.905] [h=0.809] [SIG=""]
#   M      single span "8" or inclusive range "8-11"
#   output $SSO_OUTPUT/ti/nutau_M{M}_q{q:%.5f with . -> p}[_SIG].dat
# env: SSO_TI_BACKEND=auto|gpu|cpu, SSO_SEED (optional RNG seed), SSO_OUTPUT.
using TIAnsatz, LinearAlgebra, SparseArrays, Printf, CUDA

function run_Mq(M::Int, q::Float64, tol::Float64, thetas, g::Float64, h::Float64, sig::String, gpu::Bool)
    basis, _ = canonical_basis(M); n = length(basis)
    _, index1 = canonical_basis(M + 1)
    Rd = to_device(diag(build_R(basis)), gpu)
    tb = @elapsed L1q = build_L1_q(basis, index1, M, q; J=1.0, g, h)
    K = gpu ? KHGpuC(L1q) : KHCpuC(L1q)
    applyK = x -> apply_K(K, x)
    V = gpu ? CuVector{ComplexF64} : Vector{ComplexF64}
    @printf("M=%d dim=%d q=%.5f nnz(L1_q)=%d  build %.1fs  [%s]\n",
            M, n, q, nnz(L1q), tb, gpu ? @sprintf("GPU, L1 %.2f GB", l1_bytes(K)/2^30) : "CPU"); flush(stdout)
    λmaxK = power_lmax(applyK, randn_vec(V, n))
    x0 = randn_vec(V, n); x0 ./= norm(x0)
    sfx = isempty(sig) ? "" : "_$(sig)"
    datf = joinpath(output_dir(), "nutau_M$(M)_q$(qtag(q))$(sfx).dat")
    t = @elapsed res = open(datf, "w") do io
        @printf(io, "# model=Ising J=1 g=%.4g h=%.4g M=%d dim=%d q=%.10g solver=%s\n",
                g, h, M, n, q, gpu ? "mf-L1qL1q-gpures" : "mf-L1qL1q-cpu")
        println(io, DAT_COLUMNS); flush(io)
        frontier_sweep(io, thetas, Rd, applyK, x0; λmaxK, tol)
    end
    @printf("swept %d θ in %.1fs  accepted %d/%d  [M=%d q=%.5f]\nSaved %s\n",
            length(thetas), t, count(res.accepted), length(thetas), M, q, datf)
end

function main(args)
    Mlist = parse_Mspec(length(args) >= 1 ? args[1] : "6")
    qs    = length(args) >= 2 ? parse.(Float64, split(args[2], ',')) : [1.0]
    tol   = length(args) >= 3 ? parse(Float64, args[3]) : 1e-3
    tmax  = length(args) >= 4 ? parse(Float64, args[4]) : 3.0
    nθ    = length(args) >= 5 ? parse(Int, args[5]) : 20
    g     = length(args) >= 6 ? parse(Float64, args[6]) : 0.905
    h     = length(args) >= 7 ? parse(Float64, args[7]) : 0.809
    sig   = length(args) >= 8 ? args[8] : ""
    gpu = use_gpu(); seed = seed_rng!()
    println("backend: ", gpu ? "GPU ($(CUDA.name(CUDA.device())))" : "CPU", "  seed: ", something(seed, "none"))
    @printf("H: J=1 g=%.4g h=%.4g  θ = 10^range(-1.5, %.3g; length=%d)  tol=%.1e  signature='%s'\n",
            g, h, tmax, nθ, tol, sig)
    for q in qs, M in Mlist
        println("\n========== Hamiltonian M = $M, q = $q ==========")
        run_Mq(M, q, tol, theta_grid(tmax, nθ), g, h, sig, gpu)
    end
end

main(ARGS)
