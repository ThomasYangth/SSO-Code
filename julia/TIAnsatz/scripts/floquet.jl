# Kicked-Ising Floquet ν(τ) frontier (q = 0), thermodynamic limit.
#
# U_F = e^{-iτF H_x} e^{-iτF H_z},  H_x = Σ g X_i,  H_z = Σ (h Z_i + Z_iZ_{i+1})  (J = 1).
# Kernel K_F = 2I − T − Tᵀ, T[a,b] = (U_F P_b U_F†)[P_a], applied matrix-free on the
# 4^(M+2) canvas. No projection (no local conserved density for generic U_F).
#
# usage: julia --project=julia/TIAnsatz julia/TIAnsatz/scripts/floquet.jl \
#            M [tauF=0.8] [TOL=1e-3] [g=0.905] [h=0.809] [SIG=""]
#   M      single span "8" or inclusive range "6-12"
#   θ grid 10^range(-1.5, 3; length=24) (production)
#   output $SSO_OUTPUT/ti/nutau_floquet_M{M}[_SIG]_gpures.dat
#          (the `_gpures` suffix is kept for both backends: archive-compatible names)
# env: SSO_TI_BACKEND=auto|gpu|cpu, SSO_SEED, SSO_OUTPUT.
using TIAnsatz, LinearAlgebra, Printf, CUDA

const THETAS = theta_grid(3.0, 24)

function run_M(M::Int, τF::Float64, tol::Float64, g::Float64, h::Float64, sig::String, gpu::Bool)
    basis, index = canonical_basis(M); n = length(basis)
    Rd = to_device(diag(build_R(basis)), gpu)
    tb = @elapsed K = gpu ? DenseKFGpu(basis, index, M; τ=τF, g, h, margin=1) :
                            DenseKF(basis, index, M; τ=τF, g, h, margin=1)
    applyK = x -> apply_K(K, x)
    V = gpu ? CuVector{Float64} : Vector{Float64}
    @printf("M=%d dim=%d canvas 4^%d  built %.1fs  [%s]\n", M, n, M + 2, tb,
            gpu ? @sprintf("GPU, %.2f GB", gpu_bytes(K)/2^30) : "CPU"); flush(stdout)
    λmaxK = power_lmax(applyK, randn_vec(V, n))
    x0 = randn_vec(V, n); x0 ./= norm(x0)
    sfx = isempty(sig) ? "" : "_$(sig)"
    datf = joinpath(output_dir(), "nutau_floquet_M$(M)$(sfx)_gpures.dat")
    t = @elapsed res = open(datf, "w") do io
        @printf(io, "# kicked-Ising Floquet q=0 M=%d dim=%d tauF=%.4g g=%.4g h=%.4g solver=%s\n",
                M, n, τF, g, h, gpu ? "mf-gates-gpures" : "mf-gates-cpu")
        println(io, DAT_COLUMNS); flush(io)
        frontier_sweep(io, THETAS, Rd, applyK, x0; λmaxK, tol)
    end
    @printf("swept %d θ in %.1fs  accepted %d/%d  [Floquet M=%d]\nSaved %s\n",
            length(THETAS), t, count(res.accepted), length(THETAS), M, datf)
end

function main(args)
    Mlist = parse_Mspec(length(args) >= 1 ? args[1] : "8")
    τF  = length(args) >= 2 ? parse(Float64, args[2]) : 0.8
    tol = length(args) >= 3 ? parse(Float64, args[3]) : 1e-3
    g   = length(args) >= 4 ? parse(Float64, args[4]) : 0.905
    h   = length(args) >= 5 ? parse(Float64, args[5]) : 0.809
    sig = length(args) >= 6 ? args[6] : ""
    gpu = use_gpu(); seed = seed_rng!()
    println("backend: ", gpu ? "GPU ($(CUDA.name(CUDA.device())))" : "CPU", "  seed: ", something(seed, "none"))
    @printf("Floquet: tauF=%.4g g=%.4g h=%.4g  tol=%.1e  signature='%s'\n", τF, g, h, tol, sig)
    for M in Mlist
        println("\n========== Floquet M = $M ==========")
        run_M(M, τF, tol, g, h, sig, gpu)
    end
end

main(ARGS)
