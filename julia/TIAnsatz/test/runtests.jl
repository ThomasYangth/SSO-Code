# CPU test suite (GPU tests run only when CUDA is functional).
#   julia --project=julia/TIAnsatz julia/TIAnsatz/test/runtests.jl
using TIAnsatz, ThermodynamicPauli
using LinearAlgebra, SparseArrays, Random, Test, CUDA

const J0, G0, H0 = 1.0, 0.905, 0.809          # Kim–Huse static
const TF, GF, HF = 1.0, 0.9, 0.809            # kicked Ising (production Floquet point)

# ── independent finite-ring oracle: dense 2^L Kronecker Paulis, no engine code ──
const σI = ComplexF64[1 0; 0 1]; const σX = ComplexF64[0 1; 1 0]
const σY = ComplexF64[0 -im; im 0]; const σZ = ComplexF64[1 0; 0 -1]
_pm(c) = c == 'I' ? σI : c == 'X' ? σX : c == 'Y' ? σY : σZ
function ring_string(str, pos, L)
    ops = fill('I', L)
    for (i, ch) in enumerate(str); ops[mod(pos + i - 1, L) + 1] = ch; end
    M = _pm(ops[1]); for s in 2:L; M = kron(M, _pm(ops[s])); end; M
end
ti_sum(str, L) = sum(ring_string(str, i, L) for i in 0:L-1)
ring_H(L; J, g, h) = J .* ti_sum("ZZ", L) .+ g .* ti_sum("X", L) .+ h .* ti_sum("Z", L)
function ring_ansatz(basis, c, L; q=0.0)        # Σ_a c_a Σ_j e^{iqj} τ_j(P_a)
    A = zeros(ComplexF64, 2^L, 2^L)
    for (a, k) in enumerate(basis), j in 0:L-1
        A .+= c[a] * cis(q * j) .* ring_string(int_to_pauli(k), j, L)
    end
    A
end
ring_UF(L; τ, g, h) = exp(Matrix(-im * τ .* (g .* ti_sum("X", L)))) *
                      exp(Matrix(-im * τ .* (h .* ti_sum("Z", L) .+ ti_sum("ZZ", L))))

# dense reference frontier point: top eigenvector of R c = μ (I + θK) c
function dense_point(Rd, K, θ)
    E = eigen(Hermitian(Matrix(Diagonal(Rd)) .+ 0im), Hermitian(Matrix{ComplexF64}(I + θ * K)))
    c = E.vectors[:, end]; c ./= norm(c)
    (ν = real(dot(c, Rd .* c)), σ2 = real(dot(c, K * c)))
end

@testset "TIAnsatz" begin

@testset "ThermodynamicPauli" begin
    include(joinpath(@__DIR__, "..", "ThermodynamicPauli", "test", "runtests.jl"))
end

@testset "basis + RPS metric" begin
    for M in 1:4
        basis, index = canonical_basis(M)
        @test length(basis) == 3 * 4^(M - 1) == length(index)
        @test diag(build_R(basis)) ≈ [3.0^(-pauli_weight(k)) for k in basis]
    end
end

@testset "static kernel vs finite-ring ED (q = 0)" begin
    Random.seed!(1234)
    H = PauliOp("ZZ" => J0, "X" => G0, "Z" => H0)
    for M in 1:2
        L = 2M + 4                                    # no commutator wraparound
        basis, index = canonical_basis(M)
        K = Matrix(build_K(H, basis, index)); Rd = diag(build_R(basis))
        Hr = ring_H(L; J=J0, g=G0, h=H0)
        for _ in 1:3
            c = randn(length(basis)); A = ring_ansatz(basis, c, L)
            hs = real(dot(A, A))
            @test hs ≈ 2.0^L * L * dot(c, c) rtol=1e-9          # Gram = identity
            C = A * Hr - Hr * A
            @test real(dot(C, C)) / hs ≈ dot(c, K * c) / dot(c, c) rtol=1e-8
            S_hs = 0.0; S_rps = 0.0                              # RPS weight
            for (a, k) in enumerate(basis), pos in 0:L-1
                p = abs2(dot(ring_string(int_to_pauli(k), pos, L), A) / 2.0^L)
                S_hs += p; S_rps += 3.0^(-pauli_weight(k)) * p
            end
            @test S_rps / S_hs ≈ dot(c, Rd .* c) / dot(c, c) rtol=1e-8
        end
    end
end

@testset "momentum-q kernel vs finite-ring ED" begin
    Random.seed!(20)
    for M in 1:2
        L = 2M + 4
        basis, _ = canonical_basis(M); _, index1 = canonical_basis(M + 1)
        Hr = ring_H(L; J=J0, g=G0, h=H0)
        for kq in 1:(L ÷ 2)
            q = 2π * kq / L                               # commensurate with the ring
            L1q = build_L1_q(basis, index1, M, q; J=J0, g=G0, h=H0)
            for _ in 1:2
                c = randn(ComplexF64, length(basis)); A = ring_ansatz(basis, c, L; q)
                C = A * Hr .- Hr * A
                @test real(dot(C, C)) / real(dot(A, A)) ≈ norm(L1q * c)^2 / norm(c)^2 rtol=1e-8
            end
        end
    end
end

@testset "static kernel factorizations" begin
    H = kim_huse_H(; J=J0, g=G0, h=H0)
    for M in 2:5
        basis, index = canonical_basis(M); _, index1 = canonical_basis(M + 1)
        L1 = build_L1(H, basis, index1)
        L1q0 = build_L1_q(basis, index1, M, 0.0; J=J0, g=G0, h=H0)
        Ks = build_K_sparse(H, basis, index)
        @test Matrix(L1q0) ≈ Matrix(L1) atol=1e-12                 # q = 0 reduction
        @test Matrix(L1' * L1) ≈ Matrix(Ks) atol=1e-10             # K = L1†L1
        M <= 4 && @test Matrix(build_K(H, basis, index)) ≈ Matrix(Ks) atol=1e-10
        c_H = energy_density_vector(basis, index; J=J0, g=G0, h=H0)
        @test norm(Ks * c_H) < 1e-9                                # conserved density
        x = randn(length(basis))
        @test apply_K(KHCpu(L1), x) ≈ Ks * x rtol=1e-10            # matrix-free == assembled
        for q in (0.3, 1.0, π)
            L1q = build_L1_q(basis, index1, M, float(q); J=J0, g=G0, h=H0)
            Kq = Matrix(L1q' * L1q)
            @test norm(Kq - Kq') < 1e-10 * norm(Kq)
            @test eigmin(Hermitian(Kq)) > -1e-8
            z = randn(ComplexF64, length(basis))
            @test apply_K(KHCpuC(L1q), z) ≈ Kq * z rtol=1e-10
        end
    end
end

@testset "Floquet kernel" begin
    Random.seed!(2024)
    for M in 1:2                                       # vs finite-ring ED
        L = 2M + 6
        basis, index = canonical_basis(M)
        KF = Matrix(build_KF_sparse(basis, index, M; τ=TF, g=GF, h=HF))
        UF = ring_UF(L; τ=TF, g=GF, h=HF)
        for _ in 1:3
            c = randn(length(basis)); A = ring_ansatz(basis, c, L)
            C = A * UF .- UF * A
            @test real(dot(C, C)) / real(dot(A, A)) ≈ dot(c, KF * c) / dot(c, c) rtol=1e-8
        end
    end
    basis, index = canonical_basis(2)
    @test maximum(abs, Matrix(build_KF_sparse(basis, index, 2; τ=0.0, g=GF, h=HF))) < 1e-10
    for M in 3:5                                       # matrix-free butterfly == assembled
        basis, index = canonical_basis(M)
        KF = build_KF_sparse(basis, index, M; τ=TF, g=GF, h=HF)
        @test issymmetric(Matrix(KF))
        @test eigmin(Symmetric(Matrix(KF))) > -1e-9
        for margin in (1, 2)
            D = DenseKF(basis, index, M; τ=TF, g=GF, h=HF, margin)
            x = randn(length(basis))
            @test apply_K(D, x) ≈ KF * x rtol=1e-10 atol=1e-12
        end
    end
end

@testset "exact q = 0 projected solvers: sparse == dense" begin
    H = PauliOp("ZZ" => J0, "X" => G0, "Z" => H0)
    thetas = exp10.(range(-2, 3; length=12))
    for M in 2:4
        basis, index = canonical_basis(M)
        R = build_R(basis)
        Kd = build_K(H, basis, index); Ks = build_K_sparse(H, basis, index)
        c_H = energy_density_vector(basis, index; J=J0, g=G0, h=H0)
        Q = complement_projector(c_H)
        rd = nutau_sweep(R, Kd, Q, thetas)
        rs = nutau_sweep_sparse(diag(R), Ks, c_H, thetas)
        @test rs.nu ≈ rd.nu rtol=1e-5
        @test rs.sigma2 ≈ rd.sigma2 rtol=1e-5 atol=1e-10
        @test issorted(rd.tau)
        λd, τd, _ = lambda_TI(Kd, Q); λs, τs, _ = lambda_TI_sparse(Ks, c_H)
        @test λs ≈ λd rtol=1e-7
        @test τd >= rd.tau[end] * (1 - 1e-6)                       # endpoint is the ceiling
    end
end

thetas_t = [0.1, 1.0, 10.0, 100.0]

@testset "LOBPCG frontier sweep vs dense eigensolve (CPU)" begin
    Random.seed!(7)
    M = 5
    basis, index = canonical_basis(M); _, index1 = canonical_basis(M + 1); n = length(basis)
    Rd = diag(build_R(basis))
    H = kim_huse_H(; J=J0, g=G0, h=H0)

    # momentum q ≠ 0 (complex)
    L1q = build_L1_q(basis, index1, M, 0.05; J=J0, g=G0, h=H0)
    Kq = sparse(L1q' * L1q); KC = KHCpuC(L1q); aK = x -> apply_K(KC, x)
    io = IOBuffer()
    res = frontier_sweep(io, thetas_t, Rd, aK, randn_vec(Vector{ComplexF64}, n);
                         λmaxK=power_lmax(aK, randn_vec(Vector{ComplexF64}, n)), tol=1e-8, verbose=false)
    for (i, θ) in enumerate(thetas_t)
        ref = dense_point(Rd, Matrix(Kq), θ)
        @test res.nu[i] ≈ ref.ν rtol=1e-5
        @test res.sigma2[i] ≈ ref.σ2 rtol=1e-4
        @test res.accepted[i]
        @test res.nu_bar[i] < 1e-6
    end
    rows = split(strip(String(take!(io))), '\n')
    @test length(rows) == length(thetas_t)
    @test length(split(rows[1])) == 6                              # .dat row format

    # q = 0, unprojected (the production q = 0 baseline) and energy-projected
    L1 = build_L1(H, basis, index1); K0 = Matrix(L1' * L1) |> real; KR = KHCpu(L1)
    aK0 = x -> apply_K(KR, x)
    λm = power_lmax(aK0, randn(n))
    res0 = frontier_sweep(devnull, thetas_t, Rd, aK0, randn(n); λmaxK=λm, tol=1e-8, verbose=false)
    for (i, θ) in enumerate(thetas_t)
        ref = dense_point(Rd, K0, θ)
        @test res0.nu[i] ≈ ref.ν rtol=1e-5
        @test res0.accepted[i]
    end
    ĉ = energy_density_vector(basis, index; J=J0, g=G0, h=H0)
    proj = x -> x .- ĉ .* dot(ĉ, x)
    x0 = proj(Float64[sin(0.3i + 1.0) for i in 1:n]); x0 ./= norm(x0)
    resp = frontier_sweep(devnull, thetas_t, Rd, aK0, x0; λmaxK=power_lmax(x -> proj(aK0(proj(x))), randn(n)),
                          tol=1e-8, proj, verbose=false)
    rd = nutau_sweep(build_R(basis), Symmetric(K0), complement_projector(ĉ), thetas_t)
    @test resp.nu ≈ rd.nu rtol=1e-5
    @test resp.sigma2 ≈ rd.sigma2 rtol=1e-4
    @test all(resp.accepted)

    # Floquet
    KF = Matrix(build_KF_sparse(basis, index, M; τ=TF, g=GF, h=HF))
    D = DenseKF(basis, index, M; τ=TF, g=GF, h=HF); aKF = x -> apply_K(D, x)
    resF = frontier_sweep(devnull, thetas_t, Rd, aKF, randn(n); λmaxK=power_lmax(aKF, randn(n)),
                          tol=1e-8, verbose=false)
    for (i, θ) in enumerate(thetas_t)
        ref = dense_point(Rd, KF, θ)
        @test resF.nu[i] ≈ ref.ν rtol=1e-5
        @test resF.sigma2[i] ≈ ref.σ2 rtol=1e-4
    end
end

if CUDA.functional()
    @testset "GPU kernels and sweep" begin
        Random.seed!(11); CUDA.seed!(11)
        M = 5
        basis, index = canonical_basis(M); _, index1 = canonical_basis(M + 1); n = length(basis)
        Rd = diag(build_R(basis))
        L1 = build_L1(kim_huse_H(; J=J0, g=G0, h=H0), basis, index1)
        x = randn(n)
        @test Array(apply_K(KHGpu(L1), CuArray(x))) ≈ apply_K(KHCpu(L1), x) rtol=1e-10
        L1q = build_L1_q(basis, index1, M, 0.05; J=J0, g=G0, h=H0)
        z = randn(ComplexF64, n)
        @test Array(apply_K(KHGpuC(L1q), CuArray(z))) ≈ apply_K(KHCpuC(L1q), z) rtol=1e-10
        DG = DenseKFGpu(basis, index, M; τ=TF, g=GF, h=HF)
        @test Array(apply_K(DG, CuArray(x))) ≈ apply_K(DenseKF(basis, index, M; τ=TF, g=GF, h=HF), x) rtol=1e-10

        KG = KHGpuC(L1q); aK = v -> apply_K(KG, v)
        res = frontier_sweep(devnull, thetas_t, CuArray(Rd), aK, randn_vec(CuVector{ComplexF64}, n);
                             λmaxK=power_lmax(aK, randn_vec(CuVector{ComplexF64}, n)), tol=1e-8, verbose=false)
        Kq = Matrix(L1q' * L1q)
        for (i, θ) in enumerate(thetas_t)
            @test res.nu[i] ≈ dense_point(Rd, Kq, θ).ν rtol=1e-5
            @test res.accepted[i]
        end
    end
else
    @info "CUDA not functional: GPU tests skipped"
end

end # TIAnsatz
