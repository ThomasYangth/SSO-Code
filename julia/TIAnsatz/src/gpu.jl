# GPU kernels (CUDA.jl). Every apply is CuVector -> CuVector; nothing is assembled
# beyond the single-commutator map L1 (static) or the 4^W canvas maps (Floquet).

randn_vec(::Type{<:CuVector{T}}, n::Int) where {T<:Real} = CUDA.randn(T, n)
randn_vec(::Type{<:CuVector{Complex{T}}}, n::Int) where {T<:Real} = CUDA.randn(T, n) .+ im .* CUDA.randn(T, n)

# ─────────────────────────── static, q = 0 (real) ───────────────────────────

"""
    KHGpu(L1)

q = 0 static kernel `K = L1ᵀL1` on the GPU. `L1` is purely imaginary, so the real
matrix `imag(L1)` is stored once as CSC; its transpose is a CSR *view of the same
arrays*, so both `L1 x` and `L1ᵀ y` are native SpMVs with no second copy.
"""
struct KHGpu
    n::Int; m::Int
    L1::CuSparseMatrixCSC{Float64,Int32}
    L1T::CuSparseMatrixCSR{Float64,Int32}
end

function KHGpu(L1c::SparseMatrixCSC{ComplexF64,Int})
    L1r = SparseMatrixCSC{Float64,Int}(imag(L1c))
    m, n = size(L1r)
    csc = CuSparseMatrixCSC(L1r)
    KHGpu(n, m, csc, CuSparseMatrixCSR(csc.colPtr, csc.rowVal, csc.nzVal, (n, m)))
end

l1_bytes(K::KHGpu) = 8*nnz(K.L1) + 4*nnz(K.L1) + 4*(K.n + 2)
apply_K(K::KHGpu, xg::CuVector{Float64}) = K.L1T * (K.L1 * xg)

# ─────────────────────────── static, momentum q (complex) ───────────────────────────

"""
    KHGpuC(L1_q)

Momentum-q kernel `K_q = L1_q† L1_q` on the GPU, `L1_q` stored as complex CSC with
a shared-array CSR transpose; the adjoint is applied as `L1† y = conj(L1ᵀ conj y)`.
"""
struct KHGpuC
    n::Int; m::Int
    L1::CuSparseMatrixCSC{ComplexF64,Int32}
    L1T::CuSparseMatrixCSR{ComplexF64,Int32}
end

function KHGpuC(L1c::SparseMatrixCSC{ComplexF64,Int})
    m, n = size(L1c)
    csc = CuSparseMatrixCSC(L1c)
    KHGpuC(n, m, csc, CuSparseMatrixCSR(csc.colPtr, csc.rowVal, csc.nzVal, (n, m)))
end

l1_bytes(K::KHGpuC) = 16*nnz(K.L1) + 4*nnz(K.L1) + 4*(K.n + 2)
apply_K(K::KHGpuC, xg::CuVector{ComplexF64}) = conj.(K.L1T * conj.(K.L1 * xg))

# ─────────────────────────── Floquet ───────────────────────────

# one U_F gate over the whole canvas (bit arithmetic shared with the CPU apply_KF)
function _gate_kernel!(out, ψ, W::Int64, code::Int64, i::Int64, i2::Int64,
                       gmask::Int64, c2::Float64, s2::Float64, N::Int64)
    t = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    t > N && return nothing
    @inbounds out[t] = _gate_entry(ψ, t - Int64(1), W, code, i, i2, gmask, c2, s2)
    return nothing
end

function _fold_kernel!(y, res, fold, N::Int64)
    t = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    t > N && return nothing
    @inbounds begin
        a = fold[t]
        a != 0 && CUDA.@atomic y[a] += res[t]
    end
    return nothing
end

"""
    DenseKFGpu(basis, index, M; τ, g, h, margin=1)

GPU matrix-free Floquet kernel: the CPU [`DenseKF`](@ref) butterfly with each gate
a CUDA kernel and the fold an atomic scatter-add. Canvas memory 2·8·4^(M+2) bytes
(+ 4·4^(M+2) for the fold map); M = 12 ≈ 5.1 GB fits a 10 GB MIG slice. The Int32
fold map limits the canvas to M ≤ 13.
"""
struct DenseKFGpu
    n::Int; W::Int; N::Int64; τ::Float64; g::Float64; h::Float64
    lift::CuVector{Int64}
    fold::CuVector{Int32}
    ψ::CuVector{Float64}; buf::CuVector{Float64}
    sched::Vector{Tuple{Int,Int,Int,Int,Symbol}}
end

function DenseKFGpu(basis::Vector{PauliInt}, index::Dict{PauliInt,Int}, M::Int;
                    τ::Float64=0.8, g::Float64=0.905, h::Float64=0.809, margin::Int=1)
    W = M + 2margin; N = Int64(4)^W
    DenseKFGpu(length(basis), W, N, τ, g, h,
               CuArray(_floquet_lift(basis, W, margin)), CuArray(_floquet_fold(index, W)),
               CUDA.zeros(Float64, N), CUDA.zeros(Float64, N), _floquet_schedule(W))
end

gpu_bytes(K::DenseKFGpu) = 8*length(K.ψ) + 8*length(K.buf) + 4*length(K.fold) + 8*length(K.lift)

function apply_K(K::DenseKFGpu, xg::CuVector{Float64})
    yg = 2.0 .* xg
    thr = 256; blk = cld(K.N, thr)
    for dagger in (false, true)
        sgn = dagger ? -1.0 : 1.0
        fill!(K.ψ, 0.0)
        K.ψ[K.lift .+ 1] .= xg
        cur, nxt = K.ψ, K.buf
        for (code, i, i2, gmask, ang) in (dagger ? reverse(K.sched) : K.sched)
            α = _floquet_angle(ang, K.τ, K.g, K.h)
            @cuda threads=thr blocks=blk _gate_kernel!(nxt, cur, Int64(K.W), Int64(code),
                Int64(i), Int64(i2), Int64(gmask), cos(2α), sgn*sin(2α), K.N)
            cur, nxt = nxt, cur
        end
        tg = CUDA.zeros(Float64, K.n)
        @cuda threads=thr blocks=blk _fold_kernel!(tg, cur, K.fold, K.N)
        yg .-= tg
    end
    return yg
end
