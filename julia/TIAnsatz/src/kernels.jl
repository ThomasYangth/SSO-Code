# CPU kernels of the quadratic forms (assembled and matrix-free).
#
# Static (Hamiltonian) case: σ²(c) = c'Kc / c'c with
#     K_q = L1_q† L1_q,   L1_q[c,b] = Σ_δ ([H,P_b] at offset δ)[P_c] · e^{-iqδ},
# the single-commutator map basis(M) → basis(M+1). At q = 0, K = L1†L1 equals the
# double-commutator kernel K[a,b] = ([H,[H,P_b]])[P_a] (L = [H,·] is HS-self-adjoint).
#
# Floquet case: σ²(c) = c'K_F c / c'c with
#     K_F = 2I − T − Tᵀ,  T[a,b] = (U_F P_b U_F†)[P_a],
# U_F = e^{-iτ H_x} e^{-iτ H_z}, H_x = Σ g X_i, H_z = Σ (h Z_i + Z_iZ_{i+1}); i.e.
# σ² = per-site ‖U_F A U_F† − A‖²_HS / ‖A‖²_HS = ‖[A, U_F]‖² / ‖A‖².

# ─────────────────────────── static, assembled ───────────────────────────

"""
    build_K(H, basis, index) -> Symmetric{Float64}

Dense double-commutator kernel `K[a,b] = (𝓛² P_b)[P_a]`, `𝓛 = [H,·]` (q = 0).
Real symmetric PSD. Practical to M ≈ 7.
"""
function build_K(H::PauliOp, basis::Vector{PauliInt}, index::Dict{PauliInt,Int})
    L = Liouvillian(H)
    n = length(basis)
    K = zeros(ComplexF64, n, n)
    for (b, Pb) in enumerate(basis)
        seed = PauliOp{ComplexF64}(int_to_pauli(Pb) => one(ComplexF64))
        D2 = apply_rule(apply_rule(seed, L), L)     # 𝓛² P_b
        for (key, coeff) in D2
            a = get(index, key, 0)
            a == 0 && continue
            K[a, b] = coeff
        end
    end
    maxim = maximum(abs.(imag.(K)); init=0.0)
    @assert maxim < 1e-8 "K has non-negligible imaginary part ($maxim) at q=0"
    Kr = real.(K)
    return Symmetric((Kr + Kr') / 2)
end

"""
    build_K_sparse(H, basis, index) -> SparseMatrixCSC{Float64}

Same kernel as [`build_K`](@ref), stored sparse; threaded over columns.
"""
function build_K_sparse(H::PauliOp, basis::Vector{PauliInt}, index::Dict{PauliInt,Int})
    L = Liouvillian(H)
    n = length(basis)
    nt = Threads.nthreads()
    Ibuf = [Int[] for _ in 1:nt]; Jbuf = [Int[] for _ in 1:nt]; Vbuf = [Float64[] for _ in 1:nt]
    Threads.@threads :static for t in 1:nt
        lo = 1 + ((t - 1) * n) ÷ nt
        hi = (t * n) ÷ nt
        for b in lo:hi
            seed = PauliOp{ComplexF64}(int_to_pauli(basis[b]) => one(ComplexF64))
            D2 = apply_rule(apply_rule(seed, L), L)
            for (key, coeff) in D2
                a = get(index, key, 0)
                a == 0 && continue
                push!(Ibuf[t], a); push!(Jbuf[t], b); push!(Vbuf[t], real(coeff))
            end
        end
    end
    K = sparse(reduce(vcat, Ibuf), reduce(vcat, Jbuf), reduce(vcat, Vbuf), n, n)
    return (K + permutedims(K)) / 2
end

"""
    build_L1(H, basisM, indexM1) -> SparseMatrixCSC{ComplexF64}

q = 0 single-commutator map `L1[c,b] = ([H,P_b])[P_c]` (rows = basis(M+1) via
`indexM1`, cols = basis(M)), via the engine's `apply_rule`. Purely imaginary.
"""
function build_L1(H::PauliOp, basisM::Vector{PauliInt}, indexM1::Dict{PauliInt,Int})
    L = Liouvillian(H)
    n = length(basisM); m = length(indexM1)
    nt = Threads.nthreads()
    Ib = [Int[] for _ in 1:nt]; Jb = [Int[] for _ in 1:nt]; Vb = [ComplexF64[] for _ in 1:nt]
    Threads.@threads :static for t in 1:nt
        lo = 1 + ((t-1)*n) ÷ nt; hi = (t*n) ÷ nt
        for b in lo:hi
            seed = PauliOp{ComplexF64}(int_to_pauli(basisM[b]) => one(ComplexF64))
            for (key, coeff) in apply_rule(seed, L)        # [H, P_b], span ≤ M+1
                r = get(indexM1, key, 0); r == 0 && continue
                push!(Ib[t], r); push!(Jb[t], b); push!(Vb[t], coeff)
            end
        end
    end
    sparse(reduce(vcat, Ib), reduce(vcat, Jb), reduce(vcat, Vb), m, n)
end

_gen(W::Int, sites, chars) = (s = fill('I', W); for (i, ch) in zip(sites, chars); s[i] = ch; end; String(s))

# accumulate [gen, Q] into `out`: 0 if [gen,Q]=0, else 2·(gen·Q) with the product phase.
@inline function _commutator!(out::Dict{String,ComplexF64}, gen::String, Q::String, coeff::ComplexF64)
    commutes(gen, Q) && return
    phase, gQ = pauli_product(gen, Q)
    out[gQ] = get(out, gQ, 0.0im) + 2 * coeff * phase
end

"""
    build_L1_q(basisM, indexM1, M, q; J=1.0, g=0.905, h=0.809, margin=2)

Momentum-q single-commutator map (rows = basis(M+1), cols = basis(M)):

    L1_q[c,b] = Σ_δ (coefficient of P_c at offset δ in [H,P_b]) · e^{-iqδ},

δ = (leftmost non-I site of the output term) − (leftmost site of P_b). The
commutator is evaluated position-explicitly on a width-`M+2margin` canvas so the
offset δ, discarded by the engine's stripping, is recovered. At q = 0 this
equals [`build_L1`](@ref). `σ²(q) = ‖L1_q c‖² / ‖c‖²` is the per-site
commutator norm of the momentum-q ansatz `Σ_j e^{iqj} τ_j(O_M)`. Threaded over columns.
"""
function build_L1_q(basisM::Vector{PauliInt}, indexM1::Dict{PauliInt,Int}, M::Int, q::Float64;
                    J::Float64=1.0, g::Float64=0.905, h::Float64=0.809, margin::Int=2)
    n = length(basisM); m = length(indexM1); W = M + 2margin
    anchor = margin + 1
    nt = Threads.nthreads()
    Ib = [Int[] for _ in 1:nt]; Jb = [Int[] for _ in 1:nt]; Vb = [ComplexF64[] for _ in 1:nt]
    Threads.@threads :static for t in 1:nt
        lo = 1 + ((t-1)*n) ÷ nt; hi = (t*n) ÷ nt
        for b in lo:hi
            Pstr = int_to_pauli(basisM[b])
            canvas = fill('I', W)
            for (k, ch) in enumerate(Pstr); canvas[anchor + k - 1] = ch; end
            Q = String(canvas)
            out = Dict{String,ComplexF64}()
            for i in 1:W;    _commutator!(out, _gen(W, (i,),     ('Z',)),    Q, ComplexF64(h)); end
            for i in 1:W;    _commutator!(out, _gen(W, (i,),     ('X',)),    Q, ComplexF64(g)); end
            for i in 1:W-1;  _commutator!(out, _gen(W, (i, i+1), ('Z','Z')), Q, ComplexF64(J)); end
            for (Qout, c) in out
                abs(c) < 1e-12 && continue
                lpos = findfirst(!=('I'), Qout); lpos === nothing && continue
                δ = lpos - anchor
                key = pauli_to_int(Qout); key == 0 && continue
                r = get(indexM1, key, 0); r == 0 && continue
                push!(Ib[t], r); push!(Jb[t], b); push!(Vb[t], c * cis(-q * δ))
            end
        end
    end
    sparse(reduce(vcat, Ib), reduce(vcat, Jb), reduce(vcat, Vb), m, n)
end

# ─────────────────────────── Floquet, assembled ───────────────────────────

# Conjugate a fixed-width operator by e^{-iα g}: Q ↦ Q if [g,Q]=0, else
# cos(2α)Q − i sin(2α) gQ  (e^{-iαZ} X e^{iαZ} = cos2α·X + sin2α·Y).
function _conj_gen(op::Dict{String,ComplexF64}, g::String, α::Float64)
    c2 = cos(2α); s2 = sin(2α)
    new = Dict{String,ComplexF64}()
    for (Q, c) in op
        if commutes(g, Q)
            new[Q] = get(new, Q, 0.0im) + c
        else
            phase, gQ = pauli_product(g, Q)
            new[Q]  = get(new, Q, 0.0im)  + c * c2
            new[gQ] = get(new, gQ, 0.0im) + c * (-im * s2 * phase)
        end
    end
    new
end

"""
    heisenberg_conjugate_op(P, M, τ, g, h) -> Dict{PauliInt,Float64}

`U_F P U_F†` for a stripped Pauli `P` (span ≤ M), threaded term-by-term through
the commuting local rotations of `U_F = e^{-iτH_x} e^{-iτH_z}` (exact,
multiplicative — not the additive `apply_rule`). Returned on canonical keys with
translates accumulated (the q = 0 shifted sum). Coefficients are real.
"""
function heisenberg_conjugate_op(P::PauliInt, M::Int, τ::Float64, g::Float64, h::Float64)
    margin = 3; W = M + 2margin
    canvas = fill('I', W)
    for (k, ch) in enumerate(int_to_pauli(P)); canvas[margin + k] = ch; end
    op = Dict{String,ComplexF64}(String(canvas) => 1.0 + 0im)
    for i in 1:W;     op = _conj_gen(op, _gen(W, (i,),     ('Z',)),     h * τ); end
    for i in 1:W-1;   op = _conj_gen(op, _gen(W, (i, i+1), ('Z', 'Z')), τ);     end
    for i in 1:W;     op = _conj_gen(op, _gen(W, (i,),     ('X',)),     g * τ); end
    result = Dict{PauliInt,Float64}()
    for (Q, c) in op
        abs(c) < 1e-12 && continue
        key = pauli_to_int(Q)
        key == 0 && continue
        result[key] = get(result, key, 0.0) + real(c)
    end
    result
end

"""
    build_KF_sparse(basis, index, M; τ=0.8, g=0.905, h=0.809) -> SparseMatrixCSC{Float64}

Assembled Floquet kernel `K_F = 2I − T − Tᵀ`, `T[a,b] = (U_F P_b U_F†)[P_a]`. Real
symmetric PSD. Reference for the matrix-free applies; threaded over columns.
"""
function build_KF_sparse(basis::Vector{PauliInt}, index::Dict{PauliInt,Int}, M::Int;
                         τ::Float64=0.8, g::Float64=0.905, h::Float64=0.809)
    n = length(basis)
    nt = Threads.nthreads()
    Ibuf = [Int[] for _ in 1:nt]; Jbuf = [Int[] for _ in 1:nt]; Vbuf = [Float64[] for _ in 1:nt]
    Threads.@threads :static for t in 1:nt
        lo = 1 + ((t - 1) * n) ÷ nt; hi = (t * n) ÷ nt
        for b in lo:hi
            for (key, coeff) in heisenberg_conjugate_op(basis[b], M, τ, g, h)
                a = get(index, key, 0); a == 0 && continue
                push!(Ibuf[t], a); push!(Jbuf[t], b); push!(Vbuf[t], coeff)
            end
        end
    end
    T = sparse(reduce(vcat, Ibuf), reduce(vcat, Jbuf), reduce(vcat, Vbuf), n, n)
    KF = sparse(2.0I, n, n) - T - permutedims(T)
    return (KF + permutedims(KF)) / 2
end

# ─────────────────────────── Floquet, matrix-free (CPU) ───────────────────────────
#
# Dense real coefficient vector over the 4^W Hermitian Paulis of a W = M+2·margin
# canvas (margin = 1 suffices: the one-period light cone has radius 1). Site Pauli
# (x,z) bits: I=(0,0) X=(1,0) Z=(0,1) Y=(1,1); canvas index j = xm | (zm << W).
# Each gate e^{-iαg} is an exact rotation on the anticommuting pair (P, ±gP):
#     ψ'[j] = cos2α·ψ[j] − sin2α·ε[j]·ψ[j ⊻ gmask]   (ε = ±1, 0 if [g,P]=0)
#   Z_i  : anticommute iff site i ∈ {X,Y};  ε = +1 (X), −1 (Y)
#   X_i  : anticommute iff site i ∈ {Y,Z};  ε = +1 (Y), −1 (Z)
#   Z_iZ_{i+1}: anticommute iff exactly one of i,i+1 ∈ {X,Y}; ε from that site.
# The same bit arithmetic is the CUDA kernel in gpu.jl.

# gate schedule of U_F (conjugation order): (code, i, i2, gmask, angle) with
# code 1 = Z_i (angle hτ), 3 = Z_iZ_{i+1} (angle τ), 2 = X_i (angle gτ).
_floquet_schedule(W) = vcat([(1, i, 0, 1<<(W+i-1),               :h) for i in 1:W],
                            [(3, i, i+1, (1<<(W+i-1))|(1<<(W+i)), :t) for i in 1:W-1],
                            [(2, i, 0, 1<<(i-1),                  :g) for i in 1:W])

_floquet_angle(ang::Symbol, τ, g, h) = ang === :h ? h*τ : ang === :g ? g*τ : τ

# canvas index of each basis string (placed at offset `margin`)
function _floquet_lift(basis::Vector{PauliInt}, W::Int, margin::Int)
    lift = Vector{Int64}(undef, length(basis))
    for b in eachindex(basis)
        xm = 0; zm = 0
        for (k, ch) in enumerate(int_to_pauli(basis[b]))
            s = margin + k
            ch=='X' && (xm |= 1<<(s-1)); ch=='Z' && (zm |= 1<<(s-1))
            ch=='Y' && (xm |= 1<<(s-1); zm |= 1<<(s-1))
        end
        lift[b] = xm | (zm << W)
    end
    lift
end

# basis index of each canvas Pauli (0 = outside span ≤ M, or identity)
function _floquet_fold(index::Dict{PauliInt,Int}, W::Int)
    N = Int64(4)^W
    fold = zeros(Int32, N)
    chars = ('I','X','Z','Y')                     # index 1 + x + 2z
    buf = Vector{Char}(undef, W)
    for j in 1:N-1
        for k in 1:W
            buf[k] = chars[1 + ((j >> (k-1)) & 1) + 2*((j >> (W+k-1)) & 1)]
        end
        key = pauli_to_int(String(buf)); key == 0 && continue
        fold[j+1] = Int32(get(index, key, 0))
    end
    fold
end

@inline function _gate_entry(ψ, j::Int64, W::Int64, code::Int64, i::Int64, i2::Int64,
                             gmask::Int64, c2::Float64, s2::Float64)
    v = ψ[j+1]
    if code == 1                                   # Z_i
        (j >> (i-1)) & 1 == 1 || return v
        ε = ((j >> (W+i-1)) & 1) == 0 ? 1.0 : -1.0
    elseif code == 2                               # X_i
        (j >> (W+i-1)) & 1 == 1 || return v
        ε = ((j >> (i-1)) & 1) == 1 ? 1.0 : -1.0
    else                                           # Z_i Z_{i2}
        a1 = (j >> (i-1)) & 1; a2 = (j >> (i2-1)) & 1
        (a1 ⊻ a2) == 1 || return v
        ε = a1 == 1 ? (((j >> (W+i-1)) & 1) == 0 ? 1.0 : -1.0) :
                      (((j >> (W+i2-1)) & 1) == 0 ? 1.0 : -1.0)
    end
    return c2*v - s2*ε*ψ[(j ⊻ gmask) + 1]
end

"""
    DenseKF(basis, index, M; τ, g, h, margin=1)

CPU matrix-free Floquet kernel on the 4^(M+2·margin) canvas; `apply_KF(K, x)`
returns `K_F x` without assembling `K_F`.
"""
struct DenseKF
    n::Int; W::Int; τ::Float64; g::Float64; h::Float64
    lift::Vector{Int64}
    fold::Vector{Int32}
    ψ::Vector{Float64}; buf::Vector{Float64}
    sched::Vector{Tuple{Int,Int,Int,Int,Symbol}}
end

function DenseKF(basis::Vector{PauliInt}, index::Dict{PauliInt,Int}, M::Int;
                 τ::Float64=0.8, g::Float64=0.905, h::Float64=0.809, margin::Int=1)
    W = M + 2margin; N = Int64(4)^W
    DenseKF(length(basis), W, τ, g, h, _floquet_lift(basis, W, margin), _floquet_fold(index, W),
            zeros(N), zeros(N), _floquet_schedule(W))
end

"`K_F x = 2x − T x − Tᵀ x`: lift, conjugate by U_F (resp. U_F†), fold, subtract."
function apply_KF(K::DenseKF, x::AbstractVector{Float64})
    y = 2.0 .* Vector{Float64}(x)
    N = Int64(length(K.ψ))
    for dagger in (false, true)
        sgn = dagger ? -1.0 : 1.0
        fill!(K.ψ, 0.0)
        @inbounds for b in 1:K.n; K.ψ[K.lift[b]+1] = x[b]; end
        cur, nxt = K.ψ, K.buf
        for (code, i, i2, gmask, ang) in (dagger ? reverse(K.sched) : K.sched)
            α = _floquet_angle(ang, K.τ, K.g, K.h)
            c2 = cos(2α); s2 = sgn*sin(2α)
            @inbounds for j in Int64(0):N-1
                nxt[j+1] = _gate_entry(cur, j, Int64(K.W), Int64(code), Int64(i), Int64(i2),
                                       Int64(gmask), c2, s2)
            end
            cur, nxt = nxt, cur
        end
        @inbounds for j in 1:N
            a = K.fold[j]; a == 0 || (y[a] -= cur[j])
        end
    end
    return y
end
