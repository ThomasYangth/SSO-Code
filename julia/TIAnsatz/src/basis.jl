# Translation-invariant ansatz basis and the RPS metric.
#
# The variational operator is A = Σ_j τ_j(O_M), O_M traceless Hermitian on M
# consecutive sites, O_M = Σ_a c_a P_a. Using the *distinct stripped* Pauli strings
# of span ≤ M as the basis removes the translation redundancy of a fixed-width
# window, so the per-site Hilbert–Schmidt Gram matrix is the identity (G = I).

"""
    canonical_basis(M) -> (basis::Vector{PauliInt}, index::Dict{PauliInt,Int})

Distinct non-identity stripped Pauli strings of span ≤ `M`: the redundancy-free
translation-invariant basis, of size `3·4^(M-1)`. `index` maps a string to its
position in `basis`.
"""
function canonical_basis(M::Int)
    M >= 1 || throw(ArgumentError("M must be ≥ 1"))
    seen = Set{PauliInt}()
    basis = PauliInt[]
    for w in all_pauli_windows(M)
        k = pauli_to_int(w)          # strips leading/trailing I; "III…" -> 0
        k == 0 && continue
        if !(k in seen)
            push!(seen, k)
            push!(basis, k)
        end
    end
    index = Dict(k => i for (i, k) in enumerate(basis))
    @assert length(basis) == 3 * 4^(M - 1) "basis size $(length(basis)) ≠ 3·4^(M-1)"
    return basis, index
end

"""
    build_R(basis) -> Diagonal

RPS weight `R = diag(3^{-wt(P_a)})`: ν(c) = c'Rc / c'c is the infinite-temperature
random-product-state (RPS) weight of the ansatz operator.
"""
build_R(basis::Vector{PauliInt}) = Diagonal([3.0^(-pauli_weight(k)) for k in basis])

"""
    kim_huse_H(; J=1.0, g=0.905, h=0.809) -> PauliOp{ComplexF64}

Mixed-field Ising Hamiltonian density `J·ZZ + g·X + h·Z` (translation-summed by
the engine), H = Σ_i (J Z_iZ_{i+1} + g X_i + h Z_i).
"""
kim_huse_H(; J=1.0, g=0.905, h=0.809) =
    PauliOp{ComplexF64}("ZZ" => ComplexF64(J), "X" => ComplexF64(g), "Z" => ComplexF64(h))

"""
    energy_density_vector(basis, index; J, g, h) -> Vector{Float64}

Unit coefficient vector of the conserved energy density `J·ZZ + g·X + h·Z`
(strings absent for the given `M` are dropped). It is an exact zero mode of the
static kernel `K` (for M ≥ 2) and is projected out on the q = 0 "projected" path.
"""
function energy_density_vector(basis::Vector{PauliInt}, index::Dict{PauliInt,Int};
                               J::Real, g::Real, h::Real)
    c = zeros(Float64, length(basis))
    for (str, val) in (("Z", h), ("X", g), ("ZZ", J))
        k = pauli_to_int(str)
        haskey(index, k) && (c[index[k]] = val)
    end
    nrm = norm(c)
    nrm == 0 && throw(ArgumentError("empty energy-density vector"))
    return c / nrm
end

"""
    complement_projector(c_H) -> Q

Orthonormal `n × (n-1)` basis of the subspace orthogonal to `c_H` (via QR); the
dense q = 0 projected solver works in `range(Q)`.
"""
function complement_projector(c_H::AbstractVector)
    n = length(c_H)
    F = qr(reshape(collect(float(c_H)), n, 1))
    Qfull = F.Q * Matrix{Float64}(I, n, n)   # full n×n orthogonal (F.Q alone is thin)
    return Qfull[:, 2:n]
end
