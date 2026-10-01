"""
    ThermodynamicPauli

Pauli-string algebra in the thermodynamic limit (infinite chain). A
translation-invariant operator `Σ_j τ_j(O)` is represented by one `PauliOp`
whose keys are *stripped* Pauli strings (no leading/trailing identities), so
all translates of a string share one key. `UpdateRule`s (e.g. the
`Liouvillian` `[H,·]`) act on every window position and fold the result back
to stripped form; `apply_rule`/`propagate` therefore compute per-site
(translation-summed) coefficients directly.

Encoding: 2 bits per site, `I=00 X=01 Y=10 Z=11`, leftmost site in the lowest
bits, packed into a `UInt128` (≤ 64 sites).
"""
module ThermodynamicPauli

export PauliOp, UpdateRule, PauliInt,
    apply_rule, propagate, strip_paulis, clean!,
    pauli_product, commutes, all_pauli_windows,
    overlap_zero, overlap_plus, overlap_maxmixed,
    hilbert_schmidt, theta_inner, RPS_inner, pauli_size, exp_size, dropzeros!,
    Liouvillian, theta_norm, RPS_norm, avg_size, pauli_weight,
    pauli_to_int, int_to_pauli

# ══════════════════════════════════════════════════════════════════════
# Integer encoding
# ══════════════════════════════════════════════════════════════════════
#
# Each qubit uses 2 bits:  I=00  X=01  Y=10  Z=11
# Position 1 (leftmost in string) → bits 0-1 (lowest).
# The identity operator is integer 0.
# Stored integers are always "stripped": the lowest 2-bit pair is nonzero
# (unless the whole integer is 0).

const PauliInt = UInt128

# Bit masks for 128-bit integers (64 qubit pairs)
const _EVEN_MASK = ~PauliInt(0) ÷ 3          # 0101...01 — even-index bits
const _ODD_MASK  = _EVEN_MASK << 1            # 1010...10 — odd-index bits

const _CHAR_TO_BITS = Dict('I' => PauliInt(0), 'X' => PauliInt(1),
                           'Y' => PauliInt(2), 'Z' => PauliInt(3))
const _BITS_TO_CHAR = ('I', 'X', 'Y', 'Z')   # indexed by bits+1

"""
    pauli_to_int(s) -> PauliInt

Encode a Pauli string as an integer. Leading and trailing I's are stripped.
"""
function pauli_to_int(s::AbstractString)::PauliInt
    _encode_raw(strip_paulis(s))
end

"""Encode a Pauli string as-is (no stripping). Used for fixed-width windows."""
function _encode_raw(s::AbstractString)::PauliInt
    s == "I" && return PauliInt(0)
    n = PauliInt(0)
    for (i, c) in enumerate(s)
        n |= _CHAR_TO_BITS[c] << (2 * (i - 1))
    end
    n
end

"""
    int_to_pauli(n) -> String

Decode an integer to a stripped Pauli string.
"""
function int_to_pauli(n::PauliInt)::String
    n == 0 && return "I"
    chars = Char[]
    while n != 0
        push!(chars, _BITS_TO_CHAR[(n & 0x3) + 1])
        n >>= 2
    end
    String(chars)
end

"""Strip leading identity pairs from an integer (shift out low zero-bit pairs)."""
function strip_int(n::PauliInt)::PauliInt
    n == 0 && return PauliInt(0)
    n >> (trailing_zeros(n) & ~1)   # round trailing zeros down to even
end

"""Number of qubit positions in a stripped integer (distance from bit 0 to highest set pair)."""
function support_length(n::PauliInt)::Int
    n == 0 && return 0
    cld(128 - leading_zeros(n), 2)
end

"""Number of non-I qubits in a Pauli integer."""
function pauli_weight(n::PauliInt)::Int
    count_ones((n | (n >> 1)) & _EVEN_MASK)
end

# ══════════════════════════════════════════════════════════════════════
# String-level Pauli algebra  (used by Liouvillian construction)
# ══════════════════════════════════════════════════════════════════════

const PAULIS = ('I', 'X', 'Y', 'Z')

const PAULI_MULT = let
    d = Dict{Tuple{Char,Char},Tuple{ComplexF64,Char}}()
    for p in PAULIS; d[('I', p)] = (1, p); d[(p, 'I')] = (1, p); end
    for p in ('X', 'Y', 'Z'); d[(p, p)] = (1, 'I'); end
    d[('X', 'Y')] = (im, 'Z'); d[('Y', 'X')] = (-im, 'Z')
    d[('Y', 'Z')] = (im, 'X'); d[('Z', 'Y')] = (-im, 'X')
    d[('Z', 'X')] = (im, 'Y'); d[('X', 'Z')] = (-im, 'Y')
    d
end

"""Multiply two equal-length Pauli strings site-by-site. Returns `(phase, result)`."""
function pauli_product(a::AbstractString, b::AbstractString)
    length(a) == length(b) || throw(ArgumentError("Pauli strings must have equal length"))
    phase = ComplexF64(1)
    buf = Vector{Char}(undef, length(a))
    @inbounds for i in 1:length(a)
        p, r = PAULI_MULT[(a[i], b[i])]
        phase *= p
        buf[i] = r
    end
    phase, String(buf)
end

_anticommutes(a::Char, b::Char) = (a != 'I') & (b != 'I') & (a != b)

"""Check whether two equal-length Pauli strings commute."""
function commutes(a::AbstractString, b::AbstractString)
    length(a) == length(b) || throw(ArgumentError("Pauli strings must have equal length"))
    iseven(count(i -> _anticommutes(a[i], b[i]), 1:length(a)))
end

"""Generate all `w`-character Pauli strings (4^w strings)."""
function all_pauli_windows(w::Int)
    w == 0 && return [""]
    vec([join(t) for t in Iterators.product(ntuple(_ -> PAULIS, w)...)])
end

"""Strip leading and trailing `I` characters from a Pauli string."""
function strip_paulis(s::AbstractString)
    lo = findfirst(!=('I'), s)
    isnothing(lo) && return "I"
    hi = findlast(!=('I'), s)
    s[lo:hi]
end

# String-based weight (number of non-I characters)
pauli_weight(p::AbstractString) = count(!=('I'), p)

# ══════════════════════════════════════════════════════════════════════
# PauliOp type
# ══════════════════════════════════════════════════════════════════════

"""
    PauliOp{T<:Number}

A Pauli operator in the thermodynamic limit.

Internally keyed by `PauliInt` (2-bit-per-qubit integer encoding).
The identity is integer `0`. All keys are stripped (lowest pair nonzero).

Supports arithmetic: `α * op`, `op * α`, `op1 + op2`, `op1 - op2`, `-op`.
String-based indexing (`op["XZ"]`) is supported for convenience.
"""
struct PauliOp{T<:Number}
    terms::Dict{PauliInt,T}
end

PauliOp{T}() where {T<:Number} = PauliOp{T}(Dict{PauliInt,T}())

function PauliOp(pairs::Pair{String,T}...) where {T<:Number}
    d = Dict{PauliInt,T}()
    for (s, v) in pairs
        k = pauli_to_int(s)
        d[k] = get(d, k, zero(T)) + v
    end
    dropzeros!(PauliOp{T}(d))
end

function PauliOp{T}(pairs::Pair{String}...) where {T<:Number}
    d = Dict{PauliInt,T}()
    for (s, v) in pairs
        k = pauli_to_int(s)
        d[k] = get(d, k, zero(T)) + convert(T, v)
    end
    dropzeros!(PauliOp{T}(d))
end

_from_dict(d::Dict{PauliInt,T}) where {T<:Number} = dropzeros!(PauliOp{T}(d))

# ──── Dict-like interface ────────────────────────────────────────────

# String-based access (convenience)
Base.getindex(op::PauliOp, k::String) = op.terms[pauli_to_int(k)]
Base.setindex!(op::PauliOp, v, k::String) = (op.terms[pauli_to_int(k)] = v; op)
Base.get(op::PauliOp, k::String, default) = get(op.terms, pauli_to_int(k), default)
Base.haskey(op::PauliOp, k::String) = haskey(op.terms, pauli_to_int(k))

# Integer-based access (fast)
Base.getindex(op::PauliOp, k::PauliInt) = op.terms[k]
Base.setindex!(op::PauliOp, v, k::PauliInt) = (op.terms[k] = v; op)
Base.get(op::PauliOp, k::PauliInt, default) = get(op.terms, k, default)
Base.haskey(op::PauliOp, k::PauliInt) = haskey(op.terms, k)

Base.keys(op::PauliOp) = keys(op.terms)
Base.values(op::PauliOp) = values(op.terms)
Base.pairs(op::PauliOp) = pairs(op.terms)
Base.iterate(op::PauliOp) = iterate(op.terms)
Base.iterate(op::PauliOp, state) = iterate(op.terms, state)
Base.length(op::PauliOp) = length(op.terms)
Base.isempty(op::PauliOp) = isempty(op.terms)
Base.delete!(op::PauliOp, k::String) = (delete!(op.terms, pauli_to_int(k)); op)
Base.delete!(op::PauliOp, k::PauliInt) = (delete!(op.terms, k); op)
Base.valtype(::PauliOp{T}) where {T} = T
Base.valtype(::Type{PauliOp{T}}) where {T} = T
Base.copy(op::PauliOp{T}) where {T} = PauliOp{T}(copy(op.terms))
Base.:(==)(a::PauliOp, b::PauliOp) = a.terms == b.terms

function Base.show(io::IO, op::PauliOp)
    if isempty(op)
        print(io, "PauliOp(empty)")
        return
    end
    print(io, "PauliOp(")
    first_entry = true
    for (k, c) in op.terms
        first_entry || print(io, ", ")
        print(io, "\"$(int_to_pauli(k))\" => $c")
        first_entry = false
    end
    print(io, ")")
end

# ──── Zero removal ───────────────────────────────────────────────────

"""Remove terms with exactly zero amplitude."""
function dropzeros!(op::PauliOp)
    filter!(kv -> !iszero(kv.second), op.terms)
    op
end

# ──── Arithmetic ─────────────────────────────────────────────────────

function Base.:+(a::PauliOp, b::PauliOp)
    RT = promote_type(valtype(a), valtype(b))
    result = Dict{PauliInt,RT}(a.terms)
    for (k, c) in b.terms
        result[k] = get(result, k, zero(RT)) + c
    end
    _from_dict(result)
end

function Base.:-(op::PauliOp{T}) where {T}
    PauliOp{T}(Dict{PauliInt,T}(k => -c for (k, c) in op.terms))
end

function Base.:-(a::PauliOp, b::PauliOp)
    RT = promote_type(valtype(a), valtype(b))
    result = Dict{PauliInt,RT}(a.terms)
    for (k, c) in b.terms
        result[k] = get(result, k, zero(RT)) - c
    end
    _from_dict(result)
end

function Base.:*(α::Number, op::PauliOp)
    RT = promote_type(typeof(α), valtype(op))
    iszero(α) && return PauliOp{RT}()
    PauliOp{RT}(Dict{PauliInt,RT}(k => α * c for (k, c) in op.terms))
end

Base.:*(op::PauliOp, α::Number) = α * op
Base.:/(op::PauliOp, α::Number) = (1 / α) * op

# ══════════════════════════════════════════════════════════════════════
# UpdateRule
# ══════════════════════════════════════════════════════════════════════

"""
    UpdateRule{T}

A local update rule with a fixed window size.
Internally keyed by `PauliInt` window encodings (NOT stripped — full window width).
"""
struct UpdateRule{T<:Number}
    window_size::Int
    rules::Dict{PauliInt,Dict{PauliInt,T}}
end

function UpdateRule(rules::Dict{String,Dict{String,T}}) where {T<:Number}
    isempty(rules) && return UpdateRule{T}(0, Dict{PauliInt,Dict{PauliInt,T}}())
    w = length(first(keys(rules)))
    int_rules = Dict{PauliInt,Dict{PauliInt,T}}()
    for (k, v) in rules
        length(k) == w || throw(ArgumentError("Input window \"$k\" has length $(length(k)), expected $w"))
        v_int = Dict{PauliInt,T}()
        for (k2, c) in v
            length(k2) == w || throw(ArgumentError("Output window \"$k2\" has length $(length(k2)), expected $w"))
            v_int[_encode_raw(k2)] = c
        end
        int_rules[_encode_raw(k)] = v_int
    end
    UpdateRule{T}(w, int_rules)
end

# ══════════════════════════════════════════════════════════════════════
# Liouvillian
# ══════════════════════════════════════════════════════════════════════

"""
    Liouvillian(H::PauliOp) -> UpdateRule

Construct the update rule for the superoperator ``[H, \\cdot]``.

For Pauli strings that anticommute with a Hamiltonian term ``P``:
``[P, Q] = 2 P Q``. Commuting pairs contribute nothing.
"""
function Liouvillian(H::PauliOp{T}) where {T}
    CT = complex(T)
    isempty(H) && return UpdateRule{CT}(0, Dict{PauliInt,Dict{PauliInt,CT}}())

    # Convert to strings for algebra, then back to ints for storage
    H_strings = Dict(int_to_pauli(k) => v for (k, v) in H.terms)
    w = maximum(length(p) for p in keys(H_strings))

    rules = Dict{PauliInt,Dict{PauliInt,CT}}()

    for Q_str in all_pauli_windows(w)
        output = Dict{PauliInt,CT}()

        for (P_str, h_coeff) in H_strings
            P_padded = P_str * "I"^(w - length(P_str))
            commutes(P_padded, Q_str) && continue
            phase, R_str = pauli_product(P_padded, Q_str)
            R_int = _encode_raw(R_str)
            output[R_int] = get(output, R_int, zero(CT)) + h_coeff * 2 * phase
        end

        filter!(kv -> !iszero(kv.second), output)
        !isempty(output) && (rules[_encode_raw(Q_str)] = output)
    end

    UpdateRule{CT}(w, rules)
end

# ══════════════════════════════════════════════════════════════════════
# Core: apply_rule  (integer-based inner loop)
# ══════════════════════════════════════════════════════════════════════

"""Process all windows of one Pauli string, accumulate into `result`."""
function _apply_windows!(result::Dict{PauliInt,RT}, padded::PauliInt, coeff,
                         rules, nwin::Int, wmask::PauliInt) where {RT}
    for pos in 1:nwin
        shift = 2 * (pos - 1)
        window = (padded >> shift) & wmask
        local_rules = get(rules, window, nothing)
        isnothing(local_rules) && continue

        base = padded & ~(wmask << shift)

        for (new_window, amp) in local_rules
            new_padded = base | (new_window << shift)
            key = strip_int(new_padded)
            result[key] = get(result, key, zero(RT)) + coeff * amp
        end
    end
end

"""
    apply_rule(op, rule; parallel=false, max_weight=0) -> PauliOp

Apply a translationally invariant update rule to a Pauli operator.

When `max_weight > 0`, output terms whose Pauli weight exceeds
`max_weight` are discarded (truncation).
"""
function apply_rule(op::PauliOp{T1}, rule::UpdateRule{T2};
                    parallel::Bool=false, max_weight::Int=0) where {T1,T2}
    RT = promote_type(T1, T2)
    w = rule.window_size
    w == 0 && return PauliOp{RT}(Dict{PauliInt,RT}(op.terms))

    if parallel && Threads.nthreads() > 1
        return _apply_rule_parallel(op, rule, RT, w, max_weight)
    end

    result = Dict{PauliInt,RT}()
    wmask = (PauliInt(1) << (2 * w)) - 1
    pad_shift = 2 * (w - 1)
    z = PauliInt(0)

    for (pint, coeff) in op.terms
        if pint == z
            result[z] = get(result, z, zero(RT)) + coeff
            continue
        end
        padded = pint << pad_shift
        slen = support_length(pint)
        nwin = slen + w - 1       # == slen + 2*(w-1) - w + 1
        _apply_windows!(result, padded, coeff, rule.rules, nwin, wmask)
    end

    filter!(kv -> !iszero(kv.second), result)
    max_weight > 0 && filter!(kv -> pauli_weight(kv.first) <= max_weight, result)
    PauliOp{RT}(result)
end

# ──── Parallel apply_rule ────────────────────────────────────────────

function _apply_rule_parallel(op::PauliOp, rule::UpdateRule, ::Type{RT},
                              w::Int, max_weight::Int) where {RT}
    nt = Threads.nthreads()
    thread_dicts = [Dict{PauliInt,RT}() for _ in 1:nt]
    terms = collect(pairs(op.terms))
    wmask = (PauliInt(1) << (2 * w)) - 1
    pad_shift = 2 * (w - 1)
    z = PauliInt(0)

    if length(terms) >= nt
        # Many terms — parallelise over terms
        Threads.@threads :static for idx in eachindex(terms)
            tid = Threads.threadid()
            pint, coeff = terms[idx]
            if pint == z
                d = thread_dicts[tid]
                d[z] = get(d, z, zero(RT)) + coeff
            else
                padded = pint << pad_shift
                slen = support_length(pint)
                nwin = slen + w - 1
                _apply_windows!(thread_dicts[tid], padded, coeff,
                                rule.rules, nwin, wmask)
            end
        end
    else
        # Few long terms — parallelise over windows
        for (pint, coeff) in terms
            if pint == z
                thread_dicts[1][z] = get(thread_dicts[1], z, zero(RT)) + coeff
                continue
            end
            padded = pint << pad_shift
            slen = support_length(pint)
            nwin = slen + w - 1

            chunk = max(1, cld(nwin, nt))
            Threads.@threads :static for t in 1:min(nt, nwin)
                tid = Threads.threadid()
                istart = (t - 1) * chunk + 1
                istop = min(t * chunk, nwin)
                for pos in istart:istop
                    shift = 2 * (pos - 1)
                    window = (padded >> shift) & wmask
                    lr = get(rule.rules, window, nothing)
                    isnothing(lr) && continue
                    base = padded & ~(wmask << shift)
                    for (nw, amp) in lr
                        key = strip_int(base | (nw << shift))
                        d = thread_dicts[tid]
                        d[key] = get(d, key, zero(RT)) + coeff * amp
                    end
                end
            end
        end
    end

    # Merge
    result = thread_dicts[1]
    for i in 2:nt
        for (k, v) in thread_dicts[i]
            result[k] = get(result, k, zero(RT)) + v
        end
    end
    filter!(kv -> !iszero(kv.second), result)
    max_weight > 0 && filter!(kv -> pauli_weight(kv.first) <= max_weight, result)
    PauliOp{RT}(result)
end

"""
    propagate(op, rules; clean_tol=0.0, parallel=false, max_weight=0) -> PauliOp

Apply a sequence of `UpdateRule`s. Optionally truncate by coefficient
magnitude (`clean_tol`) and/or Pauli weight (`max_weight`) after each step.
"""
function propagate(op::PauliOp, rules; clean_tol::Real=0.0,
                   parallel::Bool=false, max_weight::Int=0)
    for rule in rules
        op = apply_rule(op, rule; parallel, max_weight)
        clean_tol > 0 && clean!(op, clean_tol)
    end
    op
end

# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

"""Remove entries whose coefficient magnitude is below `tol`."""
function clean!(op::PauliOp, tol::Real=1e-12)
    filter!(kv -> abs(kv.second) >= tol, op.terms)
    op
end

# ──── State overlaps ─────────────────────────────────────────────────

"""Per-site expectation value with |0⟩^⊗∞ (sums I-and-Z-only terms)."""
function overlap_zero(op::PauliOp)
    s = zero(valtype(op))
    for (k, c) in op.terms
        # I=00 and Z=11 both have equal bits in each pair.
        # X=01, Y=10 have differing bits.
        ((k ⊻ (k >> 1)) & _EVEN_MASK) == 0 && (s += c)
    end
    s
end

"""Per-site expectation value with |+⟩^⊗∞ (sums I-and-X-only terms)."""
function overlap_plus(op::PauliOp)
    s = zero(valtype(op))
    for (k, c) in op.terms
        # I=00 and X=01 have bit-1 = 0.  Y=10, Z=11 have bit-1 = 1.
        (k & _ODD_MASK) == 0 && (s += c)
    end
    s
end

"""Expectation value with the maximally mixed state (identity coefficient)."""
function overlap_maxmixed(op::PauliOp)
    get(op.terms, PauliInt(0), zero(valtype(op)))
end

# ──── Weight-based operations ────────────────────────────────────────

"""Multiply each coefficient by the Pauli weight |P|."""
function pauli_size(op::PauliOp)
    _from_dict(Dict(k => c * pauli_weight(k) for (k, c) in op.terms))
end

"""Multiply each coefficient by exp(θ * |P|)."""
function exp_size(op::PauliOp, θ::Real)
    _from_dict(Dict(k => c * exp(θ * pauli_weight(k)) for (k, c) in op.terms))
end

# ──── Inner products ─────────────────────────────────────────────────

"""Hilbert-Schmidt inner product: ``\\sum_P a_P^* b_P``."""
function hilbert_schmidt(a::PauliOp, b::PauliOp)
    RT = promote_type(valtype(a), valtype(b))
    s = zero(RT)
    for (k, ca) in a.terms
        cb = get(b.terms, k, nothing)
        isnothing(cb) && continue
        s += conj(ca) * cb
    end
    s
end

"""θ-weighted inner product: ``\\sum_P a_P^* b_P \\, e^{-2\\theta|P|}``."""
function theta_inner(a::PauliOp, b::PauliOp, θ::Real)
    RT = promote_type(valtype(a), valtype(b))
    s = zero(RT)
    for (k, ca) in a.terms
        cb = get(b.terms, k, nothing)
        isnothing(cb) && continue
        s += conj(ca) * cb * exp(-2θ * pauli_weight(k))
    end
    s
end

"""RPS inner product: `theta_inner` at θ = ½ln(3), i.e. ``\\sum_P a_P^* b_P \\, 3^{-|P|}``."""
RPS_inner(a::PauliOp, b::PauliOp) = theta_inner(a, b, 0.5 * log(3))

"""Exponentially weighted norm: ``\\sqrt{\\sum_P |c_P|^2 e^{-2\\theta|P|}}``."""
function theta_norm(op::PauliOp, θ::Real)
    s = 0.0
    for (k, c) in op.terms
        s += abs2(c) * exp(-2θ * pauli_weight(k))
    end
    sqrt(s)
end

"""RPS norm: `theta_norm` at θ = ½ln(3)."""
RPS_norm(op::PauliOp) = theta_norm(op, 0.5 * log(3))

"""Size norm: ``\\sum_P |c_P|^2 \\, |P|``."""
function avg_size(op::PauliOp)
    s = 0.0
    for (k, c) in op.terms
        s += abs2(c) * pauli_weight(k)
    end
    s
end

end # module
