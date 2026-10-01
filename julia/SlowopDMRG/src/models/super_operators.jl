#=
===============================================================================
SUPER-OPERATOR CONSTRUCTION
===============================================================================
=#

"""
    adjoint_mpo(W::MPO) -> MPO

Hermitian adjoint of an MPO, in the same index convention as the input: the
result `Wd` satisfies `Wd[i][s'=a, s=b] = conj(W[i][s'=b, s=a])`, so `Wd` acts as
``W^\\dagger``.

Link indices are left at prime level 0, exactly mirroring `W`, so `Wd[i]` is a
drop-in replacement for `W[i]` anywhere the caller does its own priming — in
particular as the top layer of the factored 4-legged `LH`/`RH` environments in
`dmrg_sweeps!`, where the caller applies `prime(...)` to lift the layer from
prime level 1→2 and its link chain from 0→1.

Needed because the environments contract two un-squared operator layers as a
*matrix product* `W · W`, which equals `W† W` only for Hermitian `W`. `C_H =
H_L − H_R` is Hermitian so the Hamiltonian path can pass `W` twice; the Floquet
`C_U = 𝒰 − 𝟙` is not, and must pass `adjoint_mpo(C_U)` as the top layer.

Contrast with the `replaceprime(dag(A)', 2=>0)` recipe used for `U_dag` inside
`build_floquet_superoperator`: that is also a correct adjoint, but it leaves link
indices at prime level 1, which breaks the symmetry the environment code relies
on. Use this function for anything that feeds the environments.
"""
function adjoint_mpo(W::MPO)
    Wd = copy(W)
    for i in eachindex(Wd)
        s = siteind(W, i)                       # unprimed (column) site index
        Wd[i] = swapind(dag(Wd[i]), s, prime(s))
    end
    return Wd
end


"""
    _normalize_gate_spec(gates) -> Vector{Pair{String,Float64}}

Coerce a Floquet gate specification into an **ordered** layer list.

Floquet layers generally do not commute (the transverse kick `X` does not commute
with the `ZZ`+`Z` layer), so the order in which `build_floquet_superoperator`
applies them is physically meaningful. An `AbstractDict` iterates in hash order,
which is not the insertion order and is not stable across sessions — so a `Dict`
spec silently picks an arbitrary Floquet unitary. Passing a `Vector` of pairs
fixes the order; a `Dict` is still accepted (for backwards compatibility with
existing call sites and tests) but warns.
"""
function _normalize_gate_spec(gates::AbstractVector)
    return [String(first(g)) => Float64(last(g)) for g in gates]
end

function _normalize_gate_spec(gates::AbstractDict)
    if length(gates) > 1
        @warn "Floquet gate spec given as a Dict: layer order is Julia hash order, " *
              "not insertion order, and non-commuting layers therefore define an " *
              "arbitrary unitary. Pass an ordered Vector of pairs instead, e.g. " *
              "[\"ZZ\" => -J, \"Z\" => -h, \"X\" => -g]."
    end
    return [String(k) => Float64(v) for (k, v) in gates]
end


"""
    build_super_operators(L::Int, sites, H_dict::Dict; pbc::Bool=false)

Constructs MPO super-operators ``H_L`` and ``H_R`` for the Hamiltonian ``H``.

# Arguments
- `L`: System size (number of sites)
- `sites`: ITensor site indices (should be "Pauli" type with dimension 4)
- `H_dict`: Dictionary mapping Pauli strings to coefficients
          Example: `Dict("X" => -1.0, "ZZ" => -1.0)` for transverse-field Ising
- `pbc`: Whether to use periodic boundary conditions

# Returns
- `H_L`: MPO representing left multiplication by ``H`` (``O \\mapsto H \\cdot O``)
- `H_R`: MPO representing right multiplication by ``H`` (``O \\mapsto O \\cdot H``)

# How it works
For each term in `H_dict`, e.g., coefficient ``c`` for string `"ZZ"`:
- Adds ``c \\cdot \\text{ZL}_i \\cdot \\text{ZL}_{i+1}`` to ``H_L`` at all bonds ``i``
- Adds ``c \\cdot \\text{ZR}_i \\cdot \\text{ZR}_{i+1}`` to ``H_R`` at all bonds ``i``

The commutator ``[H, \\cdot]`` is then obtained as ``C_H = H_L - H_R``.

# Example
If ``H = -J \\sum_i Z_i Z_{i+1} - h \\sum_i X_i``, then `H_dict = Dict("ZZ" => -J, "X" => -h)`.
This function builds ``H_L`` and ``H_R`` as MPOs on the Pauli sites.
"""
function build_super_operators(L::Int, sites, H_dict::AbstractDict; pbc::Bool=false)
    os_L = OpSum()
    os_R = OpSum()

    for (op_string, coeff) in H_dict
        n_ops = length(op_string)

        if n_ops == 1
            # Single-site term (e.g., "X" for transverse field)
            for i in 1:L
                os_L += coeff, string(op_string) * "L", i
                os_R += coeff, string(op_string) * "R", i
            end
        elseif n_ops >= 2
            # Multi-site term (e.g., "ZZ" for two-site, "PPX" for three-site)
            # Extract individual operators
            ops = [string(op_string[j]) for j in 1:n_ops]

            # Add terms for all consecutive sequences of n_ops sites
            for i in 1:(L - n_ops + 1)
                # Build the OpSum term with all operators
                # OpSum syntax: os += coeff, "op1", site1, "op2", site2, ...
                sites_L = [i + j - 1 for j in 1:n_ops]
                ops_L = [ops[j] * "L" for j in 1:n_ops]
                ops_R = [ops[j] * "R" for j in 1:n_ops]

                # Interleave operators and sites: [op1, site1, op2, site2, ...]
                args_L = Any[coeff]
                args_R = Any[coeff]
                for j in 1:n_ops
                    push!(args_L, ops_L[j], sites_L[j])
                    push!(args_R, ops_R[j], sites_L[j])
                end

                os_L += tuple(args_L...)
                os_R += tuple(args_R...)
            end

            if pbc && n_ops == 2
                # Add boundary term for two-site operators with PBC
                os_L += coeff, ops[1] * "L", L, ops[2] * "L", 1
                os_R += coeff, ops[1] * "R", L, ops[2] * "R", 1
            elseif pbc && n_ops > 2
                @warn "PBC not yet implemented for operator strings with length > 2"
            end
        else
            error("Invalid operator string: $(op_string)")
        end
    end

    return MPO(os_L, sites), MPO(os_R, sites)
end


#=
===============================================================================
FLOQUET UNITARY SUPEROPERATORS
===============================================================================

Build superoperators representing Floquet unitary time evolution of operators.

MATHEMATICAL BACKGROUND:
For a Floquet unitary ``U = e^{igP}`` where ``P`` is a Pauli string with ``P^2 = I``:
```math
    \\mathcal{U}[O] = U O U^\\dagger = e^{igP} O e^{-igP}
```

Using ``e^{igP} = \\cos(g)I + i\\sin(g)P`` and expanding:
```math
    e^{igP} O e^{-igP} = \\left(\\cos(g)I + i\\sin(g)P\\right) O \\left(\\cos(g)I - i\\sin(g)P\\right)
```
```math
    = \\cos^2(g) O + i\\sin(g)\\cos(g)(PO - OP) + \\sin^2(g) POP
```
```math
    = \\cos^2(g) O + i\\sin(g)\\cos(g)(P_L - P_R)[O] + \\sin^2(g) (P_L P_R)[O]
```

where ``P_L[O] = P \\cdot O`` (left multiplication) and ``P_R[O] = O \\cdot P`` (right multiplication).

The full Floquet operator for sequential gates is:
```math
    \\mathcal{U}_{\\text{total}} = \\mathcal{U}_n \\circ \\mathcal{U}_{n-1} \\circ \\cdots \\circ \\mathcal{U}_1
```

IMPLEMENTATION STRATEGY:
1. Build local gate tensors (few sites) as matrices, then convert to ITensors
2. Apply gate tensors directly to MPO via ITensor contraction
3. Compose sequentially to build full Floquet superoperator
=#

"""
    build_gate_tensor(sites, pauli_string::String, g::Real, start_site::Int)

Build ITensor for conjugation by Pauli gate: ``e^{igP} O e^{-igP}``.

Constructs the superoperator representing unitary conjugation by a local Pauli gate.
The gate acts on consecutive sites starting from `start_site`.

# Arguments
- `sites`: Pauli site indices (dimension 4)
- `pauli_string`: Pauli operators, e.g., "X" (1-site), "XX" (2-site), "YZ" (2-site)
- `g`: Rotation angle in the exponential ``e^{igP}``
- `start_site`: First site where the gate acts

# Returns
ITensor representing the gate action with indices ``(s_i', s_{i+1}', \\ldots, s_i, s_{i+1}, \\ldots)``

# Mathematical Details
The returned tensor implements:
```math
    \\mathcal{G}[O] = \\cos^2(g) O + i\\sin(g)\\cos(g)(P_L - P_R)[O] + \\sin^2(g) P_L P_R[O]
```
where ``P`` is the Pauli string specified by `pauli_string`.

# Example
```julia
sites = siteinds("Pauli", 10)
gate = build_gate_tensor(sites, "XX", 0.5, 3)  # e^{i·0.5·X₃X₄} conjugation
```
"""
function build_gate_tensor(sites, pauli_string::String, g::Real, start_site::Int)
    gate_len = length(pauli_string)

    # Helper to get L/R matrices for a single Pauli character
    function get_pauli_LR(char::Char)
        if char == 'I'
            return (diagm(ones(ComplexF64, 4)), diagm(ones(ComplexF64, 4)))
        elseif char == 'X'
            return (mat_XL, mat_XR)
        elseif char == 'Y'
            return (mat_YL, mat_YR)
        elseif char == 'Z'
            return (mat_ZL, mat_ZR)
        else
            error("Invalid Pauli character: $char. Must be 'I', 'X', 'Y', or 'Z'.")
        end
    end

    # Build left and right multiplication matrices via tensor products
    mat_PL, mat_PR = get_pauli_LR(pauli_string[1])

    for i in 2:gate_len
        mat_L_i, mat_R_i = get_pauli_LR(pauli_string[i])
        mat_PL = kron(mat_PL, mat_L_i)  # Tensor product for multi-site
        mat_PR = kron(mat_PR, mat_R_i)
    end

    # Compute gate matrix: cos²(g)I + i·sin(g)cos(g)(P_L - P_R) + sin²(g)·P_L·P_R
    c = cos(g)
    s = sin(g)
    c2 = c^2
    s2 = s^2
    sc = s * c

    dim = 4^gate_len
    gate_matrix = c2 * Matrix{ComplexF64}(I, dim, dim) +
                  im * sc * (mat_PL - mat_PR) +
                  s2 * (mat_PL * mat_PR)

    # Get site indices for the gate
    gate_sites = [sites[i] for i in start_site:(start_site + gate_len - 1)]
    gate_sites_out = [sites[i]' for i in start_site:(start_site + gate_len - 1)]

    # Convert matrix to ITensor
    # Indices ordered as (output indices..., input indices...)
    gate_tensor = itensor(gate_matrix, gate_sites_out..., gate_sites...)

    return gate_tensor
end


"""
    apply_gate_to_mpo(mpo::MPO, gate_tensor::ITensor, start_site::Int, gate_len::Int; cutoff=1E-14)

Apply a local gate tensor to an MPO by contracting, applying gate, and SVD decomposition.

# Arguments
- `mpo`: The MPO to apply the gate to (will be modified in place)
- `gate_tensor`: Gate tensor with indices (s'_i, s'_{i+1}, ..., s_i, s_{i+1}, ...)
- `start_site`: First site where gate acts
- `gate_len`: Number of sites the gate acts on
- `cutoff`: SVD cutoff for decomposition

# Algorithm
1. Contract the MPO tensors at sites [start_site, start_site+gate_len-1]
2. Apply the gate tensor
3. Use SVD to decompose back into individual MPO tensors
"""
function apply_gate_to_mpo(mpo::MPO, gate_tensor::ITensor, start_site::Int, gate_len::Int; cutoff=1E-14)
    # Contract all MPO tensors in the gate region
    contracted_mpo = mpo[start_site]
    for i in (start_site+1):(start_site+gate_len-1)
        contracted_mpo = contracted_mpo * mpo[i]
    end

    # Apply gate (contract gate with the MPO tensor)
    # The gate tensor has indices (s'_i, s'_{i+1}, ..., s_i, s_{i+1}, ...)
    # contracted_mpo has indices (link_left, s_i, s_{i+1}, ..., s'_i, s'_{i+1}, ..., link_right)
    result = replaceprime(gate_tensor' * contracted_mpo, 2=>1)

    # Now decompose result back into individual MPO tensors via SVD
    # For a 1-site gate, no decomposition needed
    if gate_len == 1
        mpo[start_site] = result
    else
        # For multi-site gates, use SVD to split the tensor
        # Work from left to right, performing SVD at each bond
        current_tensor = result
        current_link = linkind(mpo, start_site - 1)  # Left link of the first site

        for i in 0:(gate_len-2)
            site_idx = start_site + i

            # Get the site indices for this position
            s = siteind(mpo, site_idx)
            sp = s'

            # println("Current tensor:", inds(current_tensor))
            links = [l for l in [current_link, s, sp] if l != nothing]

            # Perform SVD
            U, S, V = svd(current_tensor, links...; cutoff=cutoff)

            # Absorb S into V for next iteration
            current_tensor = S * V

            # Store U as the MPO tensor at this site
            mpo[site_idx] = U
        end

        # The remaining tensor goes to the last site
        mpo[start_site + gate_len - 1] = current_tensor
    end

    return mpo
end


"""
    build_floquet_superoperator(L::Int, sites, gates; return_dag=false)

Build MPO representing sequential Floquet unitary gates acting on operators.

Constructs the superoperator for a Floquet time evolution consisting of multiple
local unitary gates. Each gate conjugates operators: ``O \\mapsto U_k O U_k^\\dagger``.

# Arguments
- `L`: System size (number of sites)
- `sites`: Pauli site indices (should be of type "Pauli" with dimension 4)
- `gates`: **Ordered** layer list, ``[\\text{Pauli string} \\Rightarrow \\text{angle}, ...]``
  - Keys: Pauli strings like "X" (1-site), "XX" (2-site), "YZ" (2-site)
  - Values: Rotation angles ``g`` in ``e^{igP}``
  - Layers are applied in the order given, innermost conjugation first. An
    `AbstractDict` is still accepted for backwards compatibility but iterates in
    hash order, which for non-commuting layers silently picks an arbitrary
    unitary — see `_normalize_gate_spec`.
- `return_dag`: If true, also return ``\\mathcal{U}^\\dagger`` (default: false).
  Note the returned dagger carries link indices at prime level 1; for the
  factored DMRG environments use `adjoint_mpo` instead.

PBC is not supported: gates are applied only at positions `1:(L - gate_len + 1)`,
so the chain is open.

# Returns
- If `return_dag=false`: `U_mpo` - MPO representing the Floquet superoperator
- If `return_dag=true`: `(U_mpo, U_dag_mpo)` - Tuple of forward and inverse operators

# Mathematical Details
For gates ``\\{(P_k, g_k)\\}`` applied sequentially, the output is:
```math
    \\mathcal{U}_{\\text{total}} = \\prod_k \\prod_{\\text{positions}} e^{ig_k P_k} (\\cdot) e^{-ig_k P_k}
```

The dagger (inverse) is obtained by reversing gate order and negating angles:
```math
    \\mathcal{U}^\\dagger = \\prod_k^{\\text{reversed}} \\prod_{\\text{positions}} e^{-ig_k P_k} (\\cdot) e^{ig_k P_k}
```

# Example
```julia
sites = siteinds("Pauli", 10)
gates = ["ZZ" => 0.1, "X" => 0.5]     # ordered: ZZ layer applied first
U = build_floquet_superoperator(10, sites, gates)

# Apply to an operator MPS
O_initial = productMPS(sites, "X")
O_evolved = apply(U, O_initial; cutoff=1E-12)
```

# Example with Dagger
```julia
U, U_dag = build_floquet_superoperator(10, sites, gates; return_dag=true)
# Verify: U_dag * U ≈ Identity
```
"""
function build_floquet_superoperator(
    L::Int,
    sites,
    gates;
    return_dag::Bool=false
)
    # Ordered layer list; a Dict spec is accepted but warns (see _normalize_gate_spec)
    gate_seq = _normalize_gate_spec(gates)

    # Start with identity MPO
    result_mpo = MPO(sites, "I")

    # Apply gates sequentially, in the given order
    for (pauli_string, angle) in gate_seq

        gate_len = length(pauli_string)

        num_positions = L - gate_len + 1

        # Apply gate at each position
        for pos in 1:num_positions
            start_site = pos

            # Build gate tensor for this position
            gate_tensor = build_gate_tensor(sites, pauli_string, angle, start_site)

            # Apply gate directly to MPO using tensor contraction and SVD
            apply_gate_to_mpo(result_mpo, gate_tensor, start_site, gate_len)
        end
    end

    # Build dagger if requested
    if return_dag
        dagmpo = copy(result_mpo)
        for i in 1:length(dagmpo)
            dagmpo[i] = replaceprime(dag(dagmpo[i])', 2=>0)
        end
        return (result_mpo, dagmpo)
    else
        return result_mpo
    end
end


#=
===============================================================================
OPERATOR BUILDER FUNCTIONS
===============================================================================
=#

"""
    build_hamiltonian_operators(L::Int, sites, H_dict::Dict; pbc::Bool=false)

Build operators for Hamiltonian-based DMRG (non-Floquet mode).

# Arguments
- `L`: System size
- `sites`: Pauli site indices
- `H_dict`: Hamiltonian dictionary (e.g., Dict("ZZ" => -1.0, "X" => -0.5))
- `pbc`: Periodic boundary conditions

# Returns
- `oper_single`: Commutator operator C_H (for diagnostics: ‖C_H·ψ‖ = tauinv)
- `oper_for_dmrg`: The operator DMRG squares internally to obtain ⟨ψ|C_H²|ψ⟩.
  Returned as C_H itself, NOT `apply(C_H, C_H)`. See NOTE below.
- `oper_left`: Left multiplication operator H_L (for Krylov basis)
- `label`: String label for this operator type ("Hamiltonian")

# NOTE (2026-07-22): the DMRG driver `dmrg_sweeps!` squares its
second argument internally — the LH/RH environments sandwich two un-squared C_H
layers (`CH[i] * prime(CH[i])`). Returning `apply(C_H, C_H)` here therefore
produced `C_H⁴` inside DMRG, not the physical `C_H²`. This function
now returns C_H so the driver's internal squaring yields the correct C_H². The
second tuple slot is still called `oper_for_dmrg` at call sites for API
compatibility with `build_floquet_operators`; that historical name is a
misnomer for the Hamiltonian path.

Because `C_H = H_L − H_R` is **Hermitian** in the operator inner product, the
environments' `CH[i] * prime(CH[i])` — a matrix product of the MPO with itself —
already equals `C_H† C_H`, so this path needs no separate top layer and does not
pass `CHtop` to `dmrg_sweeps!`. Contrast `build_floquet_operators`, whose
`C_U = 𝒰 − 𝟙` is not Hermitian and therefore returns an explicit adjoint for the
top layer. This function deliberately keeps its 4-tuple return shape; ~20 call
sites across `scripts/` and `test/` destructure it positionally.
"""
function build_hamiltonian_operators(L::Int, sites, H_dict::AbstractDict; pbc::Bool=false)
    logwrite("Building Hamiltonian operators...")

    # Build H_L and H_R
    H_L, H_R = build_super_operators(L, sites, H_dict; pbc=pbc)
    H_L = movedevice(H_L)
    H_R = movedevice(H_R)

    # Commutator C_H = H_L - H_R (Hermitian in the operator-space inner product)
    C_H = add(H_L, -H_R)
    logwrite("   Max bond dimension of C_H: $(maxlinkdim(C_H))")

    # Return C_H (single, un-squared). Driver squares internally.
    return C_H, C_H, H_L, "Hamiltonian"
end


"""
    build_floquet_operators(L::Int, sites, gates; pbc::Bool=false)

Build operators for Floquet-based DMRG, i.e. for minimizing
``\\|U^\\dagger O U - O\\|^2`` instead of ``\\|[H, O]\\|^2``.

Defines the Floquet superoperator ``C_U = \\mathcal{U} - \\mathbb{1}`` where
``\\mathcal{U}[O] = U O U^\\dagger`` is the conjugation MPO built by
`build_floquet_superoperator`. Note the objective is insensitive to the
conjugation direction: conjugation is an isometry of the Hilbert–Schmidt norm, so
``\\|U^\\dagger O U - O\\| = \\|U O U^\\dagger - O\\|``. (Direction would matter
only for a λ-resonance generalization ``C_U = \\mathcal{U} - e^{i\\lambda}
\\mathbb{1}``, which is not implemented.)

# Arguments
- `L`: System size
- `sites`: Pauli site indices
- `gates`: **Ordered** Floquet layer list, e.g. for kicked Ising
  `["ZZ" => -J, "Z" => -h, "X" => -g]`. See `_normalize_gate_spec` — an
  `AbstractDict` is accepted but its hash iteration order makes the unitary
  arbitrary when layers do not commute.
- `pbc`: Periodic boundary conditions (currently not supported for Floquet)

# Returns
- `oper_single`: ``C_U``, used for the `tauinv = ‖C_U·ψ‖` diagnostic. That
  diagnostic is already adjoint-correct (it computes `sqrt(inner(Oψ, Oψ))`), so
  no separate operator is needed there.
- `oper_bottom`: ``C_U`` again — the *bottom* layer of the factored two-layer
  environments in `dmrg_sweeps!`.
- `oper_top`: ``C_U^\\dagger`` via `adjoint_mpo` — the *top* layer. Passing this
  is what makes the environments compute ``\\langle\\psi|C_U^\\dagger
  C_U|\\psi\\rangle = \\|C_U\\psi\\|^2``.
- `oper_for_init`: the conjugation operator ``\\mathcal{U}`` itself. Note this
  slot holds `H_L` in `build_hamiltonian_operators`, where it seeds the Krylov
  basis instead — the two paths use it for different purposes.
- `label`: String label for this operator type ("Floquet")

# NOTE (2026-08-05): supersedes the earlier broken construction, which returned
the pre-squared `2I − U − U†` as the driver's second argument and so actually
minimized `⟨ψ|(2I−U−U†)²|ψ⟩`. The driver contracts two un-squared layers, and
`C_U² ≠ C_U†C_U` for the non-Hermitian `C_U`, so the fix is to hand it the
adjoint as the top layer rather than a second copy of the bottom one. The
pre-squared MPO is no longer built at all.
"""
function build_floquet_operators(L::Int, sites, gates; pbc::Bool=false)
    logwrite("Building Floquet operators...")

    if pbc
        @warn "PBC not yet supported for Floquet operators, proceeding with OBC"
    end

    # Conjugation superoperator 𝒰
    U = movedevice(build_floquet_superoperator(L, sites, gates))
    logwrite("   Max bond dimension of U: $(maxlinkdim(U))")

    # C_U = 𝒰 - 𝟙
    I_mpo = movedevice(MPO(sites, "I"))
    C_U = add(U, -1.0 * I_mpo)
    logwrite("   Max bond dimension of C_U = (U - I): $(maxlinkdim(C_U))")
    truncate!(C_U; cutoff=0)
    logwrite("   Max bond dimension of C_U after truncation: $(maxlinkdim(C_U))")

    # C_U† for the top environment layer. Links stay at prime level 0 so this is
    # a drop-in replacement for C_U in the environment contractions.
    C_U_dag = adjoint_mpo(C_U)

    return C_U, C_U, C_U_dag, U, "Floquet"
end
