#=
===============================================================================
KRYLOV BASIS CONSTRUCTION
===============================================================================

The Krylov space ``\\mathrm{span}\\{I, H, H^2, \\ldots, H^n\\}`` contains trivial solutions to the
commutator problem. We must orthogonalize against this subspace to find
non-trivial conserved quantities.

We represent these operators as MPS in the Pauli basis and generate them
by repeated application of ``H_L`` (left multiplication by ``H``).
=#

"""
    build_krylov_basis_raw(H_L::MPO, sites, n::Int; cutoff=1E-12)

Constructs the raw (unnormalized) Krylov basis ``\\{I, H, H^2, \\ldots, H^n\\}`` as MPS.

# Arguments
- `H_L`: Left multiplication super-operator MPO
- `sites`: Pauli site indices
- `n`: Maximum power of ``H`` to include
- `cutoff`: Truncation cutoff for MPS compression

# Returns
- Vector of ``(n+1)`` MPS representing ``I, H, H^2, \\ldots, H^n``

# Algorithm
1. Start with ``|I\\rangle`` (product state with all sites in state 1)
2. For ``k = 1`` to ``n``:
   - Apply ``H_L`` to ``|H^{k-1}\\rangle`` to get ``|H^k\\rangle``
   - Normalize to prevent exponential growth
   - Store the result

Note: These vectors are NOT orthogonalized — the DMRG driver handles that
internally, per two-site block, by modified Gram-Schmidt on the projected
tower tensors (see `dmrg_sweeps!`).
"""
function build_krylov_basis_raw(H_L::MPO, sites, n::Int; cutoff=1E-12)
    logwrite("Constructing Krylov tower {I, H, ..., H^$n} ...")

    # Start with identity operator: |I⟩ = |1,1,1,...,1⟩
    v0_mps = movedevice(productMPS(sites, "I"))
    krylov_vectors_raw = [v0_mps]

    logwrite("  Generating raw Krylov vectors |H^k>...")
    for k in 1:n
        # Apply H_L to get next power: |Hᵏ⟩ = H_L|Hᵏ⁻¹⟩
        vk_mps = noprime(contract(H_L, krylov_vectors_raw[end]; cutoff=1E-12, method="naive"))
        logwrite("    Generated |H^$k> with max bond dim: $(maxlinkdim(vk_mps))")
        sanity_check(vk_mps)

        # Normalize to prevent exponential growth
        push!(krylov_vectors_raw, normalize(vk_mps))
        logwrite("    After normalization, max bond dim: $(maxlinkdim(krylov_vectors_raw[end]))")
        sanity_check(krylov_vectors_raw[end])
    end

    return krylov_vectors_raw
end

"""
    build_operator_mps(L::Int, sites, H_dict::Dict; pbc::Bool=false, k::DTYPE=0.0)

Constructs an MPS representing a momentum eigenstate of the Hamiltonian operator.

For a translationally invariant Hamiltonian, we can construct operator eigenstates
with definite momentum ``k``:
```math
    H_k = \\sum_x e^{-ikx} T^x H
```
where ``T^x`` is translation by ``x`` sites.

# Arguments
- `L`: System size
- `sites`: Pauli site indices
- `H_dict`: Hamiltonian terms
- `pbc`: Periodic boundary conditions
- `k`: Momentum (typically ``k = 2\\pi/L`` for the first excited mode)

# Returns
- MPS representing the operator ``H_k`` in the Pauli basis

This is used as an initial guess for DMRG (`init_index=2`).
"""
function build_operator_mps(L::Int, sites, H_dict::AbstractDict; pbc::Bool=false, k::DTYPE=0.0)
    state = nothing
    bare_string = fill("I", L)

    # Helper function to add a Pauli string with coefficient
    function add_state!(coef, op_sites)
        this_string = copy(bare_string)
        for (op, site) in op_sites
            this_string[site] = string(op)
        end
        logwrite("Adding state with coef=$coef and string=$(join(this_string))")

        if isnothing(state)
            state = coef * productMPS(sites, this_string)
        else
            state = add(state, coef * productMPS(sites, this_string))
        end
    end

    # Build the momentum superposition
    for (op_string, coeff) in H_dict
        opsl = length(op_string)

        # Add all translations with phase factors exp(-ikx)
        for i = 1:(L-opsl+1)
            add_state!(coeff*exp(-1im*k*(i-1)), zip(collect(op_string), i:i+opsl-1))
        end

        # Handle periodic boundary wrapping
        if pbc && opsl > 1
            for i = 1:opsl-1
                add_state!(coeff*exp(1im*k*i),
                    vcat(
                        collect(zip(collect(op_string[end-i+1:end]), 1:i)),
                        collect(zip(collect(op_string[1:end-i]), L-(opsl-i-1):L))
                    )
                )
            end
        end
    end

    return movedevice(state)
end
