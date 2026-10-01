#=
===============================================================================
EXACT DIAGONALIZATION (FOR SMALL SYSTEMS)
===============================================================================
=#

# Maximum system size for exact diagonalization. `run_dmrg_for_size` routes
# L ≤ L_MAX_EXACT here instead of sweeping, which makes this the small-L
# reference ("oracle") the tests compare the sweeper against.
const L_MAX_EXACT = 6

"""
    exact_ground_state(C_H::MPO, M::MPO, thetasq::DTYPE;
                      C_H_top=nothing, orthos::Vector{MPS}=[], ortho_weight=0.0,
                      psi0_in=nothing, krylov_kwargs=Dict(...))

Finds the exact ground state of the super-Hamiltonian for small systems (``L \\leq 6``).

# Arguments
- `C_H`: Slowness super-operator MPO, applied as the *first* (inner) layer.
  ``[H, \\cdot]`` on the Hamiltonian path, ``C_U = \\mathcal{U} - \\mathbb{1}`` on
  the Floquet path.
- `M`: Kinetic term MPO (diagonal in Pauli basis)
- `thetasq`: ``\\theta^2`` (squared penalty parameter)
- `C_H_top`: MPO applied as the *second* (outer) layer, i.e. the adjoint of
  `C_H`. `nothing` reuses `C_H`, which is correct exactly when `C_H` is Hermitian
  as an operator — true for ``C_H = H_L - H_R``, false for the Floquet
  ``C_U``. Mirrors the `CHtop` keyword of `dmrg_sweeps!`; pass
  `adjoint_mpo(C_U)` on the Floquet path so this computes
  ``C_U^\\dagger C_U`` rather than ``C_U^2``.
- `orthos`: Vector of MPS to orthogonalize against (Krylov basis)
- `ortho_weight`: Weight for orthogonalization penalty
- `psi0_in`: Initial guess (MPS or vector)
- `krylov_kwargs`: Parameters for KrylovKit eigensolver

# Returns
- `energy`: Ground state energy ``E[O] = \\theta^2 \\|[H,O]\\|^2 - M[O]``
- `ground_state_mps`: MPS representing the ground state operator
- `converged`: Convergence flag (0 or 1)

# Algorithm
Uses `KrylovKit.eigsolve` with a custom Hamiltonian action:
```math
    H_{\\text{eff}} = P(\\theta^2 C_H^2 - M)P + \\text{ortho\\_weight} \\cdot (1-P)
```
where ``P`` projects out the orthogonalization subspace. The action is computed
by converting MPS ↔ vector, applying MPOs, and projecting.
"""
function exact_ground_state(C_H::MPO, M::MPO, thetasq::DTYPE;
                           C_H_top::Union{MPO,Nothing}=nothing,
                           orthos::Vector{MPS}=[], ortho_weight=0.0,
                           psi0_in=nothing,
                           krylov_kwargs=Dict(:krylovdim=>100, :tol=>eps(DTYPE)^(2/3), :maxiter=>1000))

    # Outer layer of the C†C sandwich. `C_top === C_H` unless an explicit adjoint
    # is supplied, so the Hamiltonian path is unchanged.
    C_top = C_H_top === nothing ? C_H : C_H_top

    # Get site indices (only physical indices, not primed)
    sites = [s for s in vcat(siteinds(C_H)...) if plev(s)==0]
    L = length(sites)

    if L > L_MAX_EXACT
        throw(ArgumentError("Exact diagonalization is only feasible for L <= $L_MAX_EXACT."))
    end

    # Combiner to convert between MPS and vector representations
    comb = movedevice(combiner(sites...))
    newind = inds(comb)[1]

    # Prepare orthogonalization vectors by QR decomposition
    ortho_arrays = []
    if length(orthos) > 0
        ortho_matrix_cols = []
        for psi in orthos
            psi_tensor = reduce(*, psi)
            psi_vec = array(psi_tensor*comb, newind)
            push!(ortho_matrix_cols, psi_vec)
        end

        # QR to ensure orthonormality
        ortho_matrix = hcat(ortho_matrix_cols...)
        F = qr(ortho_matrix)
        Q = Matrix(F.Q)

        for i in 1:size(Q, 2)
            push!(ortho_arrays, movedevice(Q[:, i]))
        end
    end

    # Project a vector orthogonal to the Krylov subspace
    function orthogonalize_against_orthos(v)
        v_ortho = copy(v)
        for ortho_vec in ortho_arrays
            overlap = dot(ortho_vec, v_ortho)
            v_ortho -= overlap * ortho_vec
        end
        return v_ortho
    end

    # Define the effective Hamiltonian action: H_eff|v⟩ = P(theta²C_H² - M)P|v⟩ + penalty
    function hamiltonian_action(v)
        # Orthogonalize input: v_ortho = P|v⟩
        v_ortho = orthogonalize_against_orthos(v)

        # Convert to ITensor
        v_tensor = ITensor(v_ortho, newind) * comb

        # Apply C_H then its adjoint to get C_H†C_H|v⟩. `C_top === C_H` on the
        # Hamiltonian path (C_H Hermitian), so this is the same C_H² as before;
        # on the Floquet path C_top = C_U† and this is the physical ‖C_U v‖².
        Hv_tensor = copy(v_tensor)
        for Hblock in C_H
            Hv_tensor = noprime(Hblock*Hv_tensor)
        end
        for Hblock in C_top
            Hv_tensor = noprime(Hblock*Hv_tensor)
        end

        # Apply M to get M|v⟩
        Mv_tensor = copy(v_tensor)
        for Mblock in M
            Mv_tensor = noprime(Mblock*Mv_tensor)
        end

        # Compute full action: (theta²C_H² - M)|v⟩
        Hv_vec = array(Hv_tensor * comb, newind)
        Mv_vec = array(Mv_tensor * comb, newind)
        Hv_vec = thetasq*Hv_vec - Mv_vec

        # Project result back to orthogonal complement: P·(result)
        Hv_ortho = orthogonalize_against_orthos(Hv_vec)

        # Add penalty term for components in Krylov subspace
        penalty = ortho_weight * (v - v_ortho)

        return Hv_ortho + penalty
    end

    # Prepare initial vector
    if isnothing(psi0_in)
        # Random initial vector orthogonal to Krylov space
        dim_total = prod([dim(s) for s in sites])
        v0 = movedevice(randn(CDTYPE, dim_total))
        v0 = orthogonalize_against_orthos(v0)
        v0 = v0 / norm(v0)
        v0 = movedevice(v0)
    else
        if isa(psi0_in, MPS)
            psi0_device = movedevice(psi0_in)
            psi0_tensor = reduce(*, psi0_device)
            v0 = array(psi0_tensor*comb, newind)
            v0 = v0 / norm(v0)
        elseif isa(psi0_in, AbstractArray)
            v0 = movedevice(psi0_in)
            v0 = v0 / norm(v0)
        else
            error("Unsupported type for psi0_in: $(typeof(psi0_in))")
        end
    end

    # Find ground state using KrylovKit
    num_solve = 3

    # Try eigsolve with increasing krylovdim if LAPACK errors occur
    local evals, evecs, info
    converged = -1

    try
        evals, evecs, info = eigsolve(hamiltonian_action, v0, num_solve, :SR;
                                      ishermitian=true, krylov_kwargs...)
        converged = min(1, info.converged)
    catch e
        if isa(e, LinearAlgebra.LAPACKException)
            @warn "LAPACK error in exact diagonalization: $e. Trying with larger krylovdim..."
            # Retry with doubled krylovdim
            krylov_kwargs_retry = merge(krylov_kwargs, Dict(:krylovdim => 2*get(krylov_kwargs, :krylovdim, 100)))
            try
                evals, evecs, info = eigsolve(hamiltonian_action, v0, num_solve, :SR;
                                              ishermitian=true, krylov_kwargs_retry...)
                converged = min(1, info.converged)
                @warn "Retry with krylovdim=$(krylov_kwargs_retry[:krylovdim]) succeeded!"
            catch e2
                @warn "Exact diagonalization failed after retry: $e2. Returning failure flag (converged=-1)."
                # Return failure flag so caller can fall back to iterative DMRG
                return 0.0, nothing, -1
            end
        else
            rethrow(e)
        end
    end

    # Check if we have valid results
    if converged < 0 || length(evals) == 0 || length(evecs) == 0
        @warn "Exact diagonalization failed to produce valid results (converged=$converged). Returning failure flag."
        return 0.0, nothing, -1
    end

    energy = real(evals[1])
    ground_state_vec = evecs[1]

    # Convert back to MPS
    ground_state_tensor = ITensor(ground_state_vec, newind) * comb
    ground_state_mps = MPS(ground_state_tensor, sites)

    return energy, ground_state_mps, converged
end
