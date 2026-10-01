#=
===============================================================================
MAIN DMRG ROUTINE
===============================================================================
=#

"""
    _run_dmrg_core(L::Int, sites, oper_single::MPO, oper_for_dmrg::MPO,
                   M_mpo::MPO, metadata::Dict, psi0::MPS, ortho_states::Vector{MPS},
                   theta::DTYPE, ortho_weight::DTYPE, initial_noise::DTYPE, final_noise::DTYPE,
                   noise_decay_factor::DTYPE, sweeps_per_iteration::Int, max_iter::Int,
                   energy_conv_threshold::DTYPE, operator_label::String,
                   obs_conv_threshold::DTYPE=0.01, obs_conv_consecutive::Int=1)

Core DMRG routine for both Hamiltonian and Floquet modes.

# Arguments
- `L`: System size
- `sites`: Pauli site indices
- `oper_single`: Single operator (C_H for Hamiltonian, C_U = 𝒰−𝟙 for Floquet) —
  for the `tauinv = ‖oper_single·ψ‖` diagnostic, which is adjoint-correct as
  written and so needs no companion operator
- `oper_for_dmrg`: Historical name; actually holds the un-squared operator that
  the DMRG driver sandwiches into `C†C` via the two `CH[i]·prime(CHt[i])` layers
  in the LH/RH environment updates in `dmrg_sweeps!` — i.e. the **bottom** layer.
  For the Hamiltonian path this is C_H, so the minimized quantity is
  θ²·⟨ψ|C_H²|ψ⟩ − ⟨ψ|M|ψ⟩. For the Floquet path it is C_U.
- `M_mpo`: Kinetic term operator
- `metadata`: Metadata dictionary for saving
- `psi0`: Initial MPS
- `ortho_states`: States to orthogonalize against (e.g., Krylov basis)
- Other parameters: DMRG configuration options
- `oper_top`: **Top** layer of the `C†C` sandwich, i.e. the adjoint of
  `oper_for_dmrg`. `nothing` reuses `oper_for_dmrg`, correct whenever that
  operator is Hermitian (the Hamiltonian path). The Floquet path must pass
  `adjoint_mpo(C_U)`, since `C_U² ≠ C_U†C_U`. Forwarded as `CHtop` to
  `dmrg_sweeps!` and as `C_H_top` to `exact_ground_state`.
- `operator_label`: String label ("Hamiltonian" or "Floquet")
- `obs_conv_threshold`: Relative change threshold for observable convergence (default: 0.01 = 1%)
- `obs_conv_consecutive`: Number of consecutive iterations with small changes to declare convergence (default: 1)

# Returns
- `psi`: Final MPS representing the conserved quantity
"""
function _run_dmrg_core(L::Int, sites, oper_single::MPO, oper_for_dmrg::MPO,
                        M_mpo::MPO, metadata::Dict, psi0_in::MPS, ortho_states::Vector{MPS},
                        theta::DTYPE, ortho_weight::DTYPE, initial_noise::DTYPE, final_noise::DTYPE,
                        noise_decay_factor::DTYPE, sweeps_per_iteration::Int, max_iter::Int,
                        energy_conv_threshold::DTYPE, operator_label::String,
                        obs_conv_threshold::DTYPE=0.01, obs_conv_consecutive::Int=1,
                        debuglogger::Union{DebugLogger,Nothing}=nothing;
                        oper_top::Union{MPO,Nothing}=nothing,
                        maxdim_cap::Int=128,
                        thetasq_out::Union{Ref{DTYPE},Nothing}=nothing,
                        scheduling::Symbol=:fixed_theta,
                        local_solver::Symbol=:eigsolve,
                        epsilon_loc::DTYPE=DTYPE(1e-4),
                        eps_anneal_sweeps::Int=1,
                        eps_anneal_start::DTYPE=DTYPE(0),
                        qcqp_theta_sq_max::DTYPE=DTYPE(1e8),
                        qcqp_bisect_tol::Real=1e-3,
                        qcqp_warm_accept_tol::Real=5e-2,
                        qcqp_warm_accept_tol_min::Real=5e-3,
                        qcqp_warm_accept_tol_trigger::Real=3.0,
                        qcqp_use_secant::Bool=true,
                        qcqp_slope_init::Real=1.0,
                        qcqp_thetasq_ema::Union{Ref{DTYPE},Nothing}=nothing,
                        qcqp_ema_alpha::Real=0.1,
                        save_mps::Bool=true,
                        qcqp_lobpcg_cheby_degree::Int=0,   # 0 = adaptive in κ(B); >0 pins it
                        qcqp_lobpcg_cheby_min::Int=2,
                        qcqp_lobpcg_cheby_max::Int=8,
                        qcqp_lobpcg_cheby_eta::Real=0.1,
                        qcqp_lobpcg_rel_tol::Real=1e-6,
                        qcqp_lobpcg_inner_maxiter::Int=12,
                        qcqp_lobpcg_max_restarts::Int=8,
                        qcqp_lobpcg_maxiter::Int=100,
                        qcqp_lobpcg_power_iters::Int=10,
                        krylovdim::Int=25,
                        qcqp_verbose::Bool=false,
                        save_callback::Union{Function,Nothing}=nothing)

    # psi0 = reference initial state (normalized, never modified)
    psi0 = copy(psi0_in)
    psi0 /= norm(psi0)

    # psi = working state that gets optimized
    psi = copy(psi0_in)

    need_dmrg = (L > L_MAX_EXACT)  # Flag to determine if DMRG is needed

    #--------------------------------------------------------------------------
    # EXACT DIAGONALIZATION (L ≤ 6)
    #--------------------------------------------------------------------------

    if L <= L_MAX_EXACT
        logwrite("\nL is small enough for exact diagonalization. Attempting exact solution...")
        energy, psi_result, converged = exact_ground_state(oper_for_dmrg, M_mpo, theta^2;
                                                            C_H_top=oper_top,
                                                            psi0_in=psi,
                                                            orthos=ortho_states,
                                                            ortho_weight=ortho_weight*theta^2)
        logwrite("Exact diagonalization convergence status: $(converged)")

        # Check if exact diagonalization converged
        if converged >= 1 && !isnothing(psi_result)
            psi = psi_result
            logwrite("Exact ground state energy: $energy")
            logwrite("Exact ground state MPS has bond dimension: $(linkdims(psi))")
            logwrite("Exact diagonalization converged successfully!")

            # Compute diagnostics
            naive_cons = Dict(:alg=>"naive", :truncate=>false)

            Npsi = norm(psi)
            # Compute ||O·ψ|| robustly to avoid sqrt of small negative numbers from roundoff
            tauinv_mps = apply(oper_single, psi; naive_cons...)
            tauinv = sqrt(abs(inner(tauinv_mps, tauinv_mps)))  # ||O·ψ|| = sqrt(⟨O·ψ|O·ψ⟩)
            nu = real(inner(psi, apply(M_mpo, psi; naive_cons...)))  # ⟨ψ|M|ψ⟩
            orthoovlps = [abs(inner(psi, u)) for u in ortho_states]
            opsizes = get_size_distribution(psi)
            k_weights = calculate_k_space_decomposition(psi)[1]
            psi0ovlp = abs(inner(psi / norm(psi), psi0))

            logwrite("Norm psi: $Npsi")
            logwrite("tauinv ($operator_label): $tauinv, tau = $(1/tauinv)")
            logwrite("nu (M expectation): $nu")
            logwrite("Overlap with initial state psi0: $psi0ovlp")
            for i in eachindex(ortho_states)
                logwrite("Ortho state $i overlap: $(orthoovlps[i])")
            end
            logwrite("Size distribution of psi:")
            logwrite(opsizes)
            logwrite("K-space weights of psi:")
            logwrite(k_weights)
        else
            logwrite("\n!!! WARNING: Exact diagonalization did not converge (converged = $converged)")
            logwrite("!!! Will fall back to iterative DMRG using the unconverged result as initial guess...")
            need_dmrg = true
        end
    end

    #--------------------------------------------------------------------------
    # ITERATIVE DMRG (L > 6 OR EXACT DIAG FAILED)
    #--------------------------------------------------------------------------

    if need_dmrg
        logwrite("\n4. Starting DMRG calculation with built-in orthogonalization...")

        psi = movedevice(psi)

        psi /= norm(psi)
        psi = complex(psi)  # Ensure complex type for generality

        energy = 0.0
        last_energy = Inf
        current_noise = initial_noise

        # Observable convergence tracking
        obs_history = Vector{NamedTuple{(:tauinv, :nu, :k_weights), Tuple{DTYPE, DTYPE, Vector{Float64}}}}()
        obs_converged_count = 0

        # Adaptive noise DMRG loop
        for iter in 1:max_iter
            logwrite("\n--- Iteration $iter / $max_iter (L=$L) ---")
            logwrite("Energy from previous iteration: $last_energy")
            logwrite("Current noise level: $current_noise / final_noise: $final_noise")

            # Adaptive minimum bond dimension based on noise level
            # As noise decreases, allow larger bond dimensions
            mindim = max(2^5, 2^(8-floor(Int, log10(current_noise/final_noise))))
            logwrite("Setting minimum bond dimension to: $mindim")

            # Bond-dimension ramp — keyed to the warm-start bond dimension
            # `warm_chi = maxlinkdim(psi)`. Philosophy: never spend a sweep at
            # (or below) the incoming χ. A state warm-started already converged
            # at χ=warm_chi gains nothing from re-sweeping at warm_chi, so the
            # ramp GROWS PAST it immediately — sweep 1 at 2·warm_chi, sweep 2 at
            # 4·warm_chi, then the cap:
            #   warm χ=128, cap=512 → 256, 512, 512
            #   warm χ=256, cap=512 → 512 straight away
            #   warm χ=512, cap=512 → 512 (already at the cap)
            # A COLD start (small warm_chi) instead falls back to the standard
            # 32→64→cap warmup, because the `min(32,·)`/`min(64,·)` floors
            # dominate when 2·warm_chi/4·warm_chi are still tiny. `maxdim_cap`
            # is the ceiling; `setmindim!` below pins χ ≥ warm_chi so the
            # warm-start structure is never truncated away (also the LOBPCG
            # CholQR-crash guard — job 12900383).
            warm_chi = maxlinkdim(psi)
            sweeps = Sweeps(sweeps_per_iteration)
            setmaxdim!(sweeps,
                min(maxdim_cap, max(2 * warm_chi, min(32, maxdim_cap), mindim)),
                min(maxdim_cap, max(4 * warm_chi, min(64, maxdim_cap), mindim)),
                max(maxdim_cap, mindim),
            )
            # Force the SVD to keep χ ≥ warm_chi across every sweep. Without
            # this the mindim inside truncate() defaults to 1 and the first
            # sweep after a warm-start can drop bond dim aggressively before
            # the noise re-grows it — which not only wastes compute rebuilding
            # structure the warm-start already had, but produces near-singular
            # search directions that trigger LOBPCG's CholQR PosDefException
            # at the first block of the fresh sweep. Observed at step
            # boundaries of the ε walk (job 12900383 crashed exactly here).
            setmindim!(sweeps, warm_chi)
            setcutoff!(sweeps, 0)  # No truncation cutoff (rely on bond dimension only)
            setnoise!(sweeps, current_noise)

            # Run DMRG with custom effective Hamiltonian. `oper_for_dmrg` is the
            # bottom layer and `oper_top` the top layer of the C†C sandwich the
            # driver's LH/RH environments build, so this minimizes
            #     E[ψ] = θ²·⟨ψ|C†C|ψ⟩ − ⟨ψ|M|ψ⟩
            # subject to ψ ⊥ ortho_states.
            # Hamiltonian path: C = C_H (Hermitian) and oper_top = nothing, so
            #     C†C = C_H²  and  E[ψ] = θ²·‖[H,ψ]‖² − ⟨ψ|M|ψ⟩.
            # Floquet path: C = C_U = 𝒰−𝟙 and oper_top = adjoint_mpo(C_U), so
            #     C†C = C_U†C_U  and  E[ψ] = θ²·‖U†ψU − ψ‖² − ⟨ψ|M|ψ⟩.
            energy, psi, sweep_converged = dmrg_sweeps!(
                psi, oper_for_dmrg, M_mpo, theta^2, sweeps;
                CHtop = oper_top,
                thetasq_out = thetasq_out,
                dispon = 1,
                orthogonalize_states = ortho_states,
                stop_tol = energy_conv_threshold,
                subspace_coeffs = (DTYPE(1/theta^2), DTYPE(ortho_weight)),
                eig_kwargs = Dict(:nev=>1, :krylovdim=>krylovdim),
                scheduling = scheduling,
                local_solver = local_solver,
                epsilon_loc = epsilon_loc,
                eps_anneal_sweeps = eps_anneal_sweeps,
                eps_anneal_start = eps_anneal_start,
                qcqp_theta_sq_max = qcqp_theta_sq_max,
                qcqp_bisect_tol = qcqp_bisect_tol,
                qcqp_warm_accept_tol = qcqp_warm_accept_tol,
                qcqp_warm_accept_tol_min = qcqp_warm_accept_tol_min,
                qcqp_warm_accept_tol_trigger = qcqp_warm_accept_tol_trigger,
                qcqp_use_secant = qcqp_use_secant,
                qcqp_slope_init = qcqp_slope_init,
                qcqp_thetasq_ema = qcqp_thetasq_ema,
                qcqp_ema_alpha = qcqp_ema_alpha,
                qcqp_lobpcg_cheby_degree = qcqp_lobpcg_cheby_degree,
                qcqp_lobpcg_cheby_min = qcqp_lobpcg_cheby_min,
                qcqp_lobpcg_cheby_max = qcqp_lobpcg_cheby_max,
                qcqp_lobpcg_cheby_eta = qcqp_lobpcg_cheby_eta,
                qcqp_lobpcg_rel_tol = qcqp_lobpcg_rel_tol,
                qcqp_lobpcg_inner_maxiter = qcqp_lobpcg_inner_maxiter,
                qcqp_lobpcg_max_restarts = qcqp_lobpcg_max_restarts,
                qcqp_lobpcg_maxiter = qcqp_lobpcg_maxiter,
                qcqp_lobpcg_power_iters = qcqp_lobpcg_power_iters,
                qcqp_verbose = qcqp_verbose,
                debuglogger = debuglogger,
                save_callback = save_callback,
            )

            # Compute diagnostics
            naive_cons = Dict(:alg=>"naive", :truncate=>false)

            Npsi = norm(psi)
            # Compute ||O·ψ|| robustly to avoid sqrt of small negative numbers from roundoff
            tauinv_mps = apply(oper_single, psi; naive_cons...)
            tauinv = sqrt(abs(inner(tauinv_mps, tauinv_mps)))  # ||O·ψ|| = sqrt(⟨O·ψ|O·ψ⟩)
            nu = real(inner(psi, apply(M_mpo, psi; naive_cons...)))
            orthoovlps = [abs(inner(psi, u)) for u in ortho_states]
            psi0ovlp = abs(inner(psi / norm(psi), psi0))

            logwrite("Norm psi: $Npsi")
            logwrite("tauinv ($operator_label): $tauinv, tau = $(1/tauinv)")
            logwrite("nu (M expectation): $nu")
            logwrite("Overlap with initial state psi0: $psi0ovlp")
            for i in eachindex(ortho_states)
                logwrite("Ortho state $i overlap: $(orthoovlps[i])")
            end

            opsizes = get_size_distribution(psi)
            logwrite("Size distribution of psi:")
            logwrite(opsizes)

            k_weights = calculate_k_space_decomposition(psi)[1]
            logwrite("K-space weights of psi:")
            logwrite(k_weights)

            # Store current observables in history
            push!(obs_history, (tauinv=tauinv, nu=nu, k_weights=k_weights))

            converged = 0

            # Check energy convergence (original criterion)
            energy_converged = abs(energy - last_energy) < energy_conv_threshold && current_noise <= final_noise

            # Check observable convergence (new criterion)
            observable_converged = false
            if length(obs_history) >= 2
                # Compare current vs previous iteration
                curr = obs_history[end]
                prev = obs_history[end-1]

                # Relative changes
                tauinv_change = abs(curr.tauinv - prev.tauinv) / (abs(prev.tauinv) + 1e-12)
                nu_change = abs(curr.nu - prev.nu) / (abs(prev.nu) + 1e-12)
                # For k_weights, use maximum relative change (normalized by previous value)
                k_weights_change = maximum(abs.(curr.k_weights .- prev.k_weights) ./ (abs.(prev.k_weights) .+ 1e-12))

                logwrite("Observable changes: tauinv: $(tauinv_change), ⟨M⟩: $(nu_change), max Δk: $(k_weights_change)")

                # Check if observables changed less than threshold (removed k_weights_change condition)
                if tauinv_change < obs_conv_threshold &&
                   nu_change < obs_conv_threshold
                    obs_converged_count += 1
                    logwrite("Observable convergence count: $obs_converged_count / $obs_conv_consecutive")
                    if obs_converged_count >= obs_conv_consecutive
                        observable_converged = true
                    end
                else
                    obs_converged_count = 0  # Reset counter if not converged
                end
            end

            # Converged if either energy OR observables satisfy criteria
            if energy_converged || observable_converged
                if energy_converged
                    logwrite("\nConvergence reached for L=$(L)! (energy criterion)")
                end
                if observable_converged
                    logwrite("\nConvergence reached for L=$(L)! (observable criterion: $obs_conv_consecutive consecutive iterations with <$(obs_conv_threshold*100)% change)")
                end
                converged = 1
            end

            if converged == 1
                break
            end

            # Update noise schedule
            last_energy = energy
            current_noise = max(current_noise * noise_decay_factor, final_noise)
            if sweep_converged
                current_noise = final_noise  # Jump to final noise if sweep converged
            end

            if iter == max_iter
                @warn "DMRG did not converge within $max_iter iterations for L=$L."
            end
        end

        logwrite("\n-----------------------------------------")
        logwrite("DMRG for L=$L finished.")
        logwrite("Final ground state energy: $energy")
        logwrite("-----------------------------------------")

    end

    #--------------------------------------------------------------------------
    # SAVE RESULTS
    #--------------------------------------------------------------------------

    # The output filename is derived only from (L, J, hx, hz, orth, pbc, theta),
    # so two concurrent runs sharing those values write the SAME file. That is
    # not benign: it corrupted two tasks of array 12974859 with
    # `H5Error("Error opening group //psi")` when one read a partially-written
    # file. Benchmark/grid callers that do not need the artifact should pass
    # `save_mps = false`.
    if !save_mps
        logwrite("Skipping MPS save (save_mps=false)")
    else
    logwrite("Saving MPS...")
    metadata["conv"] = converged
    @time saveMPS(
        cpuarray(psi),
        metadata,
        other_data = Dict("Npsi"=>Npsi, "tauinv"=>tauinv,
            "nu"=>nu, "ovlps"=>cpuarray(orthoovlps), "sizes"=>cpuarray(opsizes),
            "kwt"=>k_weights, "psi0ovlp"=>psi0ovlp)
    )
    end

    return psi
end


"""
    run_dmrg_for_size(L::Int, H_dict::Dict; kwargs...)

Runs DMRG to find an approximate conserved quantity for a system of size ``L``.

# Arguments
- `L`: System size (number of sites)
- `H_dict`: Hamiltonian dictionary, e.g., `Dict("ZZ" => -1.0, "X" => -0.5)`.
  Under `floquet=true` this is instead the Floquet **gate spec**, and should be an
  **ordered** vector of pairs so non-commuting layers are applied in a defined
  order — for kicked Ising, `["ZZ" => -J, "Z" => -h, "X" => -g]`. A `Dict` is
  accepted for both modes but warns under `floquet=true` (hash iteration order
  would pick an arbitrary unitary).

# Keyword Arguments
- `psi0_in`: Initial MPS guess (`nothing` = random)
- `theta`: Penalty parameter for commutator weight (default: 1.0)
- `num_ortho`: Number of Krylov vectors to orthogonalize against (default: 3)
- `ortho_weight`: Weight for orthogonalization penalty (default: 1.0)
- `initial_noise`: Starting noise level for DMRG (default: 1E-4)
- `final_noise`: Final noise level (default: 1E-8)
- `noise_decay_factor`: Multiplicative decay per iteration (default: 0.8)
- `sweeps_per_iteration`: Number of DMRG sweeps per iteration (default: 5)
- `max_iter`: Maximum iterations (default: 20)
- `energy_conv_threshold`: Energy convergence threshold (default: 1E-8)
- `obs_conv_threshold`: Observable relative change threshold for early termination (default: 0.01 = 1%)
- `obs_conv_consecutive`: Number of consecutive iterations below threshold to trigger early termination (default: 1)
- `init_index`: Initial state choice
  - 0: random
  - Hamiltonian mode: 1=``H^n``, 2=momentum mode ``H_k``
  - Floquet mode: 1=Hamiltonian from H_dict, 2=hydrodynamic operator from H_dict
- `override`: Force recalculation even if converged result exists
- `pbc`: Periodic boundary conditions
- `floquet`: Use Floquet mode (default: false)
- `load_arb_init`: If true, ignore init_index when loading existing files and accept any init value (default: false)

# Returns
- `psi0`: MPS representing the approximate conserved quantity

# Workflow
1. Check for existing converged results (load if found)
2. Build operators (bottom and top layers of the ``C^\\dagger C`` sandwich the
   DMRG environments contract):
   - Hamiltonian mode: ``C_H = H_L - H_R`` for both layers (it is Hermitian, so
     ``C_H \\cdot C_H = C_H^\\dagger C_H``)
   - Floquet mode: ``C_U = \\mathcal{U} - \\mathbb{1}`` and its adjoint
     ``C_U^\\dagger``, giving ``\\|U^\\dagger O U - O\\|^2``
3. Build orthogonalization basis:
   - Hamiltonian mode: Krylov basis ``\\{I, H, H^2, \\ldots, H^n\\}``
   - Floquet mode: Identity only
4. Choose initial guess based on `init_index`
5. Run DMRG (exact for ``L \\leq 6``, iterative otherwise)
6. Save results to disk

# Output Diagnostics
- `Npsi`: Norm of result (should be 1)
- `tauinv`: Operator norm (``\\|[H,\\psi]\\|`` for Hamiltonian,
  ``\\|C_U\\psi\\| = \\|U^\\dagger \\psi U - \\psi\\|`` for Floquet). Computed as
  ``\\sqrt{\\langle C\\psi | C\\psi \\rangle}``, which is adjoint-correct for a
  non-Hermitian ``C``.
- `nu`: ``M`` expectation value
- `ovlps`: Overlaps with orthogonalization basis
- `sizes`: Operator size distribution
- `kwt`: Momentum space weights
"""
function run_dmrg_for_size(L::Int, H_dict::Union{AbstractDict, AbstractVector};
                           psi0_in::Union{MPS, Nothing}=nothing,
                           theta::DTYPE=1.0, num_ortho=3, ortho_weight=1.0,
                           initial_noise=1E-4, final_noise=1E-8,
                           noise_decay_factor=0.8, sweeps_per_iteration=5,
                           max_iter=20, energy_conv_threshold=1E-8,
                           obs_conv_threshold::DTYPE=0.01, obs_conv_consecutive::Int=1,
                           init_index::Int=0, override::Bool=false,
                           pbc::Bool=false,
                           floquet::Bool=false, load_arb_init::Bool=false,
                           model_metadata::Union{Dict,Nothing}=nothing,
                           maxdim_cap::Int=128,
                           thetasq_out::Union{Ref{DTYPE},Nothing}=nothing,
                           scheduling::Symbol=:fixed_theta,
                           local_solver::Symbol=:eigsolve,
                           epsilon_loc::DTYPE=DTYPE(1e-4),
                           eps_anneal_sweeps::Int=1,
                           eps_anneal_start::DTYPE=DTYPE(0),
                           qcqp_theta_sq_max::DTYPE=DTYPE(1e8),
                           qcqp_bisect_tol::Real=1e-3,
                           qcqp_warm_accept_tol::Real=5e-2,
                           qcqp_warm_accept_tol_min::Real=5e-3,
                           qcqp_warm_accept_tol_trigger::Real=3.0,
                           qcqp_use_secant::Bool=true,
                           qcqp_slope_init::Real=1.0,
                           qcqp_thetasq_ema::Union{Ref{DTYPE},Nothing}=nothing,
                           qcqp_ema_alpha::Real=0.1,
                           save_mps::Bool=true,
                           qcqp_lobpcg_cheby_degree::Int=0,   # 0 = adaptive in κ(B); >0 pins it
                           qcqp_lobpcg_cheby_min::Int=2,
                           qcqp_lobpcg_cheby_max::Int=8,
                           qcqp_lobpcg_cheby_eta::Real=0.1,
                           qcqp_lobpcg_rel_tol::Real=1e-6,
                           qcqp_lobpcg_inner_maxiter::Int=12,
                           qcqp_lobpcg_max_restarts::Int=8,
                           qcqp_lobpcg_maxiter::Int=100,
                           qcqp_lobpcg_power_iters::Int=10,
                           krylovdim::Int=25,
                           qcqp_verbose::Bool=false,
                           debuglogger::Union{DebugLogger,Nothing}=nothing,
                           save_callback::Union{Function,Nothing}=nothing)::MPS

    # Prepare metadata for file I/O
    # Start with common metadata
    metadata = Dict{String,Number}("L"=>L, "pbc"=>Int(pbc), "theta"=>theta, "orth"=>num_ortho,
                                    "init"=>init_index, "flq"=>Int(floquet))

    # Term-string → coefficient lookup that works for both spec shapes: a plain
    # Hamiltonian `Dict`, and the ordered `Vector` of pairs a Floquet gate spec
    # uses (where order matters, see `_normalize_gate_spec`).
    coeff_dict = H_dict isa AbstractDict ? H_dict : Dict(_normalize_gate_spec(H_dict))

    # Add model-specific metadata if provided, otherwise fall back to Ising defaults
    if !isnothing(model_metadata)
        merge!(metadata, model_metadata)
    else
        # Legacy Ising model defaults
        metadata["J"] = get(coeff_dict, "ZZ", 0.0)
        metadata["hx"] = get(coeff_dict, "X", 0.0)
        metadata["hz"] = get(coeff_dict, "Z", 0.0)
    end

    logwrite("="^60 * "\nStarting DMRG for L = $L\n" * "="^60)

    logwrite("Metadata:")
    for (key, value) in metadata
        logwrite("  $key = $value")
    end

    # Check for existing results
    logwrite("Looking for existing MPS files...")
    if load_arb_init
        logwrite("load_arb_init=true: searching for files with any init value")
    end
    @time prev_conv, mpsdata = load_existing_file(metadata; load_arb_init=load_arb_init)
    logwrite("prev_conv = $prev_conv")
    logwrite("mpsdata: $(typeof(mpsdata))")

    if prev_conv >= 0
        psi_prev, _, _, _, _ = mpsdata
        if override
            logwrite("NOTICE: Previous MPS exists, overriding!")
        end
    end

    # Load converged result if it exists (unless override=true)
    if prev_conv == 1 && !override
        logwrite("Converged MPS already exists for these parameters. Loading and returning it.")
        return psi_prev
    end

    # Use an unconverged saved result as the initial guess if available
    if prev_conv == 0 && !override
        logwrite("Unconverged MPS exists for these parameters. Loading and using it as initial state.")
        psi0_in = psi_prev
    end

    # Setup site indices
    if isnothing(psi0_in)
        sites = siteinds("Pauli", L)
    else
        @assert length(psi0_in) == L "Provided MPS length $(length(psi0_in)) does not match L=$L."
        sites = siteinds(psi0_in)
    end

    #--------------------------------------------------------------------------
    # BUILD OPERATORS AND ORTHOGONALIZATION BASIS
    #--------------------------------------------------------------------------

    # Construct M operator (diagonal, favors sparse operators)
    M_mpo = movedevice(MPO(sites, "M"))

    oper_top = nothing   # top layer of the C†C sandwich; nothing ⇒ reuse the bottom layer

    if floquet
        logwrite("\n1-2. Building Floquet operators C_U = (U - I) and C_U†...")
        oper_single, oper_for_dmrg, oper_top, oper_for_init, operator_label =
            build_floquet_operators(L, sites, H_dict; pbc=pbc)

        # Identity only: U·𝟙·U† = 𝟙 exactly, so the identity is the trivial fixed
        # point and must be projected out. Unlike the Hamiltonian case there is no
        # {I, H, H², …} polynomial tower of trivially-slow operators to remove.
        logwrite("\n3. Floquet mode: orthogonalizing against identity only...")
        ortho_states = [movedevice(productMPS(sites, "I"))]

        # Choose initial guess based on init_index. These read the gate spec as a
        # plain coefficient dict (order is irrelevant for a sum of Pauli strings)
        # to seed a heuristic starting operator.
        if init_index == 1
            logwrite("init_index set to 1 (Floquet mode), using Hamiltonian from H_dict as initial guess.")
            psi0_in = build_operator_mps(L, sites, coeff_dict; pbc=pbc, k=DTYPE(0.0))
            normalize!(psi0_in)
        elseif init_index == 2
            logwrite("init_index set to 2 (Floquet mode), using hydrodynamic operator from H_dict.")
            psi0_in = build_operator_mps(L, sites, coeff_dict; pbc=pbc, k=DTYPE(2*pi/L))
            normalize!(psi0_in)
        end
    else
        logwrite("\n1-2. Building Hamiltonian operators (commutator C_H)...")
        oper_single, oper_for_dmrg, oper_left, operator_label = build_hamiltonian_operators(L, sites, coeff_dict; pbc=pbc)

        logwrite("\n3. Constructing the Krylov basis to project out...")
        # Build {I, H, H², ..., Hⁿ⁺¹} and keep first n+1 for orthogonalization
        krylov_basis_large = build_krylov_basis_raw(oper_left, sites, num_ortho+1)
        ortho_states = krylov_basis_large[1:end-1]

        # Choose initial guess based on init_index
        if init_index == 1
            logwrite("init_index set to 1, using H^$(num_ortho) as the initial guess.")
            psi0_in = krylov_basis_large[end]
        elseif init_index == 2
            logwrite("init_index set to 2, using H_{k=2pi/L} as the initial guess.")
            psi0_in = build_operator_mps(L, sites, coeff_dict; pbc=pbc, k=DTYPE(2*pi/L))
            normalize!(psi0_in)
        end
    end

    #--------------------------------------------------------------------------
    # INITIALIZE MPS
    #--------------------------------------------------------------------------

    if isnothing(psi0_in)
        logwrite("   No initial state provided, starting from random MPS.")
        psi0 = randomMPS(sites, linkdims=10)
    else
        logwrite("   Starting from provided/constructed initial state.")
        psi0 = psi0_in
    end
    psi0 = movedevice(psi0)
    psi0 /= norm(psi0)
    psi0 = complex(psi0)

    #--------------------------------------------------------------------------
    # RUN CORE DMRG
    #--------------------------------------------------------------------------

    psi0 = _run_dmrg_core(L, sites, oper_single, oper_for_dmrg, M_mpo, metadata, psi0, ortho_states,
                          theta, ortho_weight, initial_noise, final_noise, noise_decay_factor,
                          sweeps_per_iteration, max_iter, energy_conv_threshold, operator_label,
                          obs_conv_threshold, obs_conv_consecutive, debuglogger;
                          oper_top=oper_top,
                          maxdim_cap=maxdim_cap,
                          thetasq_out=thetasq_out,
                          scheduling=scheduling,
                          local_solver=local_solver,
                          epsilon_loc=epsilon_loc,
                          eps_anneal_sweeps=eps_anneal_sweeps,
                          eps_anneal_start=eps_anneal_start,
                          qcqp_theta_sq_max=qcqp_theta_sq_max,
                          qcqp_bisect_tol=qcqp_bisect_tol,
                          qcqp_warm_accept_tol=qcqp_warm_accept_tol,
                          qcqp_warm_accept_tol_min=qcqp_warm_accept_tol_min,
                          qcqp_warm_accept_tol_trigger=qcqp_warm_accept_tol_trigger,
                          qcqp_use_secant=qcqp_use_secant,
                          qcqp_slope_init=qcqp_slope_init,
                          qcqp_thetasq_ema=qcqp_thetasq_ema,
                          qcqp_ema_alpha=qcqp_ema_alpha,
                          save_mps=save_mps,
                          qcqp_lobpcg_cheby_degree=qcqp_lobpcg_cheby_degree,
                          qcqp_lobpcg_cheby_min=qcqp_lobpcg_cheby_min,
                          qcqp_lobpcg_cheby_max=qcqp_lobpcg_cheby_max,
                          qcqp_lobpcg_cheby_eta=qcqp_lobpcg_cheby_eta,
                          qcqp_lobpcg_rel_tol=qcqp_lobpcg_rel_tol,
                          qcqp_lobpcg_inner_maxiter=qcqp_lobpcg_inner_maxiter,
                          qcqp_lobpcg_max_restarts=qcqp_lobpcg_max_restarts,
                          qcqp_lobpcg_maxiter=qcqp_lobpcg_maxiter,
                          qcqp_lobpcg_power_iters=qcqp_lobpcg_power_iters,
                          krylovdim=krylovdim,
                          qcqp_verbose=qcqp_verbose,
                          save_callback=save_callback)

    return psi0
end
