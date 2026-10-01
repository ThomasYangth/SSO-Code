"""
apply_Heff_local!: two-site local matvec for the factored (two-layer C_H env)
path. Computes  Hv = Mcoef * (LM·v·M[p]·M[p+1]·RM) + CHcoef * (LH·v·CH[p]·CH[p+1]·CH[p]'·CH[p+1]'·RH),
with orthogonalization against `U` before/after.

`subspace_coeffs` is `(scale, ortho_penalty)`: `scale` multiplies the result on
the physical (I−P) subspace, `ortho_penalty` is the eigenvalue assigned to the
P = UU† subspace so `:SR` targeting skips it.

!!! warning
    Mutates `v` in place (the (I−P) projection is applied to the caller's
    vector), hence the `!`. Pass a copy if you need `v` preserved.

The C_H term is contracted in Order B: bottom layer (`Cb` at prime level 0→1)
applied first, then top layer (`Ct` primed by 1, level 1→2). The 4-legged envs
`LH`, `RH` sandwich the two layers, with the bra side at prime level 2 (see env
update code).

`Ct_p`/`Ct_p1` are the **top-layer** operator tensors, i.e. the adjoint of the
bottom layer. For the Hamiltonian path `C_H = H_L − H_R` is Hermitian, so callers
pass the same tensors as both layers and this reduces to `C_H · C_H = C_H† C_H`.
For the Floquet path `C_U = 𝒰 − 𝟙` is not Hermitian, so the caller must pass
`adjoint_mpo(C_U)` tensors here; otherwise this would compute `C_U²`, not the
physical `C_U† C_U = ‖C_U ψ‖²`.

Output prime-level 2 indices (from bra side of the top layer) are folded back to
level 0 via `replaceprime(., 2 => 0)` to match the input layout.
"""
function apply_Heff_local!(v, psiinds,
                                    LM, RM, M_p, M_p1,
                                    LH, RH, CH_p, CH_p1, Ct_p, Ct_p1,
                                    Mcoef, CHcoef,
                                    U, subspace_coeffs,
                                    dlog::DebugLogger=NoOpDebugLogger(),
                                    matvec_counter::Union{Ref{Int},Nothing}=nothing)
    if matvec_counter !== nothing
        matvec_counter[] += 1
    end

    scale, ortho_penalty = subspace_coeffs

    # Ortho projection against U
    if U !== nothing
        v_par = U * (U' * v)
        v .-= v_par
        v_par .*= ortho_penalty
    end

    psiT = ITensor(v, psiinds...)

    # === M term (single-layer application) ===
    result_M = noprime(LM * psiT * M_p * M_p1 * RM)
    Hv = Mcoef .* vec(array(result_M, psiinds...))

    # === C_H² term via factored two-layer contraction (Order B) ===
    #
    # NOTE (perf): this hand-written left-to-right pairwise sequence is a
    # reasonable heuristic but is NOT guaranteed to minimize FLOPs for our
    # (χ, χ_H, d). At χ_H ≪ χ, the χ³·χ_H²·d² LH/RH endpoints dominate; at
    # larger χ_H the interior χ²·χ_H⁴·d³ steps take over. If matvec cost
    # becomes a profiled bottleneck, consider:
    #   1. cache `ITensors.optimal_contraction_sequence(...)` at DMRG init and
    #      call `contract(tensors; sequence=cached)` here — auto-picker uses
    #      real dim tuples so it may beat this order.
    #   2. hoist `prime(Ct_p)` / `prime(Ct_p1)` out of the matvec loop
    #      (currently reallocated per call).
    tmp = LH * psiT
    tmp = tmp * CH_p            # bottom, site p:  σ_p(0) → σ_p(1)
    tmp = tmp * CH_p1           # bottom, site p+1
    tmp = tmp * prime(Ct_p)     # top,    site p:  σ_p(1) → σ_p(2)
    tmp = tmp * prime(Ct_p1)    # top,    site p+1
    tmp = tmp * RH
    result_CH = replaceprime(tmp, 2 => 0)
    Hv .+= CHcoef .* vec(array(result_CH, psiinds...))

    # Ortho back
    if U !== nothing
        Hv .-= U * (U' * Hv)
        Hv .*= scale
        Hv .+= v_par
    end

    return Hv
end

"""
    gauge_split(psi1, psi2; linktag="", svd_kwargs)

SVD-split two neighbouring MPS tensors so that the new `psi1` is an isometry on
the indices it does not share with `psi2`, absorbing S·V into `psi2` — i.e. shift
the orthogonality centre one site to the right.

Unrelated to `ortho_L` / `ortho_R` / `orthogonalize_states`, which concern
projection out of the Krylov subspace rather than MPS gauge.
"""
function gauge_split(psi1, psi2; linktag="", svd_kwargs)
    U, S, V = svd(psi1, setdiff(inds(psi1), inds(psi2))...; lefttags=linktag, svd_kwargs...)
    return U, S*V*psi2, commonind(U, S)
end


"""
    dmrg_sweeps!(psi, CH, M, thetasq, sweeps; kwargs...) -> (energy, psi, converged)

Two-site DMRG over operator-space MPS `psi` that minimizes, block by block,

    θ²·⟨ψ|C†C|ψ⟩ − ⟨ψ|M|ψ⟩      subject to  ψ ⊥ `orthogonalize_states`,

with `C = CH` (bottom layer) and `C† = CHtop` (top layer, defaults to `CH`).
The environments are the 4-legged factored ones: two un-squared `C` layers.

Two independent switches select the algorithm:

* `scheduling` — how θ² is set at a block: `:fixed_theta` (use `thetasq`) or
  `:eps_constrained` (treat θ² as the Lagrange multiplier of ⟨C†C⟩ ≤ ε and
  locate it per block with `local_step_qcqp`; ε may be annealed from
  `eps_anneal_start` to `epsilon_loc` within the call).
* `local_solver` — the numerical primitive: `:eigsolve` (ordinary Hermitian
  eigensolve of −M + θ²C†C; badly conditioned at large θ²), `:geneig`
  (KrylovKit generalized eigensolve of −M v = λ(I + θ²C†C)v) or `:lobpcg`
  (same pencil, LOBPCG + Chebyshev-in-B preconditioner; production choice).

Convergence is judged on the sweep-to-sweep relative change of ⟨M⟩ and ⟨C†C⟩
read off the environments at the last block (θ² floats, so the eigenvalue is
not comparable between sweeps). `save_callback(psi, sweep, energy)` runs after
every sweep (used for rolling checkpoints).
"""
function dmrg_sweeps!(
    psi::MPS,
    CH::MPO,
    M::MPO,
    thetasq::DTYPE,
    sw::Sweeps;
    # Top layer of the factored two-layer C_H² / C_U†C_U environments. `nothing`
    # means "reuse `CH`", which is correct whenever `CH` is Hermitian as an
    # operator (the Hamiltonian path, C_H = H_L − H_R) and reproduces the
    # pre-Floquet contraction tensor-for-tensor. The Floquet path passes
    # `adjoint_mpo(C_U)` here, since C_U² ≠ C_U†C_U for non-Hermitian C_U.
    CHtop::Union{MPO, Nothing} = nothing,
    dispon::Int = 2,
    eig_kwargs::Dict = Dict(),
    orthogonalize_states::Vector{MPS} = Vector{MPS}(), # States to orthogonalize against during Lanczos
    subspace_coeffs::Tuple{DTYPE, DTYPE} = (1.0, 0.0), # (a,b) means H -> a*H + b*I
    stop_tol::DTYPE = 1E-8,
    # === Two-axis solver selection ===
    # scheduling: how θ² is set at each two-site block.
    #   :fixed_theta      — use the outer `thetasq` verbatim
    #   :eps_constrained  — QCQP: bisect θ² so that ⟨x|C_H²|x⟩ = epsilon_loc
    # local_solver: the numerical primitive that solves the local problem.
    #   :eigsolve — ordinary Hermitian eig on (−M + θ²·C_H²)
    #   :geneig   — generalized Hermitian eig on (−M , I + θ²·C_H²)
    #   :lobpcg   — LOBPCG on (−M , I + θ²·C_H²) with Chebyshev-in-B preconditioner
    #               (use when :geneig stalls at large θ²; see _call_lobpcg_inner)
    # Envs are always the 4-legged factored representation (un-squared C_H layers).
    scheduling::Symbol = :fixed_theta,
    local_solver::Symbol = :eigsolve,
    epsilon_loc::DTYPE = DTYPE(1e-4),  # only used under :eps_constrained
    # ε-annealing (only under :eps_constrained). The default step-start behavior
    # is a *quench*: ε jumps straight to `epsilon_loc` at the first block.
    # Annealing instead ramps ε down from `eps_anneal_start` to `epsilon_loc`
    # geometrically — each two-site block multiplies ε by
    #   q = (epsilon_loc/eps_anneal_start)^(1/(2·Nsites·eps_anneal_sweeps)),
    # clamped at the `epsilon_loc` floor, so ε reaches the target after about
    # `eps_anneal_sweeps` sweeps and then holds there. Motivation: a sudden ε
    # tightening (the quench) can shock the state into a θ²-runaway /
    # local-minimum trap; annealing lets it follow the constraint adiabatically.
    # Annealing is ON by default (`eps_anneal_sweeps=1`), but only activates when
    # the caller also supplies `eps_anneal_start > epsilon_loc` (the ε the state
    # is starting from). With `eps_anneal_start=0` (default) or `≤ epsilon_loc`,
    # or `eps_anneal_sweeps=0`, it falls back to the quench → ε=epsilon_loc
    # verbatim. So a bare call without a start ε is unaffected; ε-walk drivers
    # that pass the previous step's ε anneal automatically.
    eps_anneal_sweeps::Int = 1,
    eps_anneal_start::DTYPE = DTYPE(0),
    # Carries the final accepted θ² back out to the caller under
    # :eps_constrained. θ² warm-starts site-to-site inside one call, but is
    # otherwise re-initialized from `thetasq` on every call — so an ε walk
    # re-brackets from scratch at each step even though the optimal θ² grows
    # roughly like 1/ε. Pass a Ref here and feed it back in as the next step's
    # `thetasq` to keep the multiplier warm across the whole scan.
    thetasq_out::Union{Ref{DTYPE}, Nothing} = nothing,
    qcqp_theta_sq_max::DTYPE = DTYPE(1e8),  # bracket-up ceiling; see local_step_qcqp
    qcqp_bisect_tol::Real = 1e-3,           # relative tolerance on |B - ε|/ε for bisection convergence
    qcqp_warm_accept_tol::Real = 5e-2,      # skip bisection when warm-start B within this rel-tol of ε — initial value
    # Adaptive warm-tol schedule: after each sweep, tighten the accept-window
    # in proportion to ⟨M⟩'s sweep-over-sweep rel-change. Monotone tighten
    # only, floored at `qcqp_warm_accept_tol_min`, so once we're in the
    # near-optimum regime QCQP starts actually pinning ⟨C_H²⟩ to ε_loc via
    # bracket + bisect instead of tight-warm accepting anywhere in the initial
    # window. Set `qcqp_warm_accept_tol_min = qcqp_warm_accept_tol` to disable
    # (preserve legacy behavior).
    qcqp_warm_accept_tol_min::Real = 5e-3,
    qcqp_warm_accept_tol_trigger::Real = 3.0,  # multiplier: warm_tol := clamp(trigger·ΔM/M, min, current)
    # Secant search: persistent slope for the log-log Newton update in the
    # QCQP bracketing step. Carried across every block for the whole DMRG
    # sweeper (all sweeps, both directions). Setting `qcqp_use_secant=false`
    # falls back to the legacy geometric ×2/×10 bracket.
    qcqp_use_secant::Bool = true,
    qcqp_slope_init::Real = 1.0,
    qcqp_verbose::Bool = false,
    # Optional EMA of θ² across blocks. If provided, caller re-uses it as the
    # warm-start seed for the next DMRG call (e.g., next ε step of a scan).
    qcqp_thetasq_ema::Union{Ref{DTYPE}, Nothing} = nothing,
    qcqp_ema_alpha::Real = 0.1,
    # LOBPCG-only tuning (ignored unless `local_solver = :lobpcg`).
    qcqp_lobpcg_cheby_degree::Int = 0,   # 0 = adaptive in κ(B); >0 pins the degree
    qcqp_lobpcg_cheby_min::Int = 2,
    qcqp_lobpcg_cheby_max::Int = 8,
    qcqp_lobpcg_cheby_eta::Real = 0.1,
    qcqp_lobpcg_rel_tol::Real = 1e-6,
    qcqp_lobpcg_inner_maxiter::Int = 12,
    qcqp_lobpcg_max_restarts::Int = 8,
    qcqp_lobpcg_maxiter::Int = 100,
    qcqp_lobpcg_power_iters::Int = 10,
    debuglogger::Union{DebugLogger, Nothing} = nothing,
    # Post-sweep hook: called as `save_callback(psi, k, energy)` at the end of
    # every sweep (before the early-convergence check). Use to checkpoint the
    # unconverged MPS for restart-after-timeout scenarios. `nothing` = no-op.
    save_callback::Union{Function, Nothing} = nothing,
)
    if !(scheduling in (:fixed_theta, :eps_constrained))
        error("scheduling=$scheduling not supported; use :fixed_theta or :eps_constrained")
    end
    if !(local_solver in (:eigsolve, :geneig, :lobpcg))
        error("local_solver=$local_solver not supported; use :eigsolve, :geneig, or :lobpcg")
    end

    # Bottom / top layers of the factored operator environments. `CHt === CH`
    # when no explicit top layer is given, so the Hamiltonian path contracts
    # exactly as it did before `CHtop` existed.
    CHt = CHtop === nothing ? CH : CHtop
    if length(CHt) != length(CH)
        error("CHtop has length $(length(CHt)) but CH has length $(length(CH))")
    end

    # Warm-started Lagrange multiplier (θ²) for :eps_constrained scheduling — the same object as
    # the outer `thetasq` argument, but allowed to float site-to-site during the
    # sweep. Initialized to the user's thetasq so that when the constraint is
    # already inactive at that value the first eigsolve just returns the
    # standard DMRG state.
    thetasq_ref = Ref(convert(DTYPE, thetasq))

    # Persistent secant slope for QCQP log-log Newton updates. Model is
    # ⟨C_H²⟩ ≈ A·(θ²)^(−s); an initial s=1 corresponds to the large-θ²
    # asymptote of A/(1+θ²·L²). See `local_step_qcqp` for the slope guards.
    slope_ref = qcqp_use_secant ? Ref(convert(DTYPE, qcqp_slope_init)) : nothing

    # Effective warm-accept tolerance, tightened between sweeps in proportion
    # to the sweep-over-sweep rel-change in ⟨M⟩. Monotone; floored at
    # `qcqp_warm_accept_tol_min`.
    warm_tol_effective = Ref(convert(DTYPE, qcqp_warm_accept_tol))
    warm_tol_floor = convert(DTYPE, qcqp_warm_accept_tol_min)
    warm_tol_trigger = convert(DTYPE, qcqp_warm_accept_tol_trigger)

    # Per-call counter for QCQP tracing
    qcqp_call_id = Ref(0)

    num_ortho = length(orthogonalize_states)

    # Logger selection: honor an explicit debuglogger; otherwise SLOWOP_DEBUG=1
    # auto-creates an ActiveDebugLogger writing to debug_log_dir().
    dlog = if debuglogger !== nothing
        debuglogger
    elseif debug_mode()
        create_debug_logger("dmrg_sweeps")
    else
        NoOpDebugLogger()
    end

    debuglog(dlog, "Starting dmrg_sweeps!")
    debuglog(dlog, "  Nsites=$(length(psi)), num_ortho=$num_ortho, nsweep=$(nsweep(sw))")
    nsw = nsweep(sw)
    maxdim_str = join(["$(maxdim(sw,i))" for i in 1:nsw], ", ")
    # Both lists must be built over 1:nsw. Hardcoding noise(sw,2) here made
    # `sweeps_per_iteration = 1` die with a BoundsError out of a *logging*
    # string — a one-sweep run is otherwise perfectly legal, and is what the
    # degree-sweep diagnostics want.
    noise_str = join(["$(noise(sw,i))" for i in 1:nsw], ", ")
    debuglog(dlog, "  maxdim=[$maxdim_str], noise=[$noise_str]")
    debuglog(dlog, "  thetasq=$thetasq, eig_nev=$(get(eig_kwargs, :nev, 1)), krylovdim=$(get(eig_kwargs, :krylovdim, 20))")

    Nsites = length(psi)
    Ekeep = DTYPE[]

    eig_nev = haskey(eig_kwargs, :nev) ? eig_kwargs[:nev] : 1
    eig_kwargs = filter(kv -> kv[1] != :nev, eig_kwargs)

    # Get the site and link indices from the initial MPS
    sites = siteinds(psi)
    links = linkinds(psi)

    debuglog(dlog, "  scheduling=$scheduling  local_solver=$local_solver")

    # Construct environmental tensors.
    # LH/RH hold the C_H² env as a 4-legged tensor with two separate χ_H MPO
    # bonds (bra side at prime level 2); at matvec time the two un-squared C_H
    # layers are contracted in Order B (see `apply_Heff_local!`).
    RH = Vector{ITensor}(undef, Nsites)
    RH[Nsites] = movedevice(ITensor(1.0))
    RM = Vector{ITensor}(undef, Nsites)
    RM[Nsites] = movedevice(ITensor(1.0))
    ortho_R = [Vector{ITensor}(undef, Nsites) for _ in 1:num_ortho]
    for i in 1:num_ortho
        ortho_R[i][Nsites] = movedevice(ITensor(1.0))
    end
    LH = Vector{ITensor}(undef, Nsites)
    LH[1] = movedevice(ITensor(1.0))
    LM = Vector{ITensor}(undef, Nsites)
    LM[1] = movedevice(ITensor(1.0))
    ortho_L = [Vector{ITensor}(undef, Nsites) for _ in 1:num_ortho]
    for i in 1:num_ortho
        ortho_L[i][1] = movedevice(ITensor(1.0))
    end

    # -- Warmup sweep: bring MPS into left-canonical form --
    # This sweep normalizes the MPS and sets up the left environment tensors.

    svd_kwargs = Dict(:cutoff=>cutoff(sw, 1), :maxdim=>maxdim(sw, 1))

    for n = 1:(Nsites - 1)
        psi[n], psi[n+1], links[n] = gauge_split(psi[n], psi[n+1]; linktag="Link,$n-$(n+1)", svd_kwargs)

        # -- Calculate initial Left environments --
        # LH is 4-legged: bra at prime level 2, top MPO at level 1, bottom MPO at level 0.
        LM[n + 1] = LM[n] * psi[n] * M[n] * dag(psi[n])'
        LH[n + 1] = LH[n] * psi[n] * CH[n] * prime(CHt[n]) * prime(dag(psi[n]), 2)
        for i in 1:num_ortho
            ortho_L[i][n + 1] = ortho_L[i][n] * replaceinds(orthogonalize_states[i][n], sites[n]=>sites[n]') * dag(psi[n])'
        end
    end

    # Sweep-to-sweep observables driving the convergence check. Held across
    # sweeps; NaN on the first sweep so the check cannot fire before there is
    # something to compare against.
    nu_prev = DTYPE(NaN)
    eps_prev = DTYPE(NaN)

    # ε-anneal schedule (see the eps_anneal_* kwargs). `anneal_block` counts
    # two-site blocks across all sweeps of this call; the per-block ε target is
    # `max(eps_anneal_start · q^anneal_block, epsilon_loc)`.
    anneal_on = scheduling === :eps_constrained &&
                eps_anneal_sweeps > 0 && eps_anneal_start > epsilon_loc
    anneal_q  = anneal_on ?
        (epsilon_loc / eps_anneal_start)^(one(DTYPE) / DTYPE(2 * Nsites * eps_anneal_sweeps)) :
        one(DTYPE)
    anneal_block = 0
    if anneal_on
        logwrite("[eps-anneal] ramping ε: $(eps_anneal_start) → $(epsilon_loc) over " *
                 "$(eps_anneal_sweeps) sweeps (per-block factor q=$(anneal_q))")
    end

    # -- Main DMRG sweeps --
    for k = 1:nsweep(sw)

        t_sweep_start = debug_timing_start(dlog, "Sweep $k/$(nsweep(sw))")

        # Assigned at the last block of this sweep (see below). Declared here so
        # they are locals of the sweep body rather than of the site loop.
        nu_sweep = DTYPE(NaN)
        eps_sweep = DTYPE(NaN)

        svd_kwargs = Dict(:cutoff=>cutoff(sw, k), :maxdim=>maxdim(sw, k))
        thisnoise = noise(sw, k)

        # Tally of per-block QCQP exit statuses. `local_step_qcqp` returns
        # `qcqp_status` (:tight_warm / :converged / :infeasible / :unconstrained
        # / :ceiling_hit) but nothing ever logged it, which made a real failure
        # undiagnosable: a ν(τ) ladder run froze at ⟨C_H²⟩=0.0506551 across
        # three ε steps while the target fell 0.05→0.0289→0.0167, with every
        # sweep "completing" in seconds. This field is what says why.
        sweep_qcqp_status = Dict{Symbol,Int}()
        # θ² at each block's exit. This is the discriminator between the two
        # ways a scan can freeze, which look identical in ⟨C_H²⟩ and ⟨M⟩:
        #   θ² pinned at `qcqp_theta_sq_max`  → the constraint is GENUINELY
        #     infeasible (ε below λ_min of C_H² on the tower complement); the
        #     fix is a corrected ε ladder or larger L, not a better solver.
        #   θ² far below the ceiling + :infeasible → the slope test bailed
        #     early; a false positive.
        sweep_qcqp_thetasq = Float64[]

        # Track timing statistics for this sweep
        sweep_timings = Dict{String, Vector{Float64}}(
            "ortho" => Float64[],
            "eigsolve" => Float64[],
            "svd_env" => Float64[],
            "total" => Float64[]
        )
        sweep_eigsolve_iters = Int[]

        # --- Sweep from right to left ---
        for (p, dir) in chain(zip((Nsites - 1):-1:1, repeat([:L], Nsites - 1)),
                              zip(1:(Nsites - 1), repeat([:R], Nsites - 1)))

            t_site_start = time()

            psi_block = psi[p] * psi[p + 1]
            block_inds = inds(psi_block)

            # === ORTHOGONALIZATION BASIS ===
            t_ortho_start = time()
            if num_ortho > 0
                orthogonalize_tensors = [vec(array(
                                            noprime(ortho_L[i][p]
                                                * (orthogonalize_states[i][p])
                                                * (orthogonalize_states[i][p+1])
                                                * ortho_R[i][p+1])
                                            , block_inds...))
                                            for i in 1:num_ortho]

                # Orthonormalize by modified Gram-Schmidt rather than LAPACK/
                # cuSOLVER QR. `num_ortho` is tiny (1 for Floquet, ~3 for the
                # Krylov tower) while each column has length χ²d² — so this is a
                # 4.2M x 1 problem at χ=512, and calling `qr` on it is both
                # wasteful and, on GPU, fatal: cusolverDnDgeqrf sizes its
                # workspace with 32-bit ints, and at χ=512 that overflows
                # (`InexactError: trunc(Int32, 2156004992)`, job 12299558),
                # capping χ at ~511 no matter how much VRAM is available.
                # MGS touches only BLAS-1 ops, so there is no such ceiling.
                # Repeated once ("twice is enough") for numerical stability.
                onb = similar(orthogonalize_tensors, 0)
                for v in orthogonalize_tensors
                    w = copy(v)
                    for _ in 1:2, u in onb
                        w .-= dot(u, w) .* u
                    end
                    nw = norm(w)
                    # Drop columns that were (numerically) already in the span;
                    # keeping them would put noise directions into the projector.
                    if nw > sqrt(eps(real(eltype(w)))) * max(norm(v), one(real(eltype(w))))
                        push!(onb, w ./ nw)
                    end
                end
                Q = isempty(onb) ? nothing : hcat(onb...)

                # GPU check for Q matrix
                if (p == 1 || p == Nsites-1) && dir == :L
                    debug_gpu_check(dlog, "Q matrix", Q)
                end
            else
                Q = nothing
            end
            t_ortho = time() - t_ortho_start
            push!(sweep_timings["ortho"], t_ortho)

            # Generate initial guess vector with optional noise
            initvec = vec(array(psi_block, block_inds...))
            if thisnoise > 0
                initvec .+= movedevice(thisnoise * randn(eltype(initvec), length(initvec)) ./ sqrt(length(initvec)))
                initvec ./= norm(initvec)
            end

            # Check GPU location of initvec
            if (p == 1 || p == Nsites-1) && dir == :L
                debug_gpu_check(dlog, "initvec", initvec)
            end

            # === TWO-SITE OPTIMIZATION (EIGENSOLVER) ===
            t_eig_start = time()

            # Counter for matrix-vector products
            matvec_counter = Ref(0)

            if scheduling === :eps_constrained
                # QCQP: bisect θ² locally so that ⟨x|C_H²|x⟩ = eps_block.
                # thetasq_ref warm-starts across sites; the inner numerical
                # primitive (`local_solver`) is the eigsolve at each θ² step.
                # eps_block equals epsilon_loc unless ε-annealing is active, in
                # which case it ramps from eps_anneal_start down to epsilon_loc.
                anneal_block += 1
                eps_block = anneal_on ?
                    max(eps_anneal_start * anneal_q^anneal_block, epsilon_loc) : epsilon_loc
                qcqp_call_id[] += 1
                En, gs, info = local_step_qcqp(
                    initvec, block_inds,
                    LM[p], RM[p+1], M[p], M[p+1],
                    LH[p], RH[p+1], CH[p], CH[p+1], CHt[p], CHt[p+1],
                    eps_block, thetasq_ref, Q, subspace_coeffs,
                    dispon, dlog, matvec_counter;
                    eig_kwargs = eig_kwargs,
                    inner_primitive = local_solver,
                    theta_sq_max = qcqp_theta_sq_max,
                    bisect_tol = qcqp_bisect_tol,
                    warm_accept_tol = warm_tol_effective[],
                    slope_ref = slope_ref,
                    thetasq_ema = qcqp_thetasq_ema,
                    ema_alpha = qcqp_ema_alpha,
                    lobpcg_cheby_degree = qcqp_lobpcg_cheby_degree,
                    lobpcg_cheby_min = qcqp_lobpcg_cheby_min,
                    lobpcg_cheby_max = qcqp_lobpcg_cheby_max,
                    lobpcg_cheby_eta = qcqp_lobpcg_cheby_eta,
                    lobpcg_rel_tol = qcqp_lobpcg_rel_tol,
                    lobpcg_inner_maxiter = qcqp_lobpcg_inner_maxiter,
                    lobpcg_max_restarts = qcqp_lobpcg_max_restarts,
                    lobpcg_maxiter = qcqp_lobpcg_maxiter,
                    lobpcg_power_iters = qcqp_lobpcg_power_iters,
                    verbose = qcqp_verbose,
                    call_id = qcqp_call_id[],
                )
                let st = get(info, :qcqp_status, :unknown)
                    sweep_qcqp_status[st] = get(sweep_qcqp_status, st, 0) + 1
                    tq = get(info, :qcqp_thetasq, NaN)
                    isfinite(tq) && push!(sweep_qcqp_thetasq, Float64(tq))
                end
            else  # scheduling === :fixed_theta
                if local_solver === :geneig
                    En, gs, info = local_step_geneig(
                        initvec, block_inds,
                        LM[p], RM[p+1], M[p], M[p+1],
                        LH[p], RH[p+1],
                        CH[p], CH[p+1], CHt[p], CHt[p+1],
                        thetasq, Q, subspace_coeffs,
                        dispon, dlog, matvec_counter;
                        eig_kwargs = eig_kwargs,
                    )
                else  # :eigsolve
                    local_matvec = (v) -> apply_Heff_local!(v, block_inds,
                        LM[p], RM[p+1], M[p], M[p+1],
                        LH[p], RH[p+1], CH[p], CH[p+1], CHt[p], CHt[p+1],
                        -one(DTYPE), thetasq,
                        Q, subspace_coeffs, dlog, matvec_counter)
                    En, gs, info = eigsolve(local_matvec,
                                        initvec,
                                        eig_nev, :SR; verbosity=max(0, dispon-1), ishermitian=true, eig_kwargs...)
                end
            end
            t_eig = time() - t_eig_start
            push!(sweep_timings["eigsolve"], t_eig)
            push!(sweep_eigsolve_iters, info.numops)
            if dispon == 2
                @show info
            end
            min_En_index = argmin(En)
            gs = gs[min_En_index]
            En = En[min_En_index]
            if dispon >= 2 && Q !== nothing
                @show Q' * gs
            end
            push!(Ekeep, En)

            # === SWEEP OBSERVABLES ⟨M⟩, ⟨C_H²⟩ ===
            # Read straight off the environments the sweep already maintains:
            # LM/LH span sites 1..p-1 and RM/RH span p+2..Nsites, so contracting
            # them against the block vector is a complete contraction of the
            # whole chain — these are exact global expectation values, not local
            # ones, and cost two matvecs rather than an MPO application per site.
            #
            # ⟨ψ|ψ⟩ = ‖gs‖² because the MPS is in mixed-canonical form with its
            # orthogonality centre on this block (every tensor outside it is an
            # isometry produced by the sweep's SVDs), so dividing by dot(gs,gs)
            # is all the normalization needed.
            #
            # Taken at one fixed point of the sweep — the last block — so the
            # sweep-to-sweep comparison is made in a consistent gauge.
            if dir == :R && p == Nsites - 1
                # apply_*_local return host vectors (the contraction may run on
                # GPU internally, but the result comes back to the CPU). Take the
                # dot products on the host so the eigenvector and the matvec
                # result share a device regardless of which solver produced `gs`.
                # cpuarray is a no-op when GPU is off.
                gs_cpu = cpuarray(gs)
                gs_nrm = real(dot(gs_cpu, gs_cpu))
                nu_sweep = real(dot(gs_cpu, apply_M_local(gs_cpu, block_inds,
                                                      LM[p], RM[p+1], M[p], M[p+1]))) / gs_nrm
                eps_sweep = real(dot(gs_cpu, apply_CH2_local(gs_cpu, block_inds,
                                                        LH[p], RH[p+1], CH[p], CH[p+1],
                                                        CHt[p], CHt[p+1]))) / gs_nrm
            end

            # Move the eigenvector back to the sweep's device: the QCQP inner
            # solvers return a host vector (they run their bookkeeping on the
            # CPU), while the :fixed_theta branches return one on the sweep
            # device. movedevice reconciles both — no-op on CPU, idempotent if
            # already on GPU — so the SVD and environment updates below stay
            # single-device.
            gs = movedevice(ITensor(gs, block_inds...)) # Convert back to ITensor

            # === SVD AND ENVIRONMENT UPDATE ===
            t_svd_start = time()

            if dir == :L

                # SVD to split the 2-site tensor and truncate
                if p == Nsites - 1
                    psi[p+1], S, V = svd(gs, (sites[Nsites]); lefttags="Link,$p-$(p+1)", svd_kwargs...)
                    psi[p] = S*V
                    links[p] = commonind(psi[p], psi[p+1])
                else
                    psi[p+1], S, V = svd(gs, (sites[p+1], links[p+1]); lefttags="Link,$p-$(p+1)", svd_kwargs...)
                    psi[p] = S*V
                    links[p] = commonind(psi[p], psi[p+1])
                end

                RM[p] = RM[p+1] * psi[p+1] * M[p+1] * dag(psi[p+1])'
                RH[p] = RH[p+1] * psi[p+1] * CH[p+1] * prime(CHt[p+1]) * prime(dag(psi[p+1]), 2)
                for i = 1:num_ortho
                    ortho_R[i][p] = ortho_R[i][p+1] * replaceinds(orthogonalize_states[i][p+1], sites[p+1]=>sites[p+1]') * dag(psi[p+1])'
                end

            elseif dir == :R

                # SVD to split the 2-site tensor and truncate
                if p == 1
                    psi[p], S, V = svd(gs, (sites[1]); lefttags="Link,$p-$(p+1)", svd_kwargs...)
                    psi[p+1] = S*V
                    links[p] = commonind(psi[p], psi[p+1])
                else
                    psi[p], S, V = svd(gs, (sites[p], links[p-1]); lefttags="Link,$p-$(p+1)", svd_kwargs...)
                    psi[p+1] = S*V
                    links[p] = commonind(psi[p], psi[p+1])
                end

                LM[p+1] = LM[p] * psi[p] * M[p] * dag(psi[p])'
                LH[p+1] = LH[p] * psi[p] * CH[p] * prime(CHt[p]) * prime(dag(psi[p]), 2)
                for i = 1:num_ortho
                    ortho_L[i][p+1] = ortho_L[i][p] * replaceinds(orthogonalize_states[i][p], sites[p]=>sites[p]') * dag(psi[p])'
                end

            else
                error("Invalid direction: $dir. Expected :L or :R.")
            end

            t_svd = time() - t_svd_start
            push!(sweep_timings["svd_env"], t_svd)

            # Total time for this site
            t_site_total = time() - t_site_start
            push!(sweep_timings["total"], t_site_total)

            # Log every 5th position or first/last
            if (p == 1 || p == Nsites-1 || mod(p, 5) == 0) && dir == :L
                dirstring = dir == :L ? "R->L" : "L->R"
                debuglog(dlog, "  Site $p ($dirstring): ortho=$(round(t_ortho,digits=3))s, eig=$(round(t_eig,digits=3))s ($(info.numops) matvec), svd+env=$(round(t_svd,digits=3))s, total=$(round(t_site_total,digits=3))s")
            end

            if dispon == 2
                dirstring = dir == :L ? "R -> L" : "L -> R"
                println("Sweep: $k/$(nsweep(sw)), $dirstring, Site: $p, E: $En")
                println("Current: noise $(thisnoise), maxdim $(maxdim(sw, k)), cutoff $(cutoff(sw, k))")
            end
        end

        # === SWEEP SUMMARY ===
        t_sweep_total = debug_timing_end(dlog, "Sweep $k/$(nsweep(sw))", t_sweep_start)

        # Compute statistics
        avg_ortho = mean(sweep_timings["ortho"])
        avg_eig = mean(sweep_timings["eigsolve"])
        avg_svd = mean(sweep_timings["svd_env"])
        avg_total = mean(sweep_timings["total"])
        avg_matvec = mean(sweep_eigsolve_iters)

        sum_ortho = sum(sweep_timings["ortho"])
        sum_eig = sum(sweep_timings["eigsolve"])
        sum_svd = sum(sweep_timings["svd_env"])

        debuglog(dlog, "Sweep $k Summary:")
        debuglog(dlog, "  Total time: $(round(t_sweep_total,digits=2))s")
        debuglog(dlog, "  Time breakdown:")
        debuglog(dlog, "    Orthogonalization: $(round(sum_ortho,digits=2))s ($(round(100*sum_ortho/t_sweep_total,digits=1))%), avg=$(round(avg_ortho,digits=3))s/site")
        debuglog(dlog, "    Eigensolver: $(round(sum_eig,digits=2))s ($(round(100*sum_eig/t_sweep_total,digits=1))%), avg=$(round(avg_eig,digits=3))s/site")
        debuglog(dlog, "    SVD+Environment: $(round(sum_svd,digits=2))s ($(round(100*sum_svd/t_sweep_total,digits=1))%), avg=$(round(avg_svd,digits=3))s/site")
        debuglog(dlog, "  Avg matvec per site: $(round(avg_matvec,digits=1))")
        debuglog(dlog, "  Total matvec calls: $(sum(sweep_eigsolve_iters))")

        if dispon >= 1
            logwrite("Sweep: $k/$(nsweep(sw)) Finished")
            # Contraction count on stdout, not just the debug log. This is the
            # hardware-independent cost metric for solver comparisons, and it
            # became meaningful only once the Chebyshev preconditioner, the
            # λ_max power iteration and B_expectation started bumping the
            # counter — before that it omitted all three.
            if !isempty(sweep_qcqp_status)
                tally = join(["$k=$v" for (k, v) in sort(collect(sweep_qcqp_status), by = first)], " ")
                logwrite("QCQP block exits: $tally")
                if !isempty(sweep_qcqp_thetasq)
                    ts = sort(sweep_qcqp_thetasq)
                    frac_at_ceiling = count(>=(0.5 * qcqp_theta_sq_max), ts) / length(ts)
                    logwrite("QCQP θ² at exit: min=$(round(ts[1],sigdigits=3)) med=$(round(ts[cld(end,2)],sigdigits=3)) max=$(round(ts[end],sigdigits=3)) ceiling=$(qcqp_theta_sq_max) frac_at_ceiling=$(round(frac_at_ceiling,digits=3))")
                end
            end
            logwrite("Contractions: $(sum(sweep_eigsolve_iters)) total, $(round(avg_matvec,digits=1))/site; eigensolver $(round(100*sum_eig/t_sweep_total,digits=1))% of $(round(t_sweep_total,digits=1))s")
            logwrite("Current: noise $(thisnoise), maxdim $(maxdim(sw, k)), cutoff $(cutoff(sw, k))")
            logwrite("Energy: $(Ekeep[end]), MaxLinkDim: $(maxlinkdim(psi))")
        end
        # === CONVERGENCE ===
        # Judged on the two physical observables the objective is built from,
        # ⟨M⟩ and ⟨C_H²⟩, compared sweep to sweep. Deliberately NOT on the
        # eigenvalue: under :eps_constrained scheduling θ² floats site to site,
        # so there is no fixed functional whose value is comparable across
        # sweeps. Both are relative changes; the ortho overlaps enter as an
        # absolute guard so a state that has drifted back into the Krylov
        # subspace cannot be declared converged.
        dnu = abs(nu_sweep - nu_prev) / max(abs(nu_prev), eps(DTYPE))
        deps = abs(eps_sweep - eps_prev) / max(abs(eps_prev), eps(DTYPE))
        ortho_overlaps = [abs(inner(orthogonalize_states[i], psi)) for i in 1:num_ortho]
        if dispon >= 1
            logwrite("⟨M⟩: $nu_sweep (rel. change $dnu), ⟨C_H²⟩: $eps_sweep (rel. change $deps) / tolerance $(stop_tol)")
            for i in 1:num_ortho
                logwrite("Overlap with orthogonalize_states[$i]: $(ortho_overlaps[i]) / tolerance $(stop_tol)")
            end
        end
        # Adaptive tighten the warm-accept window in proportion to how much
        # ⟨M⟩ moved sweep-over-sweep. ⟨M⟩ has no intrinsic sweep-over-sweep
        # floor (unlike ⟨C_H²⟩, which floors at warm_accept_tol), so it's the
        # honest convergence signal here. Monotone tighten only; floored at
        # `qcqp_warm_accept_tol_min`.
        if scheduling === :eps_constrained && !isnan(dnu)
            proposed = warm_tol_trigger * dnu
            new_tol = clamp(proposed, warm_tol_floor, warm_tol_effective[])
            if new_tol < warm_tol_effective[]
                debuglog(dlog, "  warm_accept_tol: $(round(warm_tol_effective[], sigdigits=3)) → $(round(new_tol, sigdigits=3)) (trigger 3·ΔM/M = $(round(proposed, sigdigits=3)))")
                warm_tol_effective[] = new_tol
            end
        end
        nu_prev, eps_prev = nu_sweep, eps_sweep

        # Post-sweep checkpoint hook. Runs after every completed sweep, before
        # the early-convergence check, so restart-after-timeout can pick up the
        # latest MPS regardless of whether the run finished cleanly.
        if save_callback !== nothing
            try
                save_callback(psi, k, Ekeep[end])
            catch e
                @warn "save_callback failed at sweep $k: $e"
            end
        end

        # NaN on the first sweep (nothing to compare against) propagates through
        # max() and fails the comparison, so the check cannot fire early.
        if max(dnu, deps, max(0, ortho_overlaps...)) < stop_tol # add 0 to ortho_overlaps to handle empty case
            logwrite("Early convergence reached. Stopping DMRG sweeps.")
            thetasq_out !== nothing && (thetasq_out[] = thetasq_ref[])
            return Ekeep[end], psi, true
        end
    end

    thetasq_out !== nothing && (thetasq_out[] = thetasq_ref[])
    return Ekeep[end], psi, false # Return final energy list and an official MPS object
end

