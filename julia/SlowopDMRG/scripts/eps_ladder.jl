#=
ε-ladder driver: the production run behind every DMRG point in
docs/dmrg_attempt.md (original name: scripts/ed_compare_L12_lobpcg.jl).

Walks the constraint downward, εᵢ = ε₀·(1/√3)^i (i = 1..MAX_STEPS, stop below
EPS_FLOOR), and at each step runs `run_dmrg_for_size` with
scheduling = :eps_constrained, local_solver = :lobpcg, warm-started from the
previous step's MPS and θ² EMA. Each completed step writes

    <data_dir()>/hk_scan_cmp/L{L}_lobpcg_sinmc_rep{REP}/done_step{NN}.h5
        psi, step, eps_target, eps_achieved (=⟨C_H²⟩), nu (=⟨M⟩),
        theta_sq_ema, eps0, L, variant

and every sweep overwrites rolling_step{NN}.h5 (partial-step resume). On
start-up the run resumes from the highest done_step present (SLOWOP_FRESH=1
ignores them), then from a matching rolling_step if one exists.

Model: H = J ΣZZ + hx ΣX + hz ΣZ, OBC (defaults J=1, hx=0.905, hz=0.809).
Warm start ("sinmc"): H_k = Σ_x (sin(kx) − c(L)) h_x at k = π/L with
c(L) = cot(π/2L)/L the exact overlap of the sin mode with H, so H_k ⊥ H.
ε₀ = ⟨C_H²⟩ of that state. Orthogonalized against {I, H, H², H³} (num_ortho=3).

χ-wall stop: if a step fails to lower ⟨C_H²⟩ (ε_new > 0.98 ε_prev) the ladder
halts after saving that step (SLOWOP_STOP_AT_WALL=0 to grind on to MAX_STEPS,
as the archived runs did — they produced the frozen tails).

Env: SLOWOP_L (12), SLOWOP_CHI (256), SLOWOP_REP (1), SLOWOP_MAXSTEPS (16),
     SLOWOP_ANNEAL (1 = ε-anneal sweeps per step; 0 = quench), SLOWOP_NOISE
     (1e-6), SLOWOP_J/HX/HZ, SLOWOP_FRESH, SLOWOP_ROLLING_RESUME (1),
     SLOWOP_STOP_AT_WALL (1), USE_CUDA=1 for the GPU extension.
=#

if get(ENV, "USE_CUDA", "0") == "1"
    using CUDA
end
using SlowopDMRG
using ITensors
using ITensorMPS
using HDF5
using Printf
using Dates

# ============ configuration ============
const L                = parse(Int, get(ENV, "SLOWOP_L", "12"))
# Hamiltonian from env (defaults = the reference chaotic Ising, so existing runs
# are unchanged). Isolate different H's by giving them a distinct SLOWOP_REP.
const H_DICT           = Dict("ZZ" => parse(Float64, get(ENV, "SLOWOP_J",  "1.0")),
                              "X"  => parse(Float64, get(ENV, "SLOWOP_HX", "0.905")),
                              "Z"  => parse(Float64, get(ENV, "SLOWOP_HZ", "0.809")))
const K                = π / L
const EPS_FLOOR        = 1e-5   # deep: ε₀~0.2, √3⁻¹⁶ ≈ 3e-5 reached before floor
const MAX_STEPS        = parse(Int, get(ENV, "SLOWOP_MAXSTEPS", "16"))
const EPS_RATIO        = 1 / sqrt(3)
const THETA_WARM_START = 1.0
const THETA_SQ_MAX     = 1e14
const NUM_ORTHO        = 3
const MAX_ITER         = 1
const SWEEPS_PER_ITER  = 6
const INITIAL_NOISE    = parse(Float64, get(ENV, "SLOWOP_NOISE", "1e-6"))
const FINAL_NOISE      = 1e-10
const MAXDIM_CAP       = parse(Int, get(ENV, "SLOWOP_CHI", "256"))
const ANNEAL           = parse(Int, get(ENV, "SLOWOP_ANNEAL", "1"))  # eps_anneal_sweeps per step (annealing on by default)
const LOCAL_SOLVER     = :lobpcg
const QCQP_BISECT_TOL      = 5e-3
const QCQP_WARM_ACCEPT_TOL = 1e-1   # initial tol; tightens sweep-over-sweep down to QCQP_WARM_ACCEPT_TOL_MIN
const QCQP_WARM_ACCEPT_TOL_MIN = 5e-3
const QCQP_USE_SECANT      = true    # log-log secant with warm-started slope carried across blocks
const KRYLOVDIM            = 30
const QCQP_EMA_ALPHA       = 0.1
const LOBPCG_CHEBY_DEGREE  = 0      # 0 = adaptive in κ(B); set >0 to pin
const LOBPCG_MAXITER       = 100    # unused by the burst loop (INNER×RESTARTS); kept for parity
const LOBPCG_REL_TOL       = 1e-6   # relative Ritz residual, replaces the 1e-14 absolute pin
const LOBPCG_INNER_MAXITER = 12
const LOBPCG_MAX_RESTARTS  = 8
const LOBPCG_POWER_ITERS   = 10

const STOP_AT_WALL     = get(ENV, "SLOWOP_STOP_AT_WALL", "1") == "1"
const REP              = get(ENV, "SLOWOP_REP", "1")
const VARIANT_TAG      = "sinmc_rep$(REP)"
const CHECKPOINT_DIR   = joinpath(SlowopDMRG.data_dir(), "hk_scan_cmp", "L$(L)_$(LOCAL_SOLVER)_$(VARIANT_TAG)")
mkpath(CHECKPOINT_DIR)

# ============ initial-state builder ============
"""
H_k = Σ_x (sin(k·x) − c) · h_x   with c = cot(π/(2L)) / L  (finite-L exact).
"""
function build_sinmc_hk(L::Int, sites, H_dict::AbstractDict; k::Real = π/L, pbc::Bool = false)
    # build_operator_mps(…; k) = Σ_x e^{-ikx} h_x with x = 0..L-1, so
    # (minus − plus)/(2i) = Σ_x sin(kx) h_x.
    plus  = SlowopDMRG.build_operator_mps(L, sites, H_dict; pbc = pbc, k = SlowopDMRG.DTYPE(k))     # Σ exp(-ikx) h_x
    minus = SlowopDMRG.build_operator_mps(L, sites, H_dict; pbc = pbc, k = SlowopDMRG.DTYPE(-k))    # Σ exp(+ikx) h_x
    Hmode = SlowopDMRG.build_operator_mps(L, sites, H_dict; pbc = pbc, k = SlowopDMRG.DTYPE(0.0))   # Σ h_x        (= H)
    c_L   = cot(pi / (2*L)) / L                                                                     # finite-L overlap
    hk    = (minus - plus) * (1 / (2im)) - c_L * Hmode
    @printf("[init-builder] variant=%s  c(L=%d)=%.8f  (TD limit 2/π=%.8f)\n",
            VARIANT_TAG, L, c_L, 2/pi)
    return hk
end

# ============ helpers ============
function measure_eps_nu(psi, oper_single, M_mpo)
    naive = Dict(:alg => :naive, :truncate => false)
    npsi2 = real(inner(psi, psi))
    CHpsi = apply(oper_single, psi; naive...)
    eps = real(inner(CHpsi, CHpsi)) / npsi2
    nu  = real(inner(psi, apply(M_mpo, psi; naive...))) / npsi2
    return eps, nu
end

function save_rolling(psi::MPS, path::String; extra::Dict{String,Any} = Dict{String,Any}())
    psi_cpu = SlowopDMRG.cpuarray(psi)
    tmp = path * ".tmp"
    h5open(tmp, "w") do f
        write(f, "psi", psi_cpu)
        for (k, v) in extra
            write(f, k, v)
        end
        write(f, "timestamp", string(now()))
    end
    mv(tmp, path; force = true)
end

# ============ main ============

SlowopDMRG.logwrite("=== H-orthogonal downward-ε scan  L=$L  variant=$VARIANT_TAG  scheduler=:eps_constrained  εᵢ=ε₀·(√3)⁻ⁱ ===")

sites = siteinds("Pauli", L)
oper_single, oper_for_dmrg, _, _ = SlowopDMRG.build_hamiltonian_operators(L, sites, H_DICT; pbc = false)
M_mpo = MPO(sites, "M")

psi = build_sinmc_hk(L, sites, H_DICT; k = K)
normalize!(psi)
psi = complex(psi)

psi         = SlowopDMRG.movedevice(psi)
oper_single = SlowopDMRG.movedevice(oper_single)
M_mpo       = SlowopDMRG.movedevice(M_mpo)

eps0, nu0 = measure_eps_nu(psi, oper_single, M_mpo)
@printf("[init] variant=%s  ε₀=⟨C_H²⟩=%.6g   ⟨M⟩=%.6g   maxlinkdim=%d\n",
        VARIANT_TAG, eps0, nu0, maxlinkdim(psi))

# Sanity: measure ⟨ψ|H⟩ (up to normalization) — should be numerically ~0 by construction.
H_mps = SlowopDMRG.build_operator_mps(L, sites, H_DICT; pbc = false, k = SlowopDMRG.DTYPE(0.0))
H_mps = SlowopDMRG.movedevice(complex(H_mps))
overlap_H = inner(H_mps, psi) / sqrt(real(inner(H_mps, H_mps)) * real(inner(psi, psi)))
@printf("[init] ⟨H|ψ⟩ (normalized) = %s   (should be ~0)\n", overlap_H)

# ============ auto-resume ============
# Every resubmit used to redo the scan from step 1. `done_step<NN>.h5` is
# written after each step completes; on startup we pick up from the highest one
# present. Set SLOWOP_FRESH=1 to ignore checkpoints and start clean.
#
# How many steps fit in a wall clock is NOT predictable from the cost of step 1:
# later steps have tighter ε, hence larger θ² and worse κ(B), so they are a
# harder numerical regime rather than merely a longer one. That unpredictability
# is exactly why losing completed steps to a resubmit is expensive.
#
# Note the ε ladder stays anchored on ε₀ measured from the freshly-built H_k
# above (not on the resumed state), so εᵢ = ε₀·(√3)⁻ⁱ is identical to a fresh
# run — that is why the builder still runs on the resume path.
const FORCE_FRESH = get(ENV, "SLOWOP_FRESH", "0") == "1"
done_path(step::Int) = joinpath(CHECKPOINT_DIR, @sprintf("done_step%02d.h5", step))

# NOTE: the previous `for s ...; resume_from = s` form was a silent no-op —
# in a script, a for-loop is soft scope and the assignment created a NEW LOCAL
# `resume_from`, leaving the global at 0, so EVERY resubmit restarted from step
# 1 (discovered when the L=12 χ=256 resume redid completed steps). Use a
# scope-safe functional form.
resume_from = FORCE_FRESH ? 0 :
    maximum(filter(s -> isfile(done_path(s)), 1:MAX_STEPS); init = 0)
if FORCE_FRESH
    @printf("[resume] SLOWOP_FRESH=1 — ignoring checkpoints, starting from step 1\n")
end

theta_sq_ema_seed = SlowopDMRG.DTYPE(THETA_WARM_START)
eps_now = eps0
nu_now  = nu0
# >0 ⇒ the loop's first step is warm-started from its OWN partial rolling checkpoint
# (an interrupted step), so it must not re-anneal ε — it is already near its target.
resumed_partial_step = 0

# A freshly HDF5-loaded MPS carries its own site Index objects (distinct from
# `sites` even at matching tag/dim). Re-tag onto the operators' `sites` so
# measurement contracts, then move to device. Shared by the done- and rolling-
# resume paths below.
function prepare_loaded_mps(psi_loaded)
    ls = siteinds(psi_loaded)
    for i in 1:L
        replaceind!(psi_loaded[i], ls[i], sites[i])
    end
    return SlowopDMRG.movedevice(psi_loaded)
end

if resume_from > 0
    ckpt = done_path(resume_from)
    @printf("[resume] found completed step %d — loading %s\n", resume_from, ckpt)
    psi_resume, loaded_ema, loaded_eps0 = h5open(ckpt, "r") do f
        (read(f, "psi", MPS),
         haskey(f, "theta_sq_ema") ? read(f, "theta_sq_ema") : nothing,
         haskey(f, "eps0")         ? read(f, "eps0")         : nothing)
    end

    # Guard: the ladder is anchored on ε₀. If the warm-start builder or H_DICT
    # changed since the checkpoint was written, the ε targets would silently
    # shift and the resumed scan would not be the scan it claims to be.
    if loaded_eps0 !== nothing && abs(loaded_eps0 - eps0) / eps0 > 1e-8
        error("[resume] ε₀ mismatch: checkpoint has $(loaded_eps0), this run computes $(eps0). " *
              "The ε ladder would shift. Re-run with SLOWOP_FRESH=1 or remove $(CHECKPOINT_DIR).")
    end

    if loaded_ema !== nothing
        theta_sq_ema_seed = SlowopDMRG.DTYPE(loaded_ema)
        @printf("[resume] carried theta_sq_ema = %.4g from checkpoint\n", theta_sq_ema_seed)
    end

    psi = prepare_loaded_mps(psi_resume)
    eps_now, nu_now = measure_eps_nu(psi, oper_single, M_mpo)
    @printf("[resume] state at step %d: ⟨C_H²⟩=%.6g  ⟨M⟩=%.6g  maxlinkdim=%d  (next target ε=%.6g)\n",
            resume_from, eps_now, nu_now, maxlinkdim(psi), eps0 * EPS_RATIO^(resume_from + 1))
end

# Partial-step resume. The interrupted step writes a per-sweep `rolling_step<NN>.h5`.
# If one exists for the next step (resume_from+1) and its ε-target matches THIS run's
# ladder, warm-start that step from the partial (unconverged) state instead of
# re-running it from scratch off the previous completed step. Without this, an
# interrupted step is fully redone every resubmit — so a walltime shorter than one
# step's wall makes zero progress. Disable with SLOWOP_ROLLING_RESUME=0.
if get(ENV, "SLOWOP_ROLLING_RESUME", "1") == "1"
    next_step = resume_from + 1
    rpath = joinpath(CHECKPOINT_DIR, @sprintf("rolling_step%02d.h5", next_step))
    if next_step <= MAX_STEPS && isfile(rpath)
        r_step, r_eps_target, r_ema, r_sweep = h5open(rpath, "r") do f
            (haskey(f, "step")         ? read(f, "step")         : nothing,
             haskey(f, "eps_target")   ? read(f, "eps_target")   : nothing,
             haskey(f, "theta_sq_ema") ? read(f, "theta_sq_ema") : nothing,
             haskey(f, "sweep")        ? read(f, "sweep")        : nothing)
        end
        expected_eps = eps0 * EPS_RATIO^next_step
        valid = (r_step == next_step) && (r_eps_target !== nothing) &&
                (abs(r_eps_target - expected_eps) / expected_eps < 1e-6)
        if valid
            psi_roll = h5open(rpath, "r") do f
                read(f, "psi", MPS)
            end
            psi = prepare_loaded_mps(psi_roll)
            if r_ema !== nothing
                theta_sq_ema_seed = SlowopDMRG.DTYPE(r_ema)
            end
            eps_now, nu_now = measure_eps_nu(psi, oper_single, M_mpo)
            resumed_partial_step = next_step
            @printf("[resume] partial rolling checkpoint for step %d found (through sweep %s) — warm-starting step %d from it: ⟨C_H²⟩=%.6g  ⟨M⟩=%.6g  θ²_ema=%.4g\n",
                    next_step, string(r_sweep), next_step, eps_now, nu_now, theta_sq_ema_seed)
        else
            @printf("[resume] rolling checkpoint for step %d present but does not match this ladder (step=%s eps_target=%s vs expected %.6g) — ignoring it.\n",
                    next_step, string(r_step), string(r_eps_target), expected_eps)
        end
    end
end

theta_sq_ema = Ref{SlowopDMRG.DTYPE}(theta_sq_ema_seed)

final_step = resume_from

for step in (resume_from + 1):MAX_STEPS
    global psi, eps_now, nu_now, final_step
    eps_target = eps0 * EPS_RATIO^step
    if eps_target < EPS_FLOOR
        @printf("\n[stop] next εᵢ=%.6g < floor=%.6g — halting scan\n", eps_target, EPS_FLOOR)
        break
    end
    final_step = step

    rolling_path = joinpath(CHECKPOINT_DIR, @sprintf("rolling_step%02d.h5", step))
    save_cb = (p_current, sweep_idx, energy) -> begin
        save_rolling(p_current, rolling_path;
            extra = Dict{String,Any}(
                "step"          => step,
                "sweep"         => sweep_idx,
                "energy"        => energy,
                "eps_target"    => Float64(eps_target),
                "theta_sq_ema"  => Float64(theta_sq_ema[]),
                "L"             => L,
                "variant"       => VARIANT_TAG,
            ),
        )
    end

    theta_seed = sqrt(max(theta_sq_ema[], SlowopDMRG.DTYPE(1e-6)))
    @printf("\n[step %d] target ε=%.6g   θ_seed=%.4g (θ²=%.4g)   warm-starting DMRG …\n",
            step, eps_target, theta_seed, theta_sq_ema[])
    t0 = time()
    psi_new = run_dmrg_for_size(L, H_DICT;
        psi0_in              = psi,
        theta                = theta_seed,
        num_ortho            = NUM_ORTHO,
        max_iter             = MAX_ITER,
        sweeps_per_iteration = SWEEPS_PER_ITER,
        initial_noise        = INITIAL_NOISE,
        final_noise          = FINAL_NOISE,
        scheduling           = :eps_constrained,
        local_solver         = LOCAL_SOLVER,
        epsilon_loc          = SlowopDMRG.DTYPE(eps_target),
        # ε-annealing: ramp ε from the previous step's target (this step's start)
        # down to eps_target over ANNEAL sweeps. ANNEAL=0 ⇒ quench (no ramp).
        # A partial-resumed step is already near its own eps_target, so re-loosening
        # it back to the previous step's target would throw away the partial progress:
        # for that one step start the anneal at eps_target (no ramp).
        eps_anneal_sweeps    = ANNEAL,
        eps_anneal_start     = SlowopDMRG.DTYPE(step == resumed_partial_step ? eps_target :
                                                eps0 * EPS_RATIO^(step - 1)),
        qcqp_theta_sq_max    = SlowopDMRG.DTYPE(THETA_SQ_MAX),
        qcqp_bisect_tol      = QCQP_BISECT_TOL,
        qcqp_warm_accept_tol = QCQP_WARM_ACCEPT_TOL,
        qcqp_warm_accept_tol_min = QCQP_WARM_ACCEPT_TOL_MIN,
        qcqp_use_secant      = QCQP_USE_SECANT,
        qcqp_thetasq_ema     = theta_sq_ema,
        qcqp_ema_alpha       = QCQP_EMA_ALPHA,
        qcqp_lobpcg_cheby_degree = LOBPCG_CHEBY_DEGREE,
        qcqp_lobpcg_maxiter      = LOBPCG_MAXITER,
        qcqp_lobpcg_rel_tol      = LOBPCG_REL_TOL,
        qcqp_lobpcg_inner_maxiter = LOBPCG_INNER_MAXITER,
        qcqp_lobpcg_max_restarts = LOBPCG_MAX_RESTARTS,
        qcqp_lobpcg_power_iters  = LOBPCG_POWER_ITERS,
        krylovdim            = KRYLOVDIM,
        maxdim_cap           = MAXDIM_CAP,
        override             = true,
        save_mps             = false,   # checkpoints are the done_step/rolling_step files
        save_callback        = save_cb,
    )
    walltime = time() - t0
    eps_prev = eps_now   # previous step's achieved ⟨C_H²⟩ (for wall detection)

    eps_new, nu_new = measure_eps_nu(psi_new, oper_single, M_mpo)
    @printf("[step %d] done.  ε_target=%.6g   ⟨C_H²⟩=%.6g   ⟨M⟩=%.6g   θ²_ema=%.4g   maxlinkdim=%d   wall=%.1fs\n",
            step, eps_target, eps_new, nu_new, theta_sq_ema[], maxlinkdim(psi_new), walltime)

    # χ representability wall: ⟨C_H²⟩ failed to fall this step (achieved ε barely
    # below the previous step's despite a √3-lower target) — θ² has run to its
    # ceiling and the state is frozen. A real step drops ε by ~1/√3≈0.58; a frozen
    # one keeps ratio ≈1. Flag it so we halt after recording this wall point
    # instead of grinding identical frozen steps to MAX_STEPS.
    walled = STOP_AT_WALL && isfinite(eps_prev) && eps_new > 0.98 * eps_prev

    psi     = psi_new
    eps_now = eps_new
    nu_now  = nu_new

    # Step-complete checkpoint — the last-good state a resubmit falls back to.
    # Written after the step's measurements. If a resubmit is interrupted DURING
    # the next step, the per-sweep `rolling_step` file for that step is used to
    # warm-start it (see the partial-step resume block above); this done_step is
    # the fallback when no valid rolling checkpoint exists.
    save_rolling(psi, done_path(step);
        extra = Dict{String,Any}(
            "step"         => step,
            "eps_target"   => Float64(eps_target),
            "eps_achieved" => Float64(eps_new),
            "nu"           => Float64(nu_new),
            "theta_sq_ema" => Float64(theta_sq_ema[]),
            "eps0"         => Float64(eps0),
            "L"            => L,
            "variant"      => VARIANT_TAG,
        ),
    )

    if walled
        logwrite("[wall] step $step froze at ⟨C_H²⟩=$(eps_new) (prev $(eps_prev), " *
                 "target $(eps_target)) — χ=$(MAXDIM_CAP) representability wall reached; " *
                 "halting ladder early (skipping the frozen tail to step $MAX_STEPS).")
        break
    end
end

@printf("\n=== scan complete: variant=%s  steps=%d  final ⟨C_H²⟩=%.6g  ⟨M⟩=%.6g ===\n",
        VARIANT_TAG, final_step, eps_now, nu_now)
