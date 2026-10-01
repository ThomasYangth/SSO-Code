#=
One ε-constrained DMRG step from a saved MPS, at a (usually larger) bond
dimension. Replaces three near-identical original drivers:

  SLOWOP_MODE=refine  — re-solve at the SOURCE's own ε target, no annealing
                        (original scripts/ed_chi_refine.jl: "χ refinement").
  SLOWOP_MODE=next    — advance one ladder step, ε_target = ε_src/√3, annealed
                        from ε_src over SLOWOP_ANNEAL sweeps
                        (original scripts/ed_step9_anneal.jl and
                        scripts/ed_onestep_anneal.jl).

SLOWOP_EPS_TARGET / SLOWOP_EPS_START override the targets explicitly. The
solver settings are those of the original drivers (θ²_max = 1e14,
bisect_tol = 5e-3, warm_accept_tol = 0.1, krylovdim = 30, noise 1e-6 → 1e-10,
num_ortho = 3, LOBPCG defaults); θ² is warm-started from the source's
`theta_sq_ema`.

Source: SLOWOP_SRC=<h5 with "psi">, or SLOWOP_SRCREP + SLOWOP_STEP for
  <data_dir()>/hk_scan_cmp/L{L}_lobpcg_sinmc_rep{SRCREP}/done_step{STEP}.h5
Output: SLOWOP_OUT (default <data_dir()>/hk_scan_cmp/L{L}_onestep/{TAG}.h5) with
  psi, eps_target, eps_achieved (=⟨C_H²⟩), nu (=⟨M⟩), eps_anneal_start,
  anneal_sweeps, chi, source, timestamp.

Env: SLOWOP_L (12), SLOWOP_CHI (256), SLOWOP_SWEEPS (6), SLOWOP_ANNEAL (1),
     SLOWOP_TAG, USE_CUDA=1 for the GPU extension.
=#
if get(ENV, "USE_CUDA", "0") == "1"
    using CUDA
end
using SlowopDMRG, ITensors, ITensorMPS, HDF5, Printf, Dates

const L       = parse(Int, get(ENV, "SLOWOP_L", "12"))
const H_DICT  = Dict("ZZ" => 1.0, "X" => 0.905, "Z" => 0.809)
const CHI     = parse(Int, get(ENV, "SLOWOP_CHI", "256"))
const NSWEEP  = parse(Int, get(ENV, "SLOWOP_SWEEPS", "6"))
const ANNEAL  = parse(Int, get(ENV, "SLOWOP_ANNEAL", "1"))
const MODE    = get(ENV, "SLOWOP_MODE", "next")
MODE in ("refine", "next") || error("SLOWOP_MODE must be refine or next, got $MODE")

const SRC = if haskey(ENV, "SLOWOP_SRC")
    ENV["SLOWOP_SRC"]
else
    joinpath(SlowopDMRG.data_dir(), "hk_scan_cmp",
             "L$(L)_lobpcg_sinmc_rep$(ENV["SLOWOP_SRCREP"])",
             @sprintf("done_step%02d.h5", parse(Int, ENV["SLOWOP_STEP"])))
end
isfile(SRC) || error("no source MPS at $SRC")

psi0, eps_src, tsq_warm = h5open(SRC, "r") do f
    (read(f, "psi", MPS),
     haskey(f, "eps_target") ? read(f, "eps_target") : nothing,
     haskey(f, "theta_sq_ema") ? read(f, "theta_sq_ema") : 1.0)
end

const EPS_TARGET = haskey(ENV, "SLOWOP_EPS_TARGET") ? parse(Float64, ENV["SLOWOP_EPS_TARGET"]) :
    (eps_src === nothing ? error("source has no eps_target; set SLOWOP_EPS_TARGET") :
     MODE == "refine" ? eps_src : eps_src * (1 / sqrt(3)))
# eps_anneal_start = 0 disables annealing (the refine mode, = ed_chi_refine.jl).
const EPS_START = haskey(ENV, "SLOWOP_EPS_START") ? parse(Float64, ENV["SLOWOP_EPS_START"]) :
    (MODE == "refine" ? 0.0 : eps_src)
const TAG = get(ENV, "SLOWOP_TAG", @sprintf("%s_chi%d_eps%.6g", MODE, CHI, EPS_TARGET))
const OUT = get(ENV, "SLOWOP_OUT", joinpath(SlowopDMRG.data_dir(), "hk_scan_cmp",
                                            "L$(L)_onestep", "$(TAG).h5"))

sites = siteinds("Pauli", L)
ss = siteinds(psi0)
for i in 1:L
    replaceind!(psi0[i], ss[i], sites[i])
end
psi0 = SlowopDMRG.movedevice(psi0)
CH, _, _, _ = build_hamiltonian_operators(L, sites, H_DICT; pbc = false)
CH = SlowopDMRG.movedevice(CH)
M  = SlowopDMRG.movedevice(MPO(sites, "M"))
meas(p) = (n = real(inner(p, p)); ch = apply(CH, p; alg = :naive, truncate = false);
           (real(inner(ch, ch)) / n, real(inner(p, apply(M, p; alg = :naive, truncate = false))) / n))

e0, nu0 = meas(psi0)
@printf("[one_step %s] mode=%s chi=%d sweeps=%d anneal=%d  src: ε=%.6g ⟨M⟩=%.6g χ=%d  ε_start=%.6g → ε_target=%.6g  θ²_warm=%.4g\n",
        TAG, MODE, CHI, NSWEEP, ANNEAL, e0, nu0, maxlinkdim(psi0), EPS_START, EPS_TARGET, tsq_warm)

theta_seed = sqrt(max(SlowopDMRG.DTYPE(tsq_warm), SlowopDMRG.DTYPE(1e-6)))
t0 = time()
psi = run_dmrg_for_size(L, H_DICT;
    psi0_in = psi0, theta = theta_seed, num_ortho = 3, max_iter = 1,
    sweeps_per_iteration = NSWEEP, initial_noise = 1e-6, final_noise = 1e-10,
    scheduling = :eps_constrained, local_solver = :lobpcg,
    epsilon_loc = SlowopDMRG.DTYPE(EPS_TARGET),
    eps_anneal_sweeps = ANNEAL, eps_anneal_start = SlowopDMRG.DTYPE(EPS_START),
    qcqp_theta_sq_max = SlowopDMRG.DTYPE(1e14), qcqp_bisect_tol = 5e-3,
    qcqp_warm_accept_tol = 1e-1, krylovdim = 30, maxdim_cap = CHI,
    override = true, save_mps = false)
wall = time() - t0
e, nu = meas(psi)
@printf("ONESTEP %s mode=%s chi=%d ε_target=%.6g | src: ε=%.6g ⟨M⟩=%.6g | result: ε=%.6g ⟨M⟩=%.6g dnu=%+.6g maxchi=%d wall=%.0fs\n",
        TAG, MODE, CHI, EPS_TARGET, e0, nu0, e, nu, nu - nu0, maxlinkdim(psi), wall)

mkpath(dirname(OUT))
h5open(OUT, "w") do f
    write(f, "psi", SlowopDMRG.cpuarray(psi))
    write(f, "eps_target", Float64(EPS_TARGET))
    write(f, "eps_achieved", Float64(e))
    write(f, "nu", Float64(nu))
    write(f, "eps_anneal_start", Float64(EPS_START))
    write(f, "anneal_sweeps", ANNEAL)
    write(f, "chi", CHI)
    write(f, "source", SRC)
    write(f, "timestamp", string(now()))
end
println("[saved] $OUT")
