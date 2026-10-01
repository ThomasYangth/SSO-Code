"""
    SlowopDMRG

DMRG/MPS variational search for *simple slow operators* (approximately conserved
quantities) of a spin chain. **Documented as an unsuccessful approach** in the
SSO-Code repository — see `docs/dmrg_attempt.md`.

An operator `O` on `L` qubits is stored as an MPS in the 4^L Pauli-string basis
(site type `"Pauli"`, local basis `I, X, Y, Z`). The quantities are

* `ν = ⟨O|M|O⟩/⟨O|O⟩`, `M = ⊗ diag(1, 1/3, 1/3, 1/3)` — weight-`w` Pauli strings
  get `3^{-w}` (the "simplicity" of `O`);
* `⟨C_H²⟩ = ‖[H,O]‖²/‖O‖²`, `C_H = H_L − H_R` the commutator superoperator;
* `τ = 1/√⟨C_H²⟩` (static-Hamiltonian convention).

The production path (`scheduling = :eps_constrained`, `local_solver = :lobpcg`)
maximizes ν subject to `⟨C_H²⟩ ≤ ε` and `O ⊥ {I, H, H², H³}` by two-site DMRG
sweeps. At every two-site block it solves a local QCQP whose Lagrange
multiplier θ² is located by secant/bisection; each inner solve is the
generalized eigenproblem `−M v = λ (I + θ² C_H²) v` by LOBPCG with a
Chebyshev-in-B preconditioner.

Configuration is read from environment variables at load time (`__init__`):

| variable              | default                                  | meaning                          |
|-----------------------|------------------------------------------|----------------------------------|
| `SLOWOP_DATA_DIR`     | `\$SSO_OUTPUT/dmrg` (`<repo>/output/dmrg`) | checkpoints, MPS files, CSVs     |
| `SLOWOP_USE_GPU`      | `1`                                      | use CUDA when the extension loads |
| `SLOWOP_DEBUG`        | `0`                                      | per-block debug log files        |
| `SLOWOP_DEBUG_LOG_DIR`| `<data dir>/debug_logs`                  | where debug logs go              |

GPU support is a package extension: `using CUDA` *before* `using SlowopDMRG`.
"""
module SlowopDMRG

using LinearAlgebra
using Printf
using Dates
using Statistics
using Logging
using Random
import Printf: @sprintf

using ITensors
using ITensorMPS
using HDF5
using DataFrames
using CSV
using FFTW
using KrylovKit
using IterativeSolvers
using IterTools

# ============================================================================
# Configuration (runtime, from the environment)
# ============================================================================

"""Repository root `<repo>` (this file lives in `<repo>/julia/SlowopDMRG/src`)."""
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

const _DATA_DIR      = Ref{String}("")
const _DEBUG_LOG_DIR = Ref{String}("")
const _DEBUG_MODE    = Ref(false)
const _USE_GPU       = Ref(true)

"""
    data_dir() -> String

Directory for MPS checkpoints, saved MPS files and tracking CSVs:
`\$SLOWOP_DATA_DIR`, else `\$SSO_OUTPUT/dmrg`, else `<repo>/output/dmrg`.
"""
data_dir() = _DATA_DIR[]

"""Directory for debug logs (only written when `SLOWOP_DEBUG=1`)."""
debug_log_dir() = _DEBUG_LOG_DIR[]

"""`true` when `SLOWOP_DEBUG=1`: `dmrg_sweeps!` then writes per-block diagnostics."""
debug_mode() = _DEBUG_MODE[]

function _load_config!()
    out = get(ENV, "SSO_OUTPUT", joinpath(REPO_ROOT, "output"))
    _DATA_DIR[] = get(ENV, "SLOWOP_DATA_DIR", joinpath(out, "dmrg"))
    _DEBUG_LOG_DIR[] = get(ENV, "SLOWOP_DEBUG_LOG_DIR", joinpath(_DATA_DIR[], "debug_logs"))
    _DEBUG_MODE[] = get(ENV, "SLOWOP_DEBUG", "0") == "1"
    _USE_GPU[] = get(ENV, "SLOWOP_USE_GPU", "1") == "1"
    mkpath(_DATA_DIR[])
    _DEBUG_MODE[] && mkpath(_DEBUG_LOG_DIR[])
    return nothing
end

# Floating-point precision. All production runs were Float64/ComplexF64.
const DTYPE = Float64
const CDTYPE = ComplexF64

# ============================================================================
# Logging helpers
# ============================================================================

const start_time = Ref(now())

"""Timestamped `println` to stdout (flushed), used for all run logs."""
function logwrite(msg...)
    elapsed_ms = (now() - start_time[]).value
    elapsed_str = if elapsed_ms < 1000
        "$(elapsed_ms) ms"
    elseif elapsed_ms < 60000
        "$(round(elapsed_ms/1000, digits=2)) s"
    elseif elapsed_ms < 3600000
        mins = div(elapsed_ms, 60000)
        secs = round((elapsed_ms % 60000)/1000, digits=2)
        "$(mins) min $(secs) s"
    else
        hours = div(elapsed_ms, 3600000)
        mins = div(elapsed_ms % 3600000, 60000)
        secs = round((elapsed_ms % 60000)/1000, digits=2)
        "$(hours) h $(mins) min $(secs) s"
    end
    println("[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] | Elapsed: $(elapsed_str) | ", msg...)
    flush(stdout)
end

include("debug_logger.jl")

# ============================================================================
# Device handling. CUDA is a weak dependency; the extension installs a
# `CUDABackend` into `BACKEND[]` from its `__init__` and adds methods to
# `_to_device` / `_to_host`. Without it everything stays on the CPU.
# ============================================================================

abstract type DeviceBackend end
struct CPUBackend <: DeviceBackend end
const BACKEND = Ref{DeviceBackend}(CPUBackend())

"""`true` iff the CUDA extension is loaded, functional, and `SLOWOP_USE_GPU=1`."""
flag_use_gpu() = _USE_GPU[] && !(BACKEND[] isa CPUBackend)

_to_device(::CPUBackend, obj) = obj
_to_host(::CPUBackend, obj) = obj

"""Move an Array / ITensor / MPS / MPO to the compute device (no-op on CPU)."""
movedevice(obj) = flag_use_gpu() ? _to_device(BACKEND[], obj) : obj

"""Bring an object back to host memory (no-op on CPU)."""
cpuarray(obj) = flag_use_gpu() ? _to_host(BACKEND[], obj) : obj

function __init__()
    _load_config!()
    global_logger(ConsoleLogger(stderr, Logging.Info))
    logwrite("SlowopDMRG: data_dir=$(data_dir())  debug=$(debug_mode())  SLOWOP_USE_GPU=$(_USE_GPU[])")
end

# ============================================================================
# Core: sweeper, local primitives, persistence
# ============================================================================
include("core/dmrg_sweeper.jl")
include("core/local_qcqp.jl")
include("core/local_geneig.jl")
include("core/persistence.jl")

# Physics models and DMRG driver (dmrgmodel.jl includes the submodules)
include("models/dmrgmodel.jl")
include("models/observables.jl")

# ============================================================================
# Public API
# ============================================================================

export run_dmrg_for_size, dmrg_sweeps!

export saveMPS, loadMPS, load_existing_file, load_all_metadata,
       encode_filename, decode_filename

export build_super_operators, build_floquet_superoperator,
       build_hamiltonian_operators, build_floquet_operators,
       build_krylov_basis_raw, build_operator_mps, adjoint_mpo

export exact_ground_state

export get_size_distribution, calculate_k_space_decomposition

export logwrite, movedevice, cpuarray, sanity_check, flag_use_gpu,
       data_dir, debug_mode

export DebugLogger, ActiveDebugLogger, NoOpDebugLogger, create_debug_logger,
       debuglog, debug_timing_start, debug_timing_end, debug_gpu_check,
       debug_tensor_info, close_debug_logger

end # module SlowopDMRG
