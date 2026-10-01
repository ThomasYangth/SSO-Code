"""
    SlowopDMRGCUDAExt

GPU backend for SlowopDMRG, loaded automatically when `CUDA` is imported
before `SlowopDMRG`. Installs `CUDABackend()` into `SlowopDMRG.BACKEND`, after
which `movedevice` puts arrays / ITensors / MPS / MPO on the GPU (ComplexF64 is
kept, no Float32 downcast) and `cpuarray` brings them back.

Only the tensor contractions of the environments and of the local matvecs run
on the device; LOBPCG / Chebyshev / bisection bookkeeping stays on the host
(see `local_step_qcqp`).
"""
module SlowopDMRGCUDAExt

using CUDA
using SlowopDMRG
using ITensors
using ITensorMPS
using ITensors: NDTensors

struct CUDABackend <: SlowopDMRG.DeviceBackend end

function __init__()
    if CUDA.functional()
        SlowopDMRG.BACKEND[] = CUDABackend()
        @info "SlowopDMRG CUDA extension: GPU backend active on $(CUDA.device()); flag_use_gpu()=$(SlowopDMRG.flag_use_gpu())"
    else
        @warn "SlowopDMRG CUDA extension loaded but CUDA.functional() == false; staying on CPU"
    end
    flush(stderr)
end

function SlowopDMRG._to_device(::CUDABackend, obj)
    if obj isa CuArray
        return obj
    elseif obj isa Array
        return CuArray(obj)
    elseif obj isa ITensor
        return NDTensors.cu(obj)
    elseif obj isa MPS
        return MPS([SlowopDMRG._to_device(CUDABackend(), t) for t in obj])
    elseif obj isa MPO
        return MPO([SlowopDMRG._to_device(CUDABackend(), t) for t in obj])
    else
        @warn "movedevice: unsupported type $(typeof(obj)); returning it unchanged"
        return obj
    end
end

function SlowopDMRG._to_host(::CUDABackend, obj)
    if obj isa ITensor
        return ITensor(SlowopDMRG._to_host(CUDABackend(), array(obj)), inds(obj)...)
    elseif obj isa CuArray
        return Array(obj)
    elseif obj isa Array
        return obj
    elseif obj isa MPS
        return MPS([SlowopDMRG._to_host(CUDABackend(), t) for t in obj])
    elseif obj isa MPO
        return MPO([SlowopDMRG._to_host(CUDABackend(), t) for t in obj])
    else
        throw(ArgumentError("cpuarray: unsupported type $(typeof(obj))"))
    end
end

end # module
