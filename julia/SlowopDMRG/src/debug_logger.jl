#=
Debug logger infrastructure. `dmrg_sweeps!` and the local solvers call
`debuglog`/`debug_timing_*` unconditionally; with `NoOpDebugLogger` (the
default, `SLOWOP_DEBUG=0`) these compile to no-ops. With `SLOWOP_DEBUG=1` an
`ActiveDebugLogger` writes per-sweep timing breakdowns, per-block QCQP/LOBPCG
diagnostics and device-placement checks to `debug_log_dir()`.
=#

"""
    DebugLogger

Abstract type for debug logging. Implementations include:
- `ActiveDebugLogger`: Writes detailed diagnostics to file
- `NoOpDebugLogger`: Does nothing (for production runs)
"""
abstract type DebugLogger end

"""
    ActiveDebugLogger(log_file::String)

Debug logger that writes to a file with timing, GPU diagnostics, and performance metrics.
"""
mutable struct ActiveDebugLogger <: DebugLogger
    log_file::String
    file_handle::Union{IOStream, Nothing}
    start_time::DateTime
    indent_level::Int

    function ActiveDebugLogger(log_file::String)
        mkpath(dirname(log_file))
        fh = open(log_file, "w")
        logger = new(log_file, fh, now(), 0)
        write_header(logger)
        return logger
    end
end

"""
    NoOpDebugLogger()

Debug logger that does nothing (used when `SLOWOP_DEBUG` is not `1`).
"""
struct NoOpDebugLogger <: DebugLogger end

function write_header(logger::ActiveDebugLogger)
    println(logger.file_handle, "="^80)
    println(logger.file_handle, "SlowopDMRG Debug Log")
    println(logger.file_handle, "Started: $(Dates.format(logger.start_time, "yyyy-mm-dd HH:MM:SS"))")
    println(logger.file_handle, "="^80)
    println(logger.file_handle, "")
    println(logger.file_handle, "Configuration:")
    println(logger.file_handle, "  flag_use_gpu(): $(flag_use_gpu())")
    println(logger.file_handle, "  backend: $(typeof(BACKEND[]))")
    println(logger.file_handle, "  OMP_NUM_THREADS: $(get(ENV, "OMP_NUM_THREADS", "unset"))")
    println(logger.file_handle, "")
    flush(logger.file_handle)
end

function debuglog(logger::ActiveDebugLogger, msg...; indent::Union{Int,Nothing}=nothing)
    if indent !== nothing
        logger.indent_level = indent
    end
    elapsed_ms = (now() - logger.start_time).value
    elapsed_str = if elapsed_ms < 1000
        @sprintf("%.3f ms", elapsed_ms)
    elseif elapsed_ms < 60000
        @sprintf("%.2f s", elapsed_ms/1000)
    elseif elapsed_ms < 3600000
        mins = div(elapsed_ms, 60000)
        secs = (elapsed_ms % 60000)/1000
        @sprintf("%d:%05.2f", mins, secs)
    else
        hours = div(elapsed_ms, 3600000)
        mins = div(elapsed_ms % 3600000, 60000)
        secs = (elapsed_ms % 60000)/1000
        @sprintf("%d:%02d:%05.2f", hours, mins, secs)
    end

    indent_str = "  " ^ logger.indent_level
    println(logger.file_handle, "[$elapsed_str] $indent_str", msg...)
    flush(logger.file_handle)
end

# No-op implementations
debuglog(logger::NoOpDebugLogger, msg...; indent::Union{Int,Nothing}=nothing) = nothing

function debug_timing_start(logger::ActiveDebugLogger, label::String)
    debuglog(logger, "▶ $label")
    logger.indent_level += 1
    return time()
end

debug_timing_start(logger::NoOpDebugLogger, label::String) = time()

function debug_timing_end(logger::ActiveDebugLogger, label::String, t_start::Float64)
    logger.indent_level -= 1
    elapsed = time() - t_start
    debuglog(logger, "◀ $label: $(round(elapsed, digits=4))s")
    return elapsed
end

debug_timing_end(logger::NoOpDebugLogger, label::String, t_start::Float64) = time() - t_start

function debug_gpu_check(logger::ActiveDebugLogger, var_name::String, obj)
    obj_type = typeof(obj)
    type_name = string(obj_type)

    # Check if type contains "CuArray" or "CUDA" in the name (works without importing CUDA types)
    is_gpu = flag_use_gpu() && (contains(type_name, "CuArray") || contains(type_name, "CUDA"))

    # Additional check: if it has a .data field, check that too
    if !is_gpu && hasproperty(obj, :data)
        data_type_name = string(typeof(getfield(obj, :data)))
        is_gpu = contains(data_type_name, "CuArray") || contains(data_type_name, "CUDA")
    end

    location = is_gpu ? "GPU" : "CPU"
    debuglog(logger, "  $var_name: type=$obj_type, location=$location, size=$(sizeof(obj)) bytes")
end

debug_gpu_check(logger::NoOpDebugLogger, var_name::String, obj) = nothing

function debug_tensor_info(logger::ActiveDebugLogger, var_name::String, tensor)
    # Check if this is an ITensor by type name (more robust than hasproperty)
    type_name = string(typeof(tensor))

    if contains(type_name, "ITensor")
        # This is an ITensor - check its storage
        is_gpu = false
        storage_detail = "unknown"

        try
            # Access ITensor storage via NDTensors interface
            if hasproperty(tensor, :store) && tensor.store !== nothing
                store = tensor.store
                store_type_name = string(typeof(store))

                # Check if storage contains data array
                if hasproperty(store, :data)
                    data = store.data
                    data_type_name = string(typeof(data))
                    is_gpu = contains(data_type_name, "CuArray") || contains(data_type_name, "CUDA")
                    storage_detail = data_type_name
                else
                    # No data field, check store type itself
                    is_gpu = contains(store_type_name, "CuArray") || contains(store_type_name, "CUDA")
                    storage_detail = store_type_name
                end
            else
                storage_detail = "no store found"
            end
        catch e
            storage_detail = "error accessing storage: $(e)"
        end

        location = is_gpu ? "GPU" : "CPU"
        debuglog(logger, "  $var_name: ITensor, location=$location, storage=$storage_detail")
    else
        debug_gpu_check(logger, var_name, tensor)
    end
end

debug_tensor_info(logger::NoOpDebugLogger, var_name::String, tensor) = nothing

function close_debug_logger(logger::ActiveDebugLogger)
    if logger.file_handle !== nothing
        println(logger.file_handle, "")
        println(logger.file_handle, "="^80)
        println(logger.file_handle, "Debug log ended: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(logger.file_handle, "="^80)
        close(logger.file_handle)
        logger.file_handle = nothing
    end
end

close_debug_logger(logger::NoOpDebugLogger) = nothing

"""
    create_debug_logger(name::String="debug")

Returns an `ActiveDebugLogger` writing to `debug_log_dir()` when
`debug_mode()` is on, a `NoOpDebugLogger` otherwise.
"""
function create_debug_logger(name::String="debug")
    if debug_mode()
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        log_file = joinpath(debug_log_dir(), "$(name)_$(timestamp).log")
        return ActiveDebugLogger(log_file)
    else
        return NoOpDebugLogger()
    end
end
