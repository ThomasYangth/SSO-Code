"""
    sanity_check(obj)

Assert that an MPS / MPO / ITensor / array holds no NaN or Inf and no entry
with `|x| ≥ 1e5`. Used while building the Krylov tower `{I, H, H², …}` to catch
overflow before it reaches the sweeps.
"""
function sanity_check(obj)
    if obj isa MPS || obj isa MPO
        for block in obj
            sanity_check(array(block))
        end
    elseif obj isa ITensor
        sanity_check(array(obj))
    elseif obj isa AbstractArray
        @assert !any(isnan, obj) "Array contains NaN values!"
        @assert !any(isinf, obj) "Array contains Inf values!"
        @assert maximum(abs, obj) < 1E5 "Array has excessively large values!"
    end
end
