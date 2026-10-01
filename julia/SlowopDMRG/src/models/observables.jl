
"""
    get_size_distribution(psi::MPS; maxsize=length(psi)) -> Vector

Operator-size (Pauli-weight) distribution of the operator `psi`: entry `w+1`
is the squared weight `Σ_{|P|=w} |c_P|²` on Pauli strings with exactly `w`
non-identity factors (`w = 0..maxsize`); the last entry collects weight above
`maxsize`. Computed by a transfer-matrix contraction (no enumeration).
"""
function get_size_distribution(psi::MPS; maxsize::Int=-1)
    if maxsize < 0
        maxsize = length(psi)
    end
    counter_tensor = zeros(eltype(psi[1]), maxsize+2, maxsize+2, 4, 4)
    for i = 1:maxsize+2
        counter_tensor[i, i, 1, 1] = 1
    end
    for i = 1:maxsize+1
        for j = 2:4
            counter_tensor[i, i+1, j, j] = 1
        end
    end
    for j = 2:4
        counter_tensor[maxsize+2, maxsize+2, j, j] = 1
    end
    link = Index(maxsize+2; tags="counter_link")
    # Initialize the counter vector
    counter_vector = zeros(eltype(counter_tensor), maxsize+2)
    # counter_vector counts the weight of operators with size i for i=0:maxsize, and the final element is the residual weight.
    counter_vector[1] = 1
    counter_vector = ITensor(counter_vector, link)
    counter_vector = movedevice(counter_vector)
    counter_tensor = movedevice(counter_tensor)
    # Contract
    for j = 1:length(psi)
        counter_vector = replaceinds(
                counter_vector * ITensor(counter_tensor, link, link', siteind(psi, j), siteind(psi, j)')  
                    * psi[j] * dag(psi[j])',
                link'=>link
        )
    end
    return real.(array(counter_vector, link))
end


using FFTW

#=
-------------------------------------------------------------------------------
K-SPACE DECOMPOSITION CALCULATION
-------------------------------------------------------------------------------
Function to calculate the k-space decomposition of an MPS using translational
symmetry analysis.
=#

"""
    calculate_k_space_decomposition(psi::MPS)

Calculate the k-space decomposition of an MPS by projecting onto momentum sectors.

For a length-L chain with translational operator T, this function computes the 
weights in each momentum sector k by constructing:
    |ψ_k⟩ = (1/√L) * Σ_x exp(-ikx) * T^x|ψ⟩

The weight in k-sector is given by ||ψ_k||² = Σ_x exp(-ikx) * ⟨ψ|T^x|ψ⟩,
which is guaranteed to be real and is obtained via FFT of translation overlaps.

Parameters:
- psi: Input MPS to decompose

Returns:
- k_weights: Vector of real weights (squared magnitudes) for each k-sector
- k_values: Vector of momentum values k = 2π*n/L for n = 0, 1, ..., L-1
"""
function calculate_k_space_decomposition(psi::MPS)
    L = length(psi)
    
    # Initialize arrays for results
    k_values = [2π * n / L for n in 0:(L-1)]
    
    logwrite("Calculating k-space decomposition for MPS of length $L")
    
    # Calculate overlaps <psi| T^x |psi> using symmetry <psi|T^(L-x)|psi> = <psi|T^x|psi>*
    translation_overlaps = Vector{CDTYPE}(undef, L)
    
    logwrite("Computing translation overlaps using conjugate symmetry...")
    
    for x in 0:(L-1)
        if x == 0
            # Identity case
            translation_overlaps[x+1] = inner(psi, psi)
            logwrite("  T^$x overlap: $(translation_overlaps[x+1])")
        elseif x <= div(L, 2)
            # Compute directly for first half (and middle point if L is even)
            translation_overlaps[x+1] = translation_overlap(psi, x)
            logwrite("  T^$x overlap: $(translation_overlaps[x+1])")
        else
            # Use conjugate symmetry: <psi|T^x|psi> = <psi|T^(L-x)|psi>*
            mirror_x = L - x
            translation_overlaps[x+1] = conj(translation_overlaps[mirror_x+1])
            logwrite("  T^$x overlap: $(translation_overlaps[x+1]) (from T^$mirror_x conjugate)")
        end
    end
    
    # Calculate k-space weights using FFT
    logwrite("Computing k-space weights using FFT...")
    
    # The k-weights are ||ψ_k||² = Σ_x exp(-ikx) * ⟨ψ|T^x|ψ⟩, obtained by FFT
    k_weights_complex = fft(translation_overlaps)
    k_weights = real.(k_weights_complex) ./ L  # Should be real by construction
    
    for (k_idx, k) in enumerate(k_values)
        logwrite("  k = $(round(k, digits=4)): weight = $(round(k_weights[k_idx], digits=6))")
        if abs(imag(k_weights_complex[k_idx])) > 1e-12
            @warn "k-weight has significant imaginary part: $(imag(k_weights_complex[k_idx]))"
        end
    end

    # Verify normalization: sum of weights should equal <psi|psi>
    total_weight = sum(k_weights)
    expected_total = real(translation_overlaps[1])
    
    logwrite("Normalization check:")
    logwrite("  Sum of k-weights: $(round(total_weight, digits=8))")
    logwrite("  Expected ⟨ψ|ψ⟩): $(round(expected_total, digits=8))")
    logwrite("  Ratio: $(round(total_weight/expected_total, digits=8))")
    
    return k_weights, k_values
end

"""
    translation_overlap(psi::MPS, translation::Int)

Calculate <psi|T^translation|psi> without constructing the translated MPS.
This is done by properly contracting the tensor network with cyclic permutation.

Parameters:
- psi: Input MPS
- translation: Number of sites to translate

Returns:
- Complex overlap value
"""
function translation_overlap(psi::MPS, translation::Int)
    L = length(psi)
    translation = mod(translation, L)
    
    if translation == 0
        return inner(psi, psi)
    end
    
    # For translation by x, we need to contract:
    # <psi| T^x |psi> = sum over physical indices with cyclic shift
    
    # Start contraction from the right
    R = ITensor(1.0)  # Initialize with scalar 1

    sites = siteinds(psi)
    
    # Contract from right to left, with physical index permutation
    for i in L:-1:1
        # Site i in <psi| contracts with the physical state that should be at site i
        # under translation T^x. This is the state that was originally at site (i-x) mod L
        j = mod(i - 1 - translation, L) + 1
        
        # Get tensors - bra from position i, ket from position j
        tensor_bra = dag(psi[i])'  # <psi| at site i
        tensor_ket = psi[j]       # |psi> from the site that translates to position i
        
        # Contract and accumulate with right environment
        R = R * tensor_bra * delta(sites[i]', sites[j]) * tensor_ket
    end
    
    # The result should be a scalar
    return scalar(R)
end