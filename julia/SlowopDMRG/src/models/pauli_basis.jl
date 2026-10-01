#=
===============================================================================
PAULI BASIS: CUSTOM ITENSOR SITE TYPE
===============================================================================

The "Pauli" site type defines a 4-dimensional Hilbert space representing the
operator basis ``\\{I, X, Y, Z\\}`` at each site. This allows us to represent operators
as MPS rather than quantum states.

HILBERT SPACE STRUCTURE:
- Dimension: 4
- Basis states:
  ``|1\\rangle \\equiv I`` (identity operator)
  ``|2\\rangle \\equiv X`` (Pauli-X operator)
  ``|3\\rangle \\equiv Y`` (Pauli-Y operator)
  ``|4\\rangle \\equiv Z`` (Pauli-Z operator)

SUPER-OPERATOR MATRICES:
The matrices `mat_XL`, `mat_YL`, `mat_ZL` implement LEFT multiplication by Pauli
matrices in the operator basis. That is, `mat_XL` applied to coefficients ``(a,b,c,d)``
representing operator ``O = a\\cdot I + b\\cdot X + c\\cdot Y + d\\cdot Z`` yields the coefficients of ``X\\cdot O``.

Similarly, `mat_XR`, `mat_YR`, `mat_ZR` implement RIGHT multiplication.

These matrices are derived from the Pauli multiplication table:
```math
\\begin{aligned}
  X\\cdot X &= I, \\quad Y\\cdot Y = I, \\quad Z\\cdot Z = I \\\\
  X\\cdot Y &= iZ, \\quad Y\\cdot Z = iX, \\quad Z\\cdot X = iY \\quad \\text{(cyclic)} \\\\
  Y\\cdot X &= -iZ, \\quad Z\\cdot Y = -iX, \\quad X\\cdot Z = -iY \\quad \\text{(anti-cyclic)}
\\end{aligned}
```

For example, `mat_XL` tells us:
- ``X\\cdot I = X``  (row 2, col 1 is 1)
- ``X\\cdot X = I``  (row 1, col 2 is 1)
- ``X\\cdot Y = iZ`` (row 4, col 3 is i)
- ``X\\cdot Z = -iY`` (row 3, col 4 is -i)
=#

function ITensors.space(::SiteType"Pauli")
    return 4
end

# Basis states for the Pauli operator space
function ITensors.state(::StateName"I", ::SiteType"Pauli")
    v = zeros(4)
    v[1] = 1.0  # |I⟩ = (1, 0, 0, 0)
    return v
end

function ITensors.state(::StateName"X", ::SiteType"Pauli")
    v = zeros(4)
    v[2] = 1.0  # |X⟩ = (0, 1, 0, 0)
    return v
end

function ITensors.state(::StateName"Y", ::SiteType"Pauli")
    v = zeros(4)
    v[3] = 1.0  # |Y⟩ = (0, 0, 1, 0)
    return v
end

function ITensors.state(::StateName"Z", ::SiteType"Pauli")
    v = zeros(4)
    v[4] = 1.0  # |Z⟩ = (0, 0, 0, 1)
    return v
end

function ITensors.state(::StateName"P", ::SiteType"Pauli")
    v = zeros(4)
    v[1] = 0.5  # P = (I + Z)/2
    v[4] = 0.5  # |P⟩ = (0.5, 0, 0, 0.5)
    return v
end

# Super-operator matrices for LEFT multiplication: O ↦ σ·O
# Rows/cols ordered as (I, X, Y, Z)
const mat_XL = [0 1 0 0; 1 0 0 0; 0 0 0 -im; 0 0 im 0]
const mat_YL = [0 0 1 0; 0 0 0 im; 1 0 0 0; 0 -im 0 0]
const mat_ZL = [0 0 0 1; 0 0 -im 0; 0 im 0 0; 1 0 0 0]
const mat_PL = (mat_ZL + I(4)) / 2  # P = (I+Z)/2

# Super-operator matrices for RIGHT multiplication: O ↦ O·σ
# Note: XR differs from XL only in signs for Y,Z components (due to non-commutativity)
const mat_XR = [0 1 0 0; 1 0 0 0; 0 0 0 im; 0 0 -im 0]
const mat_YR = [0 0 1 0; 0 0 0 -im; 1 0 0 0; 0 im 0 0]
const mat_ZR = [0 0 0 1; 0 0 im 0; 0 -im 0 0; 1 0 0 0]
const mat_PR = (mat_ZR + I(4)) / 2  # P = (I+Z)/2

# Identity operator in the Pauli basis
ITensors.op(::OpName"Id", ::SiteType"Pauli", s::Index) = op("I", s)

function ITensors.op(::OpName"I", ::SiteType"Pauli", s::Index)
    M = diagm(ones(4))  # 4×4 identity matrix
    return itensor(M, s', s)
end

"""
M operator: Kinetic term favoring sparse operators.

Diagonal in Pauli basis with eigenvalues:
- ``M[I] = 1``   (weight 0)
- ``M[X] = 1/3`` (weight 1)
- ``M[Y] = 1/3`` (weight 1)
- ``M[Z] = 1/3`` (weight 1)

For a Pauli string with ``w`` non-identity matrices, ``M`` gives eigenvalue ``3^{-w}``.
This penalizes extensive operators and favors local ones.
"""
function ITensors.op(::OpName"M", ::SiteType"Pauli", s::Index)
    mat = diagm([1.0, 1/3, 1/3, 1/3])
    return itensor(mat, s', s)
end

# Register the super-operator matrices as ITensor operators
ITensors.op(::OpName"XL", ::SiteType"Pauli", s::Index) = itensor(mat_XL, s', s)
ITensors.op(::OpName"YL", ::SiteType"Pauli", s::Index) = itensor(mat_YL, s', s)
ITensors.op(::OpName"ZL", ::SiteType"Pauli", s::Index) = itensor(mat_ZL, s', s)
ITensors.op(::OpName"PL", ::SiteType"Pauli", s::Index) = itensor(mat_PL, s', s)
ITensors.op(::OpName"XR", ::SiteType"Pauli", s::Index) = itensor(mat_XR, s', s)
ITensors.op(::OpName"YR", ::SiteType"Pauli", s::Index) = itensor(mat_YR, s', s)
ITensors.op(::OpName"ZR", ::SiteType"Pauli", s::Index) = itensor(mat_ZR, s', s)
ITensors.op(::OpName"PR", ::SiteType"Pauli", s::Index) = itensor(mat_PR, s', s)
