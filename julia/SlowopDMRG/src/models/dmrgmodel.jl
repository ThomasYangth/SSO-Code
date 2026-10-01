#=
===============================================================================
DMRG FOR FINDING CONSERVED QUANTITIES IN QUANTUM SPIN CHAINS
===============================================================================

This module implements a DMRG algorithm to find approximate conserved quantities
(local integrals of motion) in quantum spin systems by solving an optimization
problem in the space of operators.

THEORETICAL BACKGROUND:
-----------------------
We seek operators ``O`` that approximately commute with the Hamiltonian ``H``, i.e.,
``[H, O] ≈ 0``. To avoid trivial solutions (like ``O = I`` or ``O = H^n``), we work in
the orthogonal complement of the Krylov space ``\\{I, H, H^2, \\ldots, H^n\\}``.

The optimization problem is:
```math
\\begin{aligned}
    \\text{minimize} \\quad & E[O] = \\theta^2 \\|[H,O]\\|^2 - M[O] \\\\
    \\text{subject to} \\quad & O \\perp \\mathrm{span}\\{I, H, H^2, \\ldots, H^n\\}
\\end{aligned}
```

where:
- ``\\theta`` is a penalty parameter controlling the commutator weight
- ``M[O]`` is a "kinetic" term favoring operators with fewer Pauli matrices
- ``\\|[H,O]\\|^2`` is the squared commutator norm (zero for exact conserved quantities)

OPERATOR SPACE REPRESENTATION:
------------------------------
We represent operators in the Pauli basis ``\\{I, X, Y, Z\\}``. An operator on ``L`` sites
is a linear combination of Pauli strings, e.g.:
```math
    O = c_1 \\cdot I\\otimes I\\otimes \\cdots \\otimes I + c_2 \\cdot X\\otimes I\\otimes \\cdots \\otimes I + c_3 \\cdot Z\\otimes Z\\otimes I\\otimes \\cdots + \\ldots
```

This ``4^L`` dimensional space is represented as an MPS where each site has a
4-dimensional local Hilbert space (not a qubit, but an operator basis).

COMMUTATOR SUPER-OPERATOR:
--------------------------
The commutator ``[H, \\cdot]`` is a linear super-operator acting on operators. For any
operator ``O``, we have:
```math
    [H, O] = H \\cdot O - O \\cdot H
```

We construct this as ``C_H = H_L - H_R`` where:
- ``H_L`` represents left multiplication by ``H`` (maps ``O \\mapsto H \\cdot O``)
- ``H_R`` represents right multiplication by ``H`` (maps ``O \\mapsto O \\cdot H``)

These are MPOs acting on the 4-dimensional Pauli sites. When applied to an
MPS representing operator ``O``, ``C_H`` produces an MPS representing ``[H, O]``.

THE KINETIC TERM M:
------------------
``M`` is a diagonal operator in the Pauli basis that assigns weight 1 to the
identity and 1/3 to each Pauli matrix ``\\{X, Y, Z\\}``. For a Pauli string of
weight ``w`` (number of non-identity Paulis), ``M`` gives eigenvalue ``3^{-w}``.

This favors operators with smaller support, preventing trivial solutions
that are extensive linear combinations of all Pauli strings.

MAIN COMPONENTS:
---------------
1. Custom ITensor site type "Pauli" for the 4-dimensional operator space
2. Super-operator construction: ``H_L``, ``H_R``, and commutator ``C_H = H_L - H_R``
3. Krylov basis construction: ``\\{I, H, H^2, \\ldots, H^n\\}`` as MPS
4. DMRG solver with orthogonalization against the Krylov subspace
5. Analysis tools: operator size distribution, k-space decomposition

USAGE:
------
The main function is `run_dmrg_for_size(L, H_dict; ...)` which:
1. Constructs the commutator super-operator from the Hamiltonian
2. Builds the Krylov basis to orthogonalize against
3. Runs DMRG to minimize ``E[O]`` in the orthogonal complement
4. Returns an MPS representing the approximate conserved quantity

REFERENCES:
-----------
- Super-operator formalism for quantum dynamics
- Krylov subspace methods in many-body physics
- Local integrals of motion in many-body localization

=#

# Include all submodules in dependency order
include("sanity_check.jl")
include("pauli_basis.jl")
include("super_operators.jl")
include("ortho_basis.jl")
include("exact_solver.jl")
include("dmrg_solver.jl")
