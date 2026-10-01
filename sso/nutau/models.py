"""Models and base operators as QuSpin matrices.

Static models are ``sso.operators.Operator`` term maps (``sso.operators.ising``).
A Floquet model is an ordered list of layer Operators [H_1, H_2, ...] defining

    U = exp(-i H_1) exp(-i H_2) ...

Kicked Ising:  U = exp(-i g sum X) exp(-i (J sum ZZ + h sum Z)).

All matrices are built by QuSpin in ``spin_basis_1d(L, pauli=1)`` (Pauli
matrices, not spin-1/2) or one of its momentum blocks, i.e. in the same basis
as the eigenvectors V of :mod:`sso.nutau.spectrum`, so no site/bit convention
enters anywhere else.

Base operators for the filter constructions (``base_operator``):
    Zsum   sum_x Z_x            Xsum   sum_x X_x
    Z0     Z on site 0          YZcur  sum_x (Y_x Z_{x+1} - Z_x Y_{x+1})
YZcur is the nearest-neighbour energy current of the transverse-field Ising
chain, an exactly conserved, pure two-site (nu = 1/9) charge at hz = 0.

Also: the modulated energy density H_q = sum_x e^{-i q x} h_x, q = 2 pi K/L
(``modulated_density``; h_x = the model's own terms placed at site x, so
sum_x h_x = H), and real mixtures of base operators (``mixture_matrix``).
"""

import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

from sso.cli import find_value
from sso.operators import Operator, ising


def kicked_ising(g=0.9, J=1.0, h=0.809):
    """Kicked-Ising layers [g X, J ZZ + h Z] and the model tag used in file names."""
    layers = [Operator({"X": g}), Operator({"ZZ": J, "Z": h})]
    return layers, f"KickedIsing_g{g}J{J}h{h}"


_BASE_OPERATORS = {
    "Zsum": lambda: Operator({"Z": 1.0}),
    "Xsum": lambda: Operator({"X": 1.0}),
    "Z0": lambda: Operator({"Z": 1.0}, sitefun=lambda j: 1 if j == 0 else 0),
    "YZcur": lambda: Operator({"YZ": 1.0, "ZY": -1.0}),
}


def base_operator(name):
    """Operator for a base-operator key (Zsum, Xsum, Z0, YZcur)."""
    if name not in _BASE_OPERATORS:
        raise ValueError(f"unknown base operator {name!r}; known: {sorted(_BASE_OPERATORS)}")
    return _BASE_OPERATORS[name]()


def modulated_density(op, K, L):
    """H_q = sum_x e^{-2 pi i K x / L} h_x for the static model ``op`` (non-Hermitian for K != 0)."""
    phase = np.exp(-2j * np.pi * K * np.arange(L) / L)
    return Operator(dict(op.terms), sitefun=lambda x: phase[x % L])


def mixture_matrix(spec, L, basis):
    """Matrix of sum_i c_i O_i for ``spec = "Zsum:0.78,Xsum:-0.62"`` (base-operator keys)."""
    O = np.zeros((basis.Ns, basis.Ns), dtype=np.complex128)
    for item in spec.split(","):
        name, c = item.split(":")
        O += float(c) * operator_matrix(base_operator(name), L, basis)
    return O


def quspin_static(op, L, pbc=True):
    """QuSpin static list for ``op`` on an L-site chain (PBC wraps, OBC truncates)."""
    static = []
    for word in op:
        coeff = op[word]
        word = word.upper()
        n_pos = L if pbc else L + 1 - len(word)
        for j in range(n_pos):
            sf = op.sitefun(j)
            if sf == 0:
                continue
            chars = "".join(ch.lower() for ch in word if ch != "I")
            sites = [(j + i) % L for i, ch in enumerate(word) if ch != "I"]
            if chars:
                static.append([chars, [[coeff * sf] + sites]])
    return static


def full_basis(L):
    """The full 2^L spin basis (Pauli matrices)."""
    return spin_basis_1d(L, pauli=1)


def operator_matrix(op, L, basis, pbc=True):
    """Dense matrix of ``op`` in ``basis`` (zero matrix for an empty operator)."""
    static = quspin_static(op, L, pbc)
    if not static:
        return np.zeros((basis.Ns, basis.Ns), dtype=np.complex128)
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128,
                    check_symm=False, check_pcon=False, check_herm=False)
    return H.toarray()


def model_from_args(args):
    """(kind, model, name) from driver arguments.

    --model=Ising        --J --hx --hz        (defaults: Kim-Huse 1.0, 0.905, 0.809)
    --model=KickedIsing  --g --J --h          (defaults: 0.9, 1.0, 0.809)
    ``model`` is an Operator (static) or a list of layer Operators (Floquet).
    """
    model = find_value(args, "model", str, "Ising")
    if model == "Ising":
        op = ising(find_value(args, "J", float, 1.0), find_value(args, "hx", float, 0.905),
                   find_value(args, "hz", float, 0.809))
        return "static", op, op.name
    if model == "KickedIsing":
        layers, name = kicked_ising(find_value(args, "g", float, 0.9),
                                    find_value(args, "J", float, 1.0),
                                    find_value(args, "h", float, 0.809))
        return "floquet", layers, name
    raise ValueError(f"unknown model {model!r} (Ising, KickedIsing)")
