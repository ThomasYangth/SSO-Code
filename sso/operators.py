"""Translation-invariant operator specified as a map of Pauli words to couplings.

``Operator({"ZZ": J, "X": hx, "Z": hz})`` represents
``sum_j  J Z_j Z_{j+1} + hx X_j + hz Z_j``.  Words may contain ``I`` padding
(e.g. ``"ZIZ"``).  An optional ``sitefun(j)`` multiplies the term placed at
site ``j`` (used for spatially modulated operators).
"""


class Operator:
    def __init__(self, terms, name=None, sitefun=lambda _j: 1):
        self.terms = dict(terms)
        self.name = name
        self.sitefun = sitefun

    def __iter__(self):
        return iter(self.terms)

    def __getitem__(self, key):
        return self.terms[key]

    def __repr__(self):
        return " + ".join(f"({self[k]:.3f}){k}" for k in self)


def ising(J=1.0, hx=0.0, hz=0.0):
    """Mixed-field Ising chain  H = sum J ZZ + hx X + hz Z."""
    return Operator({"ZZ": J, "X": hx, "Z": hz}, name=f"Ising_J{J}X{hx}Z{hz}")
