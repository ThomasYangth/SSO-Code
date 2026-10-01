"""sso — code accompanying "Simple Slow Operators and Quantum Thermalization".

Subpackages
-----------
sso.nutau     Exact (ED) simple-slow-operator frontier nu(tau) in the doubled
              Hilbert space: static and Floquet chains, momentum sectors,
              tau = infinity commutant, crossing arcs, filtered constructions.
sso.dqt       Dynamical-typicality (DQT) / Chebyshev time evolution for the
              modulated energy density H_k: windowed norms, RPS norm, autocorrelator.
sso.plotting  Shared figure style and frontier-assembly helpers.

Shared infrastructure lives in sso.backend (numpy/cupy), sso.config (paths),
sso.cli (tiny --key=value parser) and sso.operators (Hamiltonian term maps).
"""

__version__ = "1.0.0"
