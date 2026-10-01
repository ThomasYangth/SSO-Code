r"""Dynamical-typicality (DQT) machinery for the modulated energy density of Model B.

Model B is the mixed-field Ising ring  H = sum_x J Z_x Z_{x+1} + g X_x + h Z_x
(J=1, g=-1.05, h=0.5).  The object is the s-windowed energy-density mode
H_k(s) = exp(-s^2 ad_H^2/4) H_k, k = 2 pi/L, and the frontier pair
tau_obs = sqrt(N1/N2), nu = M/N1 (see docs/dqt.md).

Modules
-------
model        sparse H_k builder (kron basis = QuSpin pauli=1), Gershgorin bracket,
             seeded Haar / random-product states
propagators  Chebyshev (production) and Krylov e^{-iHt} v engines; ``Propagator``
kernels      Gaussian time kernels G_v, -G_v'', g_s
pauli        symbolic Pauli algebra: exact RPS / HS Grams of ad_H^mu H_k (any L)
fourier      production Fourier-correlator estimator with {H_k, ad^2 H_k} control variate
combine      pooling of seed runs: nu(tau) front, C(t) & RPS-norm curves, G_RPS(t1,t2)
exact        full-ED reference N1, N2, M (small L) and analytic ||H_k||_RPS
chebmoment   Chebyshev-moment estimator (independent cross-check, small L)
figdata      loaders for the shipped fig3panel data (data/fig3panel)
"""
