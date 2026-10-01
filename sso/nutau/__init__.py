"""Exact nu(tau) frontier of simple slow operators (static and Floquet chains).

nu = <v|B|v>/<v|v> is the RPS weight (B_sigma = 3^{-#non-identity sites},
diagonal in the Pauli basis) and sigma^2 = <v|Delta^2|v>/<v|v> the detuning
variance (static Delta = E_m - E_n - lam, tau = 1/sigma; Floquet
Delta^2 = 4 sin^2((phi_m - phi_n - lam)/2), chord tau = 1/(2 arcsin(sigma/2))).

Modules
-------
conventions  tau(As) for both kinds
pauli        Pauli basis, RPS kernel, B in the energy basis (EnergyBasisB)
models       QuSpin matrices of Operators, kicked-Ising layers, base operators
spectrum     Spectrum: static eigh / Floquet Schur, momentum-resolved or full, PBC/OBC
frontier     soft filter M(theta) = Pi L B L Pi per sector, theta grid and theta = inf
crossing     branch crossings: cross-sector and within-sector superposition arcs
filters      Gaussian / hard / time-average filters of a fixed base operator
tauinf       tau = inf commutant nu_o(inf) and the H^n baseline
io           raw grid readers and slim-data loaders
"""
