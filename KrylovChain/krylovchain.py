r"""nu(tau) frontier of the su(1,1) Krylov-chain model (notation of the paper appendix).

Model.  Krylov chain O_0, O_1, ... (orthonormal), Liouvillian calL = [H, .] with
calL_{n-1,n} = calL_{n,n-1} = b_n = sqrt(n(n+1)), q_k(omega) monic of degree k with
q_0 = 1, q_{k+1} = omega q_k - b_k^2 q_{k-1} (= det[omega - calL restricted to O_0..O_{k-1}]),
B_k = prod_{l<=k} b_l, spectral density rho(omega) = omega / (2 sinh(pi omega/2)).
An operator O = sum_k c_k O_k has

    ||O||^2 = c^T c,   ||[H,O]||^2 = c^T calL^2 c,   ||O||_mu^2 = sum_k e^{-mu(k+1)} c_k^2,

i.e. calM_mu = diag(e^{-mu(k+1)}) (mu = log 3: RPS/EVN; mu = inf: the limit of e^{mu} calM_mu,
calM_inf = |0><0|).  The frontier operator at stiffness theta solves

    calM_mu c = lambda (1 + theta calL^2) c.

Method.  With calR = (1 + theta calL^2)^{-1} and c~ = sqrt(calM_mu) c / lambda,

    sqrt(calM_mu) calR sqrt(calM_mu) c~ = lambda c~,                                (E)

solved in the first N sites (c~_N; calR_N = N x N block of calR).  Then

    c = calR sqrt(calM_mu) c~_N                         (all sites; only c~ is truncated)
    ||O||^2       = c~_N^T sqrt(M) calR^2 sqrt(M) c~_N
    ||[H,O]||^2   = c~_N^T sqrt(M) calR^2 calL^2 sqrt(M) c~_N
    ||O||_mu^2    ~ lambda^2 c~_N^T c~_N         (calR -> calR_N in c~^T sqrt(M) R M R sqrt(M) c~)

and nu = ||O||_mu^2 / ||O||^2, tau = ||O|| / ||[H,O]||.  The matrix elements are integrals
(calR is never obtained by inverting a truncated calL):

    (g(calL))_mn = (1/(B_m B_n)) int q_m q_n g(omega) rho(omega) d omega
                 = int p_m p_n g rho d omega,      p_k = q_k / B_k orthonormal,
    calR: g = 1/(1+theta w^2),  calR^2: g = 1/(1+theta w^2)^2,  calR^2 calL^2: g = w^2/(1+theta w^2)^2.

The mu-norm approximation drops sum_{k>=N} e^{-mu(k+1)} c_k^2 (relative size <~ 0.4 e^{-mu N}
at mu = log 3, any theta) and always underestimates nu.  mu_norm="exact" sums it to
K = N + ceil(45/mu) with the K x N block of calR, which makes (tau, nu) exact for the
operator c.

mu = inf: (E) is 1 x 1 for every N (c~ = e_0, lambda = calR_00, c = calR e_0: the Lorentzian
1/(1+theta omega^2) on O_0) and both mu-norm forms are exact, so the result does not depend
on N; :func:`closed_form_mu_inf` gives the same curve from three scalar integrals.

Quadrature.  p_k has parity (-1)^k, so only m + n even is computed, as 2 int_0^Omega.
Composite Gauss-Legendre on panels graded geometrically from 0 up to 1 at the pole scale
1/sqrt(theta), then unit panels to Omega = 3K + margin.  sqrt(rho) p_k comes directly from
the three-term recursion (no overflow).
"""

import numpy as np

MU_EVN = np.log(3.0)


# --- chain and spectral data -----------------------------------------------------------------

def b(n):
    """Krylov coefficient b_n = sqrt(n (n + 1))."""
    n = np.asarray(n, float)
    return np.sqrt(n * (n + 1.0))


def rho(omega):
    """Spectral density rho(omega) = omega / (2 sinh(pi omega / 2)), stable for all omega."""
    a = 0.5 * np.pi * np.abs(np.asarray(omega, float))
    out = np.empty_like(a)
    small = a < 1e-6
    out[small] = 1.0 - a[small] ** 2 / 6.0
    big = ~small
    e = np.exp(-2.0 * a[big])
    out[big] = 2.0 * a[big] * np.exp(-a[big]) / (1.0 - e)
    return out / np.pi


def sqrt_rho_p(K, omega):
    """(K, n_omega) array sqrt(rho(omega)) p_k(omega), p_k = q_k / B_k, k = 0..K-1.

    p_{k+1} = (omega p_k - b_k p_{k-1}) / b_{k+1}  (the q_k recursion divided by B_{k+1}).
    """
    omega = np.asarray(omega, float)
    P = np.empty((K, omega.size))
    P[0] = np.sqrt(rho(omega))
    if K > 1:
        P[1] = omega * P[0] / b(1)
    for k in range(1, K - 1):
        P[k + 1] = (omega * P[k] - b(k) * P[k - 1]) / b(k + 1)
    return P


def M_diag(mu, K):
    """Diagonal of calM_mu: e^{-mu (k+1)}; (1, 0, 0, ...) for mu = inf."""
    if np.isinf(mu):
        d = np.zeros(K)
        d[0] = 1.0
        return d
    return np.exp(-mu * (np.arange(K) + 1.0))


# --- omega-space quadrature and matrix elements ----------------------------------------------

def quadrature_grid(theta, K, order=24, margin=60.0):
    """Nodes and weights on [0, 3K + margin], graded at the pole scale 1/sqrt(theta)."""
    top = 3.0 * K + margin
    scale = 1.0 / np.sqrt(theta) if theta > 0 else 1.0
    lo = min(scale, 1.0) * 1e-3
    edges = list(np.geomspace(lo, 1.0, max(2, int(np.ceil(np.log2(1.0 / lo) * 2)) + 1)))
    edges = np.unique([0.0] + edges + list(np.arange(2.0, np.ceil(top) + 1.0)))
    x, w = np.polynomial.legendre.leggauss(order)
    a, c = edges[:-1, None], edges[1:, None]
    nodes = (0.5 * (c - a) * x[None, :] + 0.5 * (c + a)).ravel()
    weights = (0.5 * (c - a) * w[None, :]).ravel()
    return nodes, weights


def function_of_L(P, nodes, weights, g):
    """(g(calL))_mn = int p_m p_n g rho = 2 int_0^Omega (zero for m + n odd)."""
    A = (P * (weights * g(nodes))[None, :]) @ P.T * 2.0
    K = P.shape[0]
    A[(np.add.outer(np.arange(K), np.arange(K)) % 2) == 1] = 0.0
    return 0.5 * (A + A.T)


def matrix_elements(theta, K, order=24, margin=60.0):
    """K x K blocks (R, R2, R2L2) of calR, calR^2 and calR^2 calL^2 from omega integrals."""
    nodes, weights = quadrature_grid(theta, K, order, margin)
    P = sqrt_rho_p(K, nodes)
    den = lambda w: 1.0 + theta * w * w
    R = function_of_L(P, nodes, weights, lambda w: 1.0 / den(w))
    R2 = function_of_L(P, nodes, weights, lambda w: 1.0 / den(w) ** 2)
    R2L2 = function_of_L(P, nodes, weights, lambda w: w * w / den(w) ** 2)
    return R, R2, R2L2


# --- frontier ---------------------------------------------------------------------------------

def mu_norm_size(mu, N, tail=45.0):
    """K for mu_norm="exact": N + ceil(tail/mu), so e^{-mu K} < e^{-tail}; N for mu = inf."""
    return N if np.isinf(mu) else N + int(np.ceil(tail / mu))


def solve_frontier(mu, theta, N, mu_norm="truncated", order=24, margin=60.0):
    """Top eigenpair of (E) in N sites and (nu, tau) of O = sum_k c_k O_k, c = calR sqrt(M) c~_N.

    mu_norm: "truncated" (the notes: ||O||_mu^2 = lambda^2 c~_N^T c~_N) or "exact"
    (sum_{k<K} e^{-mu(k+1)} c_k^2, K = mu_norm_size).  Returns dict(lam, ctilde, norm2,
    comm2, mu_norm2, mu_norm2_trunc, nu, tau, N, K).
    """
    if mu_norm not in ("truncated", "exact"):
        raise ValueError(mu_norm)
    K = mu_norm_size(mu, N) if mu_norm == "exact" else N
    R, R2, R2L2 = matrix_elements(theta, K, order, margin)
    sM = np.sqrt(M_diag(mu, N))
    evals, evecs = np.linalg.eigh(sM[:, None] * R[:N, :N] * sM[None, :])
    lam, ct = evals[-1], evecs[:, -1]
    x = sM * ct                                         # sqrt(M) c~_N
    norm2 = x @ R2[:N, :N] @ x
    comm2 = x @ R2L2[:N, :N] @ x
    mu_norm2_trunc = lam ** 2 * (ct @ ct)
    if mu_norm == "exact":
        c = R[:, :N] @ x                                # c_k, k < K
        mu_norm2 = float(np.sum(M_diag(mu, K) * c ** 2))
    else:
        mu_norm2 = mu_norm2_trunc
    return dict(lam=lam, ctilde=ct, norm2=norm2, comm2=comm2, mu_norm2=mu_norm2,
                mu_norm2_trunc=mu_norm2_trunc, nu=mu_norm2 / norm2,
                tau=np.sqrt(norm2 / comm2), N=N, K=K)


def closed_form_mu_inf(theta, order=24, margin=60.0):
    """mu = inf frontier from the three scalar integrals:
    nu = [int rho/(1+theta w^2)]^2 / int rho/(1+theta w^2)^2,
    tau = sqrt( int rho/(1+theta w^2)^2 / int w^2 rho/(1+theta w^2)^2 )."""
    nodes, weights = quadrature_grid(theta, 1, order, margin)
    r = rho(nodes)
    g = 1.0 / (1.0 + theta * nodes ** 2)
    I1, I2, I2w = (2.0 * np.sum(weights * r * f) for f in (g, g * g, nodes ** 2 * g * g))
    return dict(nu=I1 * I1 / I2, tau=np.sqrt(I2 / I2w))


def lorentzian_filter(mu, theta, K):
    """(nu, tau) of the explicit Lorentzian 1/(1+theta omega^2) on O_0 (c = calR e_0), with
    the mu-norm summed to K sites.  For mu = inf this is the frontier."""
    R, R2, R2L2 = matrix_elements(theta, K)
    c = R[:, 0]
    return dict(nu=np.sum(M_diag(mu, K) * c ** 2) / R2[0, 0], tau=np.sqrt(R2[0, 0] / R2L2[0, 0]))
