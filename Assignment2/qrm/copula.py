"""Gaussian and t copulas (Week 5, Sections 4-6).

Workflow followed here, exactly as in the notes:

1. transform each margin to uniforms through its own fitted CDF;
2. R = sin(pi * tau / 2) from Kendall's tau (exact for every elliptical copula,
   so both copulas get the SAME R); repair with Higham if not PSD;
3. Gaussian copula has no free parameter; t copula profiles nu over
   theta = 1/nu in [0.01, 0.49] on a 200-point grid, then refines;
4. log likelihoods are written as differences of log densities -- never as a
   ratio of densities -- to avoid underflow (Week 5, Section 6.2).
"""

import numpy as np
from scipy import optimize, stats

from .psd import chol_psd, higham_corr, min_eig

_EPS = 1e-12


def _clip(U):
    return np.clip(np.asarray(U, dtype=float), _EPS, 1.0 - _EPS)


def pseudo_uniforms(X):
    """Ranks scaled into (0, 1): rank / (n + 1)."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    return stats.rankdata(X, axis=0) / (n + 1)   # average ranks for ties


def kendall_R(U):
    """Correlation matrix R_ij = sin(pi * tau_ij / 2), repaired if not PSD."""
    U = np.asarray(U, dtype=float)
    d = U.shape[1]
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau = stats.kendalltau(U[:, i], U[:, j]).statistic
            R[i, j] = R[j, i] = np.sin(np.pi * tau / 2.0)
    if min_eig(R) < 0:
        R, _ = higham_corr(R)
    return R


# ----------------------------------------------------------------- likelihoods
def _mvn_logpdf_std(Z, R):
    """log density of N(0, R) for each row of Z."""
    d = R.shape[0]
    L = np.linalg.cholesky(R)
    sol = np.linalg.solve(L, Z.T)            # L^{-1} z
    q = np.sum(sol * sol, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + q)


def _mvt_logpdf_std(Y, R, nu):
    """log density of a multivariate t with location 0, scale R, nu dof."""
    from scipy.special import gammaln
    d = R.shape[0]
    L = np.linalg.cholesky(R)
    sol = np.linalg.solve(L, Y.T)
    q = np.sum(sol * sol, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * logdet - 0.5 * (nu + d) * np.log1p(q / nu))


def gaussian_copula_loglik_obs(U, R):
    """Per-observation Gaussian copula log density: ln f_R(z) - sum ln phi(z_i)."""
    Z = stats.norm.ppf(_clip(U))
    return _mvn_logpdf_std(Z, R) - np.sum(stats.norm.logpdf(Z), axis=1)


def t_copula_loglik_obs(U, R, nu):
    """Per-observation t copula log density: ln f_{R,nu}(y) - sum ln f_nu(y_i)."""
    Y = stats.t.ppf(_clip(U), nu)
    return _mvt_logpdf_std(Y, R, nu) - np.sum(stats.t.logpdf(Y, nu), axis=1)


def fit_t_copula_nu(U, R, n_grid=200, theta_lo=0.01, theta_hi=0.49):
    """Profile the t copula likelihood over nu with R held fixed.

    Grid over theta = 1/nu (nu between ~2.04 and 100), then a bounded scalar
    refinement between the grid neighbours of the maximum (Week 5, 5.3.2).
    Returns (nu_hat, loglik, grid_nu, grid_ll).
    """
    thetas = np.linspace(theta_lo, theta_hi, n_grid)
    lls = np.array([t_copula_loglik_obs(U, R, 1.0 / th).sum() for th in thetas])
    i = int(np.argmax(lls))
    lo = thetas[max(i - 1, 0)]
    hi = thetas[min(i + 1, n_grid - 1)]
    res = optimize.minimize_scalar(lambda th: -t_copula_loglik_obs(U, R, 1.0 / th).sum(),
                                   bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-8})
    th_hat = res.x if -res.fun >= lls[i] else thetas[i]
    nu_hat = 1.0 / th_hat
    return float(nu_hat), float(t_copula_loglik_obs(U, R, nu_hat).sum()), 1.0 / thetas, lls


# ------------------------------------------------------------------ simulation
def simulate_gaussian_copula(R, n, rng=None):
    """Uniform draws from a Gaussian copula with correlation R."""
    rng = np.random.default_rng(rng)
    L = chol_psd(R)
    Z = (L @ rng.standard_normal((R.shape[0], n))).T
    return stats.norm.cdf(Z)


def simulate_t_copula(R, nu, n, rng=None):
    """Uniform draws from a t copula via the normal variance mixture.

    Y = sqrt(W) L Z with W = nu / chi2_nu (one W per row, shared by every
    column -- the common shock that creates tail dependence). Uniforms use
    the copula's nu, NOT any margin's nu (Week 5, Section 6.3).
    """
    rng = np.random.default_rng(rng)
    L = chol_psd(R)
    Z = (L @ rng.standard_normal((R.shape[0], n))).T
    W = nu / rng.chisquare(nu, size=n)
    Y = Z * np.sqrt(W)[:, None]
    return stats.t.cdf(Y, nu)


def tail_dependence_t(rho, nu):
    """Lower (= upper) tail dependence of a t copula pair."""
    arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * stats.t.cdf(arg, nu + 1.0))
