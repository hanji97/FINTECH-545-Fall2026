"""VaR and Expected Shortfall (Weeks 4-5).

Sign convention for the whole library: inputs are P&L (profit positive);
VaR and ES are returned as POSITIVE numbers for a loss, i.e.
VaR = -F^{-1}(alpha) and ES = -E[X | X <= -VaR]. Absolute convention
(Week 4, Section 2.1): the mean is not subtracted.

Empirical quantile convention: sort the sample ascending and take the
ceil(n * alpha)-th value (Week 4: "pick the N*alpha value from the vector").
ES averages every value at or below that point.
"""

import math

import numpy as np
from scipy import stats


def var_hist(pnl, alpha=0.05):
    x = np.sort(np.asarray(pnl, dtype=float))
    k = max(int(math.ceil(len(x) * alpha)), 1)
    return float(-x[k - 1])


def es_hist(pnl, alpha=0.05):
    x = np.sort(np.asarray(pnl, dtype=float))
    k = max(int(math.ceil(len(x) * alpha)), 1)
    return float(-x[:k].mean())


def var_normal(mu, sigma, alpha=0.05):
    """VaR of a N(mu, sigma^2) P&L."""
    return float(-(mu + sigma * stats.norm.ppf(alpha)))


def es_normal(mu, sigma, alpha=0.05):
    """ES of a N(mu, sigma^2) P&L: -mu + sigma * phi(z_alpha) / alpha."""
    z = stats.norm.ppf(alpha)
    return float(-mu + sigma * stats.norm.pdf(z) / alpha)


def var_t(nu, mu, s, alpha=0.05):
    """VaR of a location-scale t P&L (s is the scale, not the sd)."""
    return float(-(mu + s * stats.t.ppf(alpha, nu)))


def es_t(nu, mu, s, alpha=0.05):
    """Closed-form ES of a location-scale t."""
    q = stats.t.ppf(alpha, nu)
    tail = stats.t.pdf(q, nu) * (nu + q * q) / ((nu - 1) * alpha)
    return float(-mu + s * tail)
