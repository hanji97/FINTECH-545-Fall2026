"""Exponentially weighted variance (Week 3, Section 5).

Data are ordered oldest -> newest (row 0 is the oldest observation). The most
recent observation gets weight (1 - lambda), the one before it
(1 - lambda) * lambda, and so on; weights are normalised to sum to one.
"""

import numpy as np


def ew_weights(n, lam):
    """Normalised exponential weights for n observations, oldest first."""
    age = np.arange(n - 1, -1, -1)            # newest observation has age 0
    w = (1.0 - lam) * lam ** age
    return w / w.sum()


def ew_variance(x, lam, demean=True):
    """Exponentially weighted variance of a 1-D series (oldest first).

    ``demean=True`` subtracts the equal-weighted sample mean, as in the
    Week 3 estimator. For already-demeaned returns this has no effect.
    """
    x = np.asarray(x, dtype=float)
    if demean:
        x = x - x.mean()
    w = ew_weights(len(x), lam)
    return float(w @ (x * x))


def n_eff_infinite(lam):
    """Effective sample size on an infinite horizon, (1 + lam) / (1 - lam)."""
    return (1.0 + lam) / (1.0 - lam)


def n_eff(weights):
    """Effective sample size of an arbitrary weight vector, 1 / sum(w^2)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(1.0 / np.sum(w * w))


def half_life(lam):
    """Half life in observations, ln(0.5) / ln(lambda)."""
    return float(np.log(0.5) / np.log(lam))
