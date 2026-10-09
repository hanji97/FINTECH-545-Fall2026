"""Multivariate normal simulation (Week 3, Section 1.2).

X = (L Z + mu)', with L a Cholesky root of Sigma and Z iid standard normals.
``chol_psd`` is used so a PSD (rank-deficient) Sigma also works.
"""

import numpy as np

from .psd import chol_psd


def simulate_mvn(cov, n, mean=None, rng=None):
    """Draw n rows from N(mean, cov). Returns an (n, d) array."""
    rng = np.random.default_rng(rng)
    cov = np.asarray(cov, dtype=float)
    d = cov.shape[0]
    L = chol_psd(cov)
    Z = rng.standard_normal((d, n))
    X = (L @ Z).T
    if mean is not None:
        X = X + np.asarray(mean, dtype=float)
    return X
