"""qrm -- a small quantitative risk management library for FinTech 545.

The library holds every reusable calculation used by the assignment scripts in
``problems/``. Each module covers one topic from the lecture notes:

    moments      -- sample moments with stated bias conventions (Week 1)
    correlation  -- complete-case / pairwise estimation with missing data (Week 3)
    psd          -- Cholesky for PSD matrices, Rebonato-Jackel, Higham (Week 3)
    ewma         -- exponentially weighted variance, n_eff, half life (Week 3)
    fitting      -- normal / Student t MLE and information criteria (Weeks 1-2)
    risk         -- VaR and ES: historical, normal, Student t (Weeks 4-5)
    simulation   -- multivariate normal draws through a Cholesky root (Week 3)
    copula       -- Gaussian and t copulas: fit, likelihood, simulate (Week 5)

Install with ``pip install -e .`` from the Assignment2 folder, then
``import qrm``.
"""

from . import moments, correlation, psd, ewma, fitting, risk, simulation, copula

__all__ = [
    "moments",
    "correlation",
    "psd",
    "ewma",
    "fitting",
    "risk",
    "simulation",
    "copula",
]

__version__ = "0.2.0"
