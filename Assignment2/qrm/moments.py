"""Sample moments (Week 1).

Conventions, stated explicitly because packages differ (Week 1, Section 4.4):

* variance uses the unbiased n-1 denominator;
* skewness uses the bias-corrected estimator (scipy ``bias=False``);
* kurtosis is reported as EXCESS kurtosis (normal = 0), bias-corrected
  (scipy ``fisher=True, bias=False``).
"""

import numpy as np
from scipy import stats


def four_moments(x):
    """Return a dict with mean, variance, skewness and excess kurtosis.

    Parameters
    ----------
    x : array-like
        One-dimensional sample. NaNs are dropped.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return {
        "mean": float(np.mean(x)),
        "var": float(np.var(x, ddof=1)),          # n-1 denominator
        "std": float(np.std(x, ddof=1)),
        "skew": float(stats.skew(x, bias=False)),
        "kurt": float(stats.kurtosis(x, fisher=True, bias=False)),  # excess
    }


def leave_one_out_kurtosis(x):
    """Excess kurtosis after dropping the single most extreme observation.

    Used to check how much a fourth-moment estimate depends on one point
    (Week 1, Section 6: "a single large move can move it by several units").
    Returns (kurtosis_without_point, index_of_dropped_point).
    """
    x = np.asarray(x, dtype=float)
    z = np.abs(x - np.mean(x))
    i = int(np.argmax(z))
    y = np.delete(x, i)
    return float(stats.kurtosis(y, fisher=True, bias=False)), i
