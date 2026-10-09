"""Correlation estimation with missing data (Week 3, Sections 3-4).

Two estimators:

* complete case: keep only the rows where every series is observed. One
  common sample, so the matrix is a genuine sample correlation matrix and is
  guaranteed PSD.
* pairwise: each entry (i, j) uses every row where i and j are both observed.
  Every entry comes from a different sample, so the matrix is NOT guaranteed
  PSD.

Covariances are rebuilt as Sigma = D C D with D = diag(full-history std),
which is item 7 of Week 3, Section 4.
"""

import numpy as np
import pandas as pd


def pairwise_counts(df):
    """Number of days each pair of columns is jointly observed."""
    m = df.notna().astype(int)
    return m.T @ m


def complete_case_corr(df):
    """Pearson correlation on rows with no missing value."""
    return df.dropna().corr(method="pearson")


def pairwise_corr(df):
    """Pearson correlation computed pair by pair on jointly observed rows.

    pandas' ``DataFrame.corr`` already uses pairwise-complete observations;
    it is wrapped here so the convention is explicit.
    """
    return df.corr(method="pearson", min_periods=2)


def cov_from_corr(C, sd):
    """Sigma = D C D, with D the diagonal matrix of standard deviations."""
    C = np.asarray(C, dtype=float)
    D = np.diag(np.asarray(sd, dtype=float))
    return D @ C @ D


def full_history_std(df):
    """Standard deviation of each column over all of its own observations (n-1)."""
    return df.std(ddof=1, skipna=True)
