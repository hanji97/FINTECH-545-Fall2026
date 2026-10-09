"""PSD tools (Week 3, Sections 2 and 6).

* ``chol_psd``      -- Cholesky root that tolerates PSD (zero-pivot) matrices.
* ``try_cholesky``  -- strict PD Cholesky (numpy); reports failure instead of raising.
* ``near_psd_rj``   -- Rebonato & Jackel spectral repair of a correlation matrix.
* ``higham_corr``   -- Higham (2002) alternating projections with Dykstra
                       correction, unweighted (W = I).
"""

import numpy as np


def min_eig(A):
    """Smallest eigenvalue of a symmetric matrix."""
    return float(np.linalg.eigvalsh(np.asarray(A, dtype=float)).min())


def frobenius(A, B):
    """Frobenius norm of A - B."""
    return float(np.linalg.norm(np.asarray(A) - np.asarray(B), ord="fro"))


def try_cholesky(A):
    """Attempt a strict (PD) Cholesky factorisation.

    Returns (L, None) on success, (None, message) on failure.
    """
    try:
        return np.linalg.cholesky(np.asarray(A, dtype=float)), None
    except np.linalg.LinAlgError as err:
        return None, str(err)


def chol_psd(A, tol=1e-8):
    """Cholesky root of a PSD matrix, following the column algorithm in Week 3.

    When a pivot is (numerically) zero, the whole column is set to zero, as in
    Week 3 Section 2.2. Tiny negative pivots from floating point error are
    clipped to zero. A pivot that is clearly negative means the matrix is not
    PSD and a ValueError is raised -- repair it first.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)
    scale = max(1.0, np.max(np.abs(np.diag(A))))
    for j in range(n):
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if s < -tol * scale:
            raise ValueError(f"matrix is not PSD: pivot {j} = {s:.3e}")
        s = max(s, 0.0)
        L[j, j] = np.sqrt(s)
        if L[j, j] <= tol * np.sqrt(scale):
            L[j:, j] = 0.0          # zero pivot -> zero column
            continue
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
    return L


def near_psd_rj(C, epsilon=0.0):
    """Rebonato & Jackel (1999) spectral repair of a correlation matrix.

    1. eigen-decompose C = S Lambda S'
    2. floor the eigenvalues at ``epsilon`` (0 by default)
    3. rescale rows with T so that the unit diagonal is restored
    4. return B B' with B = sqrt(T) S sqrt(Lambda')
    """
    C = np.asarray(C, dtype=float)
    vals, S = np.linalg.eigh(C)
    vals = np.maximum(vals, epsilon)
    t = 1.0 / (S * S @ vals)                 # t_i = 1 / sum_j s_ij^2 lambda_j
    B = np.diag(np.sqrt(t)) @ S @ np.diag(np.sqrt(vals))
    out = B @ B.T
    np.fill_diagonal(out, 1.0)               # remove float residue on the diagonal
    return out


def _proj_psd(A):
    """P_S: set negative eigenvalues to zero (W = I)."""
    vals, S = np.linalg.eigh((A + A.T) / 2.0)
    return S @ np.diag(np.maximum(vals, 0.0)) @ S.T


def _proj_unit_diag(A):
    """P_U: put ones back on the diagonal (W diagonal)."""
    out = A.copy()
    np.fill_diagonal(out, 1.0)
    return out


def higham_corr(C, tol=1e-12, max_iter=10_000):
    """Higham (2002) nearest correlation matrix, Frobenius norm, W = I.

    Alternating projections between the PSD cone and the unit-diagonal set,
    with Dykstra's correction (delta_S) so the limit is the nearest point.
    Returns (matrix, iterations used). Per Week 3 Section 6.3 the caller should
    check the smallest eigenvalue, not just convergence of the norm.
    """
    C = np.asarray(C, dtype=float)
    Y = C.copy()
    dS = np.zeros_like(C)
    gamma_prev = np.inf
    for k in range(1, max_iter + 1):
        R = Y - dS
        X = _proj_psd(R)
        dS = X - R
        Y = _proj_unit_diag(X)
        gamma = frobenius(Y, C)
        if abs(gamma - gamma_prev) < tol:
            break
        gamma_prev = gamma
    # Final clean-up: the unit-diagonal projection can leave an eigenvalue at
    # -1e-16; one more PSD pass (and diagonal reset) removes float residue.
    if min_eig(Y) < 0:
        Y = _proj_unit_diag(_proj_psd(Y))
    return Y, k
