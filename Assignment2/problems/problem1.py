"""Problem 1 -- Correlations from mismatched histories.

Run:  python problems/problem1.py
Writes output/p1.json.
"""

import numpy as np
import pandas as pd

import qrm
from common import DATA, save_json

df = pd.read_csv(DATA / "problem1.csv").set_index("Day")
cols = list(df.columns)                       # A, B, C, D, IDX

# ---------------------------------------------------------------- Predict
counts = qrm.correlation.pairwise_counts(df)
n_all = int(df.dropna().shape[0])             # days on which all five trade

# ---------------------------------------------------------------- Fit (d)
C_cc = qrm.correlation.complete_case_corr(df)
C_pw = qrm.correlation.pairwise_corr(df)
eig_cc = np.linalg.eigvalsh(C_cc.values)
eig_pw = np.linalg.eigvalsh(C_pw.values)
_, err_cc = qrm.psd.try_cholesky(C_cc.values)
_, err_pw = qrm.psd.try_cholesky(C_pw.values)

# ---------------------------------------------------------------- Fit (e)
# Tracking portfolio: +1 IDX, -0.4 A -0.3 B -0.2 C -0.1 D.
w = np.array([-0.4, -0.3, -0.2, -0.1, 1.0])
sd = qrm.correlation.full_history_std(df).values      # each series' own full history


def tracking_var(C):
    """w' D C D w with full-history standard deviations (Week 3, item 7)."""
    S = qrm.correlation.cov_from_corr(C, sd)
    return float(w @ S @ w)


var_pw = tracking_var(C_pw.values)
var_cc = tracking_var(C_cc.values)

# Benchmark: realised variance of the tracking residual on days all five trade.
resid = df.dropna().values @ w
var_realised = float(np.var(resid, ddof=1))

# ---------------------------------------------------------------- Fit (f)
C_rj = qrm.psd.near_psd_rj(C_pw.values)
C_hi, iters = qrm.psd.higham_corr(C_pw.values)
repairs = {}
for name, C in [("RJ", C_rj), ("Higham", C_hi)]:
    repairs[name] = {
        "min_eig": qrm.psd.min_eig(C),
        "frob": qrm.psd.frobenius(C, C_pw.values),
        "track_var": tracking_var(C),
        "chol_ok": qrm.psd.try_cholesky(C + 1e-12 * np.eye(5))[1] is None,
    }

# ---------------------------------------------------------------- Reconcile (h, i)
move_hi = pd.DataFrame(C_hi - C_pw.values, index=cols, columns=cols)
move_rj = pd.DataFrame(C_rj - C_pw.values, index=cols, columns=cols)
gap_est = qrm.psd.frobenius(C_cc.values, C_pw.values)

# Eigenvector of the negative eigenvalue: tells the repair which direction to fix.
vals, vecs = np.linalg.eigh(C_pw.values)
neg_vec = vecs[:, 0] * np.sign(vecs[-1, 0])

save_json("p1", {
    "cols": cols,
    "counts": counts.values.tolist(),
    "n_all": n_all,
    "C_cc": C_cc.values.tolist(), "C_pw": C_pw.values.tolist(),
    "eig_cc": eig_cc.tolist(), "eig_pw": eig_pw.tolist(),
    "chol_cc": err_cc or "succeeded", "chol_pw": err_pw or "succeeded",
    "sd": sd.tolist(),
    "var_pw": var_pw, "var_cc": var_cc, "var_realised": var_realised,
    "repairs": repairs, "higham_iters": iters,
    "move_hi": move_hi.values.tolist(), "move_rj": move_rj.values.tolist(),
    "gap_est": gap_est, "neg_vec": neg_vec.tolist(),
})

pd.set_option("display.width", 140, "display.precision", 4)
print("pair counts\n", counts, "\nall five:", n_all)
print("complete case\n", C_cc, "\neig", eig_cc, "\ncholesky:", err_cc or "ok")
print("pairwise\n", C_pw, "\neig", eig_pw, "\ncholesky:", err_pw or "ok")
print("tracking var pairwise", var_pw, "complete", var_cc, "realised", var_realised)
print(repairs, "\nHigham move\n", move_hi, "\nRJ move\n", move_rj, "\ngap est", gap_est)
print("neg eigvec", neg_vec)
