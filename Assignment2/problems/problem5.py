"""Problem 5 -- Model based simulation and residual correlation.

Run:  python problems/problem5.py
Writes output/p5.json.
"""

import numpy as np
import pandas as pd

import qrm
from common import DATA, N_SIM, SEED, save_json

ALPHA = 0.05
POS = {"P1": np.array([1e6, 1e6]), "P2": np.array([1e6, -1e6])}   # dollars in (A, B)

px = pd.read_csv(DATA / "problem5.csv").set_index("Day")
ret = (px / px.shift(1) - 1.0).dropna()          # arithmetic returns
rm = ret["MKT"].values
R_stock = ret[["A", "B"]].values
n = len(rm)

# ---------------------------------------------------------------- Fit (b): OLS
Xmat = np.column_stack([np.ones(n), rm])
coef, *_ = np.linalg.lstsq(Xmat, R_stock, rcond=None)     # rows: alpha, beta
alpha, beta = coef[0], coef[1]
resid = R_stock - Xmat @ coef
# Convention: n-1 denominators everywhere (np.cov default) so the factor-model
# covariance is directly comparable with the sample covariance in part (d).
cov_eps = np.cov(resid.T)
sd_eps = np.sqrt(np.diag(cov_eps))
rho_eps = cov_eps[0, 1] / (sd_eps[0] * sd_eps[1])
var_m = np.var(rm, ddof=1)

# ---------------------------------------------------------------- Fit (c): simulate
# Zero expected returns: market mean 0 and alpha set to 0 in the simulation.
rng = np.random.default_rng(SEED)
m_sim = rng.standard_normal(N_SIM) * np.sqrt(var_m)
results = {}
for label, ce in (("full", cov_eps), ("diag", np.diag(np.diag(cov_eps)))):
    eps = qrm.simulation.simulate_mvn(ce, N_SIM, rng=rng)
    r_sim = np.outer(m_sim, beta) + eps               # r_i = beta_i r_m + eps_i
    for p, w in POS.items():
        pnl = r_sim @ w
        results.setdefault(p, {})[f"sim_{label}"] = qrm.risk.var_hist(pnl, ALPHA)

# ---------------------------------------------------------------- Fit (d): delta normal
cov_r = np.cov(R_stock.T)
for p, w in POS.items():
    results[p]["delta_normal"] = qrm.risk.var_normal(0, np.sqrt(w @ cov_r @ w), ALPHA)
    # Closed-form VaR under each factor-model assumption (no simulation noise)
    for label, ce in (("full", cov_eps), ("diag", np.diag(np.diag(cov_eps)))):
        S = var_m * np.outer(beta, beta) + ce
        results[p][f"exact_{label}"] = qrm.risk.var_normal(0, np.sqrt(w @ S @ w), ALPHA)

# Identity check: beta beta' var_m + Sigma_eps should equal the sample covariance.
S_full = var_m * np.outer(beta, beta) + cov_eps
max_diff = float(np.max(np.abs(S_full - cov_r)))

save_json("p5", {
    "n": n, "alpha": alpha.tolist(), "beta": beta.tolist(), "sd_eps": sd_eps.tolist(),
    "rho_eps": rho_eps, "cov_eps": cov_eps.tolist(), "sd_m": float(np.sqrt(var_m)),
    "corr_AB": float(np.corrcoef(R_stock.T)[0, 1]),
    "results": results, "max_diff": max_diff,
})

print("alpha", alpha, "beta", beta, "sd_eps", sd_eps, "rho_eps", rho_eps)
print(pd.DataFrame(results).round(0))
print("max |model cov - sample cov|", max_diff)
