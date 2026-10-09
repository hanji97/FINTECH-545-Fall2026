"""Problem 2 -- A volatility estimate after the regime changed.

Run:  python problems/problem2.py
Writes output/p2.json and output/p2_returns.png.
"""

import numpy as np
import pandas as pd
from scipy import stats

import qrm
from common import DATA, save_json, savefig, plt

PV = 1_000_000
ALPHA = 0.05
K_RECENT = 35     # regime length read off the plot (returns from day 466 onward)

price = pd.read_csv(DATA / "problem2.csv")["Price"].values
r = price[1:] / price[:-1] - 1.0          # arithmetic returns, oldest first
r = r - r.mean()                          # remove the sample mean (problem statement)
n = len(r)

# ---------------------------------------------------------------- Predict: plot
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.plot(np.arange(1, n + 1), r, lw=0.7, color="#3b6ea5")
ax.axvline(n - K_RECENT + 0.5, color="#c0392b", ls="--", lw=0.8)
ax.text(n - K_RECENT - 3, r.max() * 0.9, f"last {K_RECENT} days", ha="right", color="#c0392b")
ax.set_xlabel("Day")
ax.set_ylabel("Demeaned arithmetic return")
savefig(fig, "p2_returns")

# ---------------------------------------------------------------- Predict (b)
ew = {}
for lam in (0.94, 0.97):
    w = qrm.ewma.ew_weights(n, lam)
    ew[str(lam)] = {
        "n_eff_inf": qrm.ewma.n_eff_infinite(lam),
        "n_eff_sample": qrm.ewma.n_eff(w),
        "half_life": qrm.ewma.half_life(lam),
        "weight_recent": float(w[-K_RECENT:].sum()),
    }
ew["equal_weight_recent"] = K_RECENT / n

# ---------------------------------------------------------------- Predict (c)
mom = qrm.moments.four_moments(r)
jb = stats.jarque_bera(r)

# ---------------------------------------------------------------- Fit (d)
sd_eq = r.std(ddof=1)
sd_97 = np.sqrt(qrm.ewma.ew_variance(r, 0.97))
sd_94 = np.sqrt(qrm.ewma.ew_variance(r, 0.94))
tfit = qrm.fitting.fit_t(r)     # location left free; it lands near 0 since r is demeaned

var = {
    "Normal, equal weight": PV * qrm.risk.var_normal(0, sd_eq, ALPHA),
    "Normal, EW 0.97": PV * qrm.risk.var_normal(0, sd_97, ALPHA),
    "Normal, EW 0.94": PV * qrm.risk.var_normal(0, sd_94, ALPHA),
    "Student t MLE": PV * qrm.risk.var_t(tfit["nu"], tfit["mu"], tfit["s"], ALPHA),
    "Historical": PV * qrm.risk.var_hist(r, ALPHA),
}

# ---------------------------------------------------------------- Reconcile (f)
sd_calm = r[:-K_RECENT].std(ddof=1)
sd_storm = r[-K_RECENT:].std(ddof=1)
p = K_RECENT / n
# Kurtosis of a two-component scale mixture of zero-mean normals:
#   E[x^4] = 3 (p s2^4 + (1-p) s1^4),  E[x^2] = p s2^2 + (1-p) s1^2
m2 = p * sd_storm**2 + (1 - p) * sd_calm**2
m4 = 3 * (p * sd_storm**4 + (1 - p) * sd_calm**4)
kurt_mixture = m4 / m2**2 - 3
mom_calm = qrm.moments.four_moments(r[:-K_RECENT])
mom_storm = qrm.moments.four_moments(r[-K_RECENT:])
t_sd = tfit["s"] * np.sqrt(tfit["nu"] / (tfit["nu"] - 2))

# ---------------------------------------------------------------- Reconcile (g)
se = {}
for lam, s in ((0.94, sd_94), (0.97, sd_97)):
    ne = qrm.ewma.n_eff(qrm.ewma.ew_weights(n, lam))
    se[str(lam)] = {"sd": s, "se_sd": s / np.sqrt(2 * ne),
                    "se_var_dollars": PV * 1.645 * s / np.sqrt(2 * ne)}
gap = var["Normal, EW 0.94"] - var["Normal, EW 0.97"]
se_gap = np.hypot(se["0.94"]["se_var_dollars"], se["0.97"]["se_var_dollars"])

save_json("p2", {
    "n": n, "k_recent": K_RECENT, "ew": ew, "moments": mom,
    "jb_stat": float(jb.statistic), "jb_p": float(jb.pvalue),
    "sd": {"eq": sd_eq, "ew97": sd_97, "ew94": sd_94},
    "t": tfit, "t_sd": t_sd, "var": var,
    "sd_calm": sd_calm, "sd_storm": sd_storm, "kurt_mixture": kurt_mixture,
    "mom_calm": mom_calm, "mom_storm": mom_storm,
    "se": se, "gap": gap, "se_gap": se_gap,
})

print(ew, mom, jb, sep="\n")
print({k: round(v) for k, v in var.items()})
print("t", tfit, "t sd", t_sd)
print("sd calm", sd_calm, "storm", sd_storm, "mixture kurt", kurt_mixture)
print("calm", mom_calm, "\nstorm", mom_storm)
print(se, "gap", gap, "se gap", se_gap)
