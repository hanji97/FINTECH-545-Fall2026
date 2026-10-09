"""Problem 3 -- When diversification raises VaR.

Run:  python problems/problem3.py
Writes output/p3.json and output/p3_pnl.png.
"""

import numpy as np
import pandas as pd

import qrm
from common import DATA, save_json, savefig, plt

d = pd.read_csv(DATA / "problem3.csv")
rA, rB = d["A"].values, d["B"].values

# P&L in dollars of each position (one-year scenario, arithmetic return x notional).
pos = {
    "$1M A": 1e6 * rA,
    "$1M B": 1e6 * rB,
    "$2M A": 2e6 * rA,
    "$1M A + $1M B": 1e6 * rA + 1e6 * rB,
}

# ---------------------------------------------------------------- Predict: plot
fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6), sharey=True)
bins = np.linspace(-1.8e6, 0.35e6, 90)
for ax, key, c in zip(axes, ["$2M A", "$1M A + $1M B"], ["#3b6ea5", "#d08a2b"]):
    ax.hist(pos[key] / 1e3, bins=bins / 1e3, color=c)
    ax.set_yscale("log")
    ax.set_title(key.replace("$", r"\$"))     # escape $ so matplotlib does not read mathtext
    ax.set_xlabel("One-year P&L ($000)")
axes[0].set_ylabel("Scenarios (log scale)")
savefig(fig, "p3_pnl")

# ---------------------------------------------------------------- Predict (a, b)
moments = {"A": qrm.moments.four_moments(rA), "B": qrm.moments.four_moments(rB)}
lossA, lossB = rA < -0.20, rB < -0.20
counts = {
    "A": int(lossA.sum()), "B": int(lossB.sum()),
    "at_least_one": int((lossA | lossB).sum()),
    "both": int((lossA & lossB).sum()),
    "n": len(rA),
}

# ---------------------------------------------------------------- Fit (c, d, e)
table = {}
for k, pnl in pos.items():
    table[k] = {
        "var5": qrm.risk.var_hist(pnl, 0.05),
        "es5": qrm.risk.es_hist(pnl, 0.05),
        "nvar5": qrm.risk.var_normal(pnl.mean(), pnl.std(ddof=1), 0.05),
        "var1": qrm.risk.var_hist(pnl, 0.01),
        "es1": qrm.risk.es_hist(pnl, 0.01),
        "mean": float(pnl.mean()), "sd": float(pnl.std(ddof=1)),
    }

# ---------------------------------------------------------------- Reconcile (f)
sub = {}
for m in ("var5", "es5", "nvar5", "var1", "es1"):
    s = table["$1M A"][m] + table["$1M B"][m]
    c = table["$1M A + $1M B"][m]
    sub[m] = {"sum_parts": s, "combined": c, "subadditive": bool(c <= s + 1e-9)}

save_json("p3", {"moments": moments, "counts": counts, "table": table, "sub": sub})

for k, v in table.items():
    print(k, {m: round(x) for m, x in v.items()})
print(moments, counts, sub, sep="\n")
