"""Problem 4 -- Gaussian or t copula.

Run:  python problems/problem4.py
Writes output/p4.json, output/p4_ranks.png, output/p4_nu_profile.png.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import qrm
from common import DATA, N_SIM, SEED, save_json, savefig, plt

PV = 1_000_000           # dollars held in each asset
TAIL = 0.025             # joint-tail threshold from part (b)

X = pd.read_csv(DATA / "problem4.csv")
names = list(X.columns)
x = X.values
m, d = x.shape
pairs = list(combinations(range(d), 2))

# ---------------------------------------------------------------- Predict (a)
mom = {c: qrm.moments.four_moments(X[c]) for c in names}
loo = {}
for j, c in enumerate(names):
    k_wo, i = qrm.moments.leave_one_out_kurtosis(x[:, j])
    loo[c] = {"kurt_without_max": k_wo, "day": int(i), "value": float(x[i, j]),
              "z": float((x[i, j] - x[:, j].mean()) / x[:, j].std(ddof=1))}

# ---------------------------------------------------------------- Predict (b)
Ur = qrm.copula.pseudo_uniforms(x)       # ranks / (m + 1)
k_tail = int(np.ceil(TAIL * m))          # 25 days in each series' own tail
low = np.zeros_like(x, dtype=bool)
high = np.zeros_like(x, dtype=bool)
for j in range(d):                       # exactly the 25 worst / best days of each series
    order = np.argsort(x[:, j])
    low[order[:k_tail], j] = True
    high[order[-k_tail:], j] = True
joint_data = {f"{names[i]}-{names[j]}": {"worst": int((low[:, i] & low[:, j]).sum()),
                                           "best": int((high[:, i] & high[:, j]).sum())}
              for i, j in pairs}
expected_indep = m * TAIL**2

fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
for ax, (i, j) in zip(axes, pairs):
    ax.scatter(Ur[:, i], Ur[:, j], s=2, alpha=0.5, color="#3b6ea5")
    ax.axvline(TAIL, color="#c0392b", lw=0.5); ax.axhline(TAIL, color="#c0392b", lw=0.5)
    ax.axvline(1 - TAIL, color="#c0392b", lw=0.5); ax.axhline(1 - TAIL, color="#c0392b", lw=0.5)
    ax.set_xlabel(names[i]); ax.set_ylabel(names[j]); ax.set_aspect("equal")
savefig(fig, "p4_ranks")

# ---------------------------------------------------------------- Fit (c) margins
margins = {}
for j, c in enumerate(names):
    nf = qrm.fitting.fit_normal(x[:, j])
    tf = qrm.fitting.fit_t(x[:, j])
    nf["aicc"] = qrm.fitting.aicc(nf["ll"], nf["k"], m)
    tf["aicc"] = qrm.fitting.aicc(tf["ll"], tf["k"], m)
    margins[c] = {"normal": nf, "t": tf, "choice": "t" if tf["aicc"] < nf["aicc"] else "normal"}


def margin_cdf(j, v):
    mg = margins[names[j]]
    if mg["choice"] == "t":
        p = mg["t"]; return stats.t.cdf(v, p["nu"], loc=p["mu"], scale=p["s"])
    p = mg["normal"]; return stats.norm.cdf(v, p["mu"], p["sigma"])


def margin_ppf(j, u):
    mg = margins[names[j]]
    if mg["choice"] == "t":
        p = mg["t"]; return stats.t.ppf(u, p["nu"], loc=p["mu"], scale=p["s"])
    p = mg["normal"]; return stats.norm.ppf(u, p["mu"], p["sigma"])


# ---------------------------------------------------------------- Fit (d) copulas
U = np.column_stack([margin_cdf(j, x[:, j]) for j in range(d)])   # through FITTED margins
R = qrm.copula.kendall_R(U)          # same R handed to both copulas
llg_obs = qrm.copula.gaussian_copula_loglik_obs(U, R)
nu_hat, llt, grid_nu, grid_ll = qrm.copula.fit_t_copula_nu(U, R)
llt_obs = qrm.copula.t_copula_loglik_obs(U, R, nu_hat)
llg = float(llg_obs.sum())
cop = {
    "gauss": {"ll": llg, "k": 0, "aicc": -2 * llg, "bic": -2 * llg},   # k = 0: AICc = AIC = BIC
    "t": {"ll": llt, "k": 1, "nu": nu_hat,
          "aicc": qrm.fitting.aicc(llt, 1, m), "bic": qrm.fitting.bic(llt, 1, m)},
}
dBIC = 2 * (llt - llg) - np.log(m)

fig, ax = plt.subplots(figsize=(4.5, 2.6))
ax.plot(grid_nu, grid_ll, color="#3b6ea5")
ax.axhline(llg, color="#888", ls="--", lw=0.8, label="Gaussian")
ax.axvline(nu_hat, color="#c0392b", lw=0.8, label=f"nu-hat = {nu_hat:.1f}")
ax.set_xscale("log"); ax.set_xlabel("nu"); ax.set_ylabel("copula log likelihood")
ax.legend(frameon=False)
savefig(fig, "p4_nu_profile")

# ---------------------------------------------------------------- Fit (e, f) simulate
rng = np.random.default_rng(SEED)
sims = {"Gaussian": qrm.copula.simulate_gaussian_copula(R, N_SIM, rng),
        "t": qrm.copula.simulate_t_copula(R, nu_hat, N_SIM, rng)}
risk = {}
joint_sim = {}
for key, Us in sims.items():
    Xs = np.column_stack([margin_ppf(j, Us[:, j]) for j in range(d)])
    pnl = PV * Xs.sum(axis=1)
    risk[key] = {"var5": qrm.risk.var_hist(pnl, .05), "es5": qrm.risk.es_hist(pnl, .05),
                 "var1": qrm.risk.var_hist(pnl, .01), "es1": qrm.risk.es_hist(pnl, .01)}
    # Joint tail days per 1,000 days. Margins are continuous and exactly the
    # fitted ones, so "in its own worst 2.5%" is simply U <= 0.025.
    joint_sim[key] = {f"{names[i]}-{names[j]}": {
        "worst": float(1000 * np.mean((Us[:, i] <= TAIL) & (Us[:, j] <= TAIL))),
        "best": float(1000 * np.mean((Us[:, i] >= 1 - TAIL) & (Us[:, j] >= 1 - TAIL)))}
        for i, j in pairs}
pnl_h = PV * x.sum(axis=1)
risk["Historical"] = {"var5": qrm.risk.var_hist(pnl_h, .05), "es5": qrm.risk.es_hist(pnl_h, .05),
                      "var1": qrm.risk.var_hist(pnl_h, .01), "es1": qrm.risk.es_hist(pnl_h, .01)}

# ---------------------------------------------------------------- Reconcile (i)
contrib = llt_obs - llg_obs
middle = np.all((Ur > 0.05) & (Ur < 0.95), axis=1)      # no series in its outer 5% either side
inner = {"total": float(contrib.sum()), "middle_days": int(middle.sum()),
         "middle_sum": float(contrib[middle].sum()), "outer_sum": float(contrib[~middle].sum())}

# ---------------------------------------------------------------- Reconcile (j)
iu = [(i, j) for i, j in pairs]
best = max(iu, key=lambda p: R[p])
lam_t = qrm.copula.tail_dependence_t(R[best], nu_hat)

save_json("p4", {
    "names": names, "moments": mom, "loo": loo, "joint_data": joint_data,
    "expected_indep": expected_indep, "margins": margins, "R": R.tolist(),
    "cop": cop, "dBIC": dBIC, "risk": risk, "joint_sim": joint_sim, "inner": inner,
    "best_pair": [names[best[0]], names[best[1]]], "rho_best": float(R[best]), "lam_t": lam_t,
})

print(pd.DataFrame(mom).T, loo, joint_data, sep="\n")
for c, mg in margins.items():
    print(c, mg["choice"], "N aicc", round(mg["normal"]["aicc"], 1), "t aicc", round(mg["t"]["aicc"], 1),
          "nu", round(mg["t"]["nu"], 2))
print("R\n", np.round(R, 4), "\n", cop, "dBIC", dBIC)
print(pd.DataFrame(risk).round(0)); print(joint_sim); print(inner)
print("best pair", best, R[best], "lambda_t", lam_t)
