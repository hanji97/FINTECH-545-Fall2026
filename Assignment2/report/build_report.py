"""Build the written report from output/*.json.

Every number in the PDF is read from the JSON files the problem scripts write,
so the report cannot drift from the code.

    python report/build_report.py      ->  Assignment 2.pdf  (needs pandoc + xelatex)
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
J = {k: json.load(open(OUT / f"{k}.json")) for k in ("p1", "p2", "p3", "p4", "p5")}
p1, p2, p3, p4, p5 = (J[k] for k in ("p1", "p2", "p3", "p4", "p5"))


# ------------------------------------------------------------------ helpers
def table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def usd(x):
    return f"\\${x:,.0f}" if x >= 0 else f"−\\${-x:,.0f}"


def f(x, d=4):
    return f"{x:.{d}f}"


def sci(x, d=2):
    m, e = f"{x:.{d}e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


def matrix_table(M, names, d=4):
    return table([""] + names, [[n] + [f(v, d) for v in row] for n, row in zip(names, M)])


md = []
add = md.append

# ------------------------------------------------------------------ front matter
add("""---
title: "Assignment 2 — Covariance, VaR, and Copulas"
subtitle: "FinTech 545 — Quantitative Risk Management"
geometry: margin=0.9in
fontsize: 10pt
mainfont: FreeSerif
monofont: DejaVu Sans Mono
numbersections: true
header-includes:
  - \\usepackage{float}
  - \\floatplacement{figure}{H}
  - \\usepackage{booktabs}
---

Conventions used throughout: arithmetic returns; VaR and ES at $\\alpha=5\\%$ unless stated, positive
numbers are losses, absolute (not mean-relative) convention; historical quantile = the
$\\lceil n\\alpha\\rceil$-th worst value; variance with $n-1$; skew and **excess** kurtosis bias-corrected.
All simulations use 100,000 draws and seed 545. Code: the `qrm` Python library plus one script per
problem (see README).
""")

# ================================================================== PROBLEM 1
c = p1["cols"]
counts = p1["counts"]
eig_cc, eig_pw = p1["eig_cc"], p1["eig_pw"]
rep = p1["repairs"]
w = np.array([-0.4, -0.3, -0.2, -0.1, 1.0]) * np.array(p1["sd"])
cos = abs(w @ np.array(p1["neg_vec"])) / np.linalg.norm(w)
mh = np.abs(np.array(p1["move_hi"]))
add(f"""
# Correlations from Mismatched Histories

## Predict

Days jointly observed (diagonal = days each series trades). **All five trade on {p1['n_all']} days.**

{table([''] + c, [[c[i]] + counts[i] for i in range(5)])}
**(a)** Complete case is guaranteed PSD: it is an ordinary sample correlation matrix of one common
data matrix, i.e. a Gram matrix, so $w'Cw$ is the sample variance of a real portfolio and cannot be
negative. Pairwise can fail: each entry is estimated on a different set of days, so the entries need not be
mutually consistent with any single joint distribution.

**(b)** If IDX were an exact linear combination the true matrix would be singular (smallest eigenvalue 0).
With only a small tracking error, the smallest true eigenvalue is just above 0. There is no buffer, so the
small inconsistencies created by pairwise estimation are enough to push it below zero. This data set is
highly exposed.

**(c)** The pairs involving D (93–96 days, C–D only 83), because the standard error of $\\hat\\rho$ is
roughly $(1-\\rho^2)/\\sqrt n$: about 0.10 at $n\\approx 90$ against 0.06 at $n\\approx 240$. D's
correlations are also measured on a different (later) window than everything else.

## Fit

**(d)**

{table(['', '$\\lambda_1$', '$\\lambda_2$', '$\\lambda_3$', '$\\lambda_4$', '$\\lambda_5$', 'Cholesky'],
       [['Complete case'] + [f(v) for v in eig_cc] + ['succeeds'],
        ['Pairwise'] + [f(v) for v in eig_pw] + ['**fails**']])}
Complete case ({p1['n_all']} days):

{matrix_table(p1['C_cc'], c)}
Pairwise:

{matrix_table(p1['C_pw'], c)}
**(e)** With $\\Sigma = DCD$ (pairwise $C$, full-history $\\sigma$), the portfolio
$w = (-0.4, -0.3, -0.2, -0.1, +1)$ on (A, B, C, D, IDX) has
$w'\\Sigma w =$ **{sci(p1['var_pw'])}**: a **negative variance**. (For reference, the realised variance of the
tracking residual on the {p1['n_all']} complete days is {sci(p1['var_realised'])}.)

**(f)**

{table(['Repair', 'Smallest eigenvalue', 'Frobenius distance from pairwise', 'Tracking variance'],
       [['Rebonato–Jäckel', sci(rep['RJ']['min_eig']), f(rep['RJ']['frob']), sci(rep['RJ']['track_var'])],
        ['Higham', sci(rep['Higham']['min_eig']) + ' (≈0, float)', f(rep['Higham']['frob']),
         sci(rep['Higham']['track_var'])]])}
Both repaired matrices pass Cholesky.

## Reconcile

**(g)** $w'\\Sigma w$ is the variance of portfolio $w$; PSD means it is $\\ge 0$ for every $w$. Finding a $w$
with $w'\\Sigma w<0$ is a direct proof the matrix is not PSD. It is not a random $w$: the eigenvector of
the negative eigenvalue has cosine similarity **{cos:.3f}** with the tracking portfolio (in correlation units).
The one portfolio the data says is nearly riskless is exactly the direction that went negative.

**(h)** Higham moved the **IDX row** most (IDX–A {f(p1['move_hi'][0][4])}, IDX–C {f(p1['move_hi'][2][4])},
IDX–B {f(p1['move_hi'][1][4])}); D's entries moved only about {f(mh[3, :3].max(), 3)}. That is *not* my answer to (c).
The repair knows nothing about sample sizes. It uses only geometry: the negative eigenvalue's eigenvector
(loading {f(p1['neg_vec'][4], 2)} on IDX) and the nearest point in Frobenius norm, so it changes the entries that
most cheaply remove that direction. Sample-size information would need a weighted Higham norm.

**(i)** Repair distance ≈ {f(rep['Higham']['frob'], 3)} against an estimator gap
$\\|C_{{cc}}-C_{{pw}}\\|_F$ = **{f(p1['gap_est'], 3)}**, about {p1['gap_est'] / rep['Higham']['frob']:.0f}× larger.
The two repairs give almost the same tracking variance ({sci(rep['RJ']['track_var'])} vs
{sci(rep['Higham']['track_var'])}), while complete case gives {sci(p1['var_cc'])}. **The choice of estimator
matters far more than the choice of repair.**
""")

# ================================================================== PROBLEM 2
m2 = p2["moments"]
v = p2["var"]
ew = p2["ew"]
t = p2["t"]
order = sorted(v, key=v.get)
add(f"""
# A Volatility Estimate After the Regime Changed

## Predict

![Demeaned arithmetic returns. Volatility jumps for the last {p2['k_recent']} days.](../output/p2_returns.png){{width=88%}}

**(a)** Predicted order, smallest to largest: **Student t ≈ Normal equal-weight < Historical < Normal EW 0.97 <
Normal EW 0.94.** The equal-weight methods give the last {p2['k_recent']} high-volatility days only
{100 * ew['equal_weight_recent']:.0f}% of the weight. Historical takes the 25th-worst of 500 days, and the storm
days supply many of the worst ones, so it should sit a little above. The fitted t matches the overall
variance with fatter tails, so its 5% quantile should be near (or inside) the normal one. The EW estimators
put most of their weight on the storm; 0.94 more than 0.97.

**(b)**

{table(['λ', '$n_{eff}$', 'Half life (days)', f"Weight on last {p2['k_recent']} days"],
       [[l, f(ew[l]['n_eff_sample'], 1), f(ew[l]['half_life'], 1), f"{100 * ew[l]['weight_recent']:.1f}%"]
        for l in ('0.94', '0.97')] + [['Equal', p2['n'], '–', f"{100 * ew['equal_weight_recent']:.1f}%"]])}
**(c)** Mean 0 (demeaned), sd {100 * m2['std']:.3f}%, skew {f(m2['skew'], 3)}, excess kurtosis
**{f(m2['kurt'], 2)}** (Jarque–Bera p ≈ {p2['jb_p']:.0e}). A single normal regime gives excess kurtosis ≈ 0, so
I predict it could **not** produce these moments.

## Fit

**(d)**

{table(['Method', 'One-day 5% VaR'], [[k, usd(v[k])] for k in v])}
Fitted t: $\\nu$ = {f(t['nu'], 2)}, $\\mu$ = {t['mu']:.5f}, scale $s$ = {t['s']:.5f}
(implied sd {100 * p2['t_sd']:.3f}%).

## Reconcile

**(e)** Realised order: {' < '.join(order)}. This matches the prediction. The only point worth noting is
that the t sits *below* the equal-weight normal: at the same variance a fat-tailed distribution has more mass
in the centre and the far tail, so its 5% point is slightly inside the normal's.

**(f)** sd before the break = **{100 * p2['sd_calm']:.3f}%**, last {p2['k_recent']} days = **{100 * p2['sd_storm']:.3f}%**
(2.5×). Inside each regime the excess kurtosis is ≈ 0 ({f(p2['mom_calm']['kurt'], 2)} and
{f(p2['mom_storm']['kurt'], 2)}). A mixture of two zero-mean normals with these weights and variances has
excess kurtosis {f(p2['kurt_mixture'], 2)}, close to the {f(m2['kurt'], 2)} observed. The fat tails are the
**mixing of two regimes**, not fat tails within either one. The fitted t is a scale mixture of normals, so it is
describing the *unconditional* blend of both regimes (implied sd {100 * p2['t_sd']:.2f}%), not today's
2.5% volatility.

**(g)** Standard error of the VaR from $\\sigma/\\sqrt{{2n_{{eff}}}}$: {usd(p2['se']['0.94']['se_var_dollars'])}
(λ=0.94) and {usd(p2['se']['0.97']['se_var_dollars'])} (λ=0.97). The gap between the two EW VaRs is
{usd(p2['gap'])}, about one standard error ({usd(p2['se_gap'])} if the errors were independent). It is **not
clearly larger than the noise**. Because both estimates use the same returns, though, most of the gap is the
real difference that 0.97 still puts 34% of its weight on the calm regime. *Opinion:* use **0.94 for a
trading limit**, which should react within days, and **0.97 (or slower) for a capital number**, which should
be stable and not whipsaw.
""")

# ================================================================== PROBLEM 3
mo = p3["moments"]
cn = p3["counts"]
tb = p3["table"]
sb = p3["sub"]
keys = list(tb)
add(f"""
# When Diversification Raises VaR

## Predict

![One-year P&L, log scale. Concentrated: two outcomes (no default / default). Diversified: three.](../output/p3_pnl.png){{width=90%}}

**(a)**

{table(['Bond', 'Mean', 'Sd', 'Skew', 'Excess kurt.'],
       [[b, f"{100 * mo[b]['mean']:.2f}%", f"{100 * mo[b]['std']:.2f}%", f(mo[b]['skew'], 2), f(mo[b]['kurt'], 1)]
        for b in ('A', 'B')])}
Skew ≈ −5 and kurtosis ≈ 22 describe a two-point mixture: about 96% of scenarios near +11% and about 4%
of defaults at −50% to −85%. A normal VaR places the 5% quantile in the empty region between the two
clusters. It is meaningless for these positions.

**(b)** Scenarios losing more than 20%: A **{cn['A']}** ({100 * cn['A'] / cn['n']:.2f}%), B **{cn['B']}**,
at least one **{cn['at_least_one']}** ({100 * cn['at_least_one'] / cn['n']:.2f}%), both {cn['both']}.

* **VaR: diversified larger.** \\$2M in A defaults in 4.06% < 5% of scenarios, so its 5% point is a
  no-default (profit) scenario. A VaR is negative. The diversified book has a default in 7.82% > 5%, so its
  5% point is a one-default scenario, a large loss.
* **ES: diversified smaller.** The concentrated worst 5% contains about 406 scenarios of losing ~60–80% of
  \\$2M. The diversified worst 5% is almost all single defaults, where only \\$1M is hit and the other bond
  still earns +11%.

## Fit

**(c, d, e)**

{table(['Position', 'Hist VaR 5%', 'Hist ES 5%', 'Normal VaR 5%', 'Hist VaR 1%', 'Hist ES 1%'],
       [[k, usd(tb[k]['var5']), usd(tb[k]['es5']), usd(tb[k]['nvar5']), usd(tb[k]['var1']), usd(tb[k]['es1'])]
        for k in keys])}
## Reconcile

**(f)**

{table(['Measure', 'VaR(A) + VaR(B)', 'VaR(A + B)', 'Subadditive?'],
       [['VaR 5%', usd(sb['var5']['sum_parts']), usd(sb['var5']['combined']), 'No'],
        ['ES 5%', usd(sb['es5']['sum_parts']), usd(sb['es5']['combined']), 'Yes']])}
VaR reads one point of the distribution. Combining the bonds moves the default mass from ~4% (just
outside the 5% window) to 7.8% (inside it), so the 5% point jumps from a profit to a loss. ES averages
the whole tail, sees the defaults in both cases, and remains subadditive (it is coherent).

**(g)** Normal VaR: \\$2M A {usd(tb['$2M A']['nvar5'])} versus diversified {usd(tb['$1M A + $1M B']['nvar5'])},
so it favours **diversification**. That is the choice ES also favours, but for the **wrong reason**: it only sees
the standard deviation fall by about $\\sqrt2$ (independent bonds), and its levels are wrong in both cases. It
reports a loss of {usd(tb['$2M A']['nvar5'])} where the true 5% outcome is a profit, and
{usd(tb['$1M A + $1M B']['nvar5'])} where the truth is {usd(tb['$1M A + $1M B']['var5'])}.

**(h)** At 1% both choices' quantiles fall inside the default region: {usd(tb['$1M A']['var1'])} +
{usd(tb['$1M B']['var1'])} = {usd(sb['var1']['sum_parts'])} ≥ {usd(sb['var1']['combined'])}, so VaR is subadditive
again. The failure only appears when $\\alpha$ lies **between the single-default probability (~4%) and the
any-default probability (~7.8%)**. It is a property of $\\alpha$ relative to the event probabilities, so a
VaR that looks fine at one confidence level can break at another.
""")

# ================================================================== PROBLEM 4
nm = p4["names"]
mom4 = p4["moments"]
loo = p4["loo"]
mg = p4["margins"]
cop = p4["cop"]
rk = p4["risk"]
jd = p4["joint_data"]
js = p4["joint_sim"]
inner = p4["inner"]
pair_keys = list(jd)


def nu_str(x):
    return "→ ∞" if x > 1e3 else f(x, 2)


tot = lambda dct: sum(v_['worst'] + v_['best'] for v_ in dct.values())
add(f"""
# Gaussian or t Copula

## Predict

**(a)**

{table(['', 'Mean', 'Sd', 'Skew', 'Excess kurt.', 'Kurt. without most extreme day'],
       [[c_, f"{100 * mom4[c_]['mean']:.3f}%", f"{100 * mom4[c_]['std']:.3f}%", f(mom4[c_]['skew'], 2),
         f(mom4[c_]['kurt'], 2), f(loo[c_]['kurt_without_max'], 2)] for c_ in nm])}
From the moments alone: **X1 normal** (skew and excess kurtosis ≈ 0), **X2 and X3 Student t**. The out-of-place
number is X2's kurtosis of {f(mom4['X2']['kurt'], 1)}. Almost all of it is **one day** (day {loo['X2']['day']}:
X2 = {100 * loo['X2']['value']:.1f}%, a {abs(loo['X2']['z']):.1f}σ move). Without it the kurtosis is
{f(loo['X2']['kurt_without_max'], 2)}. The same day is the extreme for all three series, so it is a joint crash.
Even without it X2 and X3 stay fat-tailed, so the choice of t for those two does not change.

**(b)**

![Rank-transformed pairs. Lines mark the 2.5% and 97.5% levels.](../output/p4_ranks.png){{width=92%}}

{table(['Pair', 'Both in worst 2.5%', 'Both in best 2.5%'], [[k, jd[k]['worst'], jd[k]['best']] for k in pair_keys])}
Under independence the expected count is $1000 \\times 0.025^2$ = **{p4['expected_indep']:.3f}** per pair per
tail. The observed 6–12 are an order of magnitude higher. With correlations around 0.5 some of that is
expected even under a Gaussian copula, but the scatter shows tight clusters in *both* corners. I predict the
**t copula wins**.

## Fit

**(c)** Margins by AICc ($n$ = 1000):

{table(['', 'Normal AICc', 't AICc', 't $\\nu$', 'Chosen'],
       [[c_, f(mg[c_]['normal']['aicc'], 1), f(mg[c_]['t']['aicc'], 1), nu_str(mg[c_]['t']['nu']), mg[c_]['choice']]
        for c_ in nm])}
For X1 the t's $\\nu$ diverges, meaning it collapses to the normal and loses on the extra parameter.

**(d)** $R = \\sin(\\pi\\tau/2)$ (PSD, no repair needed):

{matrix_table(p4['R'], nm)}
{table(['Copula', 'Log likelihood', '$\\hat\\nu$', 'k', 'AICc', 'BIC'],
       [['Gaussian', f(cop['gauss']['ll'], 2), '–', 0, f(cop['gauss']['aicc'], 2), f(cop['gauss']['bic'], 2)],
        ['t', f(cop['t']['ll'], 2), f(cop['t']['nu'], 2), 1, f(cop['t']['aicc'], 2), f(cop['t']['bic'], 2)]])}
![Profile log likelihood of the t copula over $\\nu$.](../output/p4_nu_profile.png){{width=55%}}

**(e)** Portfolio (\\$1M in each asset), one day:

{table(['', 'VaR 5%', 'ES 5%', 'VaR 1%', 'ES 1%'],
       [[k, usd(rk[k]['var5']), usd(rk[k]['es5']), usd(rk[k]['var1']), usd(rk[k]['es1'])] for k in ('Gaussian', 't', 'Historical')])}
**(f)** Joint-tail days implied per 1,000 days:

{table(['Pair', 'Data worst / best', 'Gaussian worst / best', 't worst / best'],
       [[k, f"{jd[k]['worst']} / {jd[k]['best']}",
         f"{js['Gaussian'][k]['worst']:.1f} / {js['Gaussian'][k]['best']:.1f}",
         f"{js['t'][k]['worst']:.1f} / {js['t'][k]['best']:.1f}"] for k in pair_keys])}
## Reconcile

**(g)** **The t copula wins decisively**: $\\Delta BIC = 2(\\ell_t-\\ell_G)-\\ln m$ = **{p4['dBIC']:.1f}**
(>10 is strong), with $\\hat\\nu$ = {f(cop['t']['nu'], 2)}. Summed over all pairs and both tails, the data have
{tot(jd)} joint-tail days. The Gaussian implies {tot(js['Gaussian']):.0f} and the t {tot(js['t']):.0f}. The t
reproduces the joint tails; the Gaussian under-counts them by about 40%.

**(h)** **ES at 1%** moves the most: {usd(rk['Gaussian']['es1'])} → {usd(rk['t']['es1'])}
(+{100 * (rk['t']['es1'] / rk['Gaussian']['es1'] - 1):.1f}%). The t value matches the historical
{usd(rk['Historical']['es1'])}. The 5% VaR barely moves ({usd(rk['Gaussian']['var5'])} vs {usd(rk['t']['var5'])}):
both copulas share the same $R$ and margins, so the portfolio variance and the body of the distribution are the
same. Tail dependence only changes how often *extreme* days coincide, and the 5% point is not extreme enough
to see that.

**(i)** Total $\\sum_t(\\ell_t-\\ell_{{G,t}})$ = {f(inner['total'], 1)}. The **{inner['middle_days']} days on which no
series sits in its outer 5%** contribute **{f(inner['middle_sum'], 1)}** ({100 * inner['middle_sum'] / inner['total']:.0f}%).
About half the evidence for the t comes from ordinary days. The t copula density is also shaped differently in
the centre, more peaked along the diagonal. The information criterion scores the fit of the **whole** copula
density, dominated by the middle where most data lie, not specifically the tail we care about.

**(j)** Most correlated pair {p4['best_pair'][0]}–{p4['best_pair'][1]}, $\\rho$ = {f(p4['rho_best'], 3)}:
$\\lambda = 2\\,t_{{\\nu+1}}\\!\\left(-\\sqrt{{(\\nu+1)(1-\\rho)/(1+\\rho)}}\\right)$ = **{f(p4['lam_t'], 3)}** under the
t copula, and **0** under the Gaussian, for any $\\rho<1$.
""")

# ================================================================== PROBLEM 5
r5 = p5["results"]
a, b = p5["alpha"], p5["beta"]
se5 = p5["sd_eps"]
add(f"""
# Model Based Simulation and Residual Correlation

## Predict

**(a)** With $r_i = \\beta_i r_M + \\epsilon_i$ and dollar positions $w_A, w_B$:
$$\\operatorname{{Var}}(P) = (w_A\\beta_A + w_B\\beta_B)^2\\sigma_M^2 + w_A^2\\sigma_{{\\epsilon A}}^2 + w_B^2\\sigma_{{\\epsilon B}}^2
+ 2\\,w_Aw_B\\,\\rho_\\epsilon\\sigma_{{\\epsilon A}}\\sigma_{{\\epsilon B}}.$$
A shared industry makes $\\rho_\\epsilon>0$. The shortcut sets the last term to zero.
**P1** ($w_A w_B>0$): the dropped term is positive, so the shortcut **understates** VaR.
**P2** ($w_A w_B<0$): the dropped term is negative, so the shortcut **overstates** VaR. P2 should be hit harder:
with similar betas its market term nearly cancels, so the residual terms are almost all of its variance.

## Fit

**(b)**

{table(['', '$\\alpha$', '$\\beta$', 'Residual sd'],
       [['A', f"{a[0]:.5f}", f(b[0], 4), f"{100 * se5[0]:.3f}%"], ['B', f"{a[1]:.5f}", f(b[1], 4), f"{100 * se5[1]:.3f}%"]])}
Residual correlation $\\rho_\\epsilon$ = **{f(p5['rho_eps'], 3)}**. The simulation uses $\\alpha=0$ and a zero-mean
market (zero expected returns).

**(c, d)** One-day 5% VaR:

{table(['', 'Simulated, full $\\Sigma_\\epsilon$', 'Simulated, diagonal $\\Sigma_\\epsilon$', 'Delta normal (sample cov)'],
       [[p_, usd(r5[p_]['sim_full']), usd(r5[p_]['sim_diag']), usd(r5[p_]['delta_normal'])] for p_ in ('P1', 'P2')])}
## Reconcile

**(e)** Yes. The shortcut moved P1 **down** {100 * (1 - r5['P1']['sim_diag'] / r5['P1']['sim_full']):.0f}% and P2
**up** {100 * (r5['P2']['sim_diag'] / r5['P2']['sim_full'] - 1):.0f}%. P2 is far more exposed: its market exposure
is only $\\beta_A-\\beta_B$ = {f(b[0] - b[1], 3)}, so the residual covariance is nearly all of its risk, and the
dropped cross term is large relative to the rest.

**(f)** Full-covariance model {usd(r5['P1']['sim_full'])} / {usd(r5['P2']['sim_full'])} (closed form
{usd(r5['P1']['exact_full'])} / {usd(r5['P2']['exact_full'])}) versus delta normal
{usd(r5['P1']['delta_normal'])} / {usd(r5['P2']['delta_normal'])}. They are **the same up to simulation noise**:
$\\beta\\beta'\\sigma_M^2+\\Sigma_\\epsilon$ reproduces the sample covariance to {sci(p5['max_diff'], 1)}. Week 5:
under a joint normal, OLS is just the conditional distribution. $\\hat\\beta=\\Sigma_{{YX}}\\Sigma_{{XX}}^{{-1}}$ and
$\\Sigma_\\epsilon$ is the conditional covariance, so a regression plus a full residual covariance hands back the
joint normal you started with. They **stop agreeing** when any piece is not that joint normal: a diagonal
$\\Sigma_\\epsilon$ (part c), non-normal errors or market (t errors via a copula), non-linear payoffs, factor and
residual parts estimated on different windows or weights, or time-varying $\\beta$ and volatility.

# Code

Python library `qrm/` (moments, correlation, psd, ewma, fitting, risk, simulation, copula) plus
`problems/problem1.py`–`problem5.py`. `python run_all.py` reproduces every number in this report;
see README.md.
""")

text = "\n".join(md)
(ROOT / "report" / "report.md").write_text(text)
pdf = ROOT / "Assignment 2.pdf"

# Some minimal TeX installs ship the Latin Modern font files but not lmodern.sty,
# which pandoc's template requests. The font itself is set with fontspec
# (FreeSerif), so an empty stub is enough; it is only used when the real
# package is missing.
env = dict(os.environ)
if not subprocess.run(["kpsewhich", "lmodern.sty"], capture_output=True, text=True).stdout.strip():
    stub = ROOT / "report" / "texstub"
    stub.mkdir(exist_ok=True)
    (stub / "lmodern.sty").write_text("\\ProvidesPackage{lmodern}\n")
    env["TEXINPUTS"] = f"{stub}{os.pathsep}" + env.get("TEXINPUTS", "")

subprocess.run(["pandoc", str(ROOT / "report" / "report.md"), "-o", str(pdf),
                "--pdf-engine=xelatex", "--resource-path", str(ROOT / "report")],
               check=True, cwd=ROOT / "report", env=env)
print("wrote", pdf)
