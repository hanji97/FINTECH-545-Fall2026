# FinTech 545 Risk Library

Julia implementations of the course material in Weeks 01-07, plus the risk
parity and ex-post attribution cases that build on Week 07, checked against the
class test cases. Cases 1.1-7.6 were in the first submission. This submission
adds 8.1-13.9, 25 cases. All 50 cases pass.

## Running

```
cd testfiles
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. run_tests.jl
```

About 30 seconds once the packages are compiled. Each case reads its inputs
from `testfiles/data`, writes the computed table to `testfiles/output` under the
same file name and columns as the class expected file, reads that file back and
compares it with the expected file in `data`. One line per case is printed, the
summary is saved to `output/test_results.csv`, and the script exits non-zero if
any case fails. `run_tests.jl` writes every file in `output` on each run; none
are copied from `data`.

## Layout

```
library/
  RiskLib.jl            entry point, loads every module below
  missing_cov.jl        week 03     covariance with missing data
  ew_cov.jl             week 03     exponentially weighted covariance
  psd_fixes.jl          week 03     near_psd, Higham, chol_psd
  simulate.jl           week 03     normal and PCA simulation
  return_calculate.jl   week 04     prices to returns
  fitted_model.jl       week 01-02  normal, t, t regression, AICc
  nig.jl                week 01     NIG by moments and by MLE
  risk_stats.jl         week 04-05  VaR, ES, portfolio VaR/ES by group and total
  multivariate_t.jl     week 05     Kendall correlation, multivariate t       (new)
  copula.jl             week 05     Gaussian and t copulas, tail dependence   (new)
  gbsm.jl               week 06-07  GBSM and its Greeks, implied vol
  binomial_tree.jl      week 06-07  American trees, numerical Greeks
  portfolio.jl          week 07     max Sharpe ratio, risk parity
  attribution.jl        week 07-08  ex-post return and volatility attribution (new)
testfiles/
  run_tests.jl          runs cases 1.1-13.9
  Project.toml, Manifest.toml
  data/                 class inputs and expected outputs, unchanged
  output/               computed outputs, one CSV per case, and test_results.csv
tools/
  compare_nig_fits.jl, nig_scipy_reference.py    cases 7.5 and 7.6 side by side
```

`gbsm.jl`, the two `bt_american` trees and `maxSR` are from my Assignment 3
library. New for this submission: `aggRisk` and `simulated_values` in
`risk_stats.jl`, `multivariate_t.jl`, `copula.jl`, `attribution.jl`,
`riskParity` in `portfolio.jl`, and the finite difference Greeks in
`binomial_tree.jl`.

## Results, cases 8.1-13.9

Largest gap between the computed file in `output` and the class file in `data`.

| Test | Case | max abs diff | tolerance |
|:--|:--|--:|:--|
| 8.1 | VaR, normal | 7e-18 | abs 1e-12 |
| 8.2 | VaR, generalized t | 6e-13 | rel 1e-6 |
| 8.3 | VaR, simulation from the fitted t | 6e-13 | rel 1e-6 |
| 8.4 | ES, normal | 7e-12 | abs 1e-9 |
| 8.5 | ES, generalized t | 2e-12 | rel 1e-6 |
| 8.6 | ES, simulation from the fitted t | 2e-12 | rel 1e-6 |
| 9.1 | VaR/ES by stock and total, Gaussian copula | 6e-9 | rel 1e-6 |
| 10.1 | Risk parity | 1e-10 | abs 1e-6 |
| 10.2 | Risk parity, half budget on X5 | 3e-11 | abs 1e-6 |
| 10.3 | Max Sharpe ratio, w ≥ 0 | 2e-8 | abs 1e-6 |
| 10.4 | Max Sharpe ratio, 0.1 ≤ w ≤ 0.5 | 6e-8 | abs 1e-6 |
| 11.1 | Ex-post attribution, stocks | 3e-16 | abs 1e-12 |
| 11.2 | Ex-post attribution, factors | 9e-16 | abs 1e-12 |
| 12.1 | European options, GBSM and Greeks | 9e-16 | abs 1e-9 |
| 12.2 | American options, continuous dividends, Greeks | 6e-8 | abs 1e-6 |
| 12.3 | American options, discrete dividends | 2e-15 | abs 1e-9 |
| 13.1 | Correlation from Kendall's tau | 0 | abs 1e-12 |
| 13.2 | Multivariate t, mean | 2e-19 | abs 1e-12 |
| 13.3 | Multivariate t, scale matrix | 7e-20 | rel 1e-9 |
| 13.4 | Multivariate t, ν and log likelihood | 5e-13 | rel 1e-9 |
| 13.5 | Gaussian copula log likelihood | 8e-10 | rel 1e-6 |
| 13.6 | t copula, ν and log likelihood | 1e-9 | rel 1e-6 |
| 13.7 | AICc and BIC | 2e-9 | rel 1e-6 |
| 13.8 | Tail dependence | 0 | rel 1e-6 |
| 13.9 | VaR/ES by stock and total, t copula | 1e-8 | rel 1e-6 |

Deterministic cases (closed forms, the attribution, the trees in 12.3, the
Kendall correlation and grid profile in 13.1-13.4) are held to 1e-9 or tighter.
Cases whose answer comes from an iterative optimizer (the generalized t fits
behind 8.2, 8.3, 8.5, 8.6, 9.1 and 13.5-13.9, Ipopt in 10.x) or from finite differences on a tree (12.2) are held
to 1e-6. 10.3 is checked on the absolute gap because three expected weights are
about -1e-8, Ipopt's bound tolerance around zero, where a relative gap is not
meaningful.

Cases 1.1-7.6 are unchanged from the first submission and all pass; the
largest gap is still 7.6 at 1.9e-5 relative, against a scipy fit.

## Notes on cases 8.1-13.9

**8.1-8.6, VaR and ES.** Reported as positive numbers for a loss. "Absolute"
is measured from zero and "Diff from Mean" from the mean (Week 04, 2.1). The
VaR quantile on a sample averages the order statistics at ⌊nα⌋ and ⌈nα⌉; ES is
the mean of everything at or below it. Normal ES uses the closed form in
Week 05, 1.1; t ES integrates x f(x) over the tail (Week 05, 1). 8.3 and 8.6
draw 10,000 values from the fitted t through its quantile function, with the
class setup's seed and order of draws, so they reproduce the expected files.
Against the values from the fitted distribution in 8.2 and 8.5 the gaps are
sampling error: +0.0021 (absolute) and +0.0012 (from the mean) on VaR, -0.0005
and -0.0002 on ES.

**9.1, Gaussian copula.** A is fitted as a normal and B as a generalized t, as
named in `test9_1_portfolio.csv`; the position values are Holding × Starting
Price. R is the Spearman correlation of U (Week 05, 4.1), then 100,000 PCA
draws, Φ, and each margin's quantile function (4.2). Spearman is used directly
as R here, as in 4.1; case 13 uses Kendall's tau instead because 6.5 compares
two copulas on one R, and sin(πτ/2) is exact for both. P&L is summed across
stocks within each draw before the quantile is taken, so the portfolio VaR is
below the sum of the stock VaRs: 165.14 against 211.46.

**10.1-10.2, risk parity.** The contribution of asset i to portfolio volatility
is CSDᵢ = wᵢ(Σw)ᵢ/σₚ, and the CSDᵢ add up to σₚ because σₚ is homogeneous of
degree one in w. The weights are long only and make CSDᵢ/bᵢ equal for every
asset, with b the risk budget (all ones, then 0.5 on X5). Solved with Ipopt by
minimizing the spread of CSDᵢ/(bᵢσₚ), the budget-weighted shares of portfolio
volatility, which keeps the objective free of the units of Σ. At the solution
the spread is below 1e-11 of its mean.

**10.3-10.4, maximum Sharpe ratio.** (wᵀμ - r_f)/√(wᵀΣw) with the weights
summing to one (Week 07, 5), solved with Ipopt from equal weights and from 20
random starting points, best kept. The class setup runs both cases on
`test5_3.csv`, not the `test5_2.csv` listed in `Tests.xlsx`, so that is what
the runner reads. That matrix is the non-PSD one from 5.3 (smallest eigenvalue
-0.0048), but every entry is non-negative, so w'Σw ≥ Σᵢ wᵢ²Σᵢᵢ > 0 for any
long-only w and the Sharpe ratio is defined on the whole feasible set of both
cases. No repair is applied, as in the class setup.

**11.1-11.2, ex-post attribution.** Weights drift with returns, so the factor
exposure Betasᵀwₜ is recomputed every day. Daily contributions are linked over
the period with Carino's k, which makes the return attributions add up to the
compounded portfolio return. Volatility attribution regresses each daily
contribution on the portfolio return; the slopes times σ(R) add up to σ(R).
11.1 is the stock level case (each stock is its own factor, betas = I), so the
residual is zero and its column is dropped.

**12.1-12.3, options.** GBSM with cost of carry b = r - q (Week 06, 4.3) and
the Greeks in Week 07, Table 1, with theta = -∂V/∂T per year. 12.2 prices on a
CRR tree with N = 500 (Week 07, 1.3) and takes the sensitivities by central
differences (Week 07, 3.3) with step ∛ε·max(|x|, 1). The tree value is
piecewise linear in S between nodes, so gamma uses a step of 1.5 in S instead.
The class file for 12.2 reports the partial derivatives of the tree value in
its inputs (S, X, T, r, b, σ), and the runner reports the same so the files
compare: Theta there is +∂V/∂T and Rho moves r with b held fixed. For option 1
(q = 0, where the American call is the European one) that gives Theta 13.01
against -13.02 in 12.1, and Rho -0.45, which is only the discounting effect
-T·V. Against the European value on the same tree, the early exercise premium
is 0 for the call with q = 0, 0.0056 for the call with q = 2%, and 0.048 and
0.56 for the two puts, in line with Week 07, 1.1. 12.3 uses the non-recombining
tree for cash dividends (Week 07, 2.3) with two steps per day, so the dividend
days map to steps 150 and 300 of 500. At a dividend node the exercise value
uses the cum-dividend price. The dividends are 0.01, far below the
X(1 - e^(-r(T-tₙ))) of 2.13 and 1.23 that Week 07, 2.2 requires before
exercising a call ahead of a dividend, so the American call equals the European
one; the put carries a premium of 0.29.

**13.1-13.4, multivariate t.** R = sin(πτ/2) from Kendall's tau (Week 05,
5.3.1). Its smallest eigenvalue is 0.536, so no repair was needed. A negative
eigenvalue would send the matrix through Higham (Week 03), and if it were still
not positive definite it would be shifted to a smallest eigenvalue of 1e-8 and
rescaled to a unit diagonal, since the densities need a Cholesky root. The fit
is the rank based one in
5.3.2: μ at the sample mean, scales tied to the sample standard deviations,
S(ν) = D(ν)RD(ν), and ν by a 200 point grid on θ = 1/ν ∈ [0.01, 0.49] refined
between the neighbors of the maximum. ν = 5.82 (the data were drawn at 6),
log likelihood 3811.28. By construction ν/(ν-2)·S has the sample variances on
its diagonal.

**13.5-13.9, copulas.** The margins are generalized t fits, frozen, and both
copulas get the same R, so they differ only by ν (Week 05, 6.5). The copula
log likelihood is computed as a difference of two sums of log densities rather
than a ratio of densities (6.2). The t copula has ν = 5.95 and log likelihood
128.70 against 98.65 for the Gaussian. With k = 0 and 1 and m = 250,
ΔBIC = 2(128.70 - 98.65) - ln 250 = 54.58, so the t copula is preferred.
Tail dependence runs from 0.08 to 0.13 across the pairs (6.4); every pair is 0
under the Gaussian copula (4.3). 13.9 simulates the t copula through the
mixture in 5.2, one chi-square draw per scenario shared by all five assets,
then the t CDF at the copula's ν and each margin's own quantile function (6.3).
Total VaR95 is 163.00 and ES95 228.57. The runner also prints the Gaussian
copula with the same margins and R: VaR95 167.09 and ES95 223.56. Week 05, 4.3
says tail dependence barely moves a 5% VaR and changes how often positions lose
together; here the VaR is close under both copulas and the ES, which averages
over the joint tail, is higher under the t.

## Conventions

- VaR and ES are positive for a loss, α = 0.05 unless the column says 99.
- `fit_normal` uses the (n-1) standard deviation; 7.1 and 8.1 expect it.
- σ in the generalized t is the scale, not the standard deviation:
  var = σ²ν/(ν-2).
- AICc = -2ℓ + 2k + 2k(k+1)/(n-k-1). For the copulas the margins and R are
  fitted once and shared, so k counts only what the two copulas do not share
  (0 Gaussian, 1 t) and n is the number of days (Week 05, 6.5).
- The multivariate t scale matrix S is not the covariance: cov = ν/(ν-2)·S.
- GBSM theta is -∂V/∂T per year; vega and rho are per 1.00 of vol and rate.
- Simulations use fixed seeds, so every run gives the same output files.

## Environment

Julia 1.11.9. Packages pinned in `testfiles/Manifest.toml`: CSV, DataFrames,
Distributions, StatsBase, Optim, ForwardDiff, SpecialFunctions, QuadGK, Roots,

JuMP, Ipopt.

`tools/nig_scipy_reference.py` needs Python with scipy and is only used for
the 7.6 comparison; the library and the runner have no Python dependency.
