# FinTech 545 — Risk Library

Julia implementations of the week 01–03 material, with a test harness that runs
them against the supplied expected outputs.

Tests 1.1 – 7.6, 25 cases, all passing.

## Running

```
cd testfiles
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. run_tests.jl
```

Exits non-zero if any case fails. The expected CSVs in `testfiles/data/` are the
only thing the harness compares against.

## Layout

```
library/
  RiskLib.jl                entry point, include this
  missing_cov.jl            covariance and correlation with missing data
  ew_cov.jl                 exponentially weighted covariance
  psd_fixes.jl              near_psd, Higham, chol_psd
  simulate.jl               normal and PCA simulation
  return_calculate.jl       prices to returns
  fitted_model.jl           FittedModel, AICc, normal / t / t-regression fits
  nig.jl                    NIG by method of moments and by MLE
testfiles/
  run_tests.jl              test harness
  data/                     supplied inputs and expected outputs
tools/
  compare_nig_fits.jl       tests 7.5 and 7.6 side by side
  nig_scipy_reference.py    scipy's NIG fit, used to check 7.6
```

`RiskLib.jl` loads as a plain script and the function signatures match the ones
`test_setup.jl` calls — `missing_cov`, `ewCovar`, `near_psd`,
`higham_nearestPSD`, `chol_psd!`, `simulateNormal`, `simulate_pca`,
`return_calculate`, `fit_normal`, `fit_general_t`, `fit_regression_t`, `aicc`.

## Results

| Test | Case | max abs diff |
|:--|:--|--:|
| 1.1–1.4 | covariance / correlation, skip-missing and pairwise | 0 |
| 2.1–2.3 | EW covariance λ=0.97, EW correlation λ=0.94, mixed | 4e-16 |
| 3.1–3.4 | near_psd and Higham, covariance and correlation | 2e-15 |
| 4.1 | chol_psd | 0 |
| 5.1–5.5 | 100k-draw simulation: PD, PSD, two repairs, PCA at 99% | 1e-11 |
| 6.1–6.2 | arithmetic and log returns | 0 |
| 7.1 | fit a normal | 7e-18 |
| 7.2–7.4 | generalized t, t regression, AICc | 2e-10 |
| 7.5 | NIG by method of moments | 0 |
| 7.6 | NIG by MLE | 2e-4 |

**5.1–5.5.** The RNG stream reproduces, so these match the expected files to
floating point and that is what the test checks. Sampling error against the
input covariance runs 3e-4 to 1e-2 at 100,000 draws, and is printed alongside.
For 5.5 that number also includes the bias from dropping the last 1% of the
variance.

**7.6.** The expected file came from `scipy.stats.norminvgauss.fit`, a numerical
optimiser, so agreement is bounded by its convergence. Checked on a relative
tolerance, 1.9e-5. `tools/compare_nig_fits.jl` shows the log likelihood here is
6.8e-10 higher than scipy's.

## Conventions

- `fit_normal` returns the sample standard deviation, (n-1), not the (n) MLE.
  7.1 expects the (n-1) value.
- AICc counts `k` as the distribution parameters plus the regression
  coefficients. For the fitted generalized t that is k = 3, so the correction in
  7.4 is 2·3·4/(100−3−1) = 0.25.
- Pairwise covariance (1.3, 1.4) is not PSD — each entry comes off a different
  row set. That is what test 3 repairs and test 4 then factors.
- `σ` in the generalized t is the scale, not the standard deviation:
  var = σ²ν/(ν−2), for ν > 2.
- `near_psd` and `higham_nearestPSD` both land in the PSD cone; only Higham
  lands at the nearest point, which is what the Dykstra term costs iterations
  for.
- The two NIG fits optimise different things. 7.5 matches the sample skewness
  and kurtosis exactly; 7.6 maximises the likelihood. See
  `tools/compare_nig_fits.jl`.

## Implementation

Maximum likelihood is unconstrained everywhere, with constrained parameters on
transformed scales: `σ = exp(θ)`, `ν = 2 + exp(θ)`, and for the NIG
`β = α·tanh(θ)` so `α > |β|` holds by construction.

The t fits use BFGS with a ForwardDiff gradient. The NIG fit uses Nelder-Mead
since `K₁` has no ForwardDiff rule, started from the method-of-moments fit.

The NIG has no closed-form CDF, so `nig.jl` builds an interpolated one once per
fitted model, which keeps `u` and `eval` cheap.

## Environment

Julia 1.11.9. Dependencies pinned in `testfiles/Manifest.toml`: CSV, DataFrames,
Distributions, StatsBase, Optim, ForwardDiff, SpecialFunctions, QuadGK.

`tools/nig_scipy_reference.py` needs Python with scipy and is only used for the
7.6 check; the library has no Python dependency. Set `PYTHON` if the interpreter
is not on the path.
