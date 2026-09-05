# Assignment 1 --- Univariate and Multivariate Statistics

FinTech 545. All code is Julia.

## Files

```
Assignment1_Writeup.pdf    the written answers
README.md                  this file
common.jl                  shared helpers used by every problem
problem1.jl                Reading the shape of a sample
problem2.jl                A regression whose errors are not normal
problem3.jl                Pearson against Spearman
problem4.jl                Conditional distributions
problem5.jl                Identifying an AR or MA order
run_all.jl                 runs all five in order
Project.toml               package list
Manifest.toml              exact resolved versions
```

## Data

The CSV files are not included. Put `problem1.csv` through `problem5.csv` either
in this folder, in a `data/` subfolder, or in the parent folder. The scripts
check all three.

## Running it

Julia 1.10 or later. Developed on 1.12.7, macOS, arm64. Install packages once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then either everything at once,

```bash
julia --project=. run_all.jl
```

or one problem at a time, in any order, since each script is self contained:

```bash
julia --project=. problem1.jl
julia --project=. problem2.jl
julia --project=. problem3.jl
julia --project=. problem4.jl
julia --project=. problem5.jl
```

Each script prints its results to stdout and writes its figures to `figures/`,
creating that folder if it does not exist. Problem 5 takes about 40 seconds
because it fits six models by exact maximum likelihood with six restarts each;
the rest take a few seconds.

Every number in the write up is printed by the script with the matching number.

## Conventions

The places where more than one convention was available:

- **Moments.** Skewness and kurtosis are reported bias corrected and
  standardized, not in the population form `StatsBase` returns. Both are printed
  by `problem1.jl` so the difference can be checked. Kurtosis is **excess**
  kurtosis, so a normal sample gives about zero.
- **AICc.** `2k - 2ll + (2k^2+2k)/(n-k-1)`, with `k = p + d`: mean parameters
  including the intercept, plus distributional parameters. So the Normal error
  regression has `k = 3` and the t error regression `k = 4`; an AR(2) has
  `k = 4`.
- **OLS and Normal MLE** in Problem 2 are the same model, so they share a row.
  They differ only in the residual scale, `n-p` against `n`. AICc is computed at
  the MLE value of sigma for both.
- **t regression standard errors** come from the inverse Hessian at the optimum
  by forward mode automatic differentiation. The optimizer works on `log(sigma)`
  and `nu = 2 + exp(theta)` so sigma stays positive and nu stays above 2.
- **Block ordering in Problem 4.** The Week 2 result conditions block 1 on block
  2, and the question asks for `x2` given `x1`, so the code builds the vector as
  `[x2 x1]` rather than in CSV column order. That makes `S11 = var(x2)` and
  `S22 = var(x1)`, so the code matches the notes line for line.
- **Problem 5 uses the exact likelihood**, not a conditional one, so all six
  models are scored on all 500 observations and their AICc values are
  comparable. Julia has no ARIMA routine comparable to R's `arima`, so it is
  built from the autocovariance matrix formula in the Week 2 notes. The AR(2)
  fit is cross checked against Yule Walker and conditional least squares, which
  agree to three decimals.
- **Stationarity** is imposed by parametrizing through partial autocorrelations
  (`tanh` into `(-1,1)`, then the Levinson recursion), so the optimizer cannot
  leave the stationary region. Restarts are seeded, so runs reproduce exactly.
