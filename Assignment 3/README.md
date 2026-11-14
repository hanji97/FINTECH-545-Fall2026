# Assignment 3 — Options, Option Risk, and Portfolio Construction

FinTech 545. The write-up is `Assignment3_Responses.pdf`. This folder has all the
code and data needed to reproduce every number, table and figure in it.

## Requirements

- Julia 1.11 (developed on 1.11.9)
- The packages pinned in `Project.toml` / `Manifest.toml`: CSV, DataFrames,
  Distributions, StatsBase, Optim, ForwardDiff, QuadGK, SpecialFunctions,
  Roots, JuMP, Ipopt, Plots
- Optional, only to rebuild the PDF: a LaTeX distribution with `latexmk`

## Running it

From this folder:

```
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. run_all.jl
```

`run_all.jl` first runs `tests/run_course_tests.jl`, which checks the library
against the class test cases (fits, VaR/ES, max Sharpe, GBSM Greeks, American
trees with continuous and discrete dividends) and stops if any case fails. It
then runs the five problem scripts. Total run time is about five minutes;
Problem 2 takes most of it because of the convergence sweep over N and the
fine grid of tree values behind Figure 3.

Each problem can also be run on its own and prints its results:

```
julia --project=. problem1.jl     # implied vol from the XYZ chain
julia --project=. problem2.jl     # American options, discrete dividends
julia --project=. problem3.jl     # VaR of the four option books
julia --project=. problem4.jl     # max Sharpe portfolios and PUTW
julia --project=. problem5.jl     # CAPM, Fama-French 3, Carhart 4
```

To rebuild the PDF after running the scripts (the tables in the write-up are
read from `output/tables`):

```
cd report
latexmk -pdf assignment3.tex
```

## Layout

```
data/                 input CSVs, as supplied
library/RiskLib.jl    entry point; include this to load the library
  fitted_model.jl     normal and t fits, AICc          (from my weeks 1-3 library)
  return_calculate.jl prices to returns                (from my weeks 1-3 library)
  moments.jl          bias-adjusted sample moments
  regression.jl       OLS with standard errors, t stats, R², AICc
  risk_stats.jl       VaR and ES, empirical and parametric
  gbsm.jl             generalized Black-Scholes-Merton, Greeks, implied vol
  binomial_tree.jl    CRR tree, non-recombining tree for discrete dividends
  portfolio.jl        max Sharpe ratio (JuMP + Ipopt), efficient frontier
  tables.jl           writes the CSV and LaTeX tables used by the report
problem1.jl ... problem5.jl
run_all.jl
tests/run_course_tests.jl, tests/course_data/   class test cases, unchanged
output/tables/        every table as .csv and .tex
output/figures/       figures as .pdf and .png
report/               LaTeX source of the write-up and the bibliography
```

## Conventions (also stated in the write-up)

- VaR and ES are positive numbers for a loss, measured from zero. The
  empirical quantile averages the order statistics at floor(nα) and ceil(nα),
  as in the class library; ES is the mean of all outcomes at or below it.
- Arithmetic returns throughout (Problem 3 builds them from prices).
- Standard deviations use n − 1; skewness and kurtosis are bias adjusted and
  kurtosis is excess kurtosis.
- AICc = −2ℓ + 2k + 2k(k+1)/(n−k−1); k = 2 for the normal, 3 for the t,
  p + 1 for a regression with p coefficients.
- GBSM theta is −∂V/∂T per year; vega per 1.00 of volatility.
- Problem 1: T = 35/365; parity with a dividend yield is C − P = S e^{−qT} − X e^{−rT}.
- Problem 2: N = 500 for reported values (dividends on steps 100 and 400);
  early exercise premiums difference two trees on the same grid.
- Problem 3: 255 trading days; 10 day delta normal uses the √t rule;
  the simulation uses 1,000,000 paths, the same uniforms for the normal and
  the t (common random numbers), seed `MersenneTwister(545)`; a second seed
  is used only to measure Monte Carlo error.
- Problem 4: monthly risk free rate 3%/12; Sharpe annualized by √12;
  bootstrap with 1,000 resamples of whole months, seed `MersenneTwister(545)`.
- Problem 5: Fama-French and momentum files divided by 100; annualization
  by 252 trading days; "last ten years" is the ten years ending on the last
  date in the factor files.
