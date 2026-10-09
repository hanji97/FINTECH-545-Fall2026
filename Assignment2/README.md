# Assignment 2 — Covariance, VaR, and Copulas

All code is Python. Reusable calculations live in the installable library `qrm/`;
each problem is a short script in `problems/` that calls the library.

```
Assignment2/
├── Assignment 2.pdf        written answers
├── README.md
├── pyproject.toml          makes qrm pip-installable
├── run_all.py              reproduces every number, figure and the PDF
├── qrm/                    the library
│   ├── moments.py          sample moments (n-1 variance, bias-corrected skew / excess kurtosis)
│   ├── correlation.py      complete-case vs pairwise correlation, Sigma = D C D
│   ├── psd.py              PSD-tolerant Cholesky, Rebonato-Jackel, Higham
│   ├── ewma.py             exponential weights, n_eff, half life
│   ├── fitting.py          normal / Student t MLE, AIC, AICc, BIC
│   ├── risk.py             VaR and ES (historical, normal, t)
│   ├── simulation.py       multivariate normal draws via Cholesky
│   └── copula.py           Gaussian / t copula: Kendall R, likelihoods, nu profile, simulation
├── problems/problem1.py … problem5.py, common.py
├── data/problem1.csv … problem5.csv
├── output/                 JSON results and figures written by the scripts
└── report/build_report.py  builds the PDF from output/*.json
```

## Setup

Python 3.9+.

```bash
cd Assignment2
pip install -e .                 # installs qrm and its dependencies: numpy, scipy, pandas, matplotlib
```

## Reproduce everything

```bash
python run_all.py                # problems 1-5, then the PDF
python run_all.py --no-pdf       # problems only (no pandoc / LaTeX needed)
```

Or one problem at a time (run from `Assignment2/`). Each script prints its results and writes
`output/pK.json`:

```bash
python problems/problem1.py
python problems/problem2.py
python problems/problem3.py
python problems/problem4.py
python problems/problem5.py
python report/build_report.py    # needs pandoc and xelatex
```

Runtime is a few seconds per problem.

## Conventions

These are also commented in the code.

* Returns are arithmetic, `P_t / P_{t-1} - 1`.
* VaR and ES are positive numbers for a loss, absolute convention (mean not subtracted).
  The historical quantile is the `ceil(n * alpha)`-th worst value; ES is the mean of
  that value and everything worse.
* Variance uses `n - 1`. Skew and **excess** kurtosis are bias-corrected
  (`scipy.stats.skew(bias=False)`, `kurtosis(fisher=True, bias=False)`).
* The Student t uses `(nu, mu, s)`, where `s` is a scale, not the standard deviation.
* Exponential weights: the newest observation gets `1 - lambda`, normalised to sum to 1.
* Copulas: margins are chosen by AICc, `R = sin(pi * tau / 2)` from Kendall's tau is
  shared by both copulas, and the t copula's `nu` is profiled on a 200-point grid in
  `1/nu` over [0.01, 0.49] and then refined.
* Simulations use seed 545 and 100,000 draws.
