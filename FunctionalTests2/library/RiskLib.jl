# FinTech 545 - Quantitative Risk Management
# Risk library, weeks 01-07. Entry point:
#
#     include("../library/RiskLib.jl")
#
# Plain script, not a module, so the functions land in the caller's scope.

using CSV
using DataFrames
using Distributions
using StatsBase
using Statistics
using LinearAlgebra
using Random
using Printf
using Optim
using ForwardDiff
using QuadGK
using Roots
using JuMP
using Ipopt
using SpecialFunctions: loggamma, besselkx

include("missing_cov.jl")        # week 03, sec. 3   covariance with missing data
include("ew_cov.jl")             # week 03, sec. 5   exponentially weighted covariance
include("psd_fixes.jl")          # week 03, sec. 2,6 near_psd, Higham, chol_psd
include("simulate.jl")           # week 03, sec. 1,7 normal and PCA simulation
include("return_calculate.jl")   # week 04, sec. 1   prices to returns
include("fitted_model.jl")       # week 01-02        normal, t, t-regression, AICc
include("nig.jl")                # week 01, sec. 5.6 NIG by moments and by MLE
include("risk_stats.jl")         # week 04-05        VaR, ES, portfolio aggregation
include("multivariate_t.jl")     # week 05, sec. 5   Kendall correlation, multivariate t
include("copula.jl")             # week 05, sec. 4,6 Gaussian and t copulas
include("gbsm.jl")               # week 06-07        GBSM, Greeks, implied vol
include("binomial_tree.jl")      # week 06-07        CRR and discrete dividend trees, numerical Greeks
include("portfolio.jl")          # week 07, sec. 4-5 max Sharpe ratio; risk parity
include("attribution.jl")        #                   ex-post return and volatility attribution
