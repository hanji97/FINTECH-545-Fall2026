# FinTech 545 - Quantitative Risk Management
# Library for weeks 01-03. Entry point:
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
using Optim
using ForwardDiff
using QuadGK
using SpecialFunctions: loggamma, besselkx

include("missing_cov.jl")        # week 03 §3   covariance with missing data
include("ew_cov.jl")             # week 03 §5   exponentially weighted covariance
include("psd_fixes.jl")          # week 03 §2,6 near_psd, Higham, chol_psd
include("simulate.jl")           # week 03 §1,7 normal and PCA simulation
include("return_calculate.jl")   #              prices to returns
include("fitted_model.jl")       # week 01-02   normal, t, t-regression, AICc
include("nig.jl")                # week 01 §5.6 NIG by moments and by MLE
