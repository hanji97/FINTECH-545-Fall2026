# FinTech 545 - Quantitative Risk Management
# Library for Assignment 3. Entry point:
#
#     include("library/RiskLib.jl")
#
# Plain script, not a module, so the functions land in the caller's scope.
# fitted_model.jl and return_calculate.jl are carried over unchanged from the
# weeks 01-03 library; the rest is new for weeks 04-07.

# Off-screen GR backend for batch runs
ENV["GKSwstype"] = "100"

using CSV
using DataFrames
using Dates
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
using SpecialFunctions: loggamma

include("return_calculate.jl")   # prices to returns
include("fitted_model.jl")       # week 01-02   normal, t, t-regression, AICc
include("moments.jl")            # week 01 §4   bias-adjusted sample moments
include("regression.jl")         # week 02 §3   OLS with standard errors
include("risk_stats.jl")         # week 04-05   VaR and ES
include("gbsm.jl")               # week 06-07   GBSM, Greeks, implied vol
include("binomial_tree.jl")      # week 06-07   CRR and discrete dividend trees
include("portfolio.jl")          # week 07 §4-5 max Sharpe, efficient frontier
include("tables.jl")             #              CSV and LaTeX output
