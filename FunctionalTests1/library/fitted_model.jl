# Distribution fitting. Week 01, sections 5-6. Week 02, sections 4-5.

"""
    FittedModel

    beta        regression coefficients, or nothing for a plain fit
    errorModel  fitted error distribution
    eval        u ↦ quantile
    errors      fitted residuals
    u           residuals through the fitted CDF, i.e. the copula input
"""
struct FittedModel
    beta::Union{Vector{Float64},Nothing}
    errorModel::UnivariateDistribution
    eval::Function
    errors::Vector{Float64}
    u::Vector{Float64}
end

# --- information criteria ---------------------------------------------------
"""
    aicc(d, data)
    aicc(m::FittedModel, data)

Corrected AIC (Week 02, 5.3):

    AICc = -2ℓ + 2k + 2k(k+1)/(n-k-1)

`k` is the distribution's parameters plus, for a regression, the coefficients.
The third term is the small-sample correction and vanishes as `n` grows. Lower
is better.
"""
function aicc(d::UnivariateDistribution, data)
    ll = sum(logpdf.(d, data))
    n = length(data)
    k = length(Distributions.params(d))
    return -2ll + 2k + 2k * (k + 1) / (n - k - 1)
end

function aicc(m::FittedModel, data)
    ll = sum(logpdf.(m.errorModel, data))
    n = length(data)
    k = length(Distributions.params(m.errorModel))
    m.beta === nothing || (k += length(m.beta))
    return -2ll + 2k + 2k * (k + 1) / (n - k - 1)
end

"""
    aic(m::FittedModel, data)

Uncorrected AIC, `-2ℓ + 2k`.
"""
function aic(m::FittedModel, data)
    ll = sum(logpdf.(m.errorModel, data))
    k = length(Distributions.params(m.errorModel))
    m.beta === nothing || (k += length(m.beta))
    return -2ll + 2k
end

# --- MLE driver -------------------------------------------------------------
# Every fit below is unconstrained once the positive parameters sit on a log
# scale, so one BFGS driver with a ForwardDiff gradient covers all of them. The
# restart re-enters from the solution with a fresh Hessian approximation, which
# tightens the last few digits.
function _mle(nll, θ0::Vector{Float64})
    g!(G, θ) = (G .= ForwardDiff.gradient(nll, θ))
    θ = Optim.minimizer(optimize(nll, g!, θ0, BFGS()))
    res = optimize(nll, g!, θ, BFGS())
    return Optim.minimizer(res), -Optim.minimum(res)
end

# --- normal -----------------------------------------------------------------
"""
    fit_normal(x)

Closed form: `μ` is the sample mean, `σ` the sample standard deviation. Note
this is the (n-1) estimator, not the (n) maximum likelihood one.
"""
function fit_normal(x::AbstractVector)
    m = mean(x)
    s = std(x)

    errorModel = Normal(m, s)
    errors = x .- m
    u = cdf.(errorModel, x)
    eval(u) = quantile.(errorModel, u)

    return FittedModel(nothing, errorModel, eval, collect(float.(errors)), collect(float.(u)))
end

# --- generalized t ----------------------------------------------------------
"""
    general_t_ll(μ, σ, ν, x)

Log likelihood of `μ + σ·T(ν)`:

    log f(x) = logΓ((ν+1)/2) - logΓ(ν/2) - ½log(νπ) - log σ
               - (ν+1)/2 · log(1 + z²/ν),   z = (x-μ)/σ

Written out rather than assembled from `Distributions` so it differentiates
cleanly. `σ` is the scale, not the standard deviation: `var = σ²ν/(ν-2)`, and
only for `ν > 2`.
"""
function general_t_ll(μ, σ, ν, x)
    n = length(x)
    c = loggamma((ν + 1) / 2) - loggamma(ν / 2) - log(ν * π) / 2 - log(σ)
    s = zero(c)
    for xi in x
        z = (xi - μ) / σ
        s += log1p(z * z / ν)
    end
    return n * c - (ν + 1) / 2 * s
end

# starting values from the sample moments: kurtosis gives ν, variance gives σ
function _t_moment_start(e)
    k = kurtosis(e)                       # excess
    ν = k > 0 ? 6.0 / k + 4.0 : 10.0
    ν = clamp(ν, 2.5, 100.0)
    σ = sqrt(var(e) * (ν - 2) / ν)
    return σ, ν
end

"""
    fit_general_t(x)

MLE of the location-scale t (Week 01, 5.4). `σ` and `ν` are carried as `log σ`
and `log(ν-2)`, which makes the problem unconstrained and keeps the optimiser
off the boundary. `ν` is floored above 2 because the variance does not exist at
or below it.
"""
function fit_general_t(x::AbstractVector)
    xv = collect(float.(x))
    σ0, ν0 = _t_moment_start(xv)
    θ0 = [mean(xv), log(σ0), log(ν0 - 2)]

    nll(θ) = -general_t_ll(θ[1], exp(θ[2]), 2 + exp(θ[3]), xv)
    θ, _ = _mle(nll, θ0)

    m, s, ν = θ[1], exp(θ[2]), 2 + exp(θ[3])

    errorModel = TDist(ν) * s + m
    errors = xv .- m
    u = cdf.(errorModel, xv)
    eval(u) = quantile.(errorModel, u)

    return FittedModel(nothing, errorModel, eval, errors, collect(float.(u)))
end

"""
    fit_regression_t(y, x)

Regression with t errors, `β`, `σ` and `ν` fitted jointly by MLE (Week 02, 4).
OLS is the MLE only under normal errors; with fat tails it stays unbiased but
the implied error distribution understates the tails.

The intercept is carried in `β`, so the error distribution is pinned at mean
zero — otherwise the two are unidentified. Started from OLS plus the residual
moments.
"""
function fit_regression_t(y::AbstractVector, x::AbstractMatrix)
    yv = collect(float.(y))
    X = hcat(ones(length(yv)), Matrix{Float64}(x))

    b0 = X \ yv
    e0 = yv .- X * b0
    σ0, ν0 = _t_moment_start(e0)
    θ0 = vcat(log(σ0), log(ν0 - 2), b0)

    function nll(θ)
        σ = exp(θ[1])
        ν = 2 + exp(θ[2])
        β = θ[3:end]
        return -general_t_ll(zero(eltype(θ)), σ, ν, yv .- X * β)
    end
    θ, _ = _mle(nll, θ0)

    s, ν, beta = exp(θ[1]), 2 + exp(θ[2]), θ[3:end]

    errorModel = TDist(ν) * s              # mean zero; the intercept is in beta
    function eval_model(xnew, u)
        Xn = hcat(ones(size(xnew, 1)), Matrix{Float64}(xnew))
        return Xn * beta .+ quantile.(errorModel, u)
    end

    errors = yv .- X * beta
    u = cdf.(errorModel, errors)

    return FittedModel(collect(float.(beta)), errorModel, eval_model, errors, collect(float.(u)))
end
