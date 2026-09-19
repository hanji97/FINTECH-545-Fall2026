# Normal Inverse Gaussian, fitted by moments and by maximum likelihood.
# Week 01, section 5.6.

"""
    nig_logpdf(μ, α, β, δ, x)

Log density of `NIG(μ, α, β, δ)`, with `γ = √(α²-β²)` and `s = √(δ² + (x-μ)²)`:

    f(x) = δα/π · exp(δγ + β(x-μ)) · K₁(αs) / s

Written against the scaled Bessel, `K₁(z) = besselkx(1,z)·e^(-z)`, so the
`exp(δγ)` and the Bessel decay cancel in the log. The unscaled form loses
precision once `αs` gets past about 700.
"""
function nig_logpdf(μ::Real, α::Real, β::Real, δ::Real, x::Real)
    γ = sqrt(α^2 - β^2)
    s = sqrt(δ^2 + (x - μ)^2)
    z = α * s
    return log(δ * α / π) + δ * γ + β * (x - μ) - log(s) + log(besselkx(1, z)) - z
end

nig_ll(μ, α, β, δ, x::AbstractVector) = sum(nig_logpdf(μ, α, β, δ, xi) for xi in x)

# --- interpolated CDF -------------------------------------------------------
# The NIG has no closed-form CDF, and `u` and `eval` need one. Quadrature per
# call is too slow to sit inside a simulation, so build the grid once.
# Integrating between consecutive grid points and accumulating keeps each piece
# of the integral small.
struct NIGCdf
    x::Vector{Float64}
    F::Vector{Float64}
    d::NormalInverseGaussian
end

function NIGCdf(d::NormalInverseGaussian; npoints::Integer=2000, nsd::Real=12)
    m, s = mean(d), std(d)
    xs = collect(range(m - nsd * s, m + nsd * s, length=npoints))
    f(t) = (p = pdf(d, t); isfinite(p) ? p : 0.0)

    F = Vector{Float64}(undef, npoints)
    F[1] = first(quadgk(f, xs[1] - 10 * s, xs[1]; rtol=1e-10))
    for i in 2:npoints
        F[i] = F[i-1] + first(quadgk(f, xs[i-1], xs[i]; rtol=1e-10))
    end
    clamp!(F, 0.0, 1.0)
    return NIGCdf(xs, F, d)
end

function _interp(xs::Vector{Float64}, ys::Vector{Float64}, x::Real)
    i = searchsortedlast(xs, x)
    i <= 0 && return ys[1]
    i >= length(xs) && return ys[end]
    w = (x - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] + w * (ys[i+1] - ys[i])
end

(c::NIGCdf)(x::Real) = x <= c.x[1] ? 0.0 : x >= c.x[end] ? 1.0 : _interp(c.x, c.F, x)
nig_quantile(c::NIGCdf, u::Real) = _interp(c.F, c.x, clamp(u, c.F[1], c.F[end]))

# --- method of moments ------------------------------------------------------
"""
    fit_nig_moments(x)

Match the first four sample moments. Closed form, no optimiser.

With `γ = √(α²-β²)`:

    mean = μ + δβ/γ        var    = δα²/γ³
    skew = 3β/(α√(δγ))     exkurt = 3(1 + 4β²/α²)/(δγ)

Writing `ρ = β/α` gives `skew²/exkurt = 3ρ²/(1 + 4ρ²)`, which inverts for `ρ`
alone; everything else then follows one at a time.

Errors if the sample is outside the NIG's reachable region, which requires
`exkurt > (5/3)·skew²`.
"""
function fit_nig_moments(x::AbstractVector)
    xv = collect(float.(x))
    m, v = mean(xv), var(xv)
    s, k = skewness(xv), kurtosis(xv)      # kurtosis is excess

    k > 0 || error("the NIG method of moments needs positive excess kurtosis, got $k")
    t = s^2 / k
    t < 3 / 5 || error("sample is outside the NIG region: excess kurtosis must exceed (5/3)·skew²")

    ρ² = t / (3 - 4t)
    ρ = sign(s) * sqrt(ρ²)

    D = 3 * (1 + 4ρ²) / k                  # δγ
    α = sqrt(D / (v * (1 - ρ²)^2))
    β = ρ * α
    γ = α * sqrt(1 - ρ²)
    δ = D / γ
    μ = m - δ * β / γ

    return _nig_fitted_model(NormalInverseGaussian(μ, α, β, δ), xv)
end

# --- maximum likelihood -----------------------------------------------------
"""
    fit_nig_mle(x; start=nothing)

MLE. The constraints `δ > 0` and `α > |β|` are carried in the parameterisation
rather than handed to the optimiser:

    α = exp(θ₂),   β = α·tanh(θ₃),   δ = exp(θ₄)

so `γ = √(α²-β²)` cannot go imaginary mid-search. Nelder-Mead, because `K₁` has
no ForwardDiff rule and four parameters do not need a gradient.

Started from [`fit_nig_moments`](@ref); a cold start on this likelihood is not
reliable.
"""
function fit_nig_mle(x::AbstractVector; start=nothing)
    xv = collect(float.(x))

    d0 = start === nothing ? fit_nig_moments(xv).errorModel : start
    θ0 = [d0.μ, log(d0.α), atanh(clamp(d0.β / d0.α, -0.999, 0.999)), log(d0.δ)]

    function nll(θ)
        α = exp(θ[2])
        β = α * tanh(θ[3])
        δ = exp(θ[4])
        return -nig_ll(θ[1], α, β, δ, xv)
    end

    θ = Optim.minimizer(optimize(nll, θ0, NelderMead(),
                                 Optim.Options(iterations=100_000)))
    θ = Optim.minimizer(optimize(nll, θ, NelderMead(),
                                 Optim.Options(iterations=100_000)))

    α = exp(θ[2])
    β = α * tanh(θ[3])
    δ = exp(θ[4])
    return _nig_fitted_model(NormalInverseGaussian(θ[1], α, β, δ), xv)
end

# shared by both fits: build the CDF once and close over it
function _nig_fitted_model(errorModel::NormalInverseGaussian, x::Vector{Float64})
    F = NIGCdf(errorModel)
    errors = x .- mean(errorModel)
    u = [F(xi) for xi in x]
    eval(u) = [nig_quantile(F, ui) for ui in u]
    return FittedModel(nothing, errorModel, eval, errors, u)
end
