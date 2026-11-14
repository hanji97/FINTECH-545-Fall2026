# VaR and ES. Week 04 §2 and Week 05 §1.
#
# Both are reported as positive numbers for a loss:
#   VaR_α(x) = -F⁻¹(α)        ES_α(x) = -E[x | x ≤ -VaR_α(x)]
# measured from zero (the "absolute" convention in Week 04, 2.1), on P&L or on
# returns with gains positive.

"""
    VaR(x; alpha=0.05)

Empirical VaR from a sample. The α quantile is the average of the order
statistics at ⌊nα⌋ and ⌈nα⌉, the same estimator as the course library, so
the numbers are comparable with the class test cases.
"""
function VaR(x::AbstractVector; alpha=0.05)
    s = sort(x)
    n = length(s)
    up = ceil(Int, n * alpha)
    dn = max(floor(Int, n * alpha), 1)
    return -(s[up] + s[dn]) / 2
end

"""
    ES(x; alpha=0.05)

Empirical ES: the mean of every outcome at or below the VaR point.
"""
function ES(x::AbstractVector; alpha=0.05)
    s = sort(x)
    v = -VaR(s; alpha=alpha)
    return -mean(s[s .<= v])
end

VaR(d::UnivariateDistribution; alpha=0.05) = -quantile(d, alpha)

"""
    ES(d::UnivariateDistribution; alpha=0.05)

    ES = -(1/α) ∫_{-∞}^{-VaR} x f(x) dx      (Week 05, 1)

by quadrature. For the normal this equals the closed form
`-μ + σ φ(Φ⁻¹(α))/α`, which `ES_normal` uses directly.
"""
function ES(d::UnivariateDistribution; alpha=0.05)
    v = quantile(d, alpha)
    return -quadgk(x -> x * pdf(d, x), quantile(d, 1e-12), v)[1] / alpha
end

VaR_normal(μ, σ; alpha=0.05) = -(μ + σ * quantile(Normal(), alpha))
ES_normal(μ, σ; alpha=0.05) = -μ + σ * pdf(Normal(), quantile(Normal(), alpha)) / alpha
