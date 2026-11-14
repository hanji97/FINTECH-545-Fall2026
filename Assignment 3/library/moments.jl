# Sample moments. Week 01 §3-4.

"""
    sample_moments(x)

First four moments as reported in the write-up: mean, standard deviation
(n-1), skewness and EXCESS kurtosis, both bias adjusted.

StatsBase's `skewness` and `kurtosis` are the population (biased) estimators
g₁ and g₂ (Week 01, 4.4). The adjusted versions are

    G₁ = g₁ √(n(n-1)) / (n-2)
    G₂ = ((n+1) g₂ + 6)(n-1) / ((n-2)(n-3))

At the sample sizes in this assignment (240 to 1,257) they differ from g₁, g₂
by under 1.5%, so the choice does not change any conclusion.
"""
function sample_moments(x::AbstractVector)
    n = length(x)
    g1 = skewness(x)
    g2 = kurtosis(x)                      # excess
    return (n=n, mean=mean(x), std=std(x), var=var(x),
            skew=g1 * sqrt(n * (n - 1)) / (n - 2),
            xkurt=((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3)))
end
