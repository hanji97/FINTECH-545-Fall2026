# OLS with standard errors. Week 02 §3.3 and §5, used for CAPM and the factor
# models in Week 07 §5.3-7.

"""
    ols(y, X, names)

Regress `y` on `[1 X]`. Returns a NamedTuple with the coefficients (intercept
first, called alpha), their standard errors and t statistics, R², the
residuals, and AICc.

    s² = e'e / (n - p),   se(β̂) = √diag(s² (X'X)⁻¹)        (Week 02, 3.3)

`p` counts the intercept. AICc uses the normal log likelihood at the MLE of
the error variance, e'e/n, and k = p + 1 (the coefficients plus σ), following
the convention in `aicc(::FittedModel)` that k is the error model's parameters
plus the regression coefficients.
"""
function ols(y::AbstractVector, X::AbstractMatrix, names::Vector{String})
    n = length(y)
    Z = hcat(ones(n), X)
    p = size(Z, 2)
    ZtZi = inv(Z' * Z)
    β = ZtZi * (Z' * y)
    e = y .- Z * β
    sse = sum(abs2, e)
    s2 = sse / (n - p)
    se = sqrt.(diag(s2 .* ZtZi))
    tstat = β ./ se
    pval = 2 .* ccdf.(TDist(n - p), abs.(tstat))
    r2 = 1 - sse / sum(abs2, y .- mean(y))
    ll = sum(logpdf.(Normal(0, sqrt(sse / n)), e))
    k = p + 1
    return (names=vcat(["alpha"], names), beta=β, se=se, t=tstat, p=pval,
            r2=r2, adj_r2=1 - (1 - r2) * (n - 1) / (n - p), resid=e, n=n, k=k,
            loglik=ll, aicc=-2ll + 2k + 2k * (k + 1) / (n - k - 1), s=sqrt(s2))
end

ols(y::AbstractVector, x::AbstractVector, name::String) = ols(y, reshape(x, :, 1), [name])
