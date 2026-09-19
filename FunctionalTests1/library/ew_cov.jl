# Exponentially weighted covariance. Week 03, section 5.

"""
    ew_weights(m, λ)

Normalized exponential weights, oldest observation first:

    wᵢ ∝ (1-λ) λ^(m-i),   i = 1 … m

Row m carries the most weight and the weights sum to one.
"""
function ew_weights(m::Integer, λ::Real)
    w = [(1 - λ) * λ^(m - i) for i in 1:m]
    return w ./ sum(w)
end

"""
    ewCovar(x, λ)

Exponentially weighted covariance of the columns of `x`:

    Σ = Σᵢ wᵢ (xᵢ - x̄_w)(xᵢ - x̄_w)′

where `x̄_w` is the weighted mean. Folding √w into the centred data makes this a
single matrix product. No (n-1) correction: the weights already sum to one.
"""
function ewCovar(x::AbstractMatrix, λ::Real)
    w = ew_weights(size(x, 1), λ)
    xm = sqrt.(w) .* (x .- (w' * x))
    return xm' * xm
end

"""
    ewCorr(x, λ)

EW covariance rescaled to a unit diagonal.
"""
function ewCorr(x::AbstractMatrix, λ::Real)
    c = ewCovar(x, λ)
    s = 1 ./ sqrt.(diag(c))
    return s .* c .* s'
end

"""
    ewCovar(x, λ_var, λ_corr)

Standard deviations from the `λ_var` covariance, correlation from `λ_corr`,
multiplied back together. Used by test 2.3.
"""
function ewCovar(x::AbstractMatrix, λ_var::Real, λ_corr::Real)
    sd = sqrt.(diag(ewCovar(x, λ_var)))
    R = ewCorr(x, λ_corr)
    return sd .* R .* sd'
end
