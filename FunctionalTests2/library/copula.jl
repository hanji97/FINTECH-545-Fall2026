# Gaussian and t copulas. Week 05, sections 4 and 6.
#
# The margins are fitted first and frozen; U holds each observation pushed
# through its own fitted CDF (FittedModel.u). Both copulas get the same R from
# Kendall's tau, so they differ by one parameter, ν (Week 05, 6.5).

"""
    gaussian_copula_ll(U, R)

Copula log likelihood as the difference of two sums of log densities
(Week 05, 6.2), which avoids the underflow of a ratio of products:

    ℓ = Σₖ ln φ_R(zₖ) - Σₖ Σᵢ ln φ(zₖᵢ),   zₖᵢ = Φ⁻¹(uₖᵢ)
"""
function gaussian_copula_ll(U::AbstractMatrix, R::AbstractMatrix)
    Z = quantile.(Normal(), U)
    joint = MvNormal(zeros(size(R, 1)), Matrix(Symmetric(R)))
    return sum(logpdf(joint, Z[k, :]) for k in axes(Z, 1)) - sum(logpdf.(Normal(), Z))
end

"""
    t_copula_ll(U, R, ν)

Same form with the t (Week 05, 6.2): yₖᵢ = t_ν⁻¹(uₖᵢ), the joint density is the
multivariate t with μ = 0 and Σ = R, and the margins are t_ν carrying the same ν.
"""
function t_copula_ll(U::AbstractMatrix, R::AbstractMatrix, ν::Real)
    tν = TDist(ν)
    Y = quantile.(tν, U)
    joint = MvTDist(ν, zeros(size(R, 1)), Matrix(Symmetric(R)))
    return sum(logpdf(joint, Y[k, :]) for k in axes(Y, 1)) - sum(logpdf.(tν, Y))
end

"""
    fit_gaussian_copula(U)

R from Kendall's tau and the copula log likelihood at R. Returns (R, ℓ).
The Gaussian copula has no free parameter once R is fixed.
"""
function fit_gaussian_copula(U::AbstractMatrix)
    R = kendall_correlation(U)
    return R, gaussian_copula_ll(U, R)
end

"""
    fit_t_copula(U)

Week 05, 6.1: R from Kendall's tau, held fixed, and ν by the same profile
search used for the multivariate t (`profile_nu`). Returns (R, ν, ℓ).
"""
function fit_t_copula(U::AbstractMatrix)
    R = kendall_correlation(U)
    ν, ll = profile_nu(v -> t_copula_ll(U, R, v))
    return R, ν, ll
end

"""
    simulate_t_copula(R, ν, nsim; seed=1234)

Week 05, 6.3. Draw from the multivariate t with correlation R through the
mixture in 5.2,

    Y = √W · Z,   W = ν/Q,  Q ~ χ²_ν,   Z ~ N(0, R)

with one W per draw shared by every asset, then map each column to U with the
t CDF at the copula's ν (not the margins' ν). Z comes from the PCA simulation
in Week 03. Returns an nsim × n matrix of uniforms.
"""
function simulate_t_copula(R::AbstractMatrix, ν::Real, nsim::Integer; seed::Integer=1234)
    Z = simulate_pca(R, nsim; seed=seed)
    Random.seed!(seed + 1)
    W = ν ./ rand(Chisq(ν), nsim)
    return cdf.(TDist(ν), sqrt.(W) .* Z)
end

"""
    simulate_gaussian_copula(R, nsim; seed=1234)

Week 05, 4.2: correlated standard normals, then Φ to get uniforms.
"""
function simulate_gaussian_copula(R::AbstractMatrix, nsim::Integer; seed::Integer=1234)
    return cdf.(Normal(), simulate_pca(R, nsim; seed=seed))
end

"""
    tail_dependence_t(ρ, ν)

Lower tail dependence of a pair under the t copula (Week 05, 6.4):

    λ = 2 t_{ν+1}( -√((ν+1)(1-ρ)/(1+ρ)) )

Zero for every pair under the Gaussian copula (4.3).
"""
tail_dependence_t(ρ::Real, ν::Real) = 2 * cdf(TDist(ν + 1), -sqrt((ν + 1) * (1 - ρ) / (1 + ρ)))

"""
    copula_aicc(ll, k, m), copula_bic(ll, k, m)

Information criteria from Week 02 applied to the copula log likelihood only,
with m observations (Week 05, 6.5). R is estimated once and shared by both
copulas, so k counts only the parameters they do not share: 0 for the
Gaussian, 1 (ν) for the t. The copula with the smaller value is preferred.
"""
copula_aicc(ll, k, m) = -2ll + 2k + (2k^2 + 2k) / (m - k - 1)
copula_bic(ll, k, m) = k * log(m) - 2ll
