# Multivariate t. Week 05, section 5.
#
# Fitted the rank based way in 5.3: μ at the sample mean, the correlation from
# Kendall's tau, the scales tied to the sample standard deviations, and ν by a
# one dimensional profile.

"""
    kendall_correlation(X)

Correlation matrix from Kendall's tau (Week 05, 5.3.1):

    ρ = sin(πτ/2)

exact for any elliptical copula, Gaussian or t, and free of ν. The tau matrix
is PSD on complete data but the entrywise sine is not guaranteed to keep it
that way, so the result goes through `fix_correlation`. Ranks are unchanged by
the marginal CDFs, so this gives the same matrix on returns or on U.
"""
function kendall_correlation(X::AbstractMatrix)
    R = sin.(π .* corkendall(Matrix{Float64}(X)) ./ 2)
    R[diagind(R)] .= 1.0
    return fix_correlation(R)
end

"""
    fix_correlation(R; tol=1e-8, floor=1e-8)

Repair with Higham (Week 03) when the smallest eigenvalue is below -tol. The
copula and t densities need R⁻¹ and log|R|, so R has to be strictly positive
definite. If the smallest eigenvalue λ is still below `floor`, the matrix is
shifted and rescaled to a unit diagonal,

    R ← (R + sI) / (1 + s),   s = (floor - λ) / (1 - floor)

which moves the smallest eigenvalue to (λ + s)/(1 + s) = floor. Symmetrized
because Higham can leave the last bits unequal.
"""
function fix_correlation(R::AbstractMatrix; tol=1e-8, floor=1e-8)
    if eigmin(Symmetric(R)) < -tol
        R = higham_nearestPSD(R)
    end
    R = (R + R') ./ 2
    λ = eigmin(Symmetric(R))
    if λ < floor
        s = (floor - λ) / (1 - floor)
        R = (R + s * I) ./ (1 + s)
    end
    return R
end

"""
    mvt_scale(sd, R, ν)

Scale matrix of a multivariate t whose standard deviations are `sd`
(Week 05, 5.3.2). Since cov = ν/(ν-2)·Σ, the scale of variable i is

    dᵢ(ν) = σ̂ᵢ √((ν-2)/ν),   Σ(ν) = D(ν) R D(ν)
"""
function mvt_scale(sd::AbstractVector, R::AbstractMatrix, ν::Real)
    d = sd .* sqrt((ν - 2) / ν)
    return (d * d') .* R
end

"""
    mvt_loglikelihood(X, μ, sd, R, ν)

ℓ(ν) = Σₖ ln f(xₖ; μ̂, Σ(ν), ν), with f the multivariate t density in 5.1.
"""
function mvt_loglikelihood(X::AbstractMatrix, μ, sd, R, ν)
    d = MvTDist(ν, collect(float.(μ)), Matrix(Symmetric(mvt_scale(sd, R, ν))))
    return sum(logpdf(d, X[k, :]) for k in axes(X, 1))
end

"""
    profile_nu(ll; lo=0.01, hi=0.49, n=200)

Maximize a log likelihood `ll(ν)` over ν by the search in Week 05, 5.3.2: a
grid of `n` points on θ = 1/ν ∈ [lo, hi], then `n` more points between the
neighbors of the grid maximum. Searching in θ puts the grid points at small
ν, where the likelihood changes fastest; hi = 0.49 keeps ν above 2 and
lo = 0.01 caps it at 100.

Returns (ν̂, ℓ(ν̂)).
"""
function profile_nu(ll; lo=0.01, hi=0.49, n=200)
    θ = range(lo, hi, length=n)
    llθ = [ll(1 / t) for t in θ]
    i = argmax(llθ)

    θfine = range(θ[max(i - 1, 1)], θ[min(i + 1, n)], length=n)
    llfine = [ll(1 / t) for t in θfine]
    j = argmax(llfine)
    return 1 / θfine[j], llfine[j]
end

"""
    fit_multivariate_t(X)

Rank based fit of a multivariate t to the rows of `X` (Week 05, 5.3). Returns
the mean vector, the scale matrix S, ν and the log likelihood. The covariance
implied by the fit is ν/(ν-2)·S, which equals the sample covariance on the
diagonal by construction.
"""
function fit_multivariate_t(X::AbstractMatrix)
    Xf = Matrix{Float64}(X)
    μ = vec(mean(Xf, dims=1))
    sd = vec(std(Xf, dims=1))
    R = kendall_correlation(Xf)
    ν, ll = profile_nu(v -> mvt_loglikelihood(Xf, μ, sd, R, v))
    return μ, mvt_scale(sd, R, ν), ν, ll
end
