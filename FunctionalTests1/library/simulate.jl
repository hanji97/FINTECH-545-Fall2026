# Monte Carlo simulation. Week 03, sections 1.2 and 7.

"""
    simulateNormal(N, cov; mean=Float64[], seed=1234, fixMethod=near_psd)

`N` draws from a multivariate normal with the given covariance, returned `N × n`.
Find a root `L` with `L L′ = Σ`, draw standard normals `z`, take `L z`.

Which root depends on `Σ`:

    cholesky              positive definite
    chol_psd!             singular but PSD
    chol_psd! + fixMethod indefinite, repair first

`seed` is fixed so runs are reproducible.
"""
function simulateNormal(N::Integer, cov::AbstractMatrix; mean=Float64[],
                        seed::Integer=1234, fixMethod=near_psd)
    n, m = size(cov)
    n == m || throw(DimensionMismatch("covariance matrix is not square ($n,$m)"))

    μ = zeros(n)
    if !isempty(mean)
        length(mean) == n || throw(DimensionMismatch("mean ($(length(mean))) does not match cov ($n,$n)"))
        μ .= mean
    end

    Σ = Matrix{Float64}(cov)
    L = Matrix{Float64}(undef, n, n)
    try
        L = Matrix(cholesky(Symmetric(Σ)).L)
    catch e
        e isa LinearAlgebra.PosDefException || rethrow(e)
        try
            chol_psd!(L, Σ)
        catch
            chol_psd!(L, fixMethod(Σ))
        end
    end

    Random.seed!(seed)
    z = randn(n, N)
    return (L * z)' .+ μ'
end

"""
    simulate_pca(a, nsim; pctExp=1.0, mean=Float64[], seed=1234)

Simulate through the principal components (Week 03, 7). Decompose `Σ = V Λ V′`,
order the eigenvalues largest first, keep enough to explain `pctExp` of the
total variance, simulate with `B = V_k Λ_k^½`.

Eigenvalues at or below 1e-8 are dropped as numerical noise. With `pctExp = 1`
this reproduces `Σ`; below 1 the truncation is deliberate and the simulated
covariance is short by the tail of `Λ` that was dropped.
"""
function simulate_pca(a::AbstractMatrix, nsim::Integer; pctExp::Real=1.0,
                      mean=Float64[], seed::Integer=1234)
    n = size(a, 1)
    μ = zeros(n)
    isempty(mean) || (μ .= mean)

    F = eigen(Symmetric(Matrix(a)))
    order = reverse(axes(F.values, 1))          # julia returns them ascending
    vals = F.values[order]
    vecs = F.vectors[:, order]

    total = sum(vals)
    keep = findall(>=(1e-8), vals)

    if pctExp < 1
        cum = 0.0
        nval = 0
        for i in eachindex(keep)
            cum += vals[i] / total
            nval += 1
            cum >= pctExp && break
        end
        nval < length(keep) && (keep = keep[1:nval])
    end
    vals = vals[keep]
    vecs = vecs[:, keep]

    B = vecs * diagm(sqrt.(vals))
    Random.seed!(seed)
    z = randn(length(vals), nsim)
    return (B * z)' .+ μ'
end
