# Non-PSD repair and PSD Cholesky. Week 03, sections 2.2, 6.1, 6.2.

"""
    near_psd(a; epsilon=0.0)

Rebonato and Jäckel (Week 03, 6.1). Work in correlation space, floor the
eigenvalues at `epsilon`, rescale so the reconstruction keeps a unit diagonal:

    T_ii = 1 / Σₖ v_ik² λ'ₖ ,   B = T^½ V Λ'^½ ,   out = B B′

Single pass, no iteration. Lands in the PSD cone but not at the nearest point;
`higham_nearestPSD` does that.

A covariance is converted to correlation on the way in and the variances put
back on the way out, so the diagonal is preserved.
"""
function near_psd(a::AbstractMatrix; epsilon::Real=0.0)
    n = size(a, 1)
    out = Matrix{Float64}(a)

    invSD = nothing
    if count(x -> isapprox(x, 1.0), diag(out)) != n     # input was a covariance
        invSD = diagm(1 ./ sqrt.(diag(out)))
        out = invSD * out * invSD
    end

    vals, vecs = eigen(Symmetric(out))
    vals = max.(vals, epsilon)
    T = 1 ./ ((vecs .* vecs) * vals)
    B = diagm(sqrt.(T)) * vecs * diagm(sqrt.(vals))
    out = B * B'

    if invSD !== nothing
        SD = diagm(1 ./ diag(invSD))
        out = SD * out * SD
    end
    return out
end

# --- Higham -----------------------------------------------------------------
# projection onto the PSD cone: clip the eigenvalues at zero
function _proj_psd(A::AbstractMatrix)
    F = eigen(Symmetric(A))
    return F.vectors * diagm(max.(F.values, 0.0)) * F.vectors'
end

# P_S, taken in the W^½ metric
_proj_s(A, W05, iW05) = iW05 * _proj_psd(W05 * A * W05) * iW05

# P_U: projection onto matrices with a unit diagonal
function _proj_u(A::AbstractMatrix)
    out = copy(A)
    for i in axes(out, 1)
        out[i, i] = 1.0
    end
    return out
end

# squared weighted Frobenius norm
function _wnorm(A::AbstractMatrix, W05::AbstractMatrix)
    M = W05 * A * W05
    return sum(M .* M)
end

"""
    higham_nearestPSD(pc; W=nothing, epsilon=1e-9, maxIter=100, tol=1e-9, verbose=false)

Alternating projections between the PSD cone and the unit-diagonal set
(Week 03, 6.2). The `ΔS` term is Dykstra's correction; without it the iteration
converges to some point in the intersection, with it the limit is the nearest
correlation matrix in the weighted Frobenius norm.

Stops when the norm stops moving *and* the smallest eigenvalue has cleared
`-epsilon`. Both are needed, since the norm can flatten while the matrix is
still slightly indefinite.
"""
function higham_nearestPSD(pc::AbstractMatrix; W=nothing, epsilon::Real=1e-9,
                           maxIter::Integer=100, tol::Real=1e-9, verbose::Bool=false)
    n = size(pc, 1)
    W === nothing && (W = diagm(ones(n)))
    W05 = sqrt.(W)
    iW05 = inv(W05)

    Yk = Matrix{Float64}(pc)
    invSD = nothing
    if count(x -> isapprox(x, 1.0), diag(Yk)) != n
        invSD = diagm(1 ./ sqrt.(diag(Yk)))
        Yk = invSD * Yk * invSD
    end
    Y0 = copy(Yk)

    ΔS = zeros(n, n)
    norm_prev = Inf
    iter = 1
    while iter <= maxIter
        Rk = Yk .- ΔS
        Xk = _proj_s(Rk, W05, iW05)
        ΔS = Xk .- Rk
        Yk = _proj_u(Xk)

        norm_cur = _wnorm(Yk .- Y0, W05)
        min_eig = minimum(eigvals(Symmetric(Yk)))
        if abs(norm_cur - norm_prev) < tol && min_eig > -epsilon
            break
        end
        norm_prev = norm_cur
        iter += 1
    end
    verbose && println(iter <= maxIter ? "Higham converged in $iter iterations." :
                                         "Higham failed to converge in $maxIter iterations.")

    if invSD !== nothing
        SD = diagm(1 ./ diag(invSD))
        Yk = SD * Yk * SD
    end
    return Yk
end

# --- Cholesky for the PSD case ----------------------------------------------
"""
    chol_psd!(root, a; epsilon=-1e-8)

Cholesky that tolerates a zero eigenvalue (Week 03, 2.2). The recursion divides
by the diagonal root, which `LinearAlgebra.cholesky` cannot do when the matrix
is singular. Here a pivot at zero, or just below within `epsilon`, is set to
zero and its column with it.

Writes into `root` and returns it.
"""
function chol_psd!(root::AbstractMatrix, a::AbstractMatrix; epsilon::Real=-1e-8)
    n = size(a, 1)
    fill!(root, 0.0)

    for j in 1:n
        s = j > 1 ? dot(view(root, j, 1:j-1), view(root, j, 1:j-1)) : 0.0
        temp = a[j, j] - s
        if epsilon <= temp <= 0
            temp = 0.0
        end
        root[j, j] = sqrt(temp)          # a genuinely negative pivot throws here

        if root[j, j] != 0.0
            ir = 1.0 / root[j, j]
            for i in (j+1):n
                s = j > 1 ? dot(view(root, i, 1:j-1), view(root, j, 1:j-1)) : 0.0
                root[i, j] = (a[i, j] - s) * ir
            end
        end
    end
    return root
end

"""
    chol_psd(a; epsilon=-1e-8)

Allocating form of [`chol_psd!`](@ref).
"""
chol_psd(a::AbstractMatrix; epsilon::Real=-1e-8) =
    chol_psd!(zeros(Float64, size(a, 1), size(a, 2)), a; epsilon=epsilon)
