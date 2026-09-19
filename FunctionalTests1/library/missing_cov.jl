# Covariance and correlation with missing data. Week 03, section 3.

"""
    missing_cov(x; skipMiss=true, fun=cov)

Covariance of a matrix containing `missing`, or correlation with `fun=cor`.

    skipMiss=true    drop every row holding a missing value, compute once on the rest
    skipMiss=false   pairwise: for entry (i,j) drop only rows missing in column i or j

Pairwise keeps more data, but each entry comes off a different row set, so the
result is not guaranteed PSD.
"""
function missing_cov(x; skipMiss::Bool=true, fun=cov)
    n, m = size(x)
    missing_rows = [Set(findall(ismissing, view(x, :, j))) for j in 1:m]

    if all(isempty, missing_rows)
        return fun(_dense(x))
    end

    if skipMiss
        drop = reduce(union, missing_rows)
        keep = [i for i in 1:n if !(i in drop)]
        return fun(_dense(view(x, keep, :)))
    end

    out = Matrix{Float64}(undef, m, m)
    for j in 1:m, i in j:m
        drop = union(missing_rows[i], missing_rows[j])
        keep = [r for r in 1:n if !(r in drop)]
        # for i == j this is a 2-column block of the same series, whose
        # off-diagonal is the variance
        out[i, j] = out[j, i] = fun(_dense(view(x, keep, [i, j])))[1, 2]
    end
    return out
end

# drop the Union{Missing,T} once the missing rows are gone
_dense(x) = convert(Matrix{Float64}, x)
