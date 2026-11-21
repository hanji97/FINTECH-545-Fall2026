# Ex-post return and volatility attribution. The factor betas are the ones
# estimated with the factor models in Week 07, sections 5.3-6; the attribution
# itself is the Week 08 step that Week 07, section 7 points to.
#
# Weights drift with returns between rebalances, so the exposure to each factor
# is recomputed every period from the current weights. Single period
# contributions are linked over time with Carino's scaling (Carino 1999), which
# makes the attributed returns add up to the compounded portfolio return.

# log(1+r)/r, with its limit 1 at r = 0
_carino(r) = r == 0 ? 1.0 : log1p(r) / r

"""
    expost_factor(w, upReturns, upFfData, Betas)

`w` are the starting weights, `upReturns` the stock returns over the holding
period (one column per stock), `upFfData` the factor returns (one column per
factor) and `Betas` the stocks × factors loadings. Passing the stock returns as
the factors with `Betas = I` gives the attribution to the stocks themselves.

For each period t, with wₜ the weights at the start of the period:

    factor weight   bₜ = Betas' wₜ
    portfolio       Rₜ = wₜ' rₜ
    residual        αₜ = Rₜ - bₜ' fₜ
    next weights    wₜ₊₁ = wₜ ∘ (1 + rₜ) / (1 + Rₜ)

Return attribution to factor j is Σₜ kₜ bₜⱼ fₜⱼ with the Carino factors
kₜ = [ln(1+Rₜ)/Rₜ] / [ln(1+R)/R], R the compounded portfolio return.

Volatility attribution regresses each component series (bₜⱼ fₜⱼ and αₜ) on Rₜ.
The components add up to Rₜ, so their slopes add up to 1 and the component
volatilities βⱼ σ(R) add up to σ(R).

Returns a NamedTuple; `Attribution` is the table with rows TotalReturn,
Return Attribution and Vol Attribution.
"""
function expost_factor(w::AbstractVector, upReturns::DataFrame, upFfData::DataFrame,
                       Betas::AbstractMatrix)
    factors = names(upFfData)
    r = Matrix{Float64}(upReturns)
    f = Matrix{Float64}(upFfData[:, factors])
    n = size(r, 1)

    weights = Matrix{Float64}(undef, n, length(w))
    factorWeights = Matrix{Float64}(undef, n, length(factors))
    pReturn = Vector{Float64}(undef, n)
    resid = Vector{Float64}(undef, n)

    wt = collect(float.(w))
    for t in 1:n
        weights[t, :] = wt
        factorWeights[t, :] = Betas' * wt
        grown = wt .* (1 .+ r[t, :])
        pReturn[t] = sum(grown) - 1
        resid[t] = pReturn[t] - factorWeights[t, :]' * f[t, :]
        wt = grown ./ sum(grown)
    end

    compound(x) = exp(sum(log1p.(x))) - 1
    totalR = compound(pReturn)
    k = _carino.(pReturn) ./ _carino(totalR)

    contrib = hcat(f .* factorWeights, resid)          # bₜⱼ fₜⱼ and αₜ, columns sum to Rₜ
    cols = vcat(factors, "Alpha")

    totalRet = vcat([compound(f[:, j]) for j in eachindex(factors)], compound(resid), totalR)
    retAttrib = vcat(vec(sum(contrib .* k, dims=1)), totalR)

    X = hcat(ones(n), pReturn)
    slopes = (X \ contrib)[2, :]
    volAttrib = vcat(slopes .* std(pReturn), std(pReturn))

    Attribution = DataFrame(:Value => ["TotalReturn", "Return Attribution", "Vol Attribution"])
    for (j, c) in enumerate(vcat(cols, "Portfolio"))
        Attribution[!, c] = [totalRet[j], retAttrib[j], volAttrib[j]]
    end

    return (Attribution=Attribution, weights=weights, factorWeights=factorWeights,
            portfolioReturn=pReturn, residReturn=resid, carinoK=k)
end
