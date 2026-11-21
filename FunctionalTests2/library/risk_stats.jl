# VaR and ES. Week 04, sections 2 and 5; Week 05, section 1.
#
# Both are reported as positive numbers for a loss:
#   VaR_α(x) = -F⁻¹(α)        ES_α(x) = -E[x | x ≤ -VaR_α(x)]
# measured from zero (the "absolute" convention in Week 04, 2.1), on P&L or on
# returns with gains positive. The "diff from mean" convention in 2.1 is the
# same function applied to x - E[x].

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

# --- portfolio VaR and ES from simulated returns (Week 04, section 5) -------

"""
    simulated_values(portfolio, simRet)

Steps 3-5 of the Monte Carlo VaR algorithm in Week 04, 5. `portfolio` has
columns `Stock` and `currentValue`; `simRet` has one column per stock and one
row per draw. Returns one row per position and draw with the repriced value
and the P&L.
"""
function simulated_values(portfolio::DataFrame, simRet::DataFrame)
    nSim = nrow(simRet)
    values = crossjoin(portfolio[:, [:Stock, :currentValue]], DataFrame(iteration=1:nSim))
    r = [simRet[values.iteration[k], values.Stock[k]] for k in 1:nrow(values)]
    values.simulatedValue = values.currentValue .* (1 .+ r)
    values.pnl = values.simulatedValue .- values.currentValue
    return values
end

# VaR and ES of one P&L vector, in dollars and as a fraction of the value at risk
function _risk_row(pnl, currentValue)
    v95, e95 = VaR(pnl; alpha=0.05), ES(pnl; alpha=0.05)
    v99, e99 = VaR(pnl; alpha=0.01), ES(pnl; alpha=0.01)
    return (currentValue=currentValue, VaR95=v95, ES95=e95, VaR99=v99, ES99=e99,
            VaR95_Pct=v95 / currentValue, ES95_Pct=e95 / currentValue,
            VaR99_Pct=v99 / currentValue, ES99_Pct=e99 / currentValue)
end

"""
    aggRisk(values, aggLevel)

VaR and ES at 95% and 99% for every group in `aggLevel` and for the whole
portfolio, which is reported as a row with the group set to "Total".

The P&L is summed across positions within each draw before the quantile is
taken. Adding the position VaRs would assume the worst days line up, and VaR is
not subadditive in general (Week 04, 3.1), so the portfolio numbers cannot be
built from the position numbers.
"""
function aggRisk(values::DataFrame, aggLevel::Vector{Symbol})
    rows = DataFrame()
    if !isempty(aggLevel)
        byDraw = combine(groupby(values, vcat(aggLevel, [:iteration])),
                         :currentValue => sum => :currentValue, :pnl => sum => :pnl)
        for g in groupby(byDraw, aggLevel)
            push!(rows, merge(NamedTuple(g[1, aggLevel]), _risk_row(g.pnl, g.currentValue[1])))
        end
    end
    total = combine(groupby(values, :iteration),
                    :currentValue => sum => :currentValue, :pnl => sum => :pnl)
    totalRow = _risk_row(total.pnl, total.currentValue[1])
    isempty(aggLevel) && return DataFrame([totalRow])
    labels = NamedTuple{Tuple(aggLevel)}(Tuple(fill("Total", length(aggLevel))))
    push!(rows, merge(labels, totalRow); promote=true)
    return rows
end
