# Portfolio construction. Week 07 §4-5.
#
# The optimizer set-up follows the course's optimizers.jl: JuMP with Ipopt,
# weights summing to one and box bounds on each weight.

"""
    sharpe(w, μ, Σ, rf) = (w'μ - rf) / √(w'Σw)
"""
sharpe(w, μ, Σ, rf) = (w' * μ - rf) / sqrt(w' * Σ * w)

"""
    maxSR(Σ, μ, rf, bounds; starts=20)

Maximum Sharpe ratio portfolio with `sum(w) = 1` and `bounds[i,1] ≤ wᵢ ≤ bounds[i,2]`.

The Sharpe ratio is not concave in w, so Ipopt can in principle stop at a
local optimum. It is solved from equal weights and from `starts` random
feasible points, and the best solution is kept.
"""
function maxSR(Σ::AbstractMatrix, μ::AbstractVector, rf::Real, bounds::AbstractMatrix;
               starts::Int=20, rng=MersenneTwister(545))
    n = length(μ)
    best_w, best_sr = Float64[], -Inf
    for s in 0:starts
        w0 = s == 0 ? fill(1 / n, n) : rand(rng, Dirichlet(ones(n)))
        w0 = clamp.(w0, bounds[:, 1], bounds[:, 2])
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, bounds[i, 1] <= w[i=1:n] <= bounds[i, 2], start = w0[i])
        @constraint(model, sum(w) == 1)
        @objective(model, Max, (sum(w[i] * μ[i] for i in 1:n) - rf) /
                               sqrt(sum(w[i] * Σ[i, j] * w[j] for i in 1:n, j in 1:n)))
        optimize!(model)
        is_solved_and_feasible(model) || continue
        x = value.(w)
        sr = sharpe(x, μ, Σ, rf)
        if sr > best_sr
            best_w, best_sr = x, sr
        end
    end
    return best_w
end

maxSR(Σ, μ, rf; kw...) = maxSR(Σ, μ, rf, hcat(zeros(length(μ)), ones(length(μ))); kw...)

"""
    maxSR_unconstrained(Σ, μ, rf)

With shorting allowed the tangency portfolio has the closed form
`w ∝ Σ⁻¹(μ - rf)`, scaled to sum to one.
"""
function maxSR_unconstrained(Σ, μ, rf)
    raw = Σ \ (μ .- rf)
    return raw ./ sum(raw)
end

"""
    maxSR_qp(Σ, μ, rf, bounds)

The same problem as `maxSR`, solved as a convex quadratic program. Because the
Sharpe ratio does not change when w is scaled, substitute y = κw:

    min y'Σy   s.t.  (μ - rf)'y = 1,  sum(y) = κ ≥ 0,  lᵢκ ≤ yᵢ ≤ uᵢκ

and recover w = y/κ. With Σ positive semi-definite this is convex, so the
solution is the global maximum with no restarts. It needs at least one asset
with μᵢ > rf. Used inside the bootstrap in Problem 4, where `maxSR` with
restarts would be slow, and checked against `maxSR` on the full sample.
"""
function maxSR_qp(Σ, μ, rf, bounds)
    n = length(μ)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, y[1:n])
    @variable(model, κ >= 0)
    @constraint(model, sum((μ[i] - rf) * y[i] for i in 1:n) == 1)
    @constraint(model, sum(y) == κ)
    for i in 1:n
        isfinite(bounds[i, 1]) && @constraint(model, y[i] >= bounds[i, 1] * κ)
        isfinite(bounds[i, 2]) && @constraint(model, y[i] <= bounds[i, 2] * κ)
    end
    @objective(model, Min, sum(y[i] * Σ[i, j] * y[j] for i in 1:n, j in 1:n))
    optimize!(model)
    # Return nothing rather than an unconverged iterate, so a caller looping
    # over many problems (the bootstrap) can count and drop failures.
    is_solved_and_feasible(model) || return nothing
    k = value(κ)
    (isfinite(k) && k > 1e-10) || return nothing
    w = value.(y) ./ k
    all(isfinite, w) || return nothing
    return w
end

"""
    min_variance(Σ, μ, target, bounds)

One frontier point (Week 07, 4.3): min w'Σw s.t. sum(w) = 1, w'μ = target,
and the bounds. `target = nothing` gives the global minimum variance portfolio.
"""
function min_variance(Σ, μ, target, bounds)
    n = length(μ)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, bounds[i, 1] <= w[i=1:n] <= bounds[i, 2], start = 1 / n)
    @constraint(model, sum(w) == 1)
    target === nothing || @constraint(model, sum(w[i] * μ[i] for i in 1:n) == target)
    @objective(model, Min, sum(w[i] * Σ[i, j] * w[j] for i in 1:n, j in 1:n))
    optimize!(model)
    return is_solved_and_feasible(model) ? value.(w) : nothing
end

"""
    efficient_frontier(Σ, μ, bounds; npoints=150)

The efficient (upper) branch: targets from the minimum variance portfolio's
return up to the highest return the bounds allow.
"""
function efficient_frontier(Σ, μ, bounds; npoints=150)
    gmv = min_variance(Σ, μ, nothing, bounds)
    top = maximum(μ[i] for i in eachindex(μ) if bounds[i, 2] > 0)
    pts = Tuple{Float64,Float64}[]
    for R in range(gmv' * μ, top, length=npoints)
        w = min_variance(Σ, μ, R, bounds)
        w === nothing || push!(pts, (sqrt(w' * Σ * w), w' * μ))
    end
    return pts
end
