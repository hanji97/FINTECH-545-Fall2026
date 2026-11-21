# Portfolio construction. Week 07, sections 4-5, and risk parity.
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
starting points clamped to the bounds, and the best solution is kept.
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

# --- risk parity --------------------------------------------------------------

"""
    component_sd(w, Σ)

Each asset's contribution to portfolio volatility,

    CSDᵢ = wᵢ (Σw)ᵢ / σₚ,   σₚ = √(w'Σw)

σₚ is homogeneous of degree one in w, so by Euler's theorem the CSDᵢ add up
to σₚ.
"""
component_sd(w, Σ) = w .* (Σ * w) ./ sqrt(w' * Σ * w)

"""
    riskParity(Σ; riskBudget=ones(n))

Long-only weights at which every asset's share of portfolio volatility matches
its risk budget, i.e. CSDᵢ / bᵢ is the same for all i. Solved with Ipopt by
minimizing the squared spread of CSDᵢ / (bᵢ σₚ) around their mean, with
sum(w) = 1 and w ≥ 0, started from equal weights. Dividing by σₚ turns the CSDs
into shares of portfolio volatility, so the objective does not depend on the
units of Σ.
"""
function riskParity(Σ::AbstractMatrix; riskBudget=ones(size(Σ, 1)))
    n = size(Σ, 1)
    length(riskBudget) == n || throw(DimensionMismatch("risk budget has $(length(riskBudget)) entries for $n assets"))
    b = collect(float.(riskBudget))

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", 1e-12)
    @variable(model, w[1:n] >= 0, start = 1 / n)
    @expression(model, Σw[i=1:n], sum(Σ[i, j] * w[j] for j in 1:n))
    @expression(model, pvar, sum(w[i] * Σw[i] for i in 1:n))
    @expression(model, share[i=1:n], w[i] * Σw[i] / pvar / b[i])
    @expression(model, avg, sum(share) / n)
    @objective(model, Min, sum((share[i] - avg)^2 for i in 1:n))
    @constraint(model, sum(w) == 1)
    optimize!(model)
    is_solved_and_feasible(model) || error("risk parity did not converge: ", raw_status(model))

    x = value.(w)
    return x ./ sum(x)
end
