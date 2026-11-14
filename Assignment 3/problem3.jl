# Problem 3: VaR of an option book on ABC.
#
#     julia --project=. problem3.jl
#
# Positions are valued at the last price in problem3.csv. r = 4.25%, no
# dividends (b = r), 255 trading days a year, DaysToExpiry in trading days,
# implied vols held fixed, 10 trading day horizon, zero expected return.

include(joinpath(@__DIR__, "library", "RiskLib.jl"))
using Plots

const r = 0.0425
const DAYS = 255
const H = 10                       # horizon, trading days
const NSIM = 1_000_000              # 1% ES averages the worst 10,000 outcomes
const ALPHAS = (0.05, 0.01)

px = CSV.read(joinpath(@__DIR__, "data", "problem3.csv"), DataFrame)
rets = return_calculate(px; method="DISCRETE", dateColumn="Date").ABC   # arithmetic
S0 = px.ABC[end]

pf = CSV.read(joinpath(@__DIR__, "data", "problem3_portfolios.csv"), DataFrame)
books = unique(pf.Portfolio)

# Value of one position at price S with `elapsed` trading days gone. S may be
# a vector of simulated prices.
function pos_value(p, S, elapsed)
    p.Type == "Stock" && return p.Holding .* S
    T = (p.DaysToExpiry - elapsed) / DAYS
    return p.Holding .* [gbsm(p.Type == "Call", s, p.Strike, T, r, r, p.ImpliedVol).value for s in S]
end
book_value(bk, S, elapsed) = sum(pos_value(p, S, elapsed) for p in eachrow(pf[pf.Portfolio .== bk, :]))

# --- (a) Greeks of each book ----------------------------------------------------
# Stock: delta 1, gamma 0, theta 0. Theta is -∂V/∂T per year (Week 07), so the
# carry over the horizon is Θ·10/255.
function book_greeks(bk)
    δ = γ = θ = 0.0
    for p in eachrow(pf[pf.Portfolio .== bk, :])
        if p.Type == "Stock"
            δ += p.Holding
        else
            g = gbsm(p.Type == "Call", S0, p.Strike, p.DaysToExpiry / DAYS, r, r, p.ImpliedVol; includeGreeks=true)
            δ += p.Holding * g.delta
            γ += p.Holding * g.gamma
            θ += p.Holding * g.theta
        end
    end
    return δ, γ, θ
end

# The positions, so the reader can check the inputs.
pos = DataFrame(Book=pf.Portfolio, Type=pf.Type,
                Strike=[ismissing(x) ? "--" : @sprintf("%.0f", x) for x in pf.Strike],
                Days=[ismissing(x) ? "--" : string(x) for x in pf.DaysToExpiry],
                IV=[ismissing(x) ? "--" : @sprintf("%.0f\\%%", 100x) for x in pf.ImpliedVol],
                Holding=Float64.(pf.Holding))
write_table(pos, "p3_positions"; fmts=Dict("Holding" => "%.0f"),
            header=["Book", "Type", "Strike", "Days to expiry", "Implied vol", "Holding"], align="llrrrr")

greek = DataFrame(Book=String[], Value=Float64[], Delta=Float64[], Gamma=Float64[], Theta=Float64[],
                  Dollar_delta=Float64[], Theta_10d=Float64[], Flat_10d=Float64[])
for bk in books
    δ, γ, θ = book_greeks(bk)
    V = book_value(bk, S0, 0)
    push!(greek, (bk, V, δ, γ, θ, δ * S0, θ * H / DAYS, book_value(bk, S0, H) - V))
end
write_table(greek[:, [:Book, :Value, :Delta, :Gamma, :Theta, :Dollar_delta, :Theta_10d]], "p3_greeks";
            fmts=Dict("Value" => "%.2f", "Delta" => "%.2f", "Gamma" => "%.3f", "Theta" => "%.1f",
                      "Dollar_delta" => "%.1f", "Theta_10d" => "%.2f"),
            header=["Book", "Value", "\$\\delta\$", "\$\\Gamma\$", "\$\\Theta\$ (per year)", "\$\\delta S\$",
                    "\$\\Theta\\cdot 10/255\$"])

# --- (c) moments and the two fits -----------------------------------------------
mom = sample_moments(rets)
momtab = DataFrame(Statistic=["Mean", "Std. dev.", "Skewness", "Excess kurtosis", "Annualized vol"],
                   Value=[mom.mean, mom.std, mom.skew, mom.xkurt, mom.std * sqrt(DAYS)])
write_table(momtab, "p3_moments"; fmts=Dict("Value" => "%.5f"), header=["Statistic", "ABC daily returns"])

fn = fit_normal(rets)
ft = fit_general_t(rets)
tm = ft.errorModel
fits = DataFrame(Model=["Normal", "Student t"],
                 Location=[fn.errorModel.μ, tm.μ], Scale=[fn.errorModel.σ, tm.σ],
                 Nu=[NaN, tm.ρ.ν],
                 Std=[fn.errorModel.σ, tm.σ * sqrt(tm.ρ.ν / (tm.ρ.ν - 2))],
                 LogLik=[sum(logpdf.(fn.errorModel, rets)), sum(logpdf.(tm, rets))],
                 AICc=[aicc(fn, rets), aicc(ft, rets)])
write_table(fits, "p3_fits"; fmts=Dict("Location" => "%.6f", "Scale" => "%.6f", "Nu" => "%.3f",
                                       "Std" => "%.6f", "LogLik" => "%.2f", "AICc" => "%.2f"),
            header=["Model", "Location", "Scale", "\$\\nu\$", "Implied s.d.", "Log lik.", "AICc"])
# fit_general_t starts from the moment estimates only. Check it found the
# maximum: refit by Nelder-Mead from several starting ν and make sure none
# reaches a higher log likelihood.
ll_fit = sum(logpdf.(tm, rets))
ll_alt = maximum(begin
    θ0 = [mean(rets), log(std(rets) * sqrt((ν0 - 2) / ν0)), log(ν0 - 2)]
    -Optim.minimum(optimize(θ -> -general_t_ll(θ[1], exp(θ[2]), 2 + exp(θ[3]), rets), θ0, NelderMead(),
                            Optim.Options(iterations=20_000, g_tol=1e-12)))
end for ν0 in (2.5, 4.0, 8.0, 20.0, 60.0))
@assert ll_alt <= ll_fit + 1e-6 "t fit is not at the maximum: $ll_alt > $ll_fit"

σd = fn.errorModel.σ               # daily s.d. used by delta normal
σh = σd * sqrt(H)                  # √t rule, Week 04 §7

# --- (d) 10 day delta normal VaR and ES ------------------------------------------
# VaR = -Φ⁻¹(α) |δ S| σ_d √10 in dollars (Week 04 §4 with the √t rule);
# ES under normality is φ(Φ⁻¹(α))/α in place of -Φ⁻¹(α).
dn = DataFrame(Book=String[], VaR5=Float64[], ES5=Float64[], VaR1=Float64[], ES1=Float64[])
for row in eachrow(greek)
    sd = abs(row.Dollar_delta) * σh
    push!(dn, (row.Book, VaR_normal(0, sd; alpha=0.05), ES_normal(0, sd; alpha=0.05),
               VaR_normal(0, sd; alpha=0.01), ES_normal(0, sd; alpha=0.01)))
end
write_table(dn, "p3_deltanormal"; fmts=Dict(c => "%.2f" for c in names(dn)[2:end]),
            header=["Book", "VaR 5\\%", "ES 5\\%", "VaR 1\\%", "ES 1\\%"])

# --- (e) simulation with full revaluation ---------------------------------------
# One matrix of uniforms drives both models (common random numbers), so the
# normal and t results differ only through the distribution. Both are centred
# at zero as the problem asks: the normal keeps σ, the t keeps its scale and ν.
models = Dict("Normal" => Normal(0.0, fn.errorModel.σ), "t" => TDist(tm.ρ.ν) * tm.σ)
# 10 daily arithmetic returns per path, compounded to the horizon price
simulate_S10(U) = Dict(k => S0 .* vec(prod(1 .+ quantile.(d, U), dims=2)) for (k, d) in models)
S10 = simulate_S10(rand(MersenneTwister(545), NSIM, H))

risk_row(x) = (mean(x), VaR(x; alpha=0.05), ES(x; alpha=0.05), VaR(x; alpha=0.01), ES(x; alpha=0.01))
sim = DataFrame(Book=String[], Model=String[], Method=String[], Mean=Float64[], VaR5=Float64[],
                ES5=Float64[], VaR1=Float64[], ES1=Float64[])
pnl = Dict{Tuple{String,String},Vector{Float64}}()
for bk in books, md in ("Normal", "t")
    v0 = book_value(bk, S0, 0)
    x = book_value(bk, S10[md], H) .- v0
    pnl[(bk, md)] = x
    push!(sim, (bk, md, "Full revaluation", risk_row(x)...))
    # (g) second order approximation on the same price paths
    g = greek[greek.Book .== bk, :][1, :]
    dS = S10[md] .- S0
    approx = g.Delta .* dS .+ 0.5g.Gamma .* dS .^ 2 .+ g.Theta * H / DAYS
    push!(sim, (bk, md, "Delta-gamma-theta", risk_row(approx)...))
end
# Monte Carlo error: rerun the full revaluation with a second, unrelated seed
# and record the largest relative change in any VaR or ES.
S10b = simulate_S10(rand(MersenneTwister(2026), NSIM, H))
mc_gap = 0.0
for bk in books, md in ("Normal", "t")
    a = risk_row(pnl[(bk, md)])
    c = risk_row(book_value(bk, S10b[md], H) .- book_value(bk, S0, 0))
    global mc_gap = max(mc_gap, maximum(abs.(c[2:end] ./ a[2:end] .- 1)))
end
S10b = nothing

fr = sim[sim.Method .== "Full revaluation", Not(:Method)]
fr.ES_VaR_5 = fr.ES5 ./ fr.VaR5
write_table(fr, "p3_sim"; fmts=Dict("Mean" => "%.2f", "VaR5" => "%.2f", "ES5" => "%.2f", "VaR1" => "%.2f",
                                    "ES1" => "%.2f", "ES_VaR_5" => "%.2f"),
            header=["Book", "Model", "Mean P\\&L", "VaR 5\\%", "ES 5\\%", "VaR 1\\%", "ES 1\\%", "ES/VaR 5\\%"])
dg = sim[(sim.Method .== "Delta-gamma-theta") .& (sim.Model .== "Normal"), Not([:Method, :Model])]
write_table(dg, "p3_dgt"; fmts=Dict(c => "%.2f" for c in names(dg)[2:end]),
            header=["Book", "Mean P\\&L", "VaR 5\\%", "ES 5\\%", "VaR 1\\%", "ES 1\\%"])
CSV.write(joinpath(OUT_TABLES, "p3_sim_all.csv"), sim)

# (i) how far each number moves from the normal to the t
mv = DataFrame(Book=String[], Mean=Float64[], VaR5=Float64[], ES5=Float64[], VaR1=Float64[], ES1=Float64[])
for bk in books
    a = fr[(fr.Book .== bk) .& (fr.Model .== "Normal"), :][1, :]
    c = fr[(fr.Book .== bk) .& (fr.Model .== "t"), :][1, :]
    push!(mv, (bk, c.Mean - a.Mean, [100 * (c[k] / a[k] - 1) for k in (:VaR5, :ES5, :VaR1, :ES1)]...))
end
write_table(mv, "p3_t_vs_normal"; fmts=Dict("Mean" => "%.2f", "VaR5" => "%+.1f\\%%", "ES5" => "%+.1f\\%%",
                                            "VaR1" => "%+.1f\\%%", "ES1" => "%+.1f\\%%"),
            header=["Book", "\$\\Delta\$ mean (\\\$)", "VaR 5\\%", "ES 5\\%", "VaR 1\\%", "ES 1\\%"])

# (h) the mean P&L next to its parts: exact flat-price P&L, theta carry,
# expected gamma cost, and the financing / variance-premium split.
# Under GBSM with b = r each option satisfies Θ = r(V - SΔ) - ½σ_iv²S²Γ, so
# ΘΔt + ½ΓS²σ_real²Δt splits into r(V - Sδ)Δt + ½S²Δt Σ h_j Γ_j (σ_real² - σ_j²).
σ_real = σd * sqrt(DAYS)
carry = DataFrame(Book=String[], Flat=Float64[], Theta=Float64[], Gamma=Float64[],
                  Financing=Float64[], VarPrem=Float64[], Predicted=Float64[], Simulated=Float64[])
for row in eachrow(greek)
    bk = row.Book
    vprem = 0.0
    for p in eachrow(pf[(pf.Portfolio .== bk) .& (pf.Type .!= "Stock"), :])
        g = gbsm(p.Type == "Call", S0, p.Strike, p.DaysToExpiry / DAYS, r, r, p.ImpliedVol; includeGreeks=true)
        vprem += p.Holding * 0.5 * g.gamma * S0^2 * (σ_real^2 - p.ImpliedVol^2) * H / DAYS
    end
    fin = r * (row.Value - row.Dollar_delta) * H / DAYS
    gam10 = 0.5 * row.Gamma * (S0 * σh)^2
    sim_mean = fr[(fr.Book .== bk) .& (fr.Model .== "Normal"), :Mean][1]
    push!(carry, (bk, row.Flat_10d, row.Theta_10d, gam10, fin, vprem, fin + vprem, sim_mean))
end
write_table(carry, "p3_carry"; fmts=Dict(c => "%.2f" for c in names(carry)[2:end]),
            header=["Book", "P\\&L, price flat", "\$\\Theta\\Delta t\$", "\$\\tfrac12\\Gamma S^2\\sigma^2\\Delta t\$",
                    "\$r(V-S\\delta)\\Delta t\$", "Vol premium", "Sum", "Simulated mean"])

# Return quantiles of the two 10 day distributions, to see where the tails part.
q10 = DataFrame(Quantile=[0.001, 0.01, 0.025, 0.05, 0.10, 0.5, 0.95, 0.975, 0.99],)
for md in ("Normal", "t")
    q10[!, md] = [quantile(S10[md] ./ S0 .- 1, a) for a in q10.Quantile]
end
CSV.write(joinpath(OUT_TABLES, "p3_return_quantiles.csv"), q10)

# --- (f) P&L against the price in 10 days, with the delta-only line --------------
Sgrid = range(0.75S0, 1.25S0, length=301)
panels = []
for bk in books
    g = greek[greek.Book .== bk, :][1, :]
    full = book_value(bk, collect(Sgrid), H) .- book_value(bk, S0, 0)
    dS = Sgrid .- S0
    pl = plot(Sgrid, full, lw=2, label="Full revaluation", title=bk, titlefontsize=9,
              xlabel="ABC price in 10 days", ylabel="10 day P&L", legendfontsize=6, legend=:bottom)
    plot!(pl, Sgrid, g.Delta .* dS, lw=1.2, ls=:dash, label="Delta only")
    plot!(pl, Sgrid, g.Delta .* dS .+ 0.5g.Gamma .* dS .^ 2 .+ g.Theta * H / DAYS, lw=1, ls=:dot,
          label="Delta-gamma-theta")
    vline!(pl, [S0], color=:grey, lw=0.6, label="")
    hline!(pl, [0], color=:grey, lw=0.6, label="")
    push!(panels, pl)
end
fig = plot(panels..., layout=(2, 2), size=(900, 680), left_margin=4Plots.mm, bottom_margin=3Plots.mm)
savefig(fig, joinpath(OUT_FIGURES, "p3_pnl.pdf"))
savefig(fig, joinpath(OUT_FIGURES, "p3_pnl.png"))

# Break-even moves of the hedged short call over 10 days (full revaluation,
# vols fixed): the price changes at which the theta carry is used up.
hsc_pnl(s) = book_value("HedgedShortCall", s, H) - book_value("HedgedShortCall", S0, 0)
be_up = find_zero(hsc_pnl, (S0, 1.5S0), Bisection())
be_dn = find_zero(hsc_pnl, (0.5S0, S0), Bisection())

if abspath(PROGRAM_FILE) == @__FILE__
    println(mom)
    show(fits, allcols=true); println()
    show(greek, allcols=true); println()
    show(dn); println()
    show(sim, allrows=true, allcols=true); println()
    show(mv, allcols=true); println()
    show(q10); println()
    @printf("largest relative VaR/ES change with a second seed: %.2f%%\n", 100mc_gap)
    @printf("hedged short call break-evens over 10 days: %+.2f%% and -%.2f%%\n",
            100(be_up / S0 - 1), 100(1 - be_dn / S0))
end
