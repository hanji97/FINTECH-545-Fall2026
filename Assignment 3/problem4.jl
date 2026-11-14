# Problem 4: a Sharpe ratio that sells insurance.
#
#     julia --project=. problem4.jl
#
# 240 monthly total returns for EQ, BOND, STK and PUTW. The risk free rate is
# 3% a year, taken as 0.25% a month (3%/12). Sharpe ratios are annualized by
# √12; returns and vols in the frontier plot are annualized by 12 and √12.

include(joinpath(@__DIR__, "library", "RiskLib.jl"))
using Plots

const RF_A = 0.03
const RF = RF_A / 12
const NBOOT = 1000

df = CSV.read(joinpath(@__DIR__, "data", "problem4.csv"), DataFrame)
assets = ["EQ", "BOND", "STK", "PUTW"]
R = Matrix{Float64}(df[:, assets])
n, k = size(R)

# --- (a) moments and Sharpe ratios -----------------------------------------------
mom = DataFrame(Asset=String[], Mean=Float64[], Std=Float64[], Skew=Float64[], XKurt=Float64[],
                Ann_mean=Float64[], Ann_vol=Float64[], Sharpe=Float64[])
for (j, a) in enumerate(assets)
    m = sample_moments(R[:, j])
    push!(mom, (a, m.mean, m.std, m.skew, m.xkurt, 12m.mean, sqrt(12) * m.std,
                sqrt(12) * (m.mean - RF) / m.std))
end
write_table(mom, "p4_moments";
            fmts=Dict("Mean" => "%.4f", "Std" => "%.4f", "Skew" => "%.2f", "XKurt" => "%.2f",
                      "Ann_mean" => "%.2f", "Ann_vol" => "%.2f", "Sharpe" => "%.2f"),
            header=["Asset", "Mean", "Std. dev.", "Skew", "Excess kurt.", "Ann. mean", "Ann. vol", "Sharpe (ann.)"])

# How much of each asset's moments comes from its single most extreme month:
# its share of the sum of squared deviations, and the moments without it.
single = DataFrame(Asset=String[], Worst_month=String[], Worst=Float64[], Share_var=Float64[],
                   Share_4th=Float64[], Skew_wo=Float64[], XKurt_wo=Float64[], Sharpe_wo=Float64[])
for (j, a) in enumerate(assets)
    x = R[:, j]
    dev = x .- mean(x)
    i = argmax(abs.(dev))
    m = sample_moments(x[setdiff(1:n, i)])
    push!(single, (a, string(df.Month[i]), x[i], dev[i]^2 / sum(dev .^ 2), dev[i]^4 / sum(dev .^ 4),
                   m.skew, m.xkurt, sqrt(12) * (m.mean - RF) / m.std))
end
write_table(single, "p4_single_month";
            fmts=Dict("Worst" => "%.4f", "Share_var" => "%.2f", "Share_4th" => "%.2f", "Skew_wo" => "%.2f",
                      "XKurt_wo" => "%.2f", "Sharpe_wo" => "%.2f"),
            header=["Asset", "Month", "Return", "Share of \$\\sum d^2\$", "Share of \$\\sum d^4\$",
                    "Skew w/o", "Ex. kurt. w/o", "Sharpe w/o"])

# When EQ falls less than 5% the put expires worthless and PUTW earns its
# premium plus interest on the cash. In this file that is one of two fixed
# numbers (a low and a high volatility premium).
putw = R[:, 4]
cm = sort(collect(countmap(putw)), by=last, rev=true)
mode_ret, n_mode = cm[1]
mode2_ret, n_mode2 = cm[2]
n_worthless = count(R[:, 1] .>= -0.05)

# Pre-fit guide to the weights: with zero correlation the tangency weight of
# each asset is proportional to (mean - rf)/variance.
mv_ratio = [(m.Mean - RF) / m.Std^2 for m in eachrow(mom)]

# --- (c) three maximum Sharpe ratio portfolios ------------------------------------
μ = vec(mean(R, dims=1))
Σ = cov(R)
w_unc = maxSR_unconstrained(Σ, μ, RF)
w_long = maxSR(Σ, μ, RF)                                    # 0 ≤ w ≤ 1
cap = hcat(zeros(k), ones(k)); cap[4, 2] = 0.20
w_cap = maxSR(Σ, μ, RF, cap)
# cross-check the restarted Ipopt solutions against the convex QP form
@assert maximum(abs.(w_long .- maxSR_qp(Σ, μ, RF, hcat(zeros(k), ones(k))))) < 1e-5
@assert maximum(abs.(w_cap .- maxSR_qp(Σ, μ, RF, cap))) < 1e-5
@assert maximum(abs.(w_unc .- maxSR_qp(Σ, μ, RF, hcat(fill(-Inf, k), fill(Inf, k))))) < 1e-5

W = DataFrame(Portfolio=["Unconstrained", "Long only", "Long only, PUTW at most 20%"])
for (j, a) in enumerate(assets)
    W[!, a] = [w_unc[j], w_long[j], w_cap[j]]
end
write_table(W, "p4_weights"; fmts=Dict(a => "%.3f" for a in assets))

# --- (d) Sharpe, historical and normal VaR and ES of each portfolio -----------------
risk = DataFrame(Portfolio=String[], Sharpe=Float64[], Method=String[],
                 VaR5=Float64[], ES5=Float64[], VaR1=Float64[], ES1=Float64[])
for (nm, w) in zip(W.Portfolio, (w_unc, w_long, w_cap))
    rp = R * w
    m, s = mean(rp), std(rp)
    sr = sqrt(12) * (m - RF) / s
    push!(risk, (nm, sr, "Historical", VaR(rp; alpha=0.05), ES(rp; alpha=0.05), VaR(rp; alpha=0.01), ES(rp; alpha=0.01)))
    push!(risk, (nm, sr, "Normal", VaR_normal(m, s; alpha=0.05), ES_normal(m, s; alpha=0.05),
                 VaR_normal(m, s; alpha=0.01), ES_normal(m, s; alpha=0.01)))
end
rtab = copy(risk)
for c in (:VaR5, :ES5, :VaR1, :ES1)
    rtab[!, c] .*= 100                                       # percent of value, monthly
end
write_table(rtab, "p4_risk"; fmts=Dict("Sharpe" => "%.2f", "VaR5" => "%.2f", "ES5" => "%.2f",
                                       "VaR1" => "%.2f", "ES1" => "%.2f"),
            header=["Portfolio", "Sharpe (ann.)", "Method", "VaR 5\\%", "ES 5\\%", "VaR 1\\%", "ES 1\\%"])

# --- frontier and capital market line (annualized axes) ------------------------------
front = efficient_frontier(Σ, μ, hcat(zeros(k), ones(k)); npoints=120)
fσ = [sqrt(12) * p[1] for p in front]
fμ = [12 * p[2] for p in front]
σt = sqrt(12) * sqrt(w_long' * Σ * w_long)
μt = 12 * (w_long' * μ)
slope = (μt - RF_A) / σt
xs = range(0, maximum(sqrt(12) .* sqrt.(diag(Σ))) * 1.05, length=50)
plt = plot(fσ, fμ, lw=2, label="Long-only efficient frontier", xlabel="Annualized volatility",
           ylabel="Annualized mean return", legend=:bottomright, legendfontsize=7, size=(760, 480),
           title="Long-only frontier and capital market line (rf = 3%)", titlefontsize=10)
plot!(plt, xs, RF_A .+ slope .* xs, lw=1.2, ls=:dash, label=@sprintf("CML, slope (Sharpe) = %.2f", slope))
scatter!(plt, sqrt(12) .* sqrt.(diag(Σ)), 12 .* μ, ms=5, label="Assets",
         series_annotations=text.(assets, 8, :left, :bottom))
scatter!(plt, [σt], [μt], ms=6, marker=:star5, label="Long-only tangency")
wc_σ, wc_μ = sqrt(12) * sqrt(w_cap' * Σ * w_cap), 12 * (w_cap' * μ)
scatter!(plt, [wc_σ], [wc_μ], ms=5, marker=:diamond, label="PUTW capped at 20%")
savefig(plt, joinpath(OUT_FIGURES, "p4_frontier.pdf"))
savefig(plt, joinpath(OUT_FIGURES, "p4_frontier.png"))

# --- (e) CAPM regression of PUTW on EQ, all months and the down months ---------------
y = putw .- RF
x = R[:, 1] .- RF
capm = ols(y, x, "EQ")
down = x .< -0.05
capm_dn = ols(y[down], x[down], "EQ")
capm_up = ols(y[.!down], x[.!down], "EQ")
reg = DataFrame(Sample=["All months", "EQ excess < -5%", "Other months"],
                N=[capm.n, capm_dn.n, capm_up.n],
                Alpha=[capm.beta[1], capm_dn.beta[1], capm_up.beta[1]],
                t_alpha=[capm.t[1], capm_dn.t[1], capm_up.t[1]],
                Beta=[capm.beta[2], capm_dn.beta[2], capm_up.beta[2]],
                se_beta=[capm.se[2], capm_dn.se[2], capm_up.se[2]],
                R2=[capm.r2, capm_dn.r2, capm_up.r2])
write_table(reg, "p4_capm"; fmts=Dict("N" => "%d", "Alpha" => "%.4f", "t_alpha" => "%.2f", "Beta" => "%.3f",
                                      "se_beta" => "%.3f", "R2" => "%.3f"),
            header=["Sample", "\$n\$", "\$\\alpha\$ (monthly)", "\$t(\\alpha)\$", "\$\\beta\$", "s.e.(\$\\beta\$)", "\$R^2\$"])

# --- (h) how precisely 240 months pin down the inputs ---------------------------------
prec = DataFrame(Asset=assets, Mean=μ, SE_mean=vec(std(R, dims=1)) ./ sqrt(n))
prec.t_excess = (prec.Mean .- RF) ./ prec.SE_mean
prec.CI_low = 12 .* (prec.Mean .- 1.96 .* prec.SE_mean)
prec.CI_high = 12 .* (prec.Mean .+ 1.96 .* prec.SE_mean)
write_table(prec, "p4_precision"; fmts=Dict("Mean" => "%.4f", "SE_mean" => "%.4f", "t_excess" => "%.2f",
                                            "CI_low" => "%.3f", "CI_high" => "%.3f"),
            header=["Asset", "Mean (monthly)", "s.e. of mean", "\$t\$, mean \$-\$ rf", "95\\% CI low (ann.)", "95\\% CI high (ann.)"])

# Bootstrap: resample the 240 months with replacement (whole rows, so the
# cross-asset dependence comes along), refit the tangency weights each time.
rng = MersenneTwister(545)
boot = Dict(nm => fill(NaN, NBOOT, k) for nm in ("Unconstrained", "Long only", "Capped"))
crash = argmin(putw)                         # the month that dominates PUTW's tail
has_crash = falses(NBOOT)
n_fail = 0
for bi in 1:NBOOT
    idx = rand(rng, 1:n, n)
    has_crash[bi] = crash in idx
    Rb = R[idx, :]
    μb, Σb = vec(mean(Rb, dims=1)), cov(Rb)
    wl = maxSR_qp(Σb, μb, RF, hcat(zeros(k), ones(k)))
    wc = maxSR_qp(Σb, μb, RF, cap)
    if wl === nothing || wc === nothing      # keep the draw only if both solves succeed
        global n_fail += 1
        continue
    end
    boot["Unconstrained"][bi, :] = maxSR_unconstrained(Σb, μb, RF)
    boot["Long only"][bi, :] = wl
    boot["Capped"][bi, :] = wc
end
ok = .!isnan.(boot["Long only"][:, 1])
for nm in keys(boot)
    boot[nm] = boot[nm][ok, :]
end
has_crash = has_crash[ok]
println("bootstrap draws kept: ", count(ok), " of ", NBOOT, " (failed solves: ", n_fail, ")")
bt = DataFrame(Portfolio=String[], Asset=String[], Full_sample=Float64[], P5=Float64[], Median=Float64[], P95=Float64[])
for (nm, wfull) in (("Unconstrained", w_unc), ("Long only", w_long), ("Capped", w_cap)), (j, a) in enumerate(assets)
    v = boot[nm][:, j]
    push!(bt, (nm, a, wfull[j], quantile(v, 0.05), median(v), quantile(v, 0.95)))
end
write_table(bt, "p4_bootstrap"; fmts=Dict("Full_sample" => "%.3f", "P5" => "%.3f", "Median" => "%.3f", "P95" => "%.3f"),
            header=["Portfolio", "Asset", "Full sample", "5th pct", "Median", "95th pct"])
wl_putw = boot["Long only"][:, 4]
crash_split = (share_without=mean(.!has_crash), putw_with=median(wl_putw[has_crash]),
               putw_without=median(wl_putw[.!has_crash]), share_all_putw=mean(wl_putw .> 0.99))

if abspath(PROGRAM_FILE) == @__FILE__
    show(mom, allcols=true); println()
    show(single, allcols=true); println()
    println("PUTW months at the top value $(mode_ret): $n_mode of $n")
    show(W, allcols=true); println()
    show(risk, allcols=true, allrows=true); println()
    show(reg, allcols=true); println()
    show(prec, allcols=true); println()
    show(bt, allrows=true, allcols=true); println()
    println(crash_split)
end
