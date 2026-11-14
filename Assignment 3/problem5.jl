# Problem 5: alpha or factor exposure?
#
#     julia --project=. problem5.jl
#
# Daily fund returns (decimals) against the Fama French factors and momentum,
# which Ken French stores in percent and are divided by 100 here. Daily numbers
# are annualized with 252 trading days (the fund file has 1,257 days in five
# years, about 251 a year).

include(joinpath(@__DIR__, "library", "RiskLib.jl"))

const ANN = 252
const NOTIONAL = 10_000_000

funds = CSV.read(joinpath(@__DIR__, "data", "problem5.csv"), DataFrame; dateformat="yyyy-mm-dd")
ff = CSV.read(joinpath(@__DIR__, "data", "F-F_Research_Data_Factors_daily.CSV"), DataFrame)
mom = CSV.read(joinpath(@__DIR__, "data", "F-F_Momentum_Factor_daily.CSV"), DataFrame)
rename!(ff, strip.(names(ff)))
rename!(mom, strip.(names(mom)))                  # the file's header is "Mom   "
ff.Date = Date.(string.(ff.Date), dateformat"yyyymmdd")
mom.Date = Date.(string.(mom.Date), dateformat"yyyymmdd")
factors = innerjoin(ff, mom, on=:Date)
for c in ("Mkt-RF", "SMB", "HML", "RF", "Mom")
    factors[!, c] = factors[!, c] ./ 100
end
rename!(factors, "Mkt-RF" => "MKT")

data = innerjoin(funds, factors, on=:Date)
@assert nrow(data) == nrow(funds) "every fund day should have factor data"
fundnames = ["FUND1", "FUND2"]

specs = [("CAPM", ["MKT"]), ("FF3", ["MKT", "SMB", "HML"]), ("Carhart", ["MKT", "SMB", "HML", "Mom"])]
fits = Dict{Tuple{String,String},Any}()
for f in fundnames, (m, cols) in specs
    y = data[!, f] .- data.RF
    fits[(f, m)] = ols(y, Matrix{Float64}(data[:, cols]), cols)
end

# --- (a) excess return, vol, Sharpe and the CAPM --------------------------------------
summ = DataFrame(Fund=String[], Mean_excess=Float64[], Vol=Float64[], Sharpe=Float64[],
                 Alpha_ann=Float64[], t_alpha=Float64[], Beta=Float64[], R2=Float64[])
for f in fundnames
    ex = data[!, f] .- data.RF
    c = fits[(f, "CAPM")]
    push!(summ, (f, ANN * mean(ex), sqrt(ANN) * std(ex), sqrt(ANN) * mean(ex) / std(ex),
                 ANN * c.beta[1], c.t[1], c.beta[2], c.r2))
end
write_table(summ, "p5_summary";
            fmts=Dict("Mean_excess" => "%.4f", "Vol" => "%.4f", "Sharpe" => "%.2f", "Alpha_ann" => "%.4f",
                      "t_alpha" => "%.2f", "Beta" => "%.3f", "R2" => "%.3f"),
            header=["Fund", "Mean excess (ann.)", "Vol (ann.)", "Sharpe", "CAPM \$\\alpha\$ (ann.)",
                    "\$t(\\alpha)\$", "\$\\beta_{MKT}\$", "\$R^2\$"])

# --- (b) CAPM residuals against the omitted factors ------------------------------------
rc = DataFrame(Fund=String[], SMB=Float64[], HML=Float64[], Mom=Float64[])
for f in fundnames
    e = fits[(f, "CAPM")].resid
    push!(rc, (f, cor(e, data.SMB), cor(e, data.HML), cor(e, data.Mom)))
end
write_table(rc, "p5_resid_corr"; fmts=Dict("SMB" => "%.3f", "HML" => "%.3f", "Mom" => "%.3f"),
            header=["Fund", "SMB", "HML", "Mom"])

# A rough, pre-fit size for each omitted loading: regressing the CAPM residual
# on one factor at a time gives β ≈ ρ σ_ε / σ_f. Times the factor's average
# return in the sample, that is roughly how much alpha the factor should take.
guess = DataFrame(Fund=String[], Factor=String[], Beta_guess=Float64[], Alpha_change=Float64[])
for f in fundnames, fac in ("SMB", "HML", "Mom")
    e = fits[(f, "CAPM")].resid
    bg = cor(e, data[!, fac]) * std(e) / std(data[!, fac])
    push!(guess, (f, fac, bg, -100ANN * bg * mean(data[!, fac])))
end
write_table(guess, "p5_guess"; fmts=Dict("Beta_guess" => "%.3f", "Alpha_change" => "%.2f"),
            header=["Fund", "Factor", "Rough loading \$\\rho\\sigma_\\varepsilon/\\sigma_f\$",
                    "Implied change in \$\\alpha\$ (\\% a year)"], align="llrr")

# --- (c) the three models, coefficient (s.e.) -------------------------------------------
coef = DataFrame(Fund=String[], Model=String[], Alpha=String[], t_alpha=Float64[], MKT=String[],
                 SMB=String[], HML=String[], Mom=String[], R2=Float64[], AICc=Float64[])
cell(m, nm) = (i = findfirst(==(nm), m.names); i === nothing ? "" :
               @sprintf("%.3f (%.3f)", m.beta[i], m.se[i]))
cella(m) = @sprintf("%.2f (%.2f)", 1e4 * m.beta[1], 1e4 * m.se[1])   # basis points a day
for f in fundnames, (mname, _) in specs
    m = fits[(f, mname)]
    push!(coef, (f, mname, cella(m), m.t[1], cell(m, "MKT"), cell(m, "SMB"), cell(m, "HML"),
                 cell(m, "Mom"), m.r2, m.aicc))
end
write_table(coef, "p5_models"; fmts=Dict("t_alpha" => "%.2f", "R2" => "%.3f", "AICc" => "%.1f"),
            header=["Fund", "Model", "\$\\alpha\$, bp/day (s.e.)", "\$t(\\alpha)\$", "MKT", "SMB", "HML", "Mom",
                    "\$R^2\$", "AICc"],
            align="llrrrrrrrr")

# --- (d) expected annual return from the four factor betas ------------------------------
# Average daily factor returns over the last ten years in the factor files.
last_day = maximum(factors.Date)
recent = factors[factors.Date .> last_day - Year(10), :]
fbar = Dict(c => mean(recent[!, c]) for c in ("MKT", "SMB", "HML", "Mom", "RF"))
insample = Dict(c => mean(data[!, c]) for c in ("MKT", "SMB", "HML", "Mom", "RF"))
exp_tab = DataFrame(Fund=String[], Excess_no_alpha=Float64[], Total_no_alpha=Float64[],
                    Total_with_alpha=Float64[], Realized_total=Float64[], Realized_excess=Float64[])
for f in fundnames
    m = fits[(f, "Carhart")]
    β = Dict(zip(m.names, m.beta))
    ex = ANN * sum(β[c] * fbar[c] for c in ("MKT", "SMB", "HML", "Mom"))
    tot = ex + ANN * fbar["RF"]
    push!(exp_tab, (f, ex, tot, tot + ANN * β["alpha"], ANN * mean(data[!, f]),
                    ANN * mean(data[!, f] .- data.RF)))
end
write_table(exp_tab, "p5_expected"; fmts=Dict(c => "%.4f" for c in names(exp_tab)[2:end]),
            header=["Fund", "Excess, no \$\\alpha\$", "Total, no \$\\alpha\$", "Total, with \$\\alpha\$",
                    "Realized total", "Realized excess"])
prem = DataFrame(Series=["Mkt\$-\$RF (market excess return)", "SMB", "HML", "Mom", "RF (risk free rate)"],
                 Sample=[100ANN * insample[c] for c in ("MKT", "SMB", "HML", "Mom", "RF")],
                 Last10=[100ANN * fbar[c] for c in ("MKT", "SMB", "HML", "Mom", "RF")])
write_table(prem, "p5_premia"; fmts=Dict("Sample" => "%.2f", "Last10" => "%.2f"),
            header=["Series", "Fund sample, 1996--2001", "Last 10 years of the files"], align="lrr")

# (h) why the factor-model forecast differs from what the fund earned:
# realized excess = α + Σβ f̄_sample, forecast (no α) = Σβ f̄_recent, so
# forecast - realized = Σβ (f̄_recent - f̄_sample) - α, term by term.
hdec = DataFrame(Fund=String[], Realized=Float64[], MKT=Float64[], SMB=Float64[], HML=Float64[],
                 Mom=Float64[], Alpha=Float64[], Forecast=Float64[])
for f in fundnames
    m = fits[(f, "Carhart")]
    β = Dict(zip(m.names, m.beta))
    d(c) = 100ANN * β[c] * (fbar[c] - insample[c])
    push!(hdec, (f, 100ANN * mean(data[!, f] .- data.RF), d("MKT"), d("SMB"), d("HML"), d("Mom"),
                 -100ANN * β["alpha"], 100ANN * sum(β[c] * fbar[c] for c in ("MKT", "SMB", "HML", "Mom"))))
end
write_table(hdec, "p5_h_decomp"; fmts=Dict(c => "%.2f" for c in names(hdec)[2:end]),
            header=["Fund", "Realized excess", "\$+\\beta\\Delta\$MKT", "\$+\\beta\\Delta\$SMB",
                    "\$+\\beta\\Delta\$HML", "\$+\\beta\\Delta\$Mom", "\$-\\alpha\$", "\$=\$ Forecast, no \$\\alpha\$"])

# --- (e) the worst momentum day, replayed on each fund ------------------------------------
iw = argmin(factors.Mom)
worst_day, worst_mom = factors.Date[iw], factors.Mom[iw]
shock = DataFrame(Fund=String[], Beta_mom=Float64[], Loss=Float64[], Daily_sd=Float64[], Multiple=Float64[])
for f in fundnames
    m = fits[(f, "Carhart")]
    bm = m.beta[findfirst(==("Mom"), m.names)]
    loss = -bm * worst_mom * NOTIONAL                       # positive = a loss
    sd = std(data[!, f])
    push!(shock, (f, bm, loss, sd, loss / (sd * NOTIONAL)))
end
write_table(shock, "p5_mom_shock"; fmts=Dict("Beta_mom" => "%.3f", "Loss" => "%.0f", "Daily_sd" => "%.4f",
                                             "Multiple" => "%.2f"),
            header=["Fund", "\$\\beta_{Mom}\$", "Loss on \\\$10m", "Daily s.d.", "Multiple of s.d."])

# --- (f) FUND1's alpha change, term by term -----------------------------------------------
# With an intercept, α = ȳ - Σβ x̄ in each model, so
#   α_CAPM - α_4F = (β_MKT,4F - β_MKT,CAPM) MKT̄ + β_SMB SMB̄ + β_HML HML̄ + β_Mom Mom̄
decomp = DataFrame(Fund=String[], Term=String[], Contribution=Float64[])
for f in fundnames
    c = fits[(f, "CAPM")]
    m = fits[(f, "Carhart")]
    β = Dict(zip(m.names, m.beta))
    push!(decomp, (f, "MKT (change in \$\\beta_{MKT}\$)", 100ANN * (β["MKT"] - c.beta[2]) * insample["MKT"]))
    for t in ("SMB", "HML", "Mom")
        push!(decomp, (f, t, 100ANN * β[t] * insample[t]))
    end
    push!(decomp, (f, "Total, \$\\alpha_{CAPM}-\\alpha_{4F}\$", 100ANN * (c.beta[1] - m.beta[1])))
end
write_table(decomp, "p5_alpha_decomp"; fmts=Dict("Contribution" => "%.2f"),
            header=["Fund", "Term", "Contribution (\\% a year)"], align="llr")

if abspath(PROGRAM_FILE) == @__FILE__
    println("rows: ", nrow(data), "  ", minimum(data.Date), " to ", maximum(data.Date))
    show(summ, allcols=true); println()
    show(rc); println()
    show(coef, allcols=true, allrows=true); println()
    show(prem); println()
    println("last ten years: ", minimum(recent.Date), " to ", last_day)
    show(exp_tab, allcols=true); println()
    println("worst momentum day: ", worst_day, "  ", worst_mom)
    show(shock, allcols=true); println()
    show(decomp, allrows=true); println()
end
