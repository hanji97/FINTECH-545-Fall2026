# Class test cases 1.1 - 13.9.
#
#     cd testfiles
#     julia --project=. run_tests.jl
#
# Each case reads its inputs from data/, writes what the library computes to
# output/ under the same file name and columns as the class expected file, then
# reads that file back and compares it with the expected one in data/. The
# summary is printed and saved to output/test_results.csv. Exits non-zero if
# any case fails.

include("../library/RiskLib.jl")

const DATA = joinpath(@__DIR__, "data")
const OUT = joinpath(@__DIR__, "output")
mkpath(OUT)

readdf(f) = CSV.read(joinpath(DATA, f), DataFrame)
readm(f) = Matrix(readdf(f))

# --- comparison -------------------------------------------------------------
struct Case
    id::String
    desc::String
    file::String
    absdiff::Float64
    reldiff::Float64
    tol::Float64
    kind::Symbol
    note::String
    ok::Bool
end

const RESULTS = Case[]

isnum(v) = nonmissingtype(eltype(v)) <: Number

"Largest absolute and relative gap over the numeric columns; labels and missing cells must match."
function gaps(got::DataFrame, want::DataFrame)
    names(got) == names(want) || return (Inf, Inf, "columns differ: $(names(got))")
    nrow(got) == nrow(want) || return (Inf, Inf, "row count differs")
    ad, rd = 0.0, 0.0
    for c in names(want)
        g, e = got[!, c], want[!, c]
        ismissing.(g) == ismissing.(e) || return (Inf, Inf, "missing cells differ in $c")
        if isnum(g) && isnum(e)
            keep = .!ismissing.(e)
            gv, ev = Float64.(g[keep]), Float64.(e[keep])
            d = abs.(gv .- ev)
            ad = max(ad, maximum(d; init=0.0))
            rd = max(rd, maximum(d ./ max.(abs.(ev), eps()); init=0.0))
        else
            string.(g) == string.(e) || return (Inf, Inf, "column $c differs")
        end
    end
    return (ad, rd, "")
end

"""
Write the computed table to output/<file>, read it back, compare it with
data/<file>. `kind` is :abs or :rel, the gap that `tol` applies to.
"""
function case(id, desc, file, got::DataFrame; tol, kind=:abs, note="")
    CSV.write(joinpath(OUT, file), got)
    saved = CSV.read(joinpath(OUT, file), DataFrame)
    ad, rd, msg = gaps(saved, readdf(file))
    ok = isempty(msg) && (kind === :abs ? ad : rd) <= tol
    push!(RESULTS, Case(id, desc, file, ad, rd, tol, kind, isempty(msg) ? note : msg, ok))
    return got
end

# A group that throws is recorded as FAIL with the error for every case in it
# that had not been recorded yet, and the remaining groups still run.
function guarded(f, ids...)
    try
        f()
    catch e
        msg = first(replace(sprint(showerror, e), '\n' => ' '), 100)
        done = Set(c.id for c in RESULTS)
        for id in ids
            id in done || push!(RESULTS, Case(id, "error", "", Inf, Inf, 0.0, :abs, msg, false))
        end
    end
end

matdf(M) = DataFrame(M, :auto)

# ============================================================================
# Test 1 - covariance and correlation with missing data
# ============================================================================
guarded("1.1", "1.2", "1.3", "1.4") do
    x = readm("test1.csv")
    case("1.1", "Covariance, missing data, skip missing rows", "testout_1.1.csv",
         matdf(missing_cov(x, skipMiss=true)), tol=1e-12)
    case("1.2", "Correlation, missing data, skip missing rows", "testout_1.2.csv",
         matdf(missing_cov(x, skipMiss=true, fun=cor)), tol=1e-12)
    case("1.3", "Covariance, missing data, pairwise", "testout_1.3.csv",
         matdf(missing_cov(x, skipMiss=false)), tol=1e-12)
    case("1.4", "Correlation, missing data, pairwise", "testout_1.4.csv",
         matdf(missing_cov(x, skipMiss=false, fun=cor)), tol=1e-12)
end

# ============================================================================
# Test 2 - exponentially weighted covariance
# ============================================================================
guarded("2.1", "2.2", "2.3") do
    x = readm("test2.csv")
    case("2.1", "EW covariance, lambda=0.97", "testout_2.1.csv", matdf(ewCovar(x, 0.97)), tol=1e-12)
    case("2.2", "EW correlation, lambda=0.94", "testout_2.2.csv", matdf(ewCorr(x, 0.94)), tol=1e-12)
    case("2.3", "EW cov: EW var lambda=0.97, EW corr lambda=0.94", "testout_2.3.csv",
         matdf(ewCovar(x, 0.97, 0.94)), tol=1e-12)
end

# ============================================================================
# Test 3 - repairing a non-PSD matrix; Test 4 - Cholesky of a PSD matrix
# ============================================================================
guarded("3.1", "3.2", "3.3", "3.4", "4.1") do
    case("3.1", "near_psd on a covariance", "testout_3.1.csv",
         matdf(near_psd(readm("testout_1.3.csv"))), tol=1e-10)
    case("3.2", "near_psd on a correlation", "testout_3.2.csv",
         matdf(near_psd(readm("testout_1.4.csv"))), tol=1e-10)
    case("3.3", "Higham on a covariance", "testout_3.3.csv",
         matdf(higham_nearestPSD(readm("testout_1.3.csv"))), tol=1e-8)
    case("3.4", "Higham on a correlation", "testout_3.4.csv",
         matdf(higham_nearestPSD(readm("testout_1.4.csv"))), tol=1e-8)
    case("4.1", "chol_psd", "testout_4.1.csv", matdf(chol_psd(readm("testout_3.1.csv"))), tol=1e-10)
end

# ============================================================================
# Test 5 - simulation, 100,000 draws
# ============================================================================
# The draws use the class setup's seed, so the simulated covariance reproduces
# the expected file. The gap to the input covariance is sampling error, plus the
# dropped 1% of variance for 5.5.
function sim_case(id, desc, input, file, draws)
    covout = cov(draws)
    mc = maximum(abs.(covout .- readm(input)))
    case(id, desc, file, matdf(covout); tol=1e-8,
         note=@sprintf("vs input cov %.2e (sampling error)", mc))
end

guarded("5.1", "5.2", "5.3", "5.4", "5.5") do
    sim_case("5.1", "Normal simulation, PD input", "test5_1.csv", "testout_5.1.csv",
             simulateNormal(100_000, readm("test5_1.csv")))
    sim_case("5.2", "Normal simulation, PSD input", "test5_2.csv", "testout_5.2.csv",
             simulateNormal(100_000, readm("test5_2.csv")))
    sim_case("5.3", "Normal simulation, non-PSD input, near_psd fix", "test5_3.csv", "testout_5.3.csv",
             simulateNormal(100_000, readm("test5_3.csv"), fixMethod=near_psd))
    sim_case("5.4", "Normal simulation, non-PSD input, Higham fix", "test5_3.csv", "testout_5.4.csv",
             simulateNormal(100_000, readm("test5_3.csv"), fixMethod=higham_nearestPSD))
    sim_case("5.5", "PCA simulation, 99% of variance explained", "test5_2.csv", "testout_5.5.csv",
             simulate_pca(readm("test5_2.csv"), 100_000, pctExp=0.99))
end

# ============================================================================
# Test 6 - returns
# ============================================================================
guarded("6.1", "6.2") do
    prices = readdf("test6.csv")
    case("6.1", "Arithmetic returns", "testout6_1.csv",
         return_calculate(prices, method="DISCRETE", dateColumn="Date"), tol=1e-12)
    case("6.2", "Log returns", "testout6_2.csv",
         return_calculate(prices, method="LOG", dateColumn="Date"), tol=1e-12)
end

# ============================================================================
# Tests 7 and 8 - distribution fitting; VaR and ES (Week 04, sec. 2; Week 05, sec. 1)
# ============================================================================
guarded("7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "8.1", "8.2", "8.3", "8.4", "8.5", "8.6") do
    x71 = readm("test7_1.csv")[:, 1]
    x72 = readm("test7_2.csv")[:, 1]
    fn = fit_normal(x71)
    ft = fit_general_t(x72)
    μn, σn = Distributions.params(fn.errorModel)
    μt, σt, tdist = Distributions.params(ft.errorModel)
    νt = dof(tdist)

    case("7.1", "Fit a normal distribution", "testout7_1.csv",
         DataFrame(mu=[μn], sigma=[σn]), tol=1e-12)
    case("7.2", "Fit a generalized t", "testout7_2.csv",
         DataFrame(mu=[μt], sigma=[σt], nu=[νt]), tol=1e-6, kind=:rel)
    df = readdf("test7_3.csv")
    fr = fit_regression_t(df.y, Matrix(select(df, Not(:y))))
    m, s, ρ = Distributions.params(fr.errorModel)
    case("7.3", "t regression", "testout7_3.csv",
         DataFrame(mu=[m], sigma=[s], nu=[dof(ρ)], Alpha=[fr.beta[1]],
                   B1=[fr.beta[2]], B2=[fr.beta[3]], B3=[fr.beta[4]]), tol=1e-6, kind=:rel)
    case("7.4", "AICc of the fitted t", "testout7_4.csv", DataFrame(AICC=[aicc(ft, x72)]),
         tol=1e-6, kind=:rel)
    x75 = readm("test7_5.csv")[:, 1]
    d = fit_nig_moments(x75).errorModel
    case("7.5", "NIG by the method of moments", "testout7_5.csv",
         DataFrame(mu=[d.μ], alpha=[d.α], beta=[d.β], delta=[d.δ]), tol=1e-10)
    d = fit_nig_mle(x75).errorModel
    case("7.6", "NIG by maximum likelihood", "testout7_6.csv",
         DataFrame(mu=[d.μ], alpha=[d.α], beta=[d.β], delta=[d.δ]), tol=1e-3, kind=:rel,
         note="expected file is a scipy fit")

    # "Absolute" is measured from zero; "diff from mean" is the same measure on
    # the distribution shifted to mean zero (Week 04, 2.1).
    case("8.1", "VaR, normal", "testout8_1.csv",
         DataFrame("VaR Absolute" => [VaR(fn.errorModel)],
                   "VaR Diff from Mean" => [VaR(Normal(0, σn))]), tol=1e-12)
    case("8.2", "VaR, generalized t", "testout8_2.csv",
         DataFrame("VaR Absolute" => [VaR(ft.errorModel)],
                   "VaR Diff from Mean" => [VaR(TDist(νt) * σt)]), tol=1e-6, kind=:rel)
    # ES of the normal in closed form (Week 05, 1.1); of the t by integrating
    # x f(x) over the tail (Week 05, 1).
    case("8.4", "ES, normal", "testout8_4.csv",
         DataFrame("ES Absolute" => [ES_normal(μn, σn)],
                   "ES Diff from Mean" => [ES_normal(0.0, σn)]), tol=1e-9)
    case("8.5", "ES, generalized t", "testout8_5.csv",
         DataFrame("ES Absolute" => [ES(ft.errorModel)],
                   "ES Diff from Mean" => [ES(TDist(νt) * σt)]), tol=1e-6, kind=:rel)

    # 8.3 and 8.6: 10,000 draws from the fitted t by its quantile function
    # (Week 05, 2.3), with the class setup's seed and order of draws. The gap
    # to the values from the fitted distribution in 8.2 and 8.5 is the
    # sampling error.
    Random.seed!(8)
    sim83 = ft.eval(rand(10_000))
    sim86 = ft.eval(rand(10_000))
    v, vm = VaR(sim83), VaR(sim83 .- mean(sim83))
    case("8.3", "VaR, simulation from the fitted t", "testout8_3.csv",
         DataFrame("VaR Absolute" => [v], "VaR Diff from Mean" => [vm]); tol=1e-6, kind=:rel,
         note=@sprintf("vs 8.2: %+.2e, %+.2e", v - VaR(ft.errorModel), vm - VaR(TDist(νt) * σt)))
    e, em = ES(sim86), ES(sim86 .- mean(sim86))
    case("8.6", "ES, simulation from the fitted t", "testout8_6.csv",
         DataFrame("ES Absolute" => [e], "ES Diff from Mean" => [em]); tol=1e-6, kind=:rel,
         note=@sprintf("vs 8.5: %+.2e, %+.2e", e - ES(ft.errorModel), em - ES(TDist(νt) * σt)))
end

# ============================================================================
# Test 9 - VaR and ES of a two stock portfolio, Gaussian copula (Week 05, sec. 4)
# ============================================================================
const riskCols = [:Stock, :VaR95, :ES95, :VaR95_Pct, :ES95_Pct]

# portfolio VaR95 against the sum of the stock VaR95s
function diversification(risk)
    isTotal = risk.Stock .== "Total"
    return @sprintf("total VaR95 %.2f vs sum of stocks %.2f", only(risk.VaR95[isTotal]), sum(risk.VaR95[.!isTotal]))
end

guarded("9.1") do
    returns = dropmissing(readdf("test9_1_returns.csv"))
    port = readdf("test9_1_portfolio.csv")
    port.currentValue = port.Holding .* port[!, "Starting Price"]
    # one model per stock, by the distribution named in the portfolio file
    stocks = unique(port.Stock)
    dist = Dict(zip(port.Stock, port.Distribution))
    models = [dist[s] == "Normal" ? fit_normal(returns[!, s]) : fit_general_t(returns[!, s])
              for s in stocks]
    U = hcat([m.u for m in models]...)
    R = corspearman(U)                                   # Spearman on U, 4.1
    simU = simulate_gaussian_copula(R, 100_000)          # 4.2 step 1 and 2a
    simRet = DataFrame([models[j].eval(simU[:, j]) for j in eachindex(stocks)], stocks)
    risk = aggRisk(simulated_values(port, simRet), [:Stock])
    case("9.1", "VaR/ES by stock and total, Gaussian copula", "testout9_1.csv",
         risk[:, riskCols], tol=1e-6, kind=:rel, note=diversification(risk))
end

# ============================================================================
# Test 10 - risk parity and maximum Sharpe ratio
# ============================================================================
guarded("10.1", "10.2") do
    cin = readdf("test5_2.csv")
    Σ = Matrix(cin)
    halfX5 = [c == "x5" ? 0.5 : 1.0 for c in names(cin)]
    for (id, desc, file, b) in (("10.1", "Risk parity, normal", "testout10_1.csv", ones(size(Σ, 1))),
                                ("10.2", "Risk parity, half risk budget on X5", "testout10_2.csv", halfX5))
        w = riskParity(Σ; riskBudget=b)
        share = component_sd(w, Σ) ./ b
        case(id, desc, file, DataFrame(W=w); tol=1e-6,
             note=@sprintf("CSD/budget spread %.1e", (maximum(share) - minimum(share)) / mean(share)))
    end
end

# The class setup runs 10.3 and 10.4 on test5_3.csv (Tests.xlsx lists test5_2.csv).
guarded("10.3", "10.4") do
    Σ = readm("test5_3.csv")
    μ = readdf("test10_3_means.csv").Mean
    rf = 0.04
    n = length(μ)
    w = maxSR(Σ, μ, rf)
    case("10.3", "Max Sharpe ratio, w >= 0", "testout10_3.csv", DataFrame(W=w); tol=1e-6,
         note=@sprintf("SR %.6f vs expected %.6f", sharpe(w, μ, Σ, rf), sharpe(readdf("testout10_3.csv").W, μ, Σ, rf)))
    w = maxSR(Σ, μ, rf, hcat(fill(0.1, n), fill(0.5, n)))
    case("10.4", "Max Sharpe ratio, 0.1 <= w <= 0.5", "testout10_4.csv", DataFrame(W=w); tol=1e-6,
         note=@sprintf("SR %.6f vs expected %.6f", sharpe(w, μ, Σ, rf), sharpe(readdf("testout10_4.csv").W, μ, Σ, rf)))
end

# ============================================================================
# Test 11 - ex-post attribution
# ============================================================================
guarded("11.1") do
    returns = readdf("test11_1_returns.csv")
    w = readdf("test11_1_weights.csv").W
    # attribution to the stocks: the stocks are their own factors, betas = I
    att = expost_factor(w, returns, returns, Matrix(1.0I, ncol(returns), ncol(returns))).Attribution
    case("11.1", "Ex-post return and vol attribution, stocks", "testout11_1.csv",
         select(att, Not(:Alpha)), tol=1e-12)
end

guarded("11.2") do
    w = readdf("test11_2_weights.csv").W
    stocks = readdf("test11_2_stock_returns.csv")
    factors = readdf("test11_2_factor_returns.csv")
    betas = readdf("test11_2_beta.csv")
    # rows in the order of the stock columns, columns in the order of the factors
    B = Matrix(betas[indexin(names(stocks), string.(betas.Stock)), names(factors)])
    att = expost_factor(w, stocks, factors, B).Attribution
    case("11.2", "Ex-post return and vol attribution, factors", "testout11_2.csv", att, tol=1e-12)
end

# ============================================================================
# Test 12 - options (Week 06, sec. 4; Week 07, sec. 1-3)
# ============================================================================
guarded("12.1", "12.2") do
    opts = dropmissing(readdf("test12_1.csv"), :ID)
    greekCols = [:Value, :Delta, :Gamma, :Vega, :Rho, :Theta]
    bs = DataFrame(ID=opts.ID)
    am = DataFrame(ID=opts.ID)
    for c in greekCols
        bs[!, c] = zeros(nrow(opts)); am[!, c] = zeros(nrow(opts))
    end
    for (k, o) in enumerate(eachrow(opts))
        call = o["Option Type"] == "Call"
        T = o.DaysToMaturity / o.DayPerYear
        b = o.RiskFreeRate - o.DividendRate
        g = gbsm(call, o.Underlying, o.Strike, T, o.RiskFreeRate, b, o.ImpliedVol; includeGreeks=true)
        bs[k, greekCols] = (g.value, g.delta, g.gamma, g.vega, g.rho, g.theta)

        # American: CRR tree with N = 500, sensitivities by finite differences.
        # The class file reports the partials of the tree value in its inputs,
        # so Theta here is ∂V/∂T and Rho holds b fixed.
        v, grad, gamma = bt_american_greeks(call, o.Underlying, o.Strike, T, o.RiskFreeRate, b, o.ImpliedVol, 500)
        am[k, greekCols] = (v, grad[1], gamma, grad[6], grad[4], grad[3])
    end
    case("12.1", "European options, GBSM and Greeks", "testout12_1.csv", bs, tol=1e-9)
    case("12.2", "American options, continuous dividends, Greeks", "testout12_2.csv", am, tol=1e-6,
         note="Theta = dV/dT, Rho at fixed b, as in the class file")
end

# comma separated list in one CSV cell; a blank cell is an empty list
parselist(T, cell) = ismissing(cell) || isempty(strip(string(cell))) ? T[] :
                     parse.(T, strip.(split(string(cell), ",")))

guarded("12.3") do
    opts = dropmissing(readdf("test12_3.csv"), :ID)
    vals = Float64[]
    for o in eachrow(opts)
        # two steps per day, so the dividend days map to twice their step index
        N = 2 * o.DaysToMaturity
        steps = 2 .* parselist(Int, o.DividendDates)
        amts = parselist(Float64, o.DividendAmts)
        push!(vals, bt_american(o["Option Type"] == "Call", o.Underlying, o.Strike,
                                o.DaysToMaturity / o.DayPerYear, o.RiskFreeRate, amts, steps,
                                o.ImpliedVol, N))
    end
    case("12.3", "American options, discrete dividends", "testout12_3.csv",
         DataFrame(ID=opts.ID, Value=vals), tol=1e-9)
end

# ============================================================================
# Test 13 - multivariate t and the t copula (Week 05, sec. 5-6)
# ============================================================================
guarded("13.1", "13.2", "13.3", "13.4", "13.5", "13.6", "13.7", "13.8", "13.9") do
    returns = dropmissing(readdf("test13_returns.csv"))
    port = readdf("test13_portfolio.csv")
    X = Matrix{Float64}(returns)
    m = size(X, 1)
    stocks = names(returns)

    R = kendall_correlation(X)
    case("13.1", "Correlation from Kendall's tau", "testout13_1.csv", matdf(R), tol=1e-12,
         note=@sprintf("min eigenvalue %.3f", eigmin(Symmetric(R))))

    μ, S, ν, ll = fit_multivariate_t(X)
    case("13.2", "Multivariate t, mean vector", "testout13_2.csv", DataFrame(mu=μ), tol=1e-12)
    case("13.3", "Multivariate t, scale matrix", "testout13_3.csv", matdf(S), tol=1e-9, kind=:rel,
         note=@sprintf("max |diag(nu/(nu-2)S) - var| %.1e", maximum(abs.(diag(S) .* ν / (ν - 2) .- vec(var(X, dims=1))))))
    case("13.4", "Multivariate t, nu by profile and log likelihood", "testout13_4.csv",
         DataFrame(nu=[ν], ll=[ll]), tol=1e-9, kind=:rel)

    # margins fitted first and frozen; both copulas get the same R
    margins = [fit_general_t(X[:, j]) for j in 1:size(X, 2)]
    U = hcat([fm.u for fm in margins]...)

    _, llG = fit_gaussian_copula(U)
    case("13.5", "Gaussian copula log likelihood", "testout13_5.csv", DataFrame(ll=[llG]),
         tol=1e-6, kind=:rel)
    Rc, νc, llT = fit_t_copula(U)
    case("13.6", "t copula, nu by profile and log likelihood", "testout13_6.csv",
         DataFrame(nu=[νc], ll=[llT]), tol=1e-6, kind=:rel)

    kG, kT = 0, 1
    choice = DataFrame(Copula=["Gaussian", "T"], LL=[llG, llT], K=[kG, kT],
                       AICC=[copula_aicc(llG, kG, m), copula_aicc(llT, kT, m)],
                       BIC=[copula_bic(llG, kG, m), copula_bic(llT, kT, m)])
    case("13.7", "Gaussian vs t copula, AICc and BIC", "testout13_7.csv", choice, tol=1e-6, kind=:rel,
         note=@sprintf("dBIC = %.2f, %s copula preferred", choice.BIC[1] - choice.BIC[2],
                       choice.BIC[2] < choice.BIC[1] ? "t" : "Gaussian"))

    pairs = [(i, j) for i in 1:size(X, 2) for j in (i+1):size(X, 2)]
    tails = DataFrame(I=first.(pairs), J=last.(pairs),
                      Rho=[Rc[i, j] for (i, j) in pairs],
                      Lambda=[tail_dependence_t(Rc[i, j], νc) for (i, j) in pairs])
    case("13.8", "Lower tail dependence under the t copula", "testout13_8.csv", tails,
         tol=1e-6, kind=:rel)

    simU = simulate_t_copula(Rc, νc, 100_000; seed=13)
    simRet = DataFrame([margins[j].eval(simU[:, j]) for j in eachindex(stocks)], stocks)
    risk = aggRisk(simulated_values(port, simRet), [:Stock])

    # the same margins and R under the Gaussian copula, for comparison
    gU = simulate_gaussian_copula(Rc, 100_000; seed=13)
    gRet = DataFrame([margins[j].eval(gU[:, j]) for j in eachindex(stocks)], stocks)
    gTotal = aggRisk(simulated_values(port, gRet), Symbol[])
    case("13.9", "VaR/ES by stock and total, t copula", "testout13_9.csv",
         risk[:, riskCols], tol=1e-6, kind=:rel,
         note=diversification(risk) * @sprintf("; Gaussian copula total VaR95 %.2f ES95 %.2f",
                                               gTotal.VaR95[1], gTotal.ES95[1]))
end

# ============================================================================
# Report
# ============================================================================
sortkey(c) = Tuple(parse.(Int, split(c.id, ".")))
sorted = sort(RESULTS, by=sortkey)

println()
println("="^118)
@printf("%-5s %-50s %-6s %11s %11s %9s  %s\n", "Test", "Description", "Status", "max abs d", "max rel d", "tol", "Note")
println("="^118)
for c in sorted
    @printf("%-5s %-50s %-6s %11.2e %11.2e %4s %.0e  %s\n", c.id, c.desc, c.ok ? "PASS" : "FAIL",
            c.absdiff, c.reldiff, c.kind, c.tol, c.note)
end
println("="^118)

CSV.write(joinpath(OUT, "test_results.csv"),
          DataFrame(Test=[c.id for c in sorted], Description=[c.desc for c in sorted],
                    File=[c.file for c in sorted], Status=[c.ok ? "PASS" : "FAIL" for c in sorted],
                    MaxAbsDiff=[c.absdiff for c in sorted], MaxRelDiff=[c.reldiff for c in sorted],
                    Tolerance=[c.tol for c in sorted], ToleranceOn=[string(c.kind) for c in sorted],
                    Note=[c.note for c in sorted]))

npass = count(c -> c.ok, sorted)
@printf("%d of %d cases passed. Computed outputs are in %s\n", npass, length(sorted), OUT)
if npass != length(sorted)
    println("FAILED: ", join([c.id for c in sorted if !c.ok], ", "))
    exit(1)
end
