# Checks the library against the class test suite before any of it is used.
#
#     julia --project=. tests/run_course_tests.jl
#
# The CSVs in tests/course_data are copied unchanged from testfiles/data in the
# class repository. Prints one line per case; exits non-zero if any case fails.

include(joinpath(@__DIR__, "..", "library", "RiskLib.jl"))

const D = joinpath(@__DIR__, "course_data")
rd(f) = CSV.read(joinpath(D, f), DataFrame)
results = Tuple{String,Float64,Bool}[]
check(name, got, expected, tol) = (d = maximum(abs.(collect(got) .- collect(expected)));
                                   push!(results, (name, d, d <= tol)))

x1 = rd("test7_1.csv")[:, 1]
x2 = rd("test7_2.csv")[:, 1]

# 7.1 normal, 7.2 generalized t, 7.4 AICc of the t
fn = fit_normal(x1)
check("7.1 fit normal", [fn.errorModel.μ, fn.errorModel.σ], Vector(rd("testout7_1.csv")[1, :]), 1e-9)
ft = fit_general_t(x2)
tm = ft.errorModel
check("7.2 fit t", [tm.μ, tm.σ, tm.ρ.ν], Vector(rd("testout7_2.csv")[1, :]), 1e-4)
check("7.4 AICc", [aicc(ft, x2)], [rd("testout7_4.csv")[1, 1]], 1e-4)

# 8.1, 8.2 VaR and 8.4, 8.5 ES: absolute and measured from the mean
check("8.1 VaR normal", [VaR(fn.errorModel), VaR(Normal(0, fn.errorModel.σ))], Vector(rd("testout8_1.csv")[1, :]), 1e-9)
check("8.2 VaR t", [VaR(tm), VaR(TDist(tm.ρ.ν) * tm.σ)], Vector(rd("testout8_2.csv")[1, :]), 1e-4)
check("8.4 ES normal", [ES(fn.errorModel), ES(Normal(0, fn.errorModel.σ))], Vector(rd("testout8_4.csv")[1, :]), 1e-6)
check("8.5 ES t", [ES(tm), ES(TDist(tm.ρ.ν) * tm.σ)], Vector(rd("testout8_5.csv")[1, :]), 1e-4)
check("8.4 closed form ES normal", [ES_normal(fn.errorModel.μ, fn.errorModel.σ)], [rd("testout8_4.csv")[1, 1]], 1e-9)

# 10.3, 10.4 maximum Sharpe ratio
Σ = Matrix(rd("test5_3.csv"))
μ = rd("test10_3_means.csv").Mean
check("10.3 max SR, w >= 0", maxSR(Σ, μ, 0.04), rd("testout10_3.csv").W, 1e-4)
check("10.4 max SR, 0.1 <= w <= 0.5", maxSR(Σ, μ, 0.04, hcat(fill(0.1, 5), fill(0.5, 5))), rd("testout10_4.csv").W, 1e-4)

# 12.1 GBSM and Greeks; 12.2 American with continuous dividends
opts = dropmissing(rd("test12_1.csv"), :ID)
e121 = rd("testout12_1.csv")
e122 = rd("testout12_2.csv")
g121 = Matrix{Float64}(undef, nrow(opts), 6)
v122 = Float64[]
d122 = Float64[]
for (k, o) in enumerate(eachrow(opts))
    call = o["Option Type"] == "Call"
    T = o.DaysToMaturity / o.DayPerYear
    b = o.RiskFreeRate - o.DividendRate
    g = gbsm(call, o.Underlying, o.Strike, T, o.RiskFreeRate, b, o.ImpliedVol; includeGreeks=true)
    g121[k, :] = [g.value, g.delta, g.gamma, g.vega, g.rho, g.theta]
    f(s) = bt_american(call, s, o.Strike, T, o.RiskFreeRate, b, o.ImpliedVol, 500)
    push!(v122, f(o.Underlying))
    # FiniteDiff.jl's default central step, cbrt(eps)|x|. A tree is piecewise
    # linear in S, so its delta depends on the bump (Problem 2h).
    push!(d122, central_diff(f, o.Underlying, cbrt(eps()) * max(abs(o.Underlying), 1.0))[1])
end
check("12.1 GBSM value and Greeks", vec(g121), vec(Matrix(e121[:, [:Value, :Delta, :Gamma, :Vega, :Rho, :Theta]])), 1e-8)
check("12.2 American tree value", v122, e122.Value, 1e-8)
check("12.2 American tree delta", d122, e122.Delta, 1e-6)

# 12.3 American with discrete dividends, N = 2 x days and dividend steps doubled
opts = dropmissing(rd("test12_3.csv"), :ID)
got = Float64[]
for o in eachrow(opts)
    steps = 2 .* parse.(Int, split(string(o.DividendDates), ","))
    amts = parse.(Float64, split(string(o.DividendAmts), ","))
    push!(got, bt_american(o["Option Type"] == "Call", o.Underlying, o.Strike,
                           o.DaysToMaturity / o.DayPerYear, o.RiskFreeRate, amts, steps,
                           o.ImpliedVol, 2 * o.DaysToMaturity))
end
check("12.3 American, discrete dividends", got, rd("testout12_3.csv").Value, 1e-8)

w = maximum(length(r[1]) for r in results)
for (name, d, ok) in results
    @printf("%-*s  max abs diff %.2e  %s\n", w, name, d, ok ? "PASS" : "FAIL")
end
write_table(DataFrame(Case=first.(results), Max_abs_diff=[@sprintf("%.1e", r[2]) for r in results],
                      Result=[r[3] ? "pass" : "FAIL" for r in results]), "tests";
            header=["Class test case", "Max. abs. difference", "Result"])
exit(all(r[3] for r in results) ? 0 : 1)
