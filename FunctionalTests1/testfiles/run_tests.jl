# Functional tests 1.1 - 7.6.
#
#     julia --project=. run_tests.jl
#
# Runs each case through the library and compares against the expected output
# in data/. Tolerances differ by case:
#
#   1.x, 2.x, 3.x, 4.1, 6.x, 7.1, 7.4, 7.5   closed form, floating point
#   7.2, 7.3, 7.6                            MLE, optimiser convergence
#   5.x                                      simulation, see sim_case

include("../library/RiskLib.jl")
using Printf

const DATA = joinpath(@__DIR__, "data")
readm(f) = Matrix(CSV.read(joinpath(DATA, f), DataFrame))
readdf(f) = CSV.read(joinpath(DATA, f), DataFrame)

# --- comparison -------------------------------------------------------------
"Largest absolute and relative gap between two numeric arrays."
function gaps(actual, expected)
    a = vec(collect(Float64.(actual)))
    e = vec(collect(Float64.(expected)))
    length(a) == length(e) || return (Inf, Inf)
    d = abs.(a .- e)
    scale = max.(abs.(e), eps())
    return (maximum(d), maximum(d ./ scale))
end

struct Case
    id::String
    desc::String
    absdiff::Float64
    reldiff::Float64
    note::String
    ok::Bool
end

const RESULTS = Case[]

function check(id, desc, actual, expected; tol, kind=:abs, note="")
    ad, rd = gaps(actual, expected)
    ok = (kind === :abs ? ad : rd) <= tol
    push!(RESULTS, Case(id, desc, ad, rd, note, ok))
    return ok
end

# ============================================================================
# Test 1 - covariance and correlation with missing data
# ============================================================================
let x = readm("test1.csv")
    check("1.1", "Covariance, missing data, skip missing rows",
          missing_cov(x, skipMiss=true), readm("testout_1.1.csv"), tol=1e-12)
    check("1.2", "Correlation, missing data, skip missing rows",
          missing_cov(x, skipMiss=true, fun=cor), readm("testout_1.2.csv"), tol=1e-12)
    check("1.3", "Covariance, missing data, pairwise",
          missing_cov(x, skipMiss=false), readm("testout_1.3.csv"), tol=1e-12)
    check("1.4", "Correlation, missing data, pairwise",
          missing_cov(x, skipMiss=false, fun=cor), readm("testout_1.4.csv"), tol=1e-12)
end

# ============================================================================
# Test 2 - exponentially weighted covariance
# ============================================================================
let x = readm("test2.csv")
    check("2.1", "EW covariance, lambda=0.97",
          ewCovar(x, 0.97), readm("testout_2.1.csv"), tol=1e-12)
    check("2.2", "EW correlation, lambda=0.94",
          ewCorr(x, 0.94), readm("testout_2.2.csv"), tol=1e-12)
    check("2.3", "EW cov: EW var lambda=0.97, EW corr lambda=0.94",
          ewCovar(x, 0.97, 0.94), readm("testout_2.3.csv"), tol=1e-12)
end

# ============================================================================
# Test 3 - repairing a non-PSD matrix
# ============================================================================
check("3.1", "near_psd on a covariance",
      near_psd(readm("testout_1.3.csv")), readm("testout_3.1.csv"), tol=1e-10)
check("3.2", "near_psd on a correlation",
      near_psd(readm("testout_1.4.csv")), readm("testout_3.2.csv"), tol=1e-10)
check("3.3", "Higham on a covariance",
      higham_nearestPSD(readm("testout_1.3.csv")), readm("testout_3.3.csv"), tol=1e-8)
check("3.4", "Higham on a correlation",
      higham_nearestPSD(readm("testout_1.4.csv")), readm("testout_3.4.csv"), tol=1e-8)

# ============================================================================
# Test 4 - Cholesky that tolerates a zero eigenvalue
# ============================================================================
check("4.1", "chol_psd",
      chol_psd(readm("testout_3.1.csv")), readm("testout_4.1.csv"), tol=1e-10)

# ============================================================================
# Test 5 - simulation
# ============================================================================
# The RNG stream reproduces, so these match the expected file to floating point
# and that is what gates PASS. The gap to the input covariance is sampling error
# at 100k draws and is reported alongside; for 5.5 it also carries the bias from
# dropping the last 1% of the variance.
function sim_case(id, desc, input_file, expected_file, simulated; tol=1e-8)
    covin = readm(input_file)
    covout = cov(simulated)
    mc, _ = gaps(covout, covin)
    check(id, desc, covout, readm(expected_file); tol=tol,
          note=@sprintf("vs input cov %.2e (sampling error)", mc))
end

sim_case("5.1", "Normal simulation, PD input, 100k draws",
         "test5_1.csv", "testout_5.1.csv",
         simulateNormal(100_000, readm("test5_1.csv")))

sim_case("5.2", "Normal simulation, PSD input, 100k draws",
         "test5_2.csv", "testout_5.2.csv",
         simulateNormal(100_000, readm("test5_2.csv")))

sim_case("5.3", "Normal simulation, non-PSD input, near_psd fix",
         "test5_3.csv", "testout_5.3.csv",
         simulateNormal(100_000, readm("test5_3.csv"), fixMethod=near_psd))

sim_case("5.4", "Normal simulation, non-PSD input, Higham fix",
         "test5_3.csv", "testout_5.4.csv",
         simulateNormal(100_000, readm("test5_3.csv"), fixMethod=higham_nearestPSD))

sim_case("5.5", "PCA simulation, 99 pct variance explained",
         "test5_2.csv", "testout_5.5.csv",
         simulate_pca(readm("test5_2.csv"), 100_000, pctExp=0.99))

# ============================================================================
# Test 6 - returns
# ============================================================================
let prices = readdf("test6.csv")
    for (id, method, desc, expected) in (("6.1", "DISCRETE", "Calculate arithmetic returns", "testout6_1.csv"),
                                         ("6.2", "LOG",      "Calculate log returns",        "testout6_2.csv"))
        got = return_calculate(prices, method=method, dateColumn="Date")
        want = readdf(expected)
        dates_ok = got.Date == want.Date
        check(id, desc, Matrix(got[!, Not(:Date)]), Matrix(want[!, Not(:Date)]);
              tol=1e-12, note = dates_ok ? "dates match" : "DATE COLUMN MISMATCH")
    end
end

# ============================================================================
# Test 7 - distribution fitting
# ============================================================================
let x = readm("test7_1.csv")[:, 1]
    fd = fit_normal(x)
    mu, sigma = Distributions.params(fd.errorModel)
    check("7.1", "Fit a normal distribution",
          [mu, sigma], vec(readm("testout7_1.csv")), tol=1e-12)
end

let x = readm("test7_2.csv")[:, 1]
    fd = fit_general_t(x)
    mu, sigma, rho = Distributions.params(fd.errorModel)
    check("7.2", "Fit a generalized t distribution",
          [mu, sigma, dof(rho)],
          vec(readm("testout7_2.csv")); tol=1e-6, kind=:rel)
    check("7.4", "AICc of the fitted t",
          [aicc(fd, x)], vec(readm("testout7_4.csv")); tol=1e-6, kind=:rel)
end

let df = readdf("test7_3.csv")
    fd = fit_regression_t(df.y, Matrix(select(df, Not(:y))))
    mu, sigma, rho = Distributions.params(fd.errorModel)
    got = vcat(mu, sigma, dof(rho), fd.beta)
    check("7.3", "T regression", got, vec(readm("testout7_3.csv")); tol=1e-6, kind=:rel)
end

let x = readm("test7_5.csv")[:, 1]
    d = fit_nig_moments(x).errorModel
    check("7.5", "Fit a NIG by the method of moments",
          collect(Distributions.params(d)), vec(readm("testout7_5.csv")), tol=1e-10)

    d = fit_nig_mle(x).errorModel
    check("7.6", "Fit a NIG by maximum likelihood",
          collect(Distributions.params(d)), vec(readm("testout7_6.csv"));
          tol=1e-3, kind=:rel, note="expected file is a scipy fit")
end

# ============================================================================
# Report
# ============================================================================
# 7.4 shares a fit with 7.2, so sort before printing
sortkey(c) = (parse(Int, split(c.id, ".")[1]), parse(Int, split(c.id, ".")[2]))

println()
println("="^106)
@printf("%-5s %-50s %-7s %12s %12s  %s\n", "Test", "Description", "Status", "max abs d", "max rel d", "Note")
println("="^106)
for c in sort(RESULTS, by=sortkey)
    @printf("%-5s %-50s %-7s %12.3e %12.3e  %s\n",
            c.id, c.desc, c.ok ? "PASS" : "FAIL", c.absdiff, c.reldiff, c.note)
end
println("="^106)

npass = count(c -> c.ok, RESULTS)
@printf("%d of %d cases passed.\n", npass, length(RESULTS))
if npass != length(RESULTS)
    println("FAILED: ", join([c.id for c in sort(RESULTS, by=sortkey) if !c.ok], ", "))
    exit(1)
end
