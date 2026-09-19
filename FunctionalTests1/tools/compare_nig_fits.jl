# Tests 7.5 and 7.6 side by side.
#
#     julia --project=../testfiles compare_nig_fits.jl
#
# Same 1,000 observations from a known NIG, fitted three ways: method of moments
# and MLE from this library, plus scipy's MLE as a check on the optimiser. The
# moment fit reproduces the sample skewness and kurtosis; the MLE reaches the
# higher log likelihood.

include("../library/RiskLib.jl")
using Printf

const PYTHON = get(ENV, "PYTHON", "python")
const DATA = joinpath(@__DIR__, "..", "testfiles", "data", "test7_5.csv")

x = Matrix(CSV.read(DATA, DataFrame))[:, 1]
truth = NormalInverseGaussian(0.02, 40.0, -8.0, 0.05)   # how test7_5.csv was drawn

mm = fit_nig_moments(x).errorModel
mle = fit_nig_mle(x).errorModel

# scipy via the reference script; a missing scipy drops the column instead of
# failing the run
function scipy_fit()
    script = joinpath(@__DIR__, "nig_scipy_reference.py")
    out = try
        read(`$PYTHON $script $DATA`, String)
    catch e
        @warn "could not run the scipy reference; skipping that column" exception = e
        return nothing
    end
    grab(k) = (m = match(Regex("\"$k\": *(-?[0-9.eE+-]+)"), out); m === nothing ? nothing : parse(Float64, m[1]))
    p = grab.(["mu", "alpha", "beta", "delta"])
    any(isnothing, p) && return nothing
    return NormalInverseGaussian(p...)
end

sp = scipy_fit()

ll(d) = sum(nig_logpdf(Distributions.params(d)..., xi) for xi in x)

cols = ["truth" => truth, "7.5 moments" => mm, "7.6 MLE (julia)" => mle]
sp === nothing || push!(cols, "7.6 MLE (scipy)" => sp)

println()
println("NIG fits on test7_5.csv, n = ", length(x))
println("="^76)
@printf("%-16s", "")
for (name, _) in cols
    @printf("%16s", name)
end
println()
println("-"^76)

for (label, f) in ["mu" => (d -> d.μ), "alpha" => (d -> d.α),
                   "beta" => (d -> d.β), "delta" => (d -> d.δ)]
    @printf("%-16s", label)
    for (_, d) in cols
        @printf("%16.5f", f(d))
    end
    println()
end

println("-"^76)
@printf("%-16s", "log likelihood")
for (name, d) in cols
    name == "truth" ? @printf("%16s", "-") : @printf("%16.4f", ll(d))
end
println()

println()
println("Moments: the sample, and what each fit implies")
println("-"^76)
for (label, f) in ["mean" => mean, "std" => std, "skewness" => skewness, "kurtosis" => kurtosis]
    @printf("%-16s%16.5f", label, f(x))
    for (name, d) in cols
        name == "truth" ? nothing : @printf("%16.5f", f(d))
    end
    println()
end
println()
println("(first column is the sample)")

if sp !== nothing
    d = maximum(abs.(collect(Distributions.params(mle)) .- collect(Distributions.params(sp))) ./
                abs.(collect(Distributions.params(sp))))
    @printf("\nJulia MLE vs scipy MLE: max relative difference %.3e\n", d)
    @printf("log likelihood gap (julia - scipy): %+.3e\n", ll(mle) - ll(sp))
end
