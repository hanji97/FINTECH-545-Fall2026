include(joinpath(@__DIR__, "common.jl"))

println("="^78)
println("PROBLEM 5 - Identifying an AR or MA Order")
println("="^78)

x = readproblem(5).x
n = length(x)
band = 1.96 / sqrt(n)

@printf("\nn = %d   mean = %+.6f   sd = %.6f\n", n, mean(x), std(x))
@printf("significance band  +/- 1.96/sqrt(n) = +/- %.6f\n", band)

L = 12
a = autocor(x, 0:L)
p = pacf(x, 1:L)
println("\nlag      ACF      PACF   |ACF|>band  |PACF|>band")
for k in 1:L
    @printf("%3d  %+8.4f  %+8.4f   %9s  %10s\n", k, a[k+1], p[k],
            abs(a[k+1]) > band ? "yes" : ".", abs(p[k]) > band ? "yes" : ".")
end
@printf("\nlast lag with |PACF| > band inside the first 6: %d\n",
        maximum([k for k in 1:6 if abs(p[k]) > band]))
@printf("last lag with |ACF|  > band inside the first 6: %d\n",
        maximum([k for k in 1:6 if abs(a[k+1]) > band]))

ts = plot(x, lw=1, color=:steelblue, legend=false, xlabel="t", ylabel="x",
          title="Problem 5: the series")
hline!(ts, [mean(x)], lw=2, color=:firebrick)
savefig(ts, joinpath(FIGS, "p5_series.png"))

ylim = 1.15 * maximum(abs, vcat(a[2:end], p))
pa = bar(1:L, a[2:end], color=:steelblue, legend=false, xlabel="lag", ylabel="ACF",
         title="ACF", bar_width=0.35, ylims=(-ylim, ylim), xticks=1:L)
hline!(pa, [band, -band], ls=:dash, lw=2, color=:firebrick)
pp = bar(1:L, p, color=:seagreen, legend=false, xlabel="lag", ylabel="PACF",
         title="PACF", bar_width=0.35, ylims=(-ylim, ylim), xticks=1:L)
hline!(pp, [band, -band], ls=:dash, lw=2, color=:firebrick)
savefig(plot(pa, pp, layout=(2, 1), size=(900, 700),
             plot_title="Problem 5: ACF decays, PACF cuts off"),
        joinpath(FIGS, "p5_acf_pacf.png"))

specs = [(1, 0, "AR(1)"), (2, 0, "AR(2)"), (3, 0, "AR(3)"),
         (0, 1, "MA(1)"), (0, 2, "MA(2)"), (0, 3, "MA(3)")]
fits = Dict{String,Any}()
println("\nfitting by exact Gaussian maximum likelihood on all $n observations")
for (pp_, qq, lbl) in specs
    f = fit_arma(x, pp_, qq)
    fits[lbl] = f
    print("  $lbl  mu $(round(f.mu, digits=4))  sigma $(round(f.sigma, digits=4))")
    pp_ > 0 && print("  phi ", round.(f.phi, digits=4))
    qq > 0 && print("  theta ", round.(f.theta, digits=4))
    println("  ll ", round(f.ll, digits=4))
end

println("\nmodel selection")
@printf("  %-8s %4s %12s %12s %12s\n", "model", "k", "loglik", "AICc", "BIC")
for (_, _, lbl) in specs
    f = fits[lbl]
    @printf("  %-8s %4d %12.4f %12.4f %12.4f\n", lbl, f.k, f.ll, f.aicc, f.bic)
end
bestlbl = argmin(Dict(lbl => fits[lbl].aicc for (_, _, lbl) in specs))
@printf("\nAICc selects %s\n", bestlbl)
@printf("  next best is %s by %.4f AICc\n",
        sort([(fits[l].aicc, l) for (_, _, l) in specs])[2][2],
        sort([fits[l].aicc for (_, _, l) in specs])[2] - fits[bestlbl].aicc)

println("\nAR(2) against AR(3)")
f2, f3 = fits["AR(2)"], fits["AR(3)"]
@printf("  AR(2) phi = %s\n", round.(f2.phi, digits=5))
@printf("  AR(3) phi = %s\n", round.(f3.phi, digits=5))
@printf("  third coefficient %+.5f\n", f3.phi[3])
@printf("  log likelihood gain %.4f for one extra parameter\n", f3.ll - f2.ll)
@printf("  AICc charges roughly 2 per parameter, so it needs a gain above 1.0; it got %.4f\n",
        f3.ll - f2.ll)
lr = 2 * (f3.ll - f2.ll)
@printf("  likelihood ratio statistic %.4f, chi-square(1) p-value %.4f\n",
        lr, ccdf(Chisq(1), lr))

println("\ncross checks on the AR(2) coefficients")
r1, r2 = a[2], a[3]
yw2 = (r2 - r1^2) / (1 - r1^2)
yw1 = r1 * (1 - yw2)
@printf("  Yule-Walker from the sample ACF     phi = [%+.5f, %+.5f]\n", yw1, yw2)
Xc = [ones(n - 2) x[2:n-1] x[1:n-2]]
bc = Xc \ x[3:n]
@printf("  conditional least squares           phi = [%+.5f, %+.5f]\n", bc[2], bc[3])
@printf("  exact Gaussian MLE                  phi = [%+.5f, %+.5f]\n",
        fits["AR(2)"].phi[1], fits["AR(2)"].phi[2])

println("\nin-sample fit, one step ahead, conditional on the observed past")
for lbl in ("AR(1)", "AR(2)", "AR(3)")
    f = fits[lbl]
    pl = length(f.phi)
    r = x .- f.mu
    pred = [sum(f.phi[i] * r[t-i] for i in 1:pl) for t in pl+1:n]
    err = r[pl+1:end] .- pred
    sst = sum((x[pl+1:end] .- mean(x[pl+1:end])) .^ 2)
    @printf("  %-6s  R2 %.6f   residual sd %.6f\n", lbl, 1 - sum(abs2, err) / sst, std(err))
end
println("\nfigures: figures/p5_series.png  figures/p5_acf_pacf.png")
