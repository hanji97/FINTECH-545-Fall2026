include(joinpath(@__DIR__, "common.jl"))

println("="^78)
println("PROBLEM 2 - A Regression Whose Errors Are Not Normal")
println("="^78)

d = readproblem(2)
x, y = d.x, d.y
n = length(x)
X = [ones(n) x]

sc = scatter(x, y, ms=3, mc=:steelblue, label="data", xlabel="x", ylabel="y",
             title="Problem 2: y against x")
savefig(sc, joinpath(FIGS, "p2_scatter.png"))

o = olsfit(X, y)
mo = fourmoments(o.resid)
@printf("\nOLS\n")
@printf("  alpha           %+.6f   se %.6f   t %+.2f\n", o.beta[1], o.se[1], o.tstat[1])
@printf("  beta            %+.6f   se %.6f   t %+.2f\n", o.beta[2], o.se[2], o.tstat[2])
@printf("  s (resid sd)    %.6f\n", o.s)
@printf("  R2 / adj R2     %.4f / %.4f\n", o.r2, o.adjr2)
@printf("  Durbin-Watson   %.4f\n", durbinwatson(o.resid))
@printf("  smallest eig    %.4f\n", o.smallest_eig)
@printf("  resid skew      %+.4f\n", mo.skew)
@printf("  resid ex kurt   %+.4f\n", mo.exkurt)

sigma_mle = sqrt(sum(abs2, o.resid) / n)
ll_n = normal_ll(o.resid, sigma_mle)
k_n = 3
@printf("\nMLE, Normal error\n")
@printf("  alpha           %+.6f\n", o.beta[1])
@printf("  beta            %+.6f\n", o.beta[2])
@printf("  sigma           %.6f\n", sigma_mle)
@printf("  log likelihood  %.4f\n", ll_n)
@printf("  k / AICc        %d / %.4f\n", k_n, aicc(ll_n, k_n, n))

function negll_t(v)
    a, b, ls, lnu = v
    e = y .- a .- b .* x
    -t_ll(e, exp(ls), 2 + exp(lnu))
end
v0 = [o.beta[1], o.beta[2], log(o.s), log(3.0)]
r1 = optimize(negll_t, v0, NelderMead(), Optim.Options(iterations=50000))
gt!(G, v) = ForwardDiff.gradient!(G, negll_t, v)
r2 = optimize(negll_t, gt!, Optim.minimizer(r1), BFGS())
v = Optim.minimizer(r2)
ll_t = -Optim.minimum(r2)
a_t, b_t = v[1], v[2]
s_t = exp(v[3])
nu_t = 2 + exp(v[4])
H = ForwardDiff.hessian(negll_t, v)
sev = sqrt.(diag(inv(H)))
k_t = 4
@printf("\nMLE, Student's t error\n")
@printf("  alpha           %+.6f   se %.6f\n", a_t, sev[1])
@printf("  beta            %+.6f   se %.6f\n", b_t, sev[2])
@printf("  sigma (scale)   %.6f\n", s_t)
@printf("  nu              %.4f\n", nu_t)
@printf("  implied sd      %.6f\n", s_t * sqrt(nu_t / (nu_t - 2)))
@printf("  log likelihood  %.4f\n", ll_t)
@printf("  k / AICc        %d / %.4f\n", k_t, aicc(ll_t, k_t, n))

println("\nmodel comparison")
@printf("  %-22s %10s %8s %12s %12s\n", "model", "loglik", "k", "AICc", "BIC")
@printf("  %-22s %10.4f %8d %12.4f %12.4f\n", "OLS", ll_n, k_n, aicc(ll_n, k_n, n), bic(ll_n, k_n, n))
@printf("  %-22s %10.4f %8d %12.4f %12.4f\n", "MLE Normal", ll_n, k_n, aicc(ll_n, k_n, n), bic(ll_n, k_n, n))
@printf("  %-22s %10.4f %8d %12.4f %12.4f\n", "MLE Student t", ll_t, k_t, aicc(ll_t, k_t, n), bic(ll_t, k_t, n))
@printf("  delta AICc (Normal - t) = %.4f\n", aicc(ll_n, k_n, n) - aicc(ll_t, k_t, n))

println("\nslope estimates side by side")
@printf("  OLS            %.6f\n", o.beta[2])
@printf("  MLE Normal     %.6f\n", o.beta[2])
@printf("  MLE t          %.6f\n", b_t)
@printf("  spread         %.6f  (%.2f%% of the OLS slope, %.3f OLS standard errors)\n",
        abs(b_t - o.beta[2]), 100 * abs(b_t - o.beta[2]) / o.beta[2],
        abs(b_t - o.beta[2]) / o.se[2])
@printf("  se(beta) OLS %.6f  vs  MLE t %.6f   ratio %.3f\n", o.se[2], sev[2], o.se[2] / sev[2])

println("\nerror quantiles, fitted Normal against fitted t")
Nerr = Normal(0, sigma_mle)
for p in (0.95, 0.975, 0.99, 0.995, 0.999)
    qn = quantile(Nerr, p)
    qt = s_t * quantile(TDist(nu_t), p)
    @printf("  %6.3f   Normal %+8.4f   t %+8.4f   wider: %s\n",
            p, qn, qt, abs(qt) > abs(qn) ? "t" : "Normal")
end
function crossover(lo, hi)
    for p in range(lo, hi, length=200000)
        s_t * quantile(TDist(nu_t), p) > quantile(Nerr, p) && return p
    end
    NaN
end
@printf("  the t overtakes the Normal at about the %.3f%% quantile\n", 100 * crossover(0.5, 0.9999))

println("\ninterval calibration of the two fitted error laws on the OLS residuals")
@printf("  %8s %10s %12s %12s\n", "level", "expected", "outside N", "outside t")
for lev in (0.90, 0.95, 0.99, 0.995)
    qn = quantile(Nerr, 0.5 + lev / 2)
    qt = s_t * quantile(TDist(nu_t), 0.5 + lev / 2)
    @printf("  %8.3f %10.1f %12d %12d\n", lev, (1 - lev) * n,
            count(>(qn), abs.(o.resid)), count(>(qt), abs.(o.resid)))
end

xs = range(minimum(x), maximum(x), length=200)
fit = plot(xs, o.beta[1] .+ o.beta[2] .* xs, lw=2.5, color=:firebrick, label="OLS")
plot!(fit, xs, a_t .+ b_t .* xs, lw=2.5, ls=:dash, color=:seagreen, label="MLE t")
scatter!(fit, x, y, ms=3, mc=:steelblue, label="data", xlabel="x", ylabel="y",
         title="Problem 2: the two fits are visually the same line")
savefig(fit, joinpath(FIGS, "p2_fits.png"))

e = o.resid
hh = histogram(e, bins=40, normalize=:pdf, alpha=0.55, color=:steelblue,
               label="OLS residuals", xlabel="residual", ylabel="density",
               title="Problem 2: residuals against the two fitted error laws")
zs = range(minimum(e) * 1.1, maximum(e) * 1.1, length=600)
plot!(hh, zs, z -> pdf(Nerr, z), lw=2.5, color=:firebrick, label="fitted Normal")
plot!(hh, zs, z -> pdf(TDist(nu_t), z / s_t) / s_t, lw=2.5, color=:seagreen, label="fitted t")
savefig(hh, joinpath(FIGS, "p2_resid.png"))
println("\nfigures: figures/p2_scatter.png  figures/p2_fits.png  figures/p2_resid.png")
