include(joinpath(@__DIR__, "common.jl"))

println("="^78)
println("PROBLEM 1 - Reading the Shape of a Sample")
println("="^78)

x = readproblem(1).x
m = fourmoments(x)

@printf("n                       %d\n", m.n)
@printf("mean                    %+.8f\n", m.mean)
@printf("variance                %.10f\n", m.var)
@printf("standard deviation      %.8f\n", m.sd)
@printf("skewness (population)   %+.6f\n", m.skew_biased)
@printf("skewness (unbiased)     %+.6f\n", m.skew)
@printf("excess kurt (pop.)      %+.6f\n", m.exkurt_biased)
@printf("excess kurt (unbiased)  %+.6f\n", m.exkurt)
@printf("min / max               %+.6f / %+.6f\n", minimum(x), maximum(x))
@printf("negative observations   %d of %d\n", count(<(0), x), m.n)

println("\nfeasibility check  excess kurtosis >= skewness^2 - 2")
@printf("  %+.4f >= %+.4f  -> %s\n", m.exkurt, m.skew^2 - 2,
        m.exkurt >= m.skew^2 - 2 ? "satisfied" : "VIOLATED")

println("\nmoment-matched NIG (four moments, closed form)")
u = 3 / (m.exkurt - 4 / 3 * m.skew^2)
rho = m.skew * sqrt(u) / 3
alpha = sqrt(u) / (m.sd * (1 - rho^2))
beta = rho * alpha
gamma = alpha * sqrt(1 - rho^2)
delta = u / gamma
mu_nig = m.mean - delta * beta / gamma
@printf("  alpha=%.4f  beta=%.4f  delta=%.6f  mu=%.6f   (|beta|<alpha: %s)\n",
        alpha, beta, delta, mu_nig, abs(beta) < alpha ? "yes" : "no")
@printf("  implied skew %+.6f   implied excess kurt %+.6f\n",
        3beta / (alpha * sqrt(delta * gamma)),
        3 / (delta * gamma) * (1 + 4beta^2 / alpha^2))

println("\nfitted Normal by matching mean and variance")
N = Normal(m.mean, m.sd)
@printf("  N(%.8f, %.10f)\n", m.mean, m.var)

q01 = quantile(N, 0.01)
q99 = quantile(N, 0.99)
below = count(<(q01), x)
above = count(>(q99), x)
@printf("\n1%% quantile of fitted Normal   %+.8f\n", q01)
@printf("observations below it          %d\n", below)
@printf("expected under the Normal      %.1f\n", 0.01 * m.n)
@printf("ratio                          %.2fx\n", below / (0.01 * m.n))
@printf("99%% quantile / above / expect  %+.8f  %d  %.1f\n", q99, above, 0.01 * m.n)

emp01 = quantile(x, 0.01)
@printf("\nempirical 1%% quantile          %+.8f\n", emp01)
@printf("Normal 1%% quantile             %+.8f  (understates the loss by %.1f%%)\n",
        q01, 100 * (emp01 / q01 - 1))
@printf("worst observation              %+.8f  = %.2f fitted sigma\n",
        minimum(x), (minimum(x) - m.mean) / m.sd)

println("\ntail and shoulder counts, observed against Normal expectation")
for p in (0.005, 0.01, 0.025, 0.05, 0.10)
    lo = quantile(N, p)
    hi = quantile(N, 1 - p)
    @printf("  %5.1f%%  left %3d vs %5.1f    right %3d vs %5.1f\n",
            100p, count(<(lo), x), p * m.n, count(>(hi), x), p * m.n)
end
inside1 = count(z -> abs(z - m.mean) <= m.sd, x)
@printf("  within +/-1 sigma  %d observed vs %.1f expected\n", inside1, 0.6827 * m.n)

h = histogram(x, bins=60, normalize=:pdf, label="sample", alpha=0.55,
              color=:steelblue, xlabel="x", ylabel="density",
              title="Problem 1: sample against the moment-matched Normal")
plot!(h, range(minimum(x), maximum(x), length=600), z -> pdf(N, z),
      lw=2.5, color=:firebrick, label="fitted Normal")
vline!(h, [q01], lw=2, ls=:dash, color=:black, label="Normal 1% quantile")
savefig(h, joinpath(FIGS, "p1_hist.png"))

srt = sort(x)
theo = [quantile(N, (i - 0.5) / m.n) for i in 1:m.n]
qq = scatter(theo, srt, ms=2.5, mc=:steelblue, label="sample",
             xlabel="Normal quantile", ylabel="sample quantile",
             title="Problem 1: QQ plot against the fitted Normal", legend=:topleft)
plot!(qq, [minimum(theo), maximum(theo)], [minimum(theo), maximum(theo)],
      lw=2, color=:firebrick, label="45 degree line")
savefig(qq, joinpath(FIGS, "p1_qq.png"))
println("\nfigures: figures/p1_hist.png  figures/p1_qq.png")
