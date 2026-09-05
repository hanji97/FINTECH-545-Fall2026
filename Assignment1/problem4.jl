include(joinpath(@__DIR__, "common.jl"))

println("="^78)
println("PROBLEM 4 - Conditional Distributions")
println("="^78)

d = readproblem(4)
x1, x2 = d.x1, d.x2
n = length(x1)

Z = [x2 x1]
mu = vec(mean(Z, dims=1))
S = cov(Z)
rho = cor(x1, x2)

println("\nvector ordered as [x2; x1] so that block 1 is the series being conditioned")
println("and block 2 is the series being observed, matching the Week 2 partition")
@printf("  n = %d\n", n)
@printf("  mu    = [mu_x2, mu_x1] = [%+.6f, %+.6f]\n", mu[1], mu[2])
@printf("  Sigma = [%9.6f  %9.6f]   S11 = var(x2), S22 = var(x1)\n", S[1, 1], S[1, 2])
@printf("          [%9.6f  %9.6f]\n", S[2, 1], S[2, 2])
@printf("  rho   = %+.6f\n", rho)

S11, S12, S21, S22 = S[1, 1], S[1, 2], S[2, 1], S[2, 2]
condvar = S11 - S12 * inv(S22) * S21
beta = S12 * inv(S22)

println("\nconditional variance of x2 given x1")
@printf("  Sigma_bar = S11 - S12 S22^-1 S21\n")
@printf("            = %.6f - (%.6f)(%.6f)^-1(%.6f)\n", S11, S12, S22, S21)
@printf("            = %.6f - %.6f = %.6f\n", S11, S12 * inv(S22) * S21, condvar)
@printf("  conditional sd                %.6f\n", sqrt(condvar))
@printf("  unconditional var(x2) = S11   %.6f\n", S11)
@printf("  shrink factor Sigma_bar/S11   %.6f\n", condvar / S11)
@printf("  1 - rho^2                     %.6f\n", 1 - rho^2)
@printf("  variance falls %.2f%%, sd falls %.2f%%\n",
        100 * (1 - condvar / S11), 100 * (1 - sqrt(condvar / S11)))

println("\nconditional mean of x2 given x1 = a")
@printf("  mu_bar = mu1 + S12 S22^-1 (a - mu2) = %+.6f %+.6f (a %+.6f)\n",
        mu[1], beta, -mu[2])
@printf("  coefficient S12 S22^-1        %+.6f\n", beta)
o = olsfit([ones(n) x1], x2)
@printf("  OLS slope of x2 on x1         %+.6f   se %.6f\n", o.beta[2], o.se[2])
@printf("  OLS intercept                 %+.6f   mu_bar intercept %+.6f\n",
        o.beta[1], mu[1] - beta * mu[2])
@printf("  OLS s^2 (n-p)                 %.6f   Sigma_bar %.6f\n", o.s2, condvar)

cm = mu[1] .+ beta .* (x1 .- mu[2])
resid = x2 .- cm
zq = quantile(Normal(), 0.975)
halfwidth = zq * sqrt(condvar)
inside = abs.(resid) .<= halfwidth

@printf("\n95%% band  mu_bar +/- %.6f x %.6f = +/- %.6f  (constant width)\n",
        zq, sqrt(condvar), halfwidth)
@printf("  inside    %d of %d\n", count(inside), n)
@printf("  coverage  %.4f   nominal 0.9500\n", mean(inside))

dist = abs.(x1 .- mu[2]) ./ std(x1)
buckets = [(0.0, 1.0, "|x1 - mu| < 1 sd"),
           (1.0, 2.0, "1 sd <= . < 2 sd"),
           (2.0, Inf, "|x1 - mu| >= 2 sd")]

println("\ncoverage by how far x1 sits from its mean")
@printf("  %-20s %5s %9s %9s %10s %12s\n",
        "bucket", "n", "coverage", "std err", "resid sd", "resid exkurt")
for (lo, hi, lbl) in buckets
    m = (dist .>= lo) .& (dist .< hi)
    nb = count(m)
    c = mean(inside[m])
    @printf("  %-20s %5d %9.4f %9.4f %10.4f %12.4f\n",
            lbl, nb, c, sqrt(c * (1 - c) / nb), std(resid[m]), fourmoments(resid[m]).exkurt)
end
cm_all = mean(inside)
@printf("  %-20s %5d %9.4f %9.4f %10.4f %12.4f\n", "all", n, cm_all,
        sqrt(cm_all * (1 - cm_all) / n), std(resid), fourmoments(resid).exkurt)

println("\nresidual shape overall")
mr = fourmoments(resid)
@printf("  mean %+.6f  sd %.6f  skew %+.4f  excess kurtosis %+.4f\n",
        mr.mean, mr.sd, mr.skew, mr.exkurt)

println("\nhalf width each bucket would need for 95% coverage")
for (lo, hi, lbl) in buckets
    m = (dist .>= lo) .& (dist .< hi)
    need = quantile(abs.(resid[m]), 0.95)
    @printf("  %-20s  needed %.4f   supplied %.4f   ratio %.2f\n",
            lbl, need, halfwidth, need / halfwidth)
end

println("\nis the residual scale a function of x1? regress |resid| on |x1 - mu|")
ab = olsfit([ones(n) abs.(x1 .- mu[2])], abs.(resid))
@printf("  slope %+.6f   se %.6f   t %+.2f   R2 %.4f\n",
        ab.beta[2], ab.se[2], ab.tstat[2], ab.r2)

xs = range(minimum(x1), maximum(x1), length=300)
band = plot(xs, mu[1] .+ beta .* (xs .- mu[2]), lw=2.5, color=:firebrick,
            label="conditional mean", xlabel="x1", ylabel="x2",
            title="Problem 4: conditional expectation with a 95% band")
plot!(xs, mu[1] .+ beta .* (xs .- mu[2]) .+ halfwidth, lw=2, ls=:dash,
      color=:black, label="95% band")
plot!(xs, mu[1] .+ beta .* (xs .- mu[2]) .- halfwidth, lw=2, ls=:dash,
      color=:black, label="")
scatter!(x1[inside], x2[inside], ms=2.6, mc=:steelblue, label="inside")
scatter!(x1[.!inside], x2[.!inside], ms=3.4, mc=:orange, label="outside")
savefig(band, joinpath(FIGS, "p4_band.png"))

rp = scatter(x1, resid, ms=2.6, mc=:steelblue, label="residual", xlabel="x1",
             ylabel="x2 - E[x2|x1]", title="Problem 4: residual spread widens with |x1|")
hline!(rp, [halfwidth, -halfwidth], lw=2, ls=:dash, color=:black, label="95% band")
savefig(rp, joinpath(FIGS, "p4_resid.png"))

edges = range(minimum(x1), maximum(x1), length=13)
cx = Float64[]
cs = Float64[]
for i in 1:length(edges)-1
    m = (x1 .>= edges[i]) .& (x1 .< edges[i+1])
    if count(m) >= 15
        push!(cx, (edges[i] + edges[i+1]) / 2)
        push!(cs, std(resid[m]))
    end
end
sp = scatter(cx, cs, ms=6, mc=:seagreen, label="residual sd in bin", xlabel="x1",
             ylabel="sd of x2 - E[x2|x1]",
             title="Problem 4: the conditional sd is not constant")
hline!(sp, [sqrt(condvar)], lw=2.5, color=:firebrick,
       label="sqrt of the partitioned conditional variance")
savefig(sp, joinpath(FIGS, "p4_scale.png"))
println("\nfigures: figures/p4_band.png  figures/p4_resid.png  figures/p4_scale.png")
