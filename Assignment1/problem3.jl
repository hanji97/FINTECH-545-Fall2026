include(joinpath(@__DIR__, "common.jl"))

println("="^78)
println("PROBLEM 3 - Pearson Against Spearman")
println("="^78)

d = readproblem(3)
M = Matrix(d)
nm = names(d)
n = size(M, 1)
println("\nn = $n, series = ", join(nm, ", "))

P = cor(M)
S = corspearman(M)

println("\nPearson correlation matrix")
print("        ")
for j in nm; @printf("%9s", j); end
println()
for i in 1:4
    @printf("%8s", nm[i])
    for j in 1:4; @printf("%9.4f", P[i, j]); end
    println()
end

println("\nSpearman rank correlation matrix")
print("        ")
for j in nm; @printf("%9s", j); end
println()
for i in 1:4
    @printf("%8s", nm[i])
    for j in 1:4; @printf("%9.4f", S[i, j]); end
    println()
end

println("\npairwise gaps, Spearman minus Pearson, sorted by absolute size")
pairs = [(i, j) for i in 1:4 for j in i+1:4]
sort!(pairs, by=ij -> -abs(S[ij[1], ij[2]] - P[ij[1], ij[2]]))
for (i, j) in pairs
    @printf("  %s-%s   Pearson %+.4f   Spearman %+.4f   gap %+.4f\n",
            nm[i], nm[j], P[i, j], S[i, j], S[i, j] - P[i, j])
end
(bi, bj) = pairs[1]
@printf("\nlargest gap: %s-%s at %+.4f\n", nm[bi], nm[bj], S[bi, bj] - P[bi, bj])

x1, x2, x3, x4 = d.x1, d.x2, d.x3, d.x4
println("\nwhat x2 actually is")
for (lbl, v) in (("x1", x1), ("x1^2", x1 .^ 2), ("x1^3", x1 .^ 3))
    @printf("  cor(%-5s, x2) = %+.6f\n", lbl, cor(v, x2))
end
Z = [ones(n) x1 .^ 3]
bc = Z \ x2
rc = x2 - Z * bc
@printf("  x2 on x1^3:  intercept %+.6f  slope %+.6f  R2 %.6f  residual sd %.6f\n",
        bc[1], bc[2], 1 - sum(abs2, rc) / sum((x2 .- mean(x2)) .^ 2), std(rc))
@printf("  monotone in x1: %s (%d of %d adjacent pairs agree in sign after sorting)\n",
        all(diff(x2[sortperm(x1)]) .> 0) ? "exactly" : "not exactly",
        count(>(0), diff(x2[sortperm(x1)])), n - 1)

println("\nrank disagreement concentrates where the cubic is flat")
ord = sortperm(x1)
rx, ry = ordinalrank(x1), ordinalrank(x2)
for (lo, hi) in ((0.0, 0.5), (0.5, 1.0), (1.0, 3.5))
    m = (abs.(x1) .>= lo) .& (abs.(x1) .< hi)
    @printf("  %.1f <= |x1| < %.1f   n=%3d   mean |rank(x1)-rank(x2)| = %6.2f\n",
            lo, hi, count(m), mean(abs.(rx[m] .- ry[m])))
end

println("\nsanity: linearity of the other informative pair")
@printf("  x1-x3 Pearson %+.4f  Spearman %+.4f   cor(x1^3,x3) %+.4f\n",
        P[1, 3], S[1, 3], cor(x1 .^ 3, x3))

plots = Any[]
for i in 1:4, j in 1:4
    if i == j
        push!(plots, histogram(M[:, i], bins=30, legend=false, color=:steelblue,
                               alpha=0.7, title=nm[i], titlefontsize=9,
                               xticks=false, yticks=false))
    else
        push!(plots, scatter(M[:, j], M[:, i], ms=1.6, mc=:steelblue, legend=false,
                             xticks=false, yticks=false,
                             title="$(nm[i]) vs $(nm[j])", titlefontsize=8))
    end
end
pm = plot(plots..., layout=(4, 4), size=(1100, 1000),
          plot_title="Problem 3: every pair")
savefig(pm, joinpath(FIGS, "p3_pairs.png"))

f = scatter(x1, x2, ms=3, mc=:steelblue, label="data", xlabel="x1", ylabel="x2",
            title="Problem 3: x1 against x2, the pair with the largest gap")
xs = range(minimum(x1), maximum(x1), length=300)
plot!(f, xs, bc[1] .+ bc[2] .* xs .^ 3, lw=2.5, color=:seagreen, label="cubic")
plot!(f, xs, mean(x2) .+ (cov(x1, x2) / var(x1)) .* (xs .- mean(x1)),
      lw=2.5, ls=:dash, color=:firebrick, label="OLS line (what Pearson sees)")
savefig(f, joinpath(FIGS, "p3_x1x2.png"))

r = scatter(ordinalrank(x1), ordinalrank(x2), ms=2.5, mc=:seagreen, legend=false,
            xlabel="rank(x1)", ylabel="rank(x2)",
            title="Problem 3: the same pair in rank space")
savefig(r, joinpath(FIGS, "p3_ranks.png"))
println("\nfigures: figures/p3_pairs.png  figures/p3_x1x2.png  figures/p3_ranks.png")
