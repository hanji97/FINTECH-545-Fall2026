# Problem 2: American options and discrete dividends.
#
#     julia --project=. problem2.jl
#
# S = 48.50, X = 50, T = 0.5, r = 4.5%, σ = 30%. Dividends of 0.25 at t = 0.10
# and 1.50 at t = 0.40. N is a multiple of 5, so with Δt = T/N the dividends
# land exactly on steps N/5 and 4N/5.

include(joinpath(@__DIR__, "library", "RiskLib.jl"))
using Plots

const S = 48.50
const X = 50.0
const T = 0.5
const r = 0.045
const σ = 0.30
const DIV_T = [0.10, 0.40]
const DIV_AMT = [0.25, 1.50]
const N_MAIN = 500                   # steps for the reported prices and Greeks

div_steps(N) = round.(Int, DIV_T ./ T .* N)

# One entry point for every price in this problem.
#   divs = false  the recombining tree with b = r (no dividends)
#   divs = true   the non-recombining tree
function tree(call, american; N=N_MAIN, s=S, amts=DIV_AMT, divs=true)
    divs || return bt_american(call, s, X, T, r, r, σ, N; american=american)
    return bt_american(call, s, X, T, r, amts, div_steps(N), σ, N; american=american)
end

# --- (a) Hull's condition, Week 07 §2.2 ---------------------------------------
# Exercising the call just before dividend n can only pay if the dividend beats
# the interest on the strike from t_n to expiry: D_n > X(1 - e^{-r(T - t_n)}).
cond = DataFrame(Dividend=["D1", "D2"], t_n=DIV_T, D_n=DIV_AMT,
                 Interest=[X * (1 - exp(-r * (T - t))) for t in DIV_T])
cond.Exercise_possible = ifelse.(cond.D_n .> cond.Interest, "yes", "no")
write_table(cond, "p2_condition"; fmts=Dict("t_n" => "%.2f", "D_n" => "%.2f", "Interest" => "%.4f"),
            header=["Dividend", "\$t_n\$", "\$D_n\$", "\$X(1-e^{-r(T-t_n)})\$", "Exercise can be optimal"])

# --- (c) prices ----------------------------------------------------------------
# No dividends: GBSM for the Europeans, the tree for the Americans. With
# dividends: the non-recombining tree for all four, plus GBSM on S less the
# present value of the dividends for the Europeans.
pvD = sum(DIV_AMT .* exp.(-r .* DIV_T))
P = Dict{String,Float64}()
for (call, nm) in ((true, "call"), (false, "put"))
    P["euro_$(nm)_nodiv_gbsm"] = gbsm(call, S, X, T, r, r, σ).value
    P["euro_$(nm)_nodiv_tree"] = tree(call, false; divs=false)
    P["amer_$(nm)_nodiv_tree"] = tree(call, true; divs=false)
    P["euro_$(nm)_div_tree"] = tree(call, false)
    P["amer_$(nm)_div_tree"] = tree(call, true)
    P["euro_$(nm)_div_gbsm_escrow"] = gbsm(call, S - pvD, X, T, r, r, σ).value
end
# "No dividends" is GBSM for the Europeans and the tree for the Americans, as
# the problem asks. The effect of the dividends is measured tree minus tree on
# the same grid, so the GBSM-vs-tree discretization gap does not leak into it.
prices = DataFrame(
    Option=["European call", "American call", "European put", "American put"],
    No_div=[P["euro_call_nodiv_gbsm"], P["amer_call_nodiv_tree"], P["euro_put_nodiv_gbsm"], P["amer_put_nodiv_tree"]],
    No_div_tree=[P["euro_call_nodiv_tree"], P["amer_call_nodiv_tree"], P["euro_put_nodiv_tree"], P["amer_put_nodiv_tree"]],
    Div_tree=[P["euro_call_div_tree"], P["amer_call_div_tree"], P["euro_put_div_tree"], P["amer_put_div_tree"]],
    GBSM_escrow=[P["euro_call_div_gbsm_escrow"], NaN, P["euro_put_div_gbsm_escrow"], NaN])
prices.Change = prices.Div_tree .- prices.No_div_tree
write_table(prices, "p2_prices";
            header=["Option", "No div.\\ (GBSM/tree)", "No div., tree", "Div., tree", "GBSM on \$S-PV(D)\$",
                    "Effect (tree)"])

# Early exercise premium, American minus European on the same tree and N, so
# the discretization error cancels out of the difference.
prem = DataFrame(Option=["Call", "Put"],
    No_div=[P["amer_call_nodiv_tree"] - P["euro_call_nodiv_tree"], P["amer_put_nodiv_tree"] - P["euro_put_nodiv_tree"]],
    Div=[P["amer_call_div_tree"] - P["euro_call_div_tree"], P["amer_put_div_tree"] - P["euro_put_div_tree"]])
write_table(prem, "p2_premium"; header=["Option", "No dividends", "With dividends"])

# --- (d) convergence in N ---------------------------------------------------------
Ns = collect(10:5:500)
conv = DataFrame(N=Ns,
                 Amer_call=[tree(true, true; N=n) for n in Ns],
                 Amer_put=[tree(false, true; N=n) for n in Ns],
                 Euro_call=[tree(true, false; N=n) for n in Ns],
                 Euro_put=[tree(false, false; N=n) for n in Ns])
for n in (1000, 1005)                            # two far points for the table only
    push!(conv, (n, tree(true, true; N=n), tree(false, true; N=n), NaN, NaN))
end
CSV.write(joinpath(OUT_TABLES, "p2_convergence_full.csv"), conv)
showN = [25, 50, 100, 105, 200, 205, 300, 305, 495, 500, 1000, 1005]
write_table(conv[in.(conv.N, Ref(showN)), [:N, :Amer_call, :Amer_put]], "p2_convergence";
            fmts=Dict("N" => "%.0f", "Amer_call" => "%.5f", "Amer_put" => "%.5f"),
            header=["\$N\$", "American call", "American put"], align="rrr")
sub = conv[conv.N .<= 500, :]
pc = plot(sub.N, sub.Amer_call, lw=1, label="", title="American call, with dividends", titlefontsize=9,
          xlabel="Steps N", ylabel="Tree value")
hline!(pc, [conv.Amer_call[end-1]], ls=:dot, color=:grey, label="N = 1000")
pp = plot(sub.N, sub.Amer_put, lw=1, label="", title="American put, with dividends", titlefontsize=9,
          xlabel="Steps N")
hline!(pp, [conv.Amer_put[end-1]], ls=:dot, color=:grey, label="N = 1000")
fig = plot(pc, pp, layout=(1, 2), size=(900, 340), legendfontsize=7, left_margin=4Plots.mm,
           bottom_margin=4Plots.mm)
savefig(fig, joinpath(OUT_FIGURES, "p2_convergence.pdf"))
savefig(fig, joinpath(OUT_FIGURES, "p2_convergence.png"))

# How much the prices and the premiums wobble over N = 400..500: the premium
# differences an American and a European tree on the same grid, so much of the
# oscillation should cancel.
late = conv[(conv.N .>= 400) .& (conv.N .<= 500), :]
swing(x) = maximum(x) - minimum(x)
swings = (call=swing(late.Amer_call), put=swing(late.Amer_put),
          prem_call=swing(late.Amer_call .- late.Euro_call), prem_put=swing(late.Amer_put .- late.Euro_put))

# --- (e) delta and gamma by central differences, N fixed ------------------------
greeks = DataFrame(Option=String[], Bump=Float64[], Delta=Float64[], Gamma=Float64[])
for (call, nm) in ((true, "American call"), (false, "American put"))
    for h in (1.0, 0.1, 0.01, 0.001)
        dl, gm = central_diff(s -> tree(call, true; s=s), S, h)
        push!(greeks, (nm, h, dl, gm))
    end
end
write_table(greeks, "p2_greeks"; fmts=Dict("Bump" => "%g", "Delta" => "%.5f", "Gamma" => "%.5f"),
            header=["Option", "Bump \$h\$", "\$\\Delta\$", "\$\\Gamma\$"])

# Value against S on a fine grid at fixed N, which is what (h) is about. The
# slope between neighbouring grid points (spacing 0.01) is the tree's local
# delta; where it jumps, the tree has a kink.
grid = range(S - 1.5, S + 1.5, length=301)
vals = [tree(true, true; s=s) for s in grid]
slope = diff(vals) ./ step(grid)
mid = (grid[1:end-1] .+ grid[2:end]) ./ 2
jumps = diff(slope)
step_idx = findall(abs.(jumps) .> 0.25 * maximum(abs.(jumps)))     # where the slope rises
big_at = mid[step_idx] .+ step(grid) / 2
# The rises come in clusters (one per terminal strike crossing, smeared by the
# non-recombining subtrees). Start a new cluster after a flat stretch > 0.1,
# then measure the spacing of the clusters and the flat piece S sits on.
starts = [1; findall(diff(big_at) .> 0.1) .+ 1]
ends = [starts[2:end] .- 1; length(big_at)]
big_gap = mean(diff(big_at[starts]))
flat_lo = maximum(big_at[ends][big_at[ends] .< S])
flat_hi = minimum(big_at[starts][big_at[starts] .> S])
p1 = plot(grid, vals, lw=1.2, label="", title="American call value, N = $N_MAIN", titlefontsize=9,
          xlabel="S", ylabel="Tree value")
p2 = plot(mid, slope, lw=1.2, label="", title="Slope between neighbouring points (local delta)",
          titlefontsize=9, xlabel="S")
for pl in (p1, p2)
    vline!(pl, [S - 1, S + 1], ls=:dash, color=:grey, lw=0.8, label="")
    vline!(pl, [S - 0.1, S + 0.1], ls=:dot, color=:firebrick, lw=0.8, label="")
end
fig = plot(p1, p2, layout=(1, 2), size=(900, 330), left_margin=5Plots.mm, bottom_margin=4Plots.mm)
savefig(fig, joinpath(OUT_FIGURES, "p2_kinks.pdf"))
savefig(fig, joinpath(OUT_FIGURES, "p2_kinks.png"))
nkinks = length(step_idx)

# --- (f) sensitivity to each dividend, one cent central difference --------------
sens = DataFrame(Option=String[], dV_dD1=Float64[], dV_dD2=Float64[])
for (call, am, nm) in ((true, false, "European call"), (true, true, "American call"),
                       (false, false, "European put"), (false, true, "American put"))
    d = Float64[]
    for j in 1:2
        up = copy(DIV_AMT); up[j] += 0.01
        dn = copy(DIV_AMT); dn[j] -= 0.01
        push!(d, (tree(call, am; amts=up) - tree(call, am; amts=dn)) / 0.02)
    end
    push!(sens, (nm, d[1], d[2]))
end
write_table(sens, "p2_div_sens"; header=["Option", "\$\\partial V/\\partial D_1\$", "\$\\partial V/\\partial D_2\$"])

# --- (g) critical price for exercising the call just before D2 ------------------
# No dividend follows D2, so from there the American call is worth the
# European one. Exercise is optimal when S - X ≥ C_GBSM(S - D2, X, T - t2).
t2, D2 = DIV_T[2], DIV_AMT[2]
hold2(s) = gbsm(true, s - D2, X, T - t2, r, r, σ).value
S_star = find_zero(s -> (s - X) - hold2(s), (X + 1e-9, 4X), Bisection())

# Same boundary read off the tree's own decision at the step-k2 nodes. Before
# D1 the tree recombines; after it each node starts its own subtree from
# S_k1 - D1, so the step-k2 prices are (S u^i d^(k1-i) - D1) u^j d^(m-j).
k1, k2 = div_steps(N_MAIN)
dt = T / N_MAIN
u = exp(σ * sqrt(dt)); d = 1 / u
m = k2 - k1
nodes_k1 = [S * u^i * d^(k1 - i) for i in 0:k1]
nodes_k2 = [(s1 - DIV_AMT[1]) * u^j * d^(m - j) for s1 in nodes_k1 for j in 0:m]
ex_k2 = [s - X > bt_american(true, s - D2, X, T - t2, r, r, σ, N_MAIN - k2) for s in nodes_k2]
tree_boundary = minimum(nodes_k2[ex_k2])
hold_k1 = [bt_american(true, s - DIV_AMT[1], X, T - DIV_T[1], r, [D2], [m], σ, N_MAIN - k1) for s in nodes_k1]
n_ex_k1 = count(nodes_k1 .- X .> hold_k1)

# Risk neutral probability, from the tree, of standing in the exercise region
# just before D2.
p = (exp(r * dt) - d) / (u - d)
w1 = [binomial(big(k1), i) * p^i * (1 - p)^(k1 - i) for i in 0:k1]
w2 = [binomial(big(m), j) * p^j * (1 - p)^(m - j) for j in 0:m]
prob_ex = Float64(sum(w1[a] * w2[c] * (nodes_k2[(a-1)*(m+1)+c] >= S_star) for a in 1:k1+1, c in 1:m+1))

# Benchmarks for (h): the closed form gamma on the escrowed price, and how far
# apart in S the terminal nodes' strike crossings are. Node i at maturity sits
# at S u^(2i-N); it crosses X at S = X u^(N-2i), so neighbouring crossings are
# a factor u² apart, about S(u² - 1) in price.
gamma_bench_call = gbsm(true, S - pvD, X, T, r, r, σ; includeGreeks=true).gamma
gamma_bench_put = gbsm(false, S - pvD, X, T, r, r, σ; includeGreeks=true).gamma
kink_spacing = S * (u^2 - 1)

# Hull's sharper test for the earlier dividend: with a later dividend still to
# come, exercising before D1 only gains the interest up to t2, not up to T.
int_to_next = X * (1 - exp(-r * (DIV_T[2] - DIV_T[1])))

if abspath(PROGRAM_FILE) == @__FILE__
    show(cond); println()
    for k in sort(collect(keys(P))); @printf("%-28s %.6f\n", k, P[k]); end
    show(prices, allcols=true); println()
    show(prem); println()
    show(conv[in.(conv.N, Ref(showN)), :], allrows=true); println()
    show(greeks, allrows=true); println()
    show(sens); println()
    @printf("PV(D) = %.6f, S - PV(D) = %.6f\n", pvD, S - pvD)
    @printf("S* = %.4f, lowest exercised step-%d node = %.4f, call nodes exercised before D1 = %d\n",
            S_star, k2, tree_boundary, n_ex_k1)
    @printf("P(in exercise region just before D2) = %.4f\n", prob_ex)
    @printf("slope rises in clusters starting at %s; mean cluster spacing %.3f vs S(u^2-1) = %.3f; S sits on a flat piece from %.2f to %.2f\n",
            join(round.(big_at[starts], digits=2), ", "), big_gap, kink_spacing, flat_lo, flat_hi)
end
