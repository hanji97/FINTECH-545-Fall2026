# Problem 1: implied volatility from a dirty chain on XYZ.
#
#     julia --project=. problem1.jl
#
# S = 182.40, valued 2026-11-13, expiry 2026-12-18, r = 4.25%, continuous
# dividend yield q = 0.60%. Time is calendar days over 365. GBSM with cost of
# carry b = r - q (Merton 1973 in the Week 06 table).

include(joinpath(@__DIR__, "library", "RiskLib.jl"))
using Plots

const S = 182.40
const r = 0.0425
const q = 0.0060
const b = r - q
const T = Dates.value(Date(2026, 12, 18) - Date(2026, 11, 13)) / 365   # 35 days

chain = CSV.read(joinpath(@__DIR__, "data", "problem1.csv"), DataFrame)
chain.Mid = (chain.Bid .+ chain.Ask) ./ 2
calls = sort(chain[chain.Type .== "Call", :], :Strike)
puts = sort(chain[chain.Type .== "Put", :], :Strike)
@assert calls.Strike == puts.Strike
K = calls.Strike

# --- (a) put call parity on the mids -----------------------------------------
# With a continuous yield, parity reads C - P = S e^{-qT} - X e^{-rT}: the Week 06
# relation C + X e^{-rT} = P + S with S replaced by the prepaid forward. The
# gap is only tradeable beyond the sum of the two bid-ask spreads.
parity = DataFrame(Strike=K, C_mid=calls.Mid, P_mid=puts.Mid,
                   Theory=S * exp(-q * T) .- K .* exp(-r * T))
parity.Gap = parity.C_mid .- parity.P_mid .- parity.Theory
parity.Band = (calls.Ask .- calls.Bid) .+ (puts.Ask .- puts.Bid)
parity.Flag = ifelse.(abs.(parity.Gap) .> parity.Band, "yes", "")
write_table(parity, "p1_parity";
            fmts=Dict("Strike" => "%.0f", "C_mid" => "%.3f", "P_mid" => "%.3f", "Theory" => "%.3f",
                      "Gap" => "%.3f", "Band" => "%.2f"),
            header=["Strike", "\$C_{mid}\$", "\$P_{mid}\$", "\$Se^{-qT}-Xe^{-rT}\$", "Gap", "Band", "Flag"])
flagged = parity.Strike[parity.Flag .== "yes"]

# --- (d) implied vol from bid, mid and ask; (e) vega and the IV per cent ------
# Vega is taken at each quote's own mid IV. A one cent price change moves the
# IV by about 0.01 / vega (vega is per 1.00 of vol).
iv = DataFrame(Type=String[], Strike=Float64[], IV_Bid=Float64[], IV_Mid=Float64[], IV_Ask=Float64[],
               Vega=Float64[], dIV_per_cent=Float64[], Spread=Float64[])
nope = DataFrame(Type=String[], Strike=Float64[], Side=String[], Reason=String[])
for (typ, book) in (("Call", calls), ("Put", puts)), row in eachrow(book)
    call = typ == "Call"
    ivs = Dict{String,Float64}()
    for side in ("Bid", "Mid", "Ask")
        σ, why = implied_vol(call, row[side], S, row.Strike, T, r, b)
        ivs[side] = σ
        isempty(why) || push!(nope, (typ, row.Strike, side, why))
    end
    vega = gbsm(call, S, row.Strike, T, r, b, ivs["Mid"]; includeGreeks=true).vega
    push!(iv, (typ, row.Strike, ivs["Bid"], ivs["Mid"], ivs["Ask"], vega, 0.01 / vega, row.Ask - row.Bid))
end
tbl = copy(iv)
for c in (:IV_Bid, :IV_Mid, :IV_Ask, :dIV_per_cent)
    tbl[!, c] .*= 100                        # vol points
end
write_table(tbl, "p1_iv";
            fmts=Dict("Strike" => "%.0f", "IV_Bid" => "%.2f", "IV_Mid" => "%.2f", "IV_Ask" => "%.2f",
                      "Vega" => "%.2f", "dIV_per_cent" => "%.3f", "Spread" => "%.2f"),
            header=["Type", "Strike", "IV bid", "IV mid", "IV ask", "Vega", "\\(\\Delta\\)IV per 1c", "Spread"])
CSV.write(joinpath(OUT_TABLES, "p1_iv_failures.csv"), nope)

# --- (f) the flagged strike in vol points -------------------------------------
# Closing a parity gap g by moving one leg takes a vol change of about g/vega.
# At one strike and one vol the call and put vegas are equal, so this is also
# the call-put IV difference parity implies.
getiv(typ, k) = iv[(iv.Type .== typ) .& (iv.Strike .== k), :][1, :]
flag = DataFrame(Strike=Float64[], Gap=Float64[], Vega_call=Float64[], Vega_put=Float64[],
                 Volpts_call_vega=Float64[], Volpts_put_vega=Float64[], IV_call=Float64[],
                 IV_put=Float64[], IV_diff=Float64[], Parity_put=Float64[])
for k in flagged
    c, p = getiv("Call", k), getiv("Put", k)
    g = parity.Gap[parity.Strike .== k][1]
    push!(flag, (k, g, c.Vega, p.Vega, 100g / c.Vega, 100g / p.Vega, 100c.IV_Mid, 100p.IV_Mid,
                 100(c.IV_Mid - p.IV_Mid), gbsm(false, S, k, T, r, b, c.IV_Mid).value))
end
CSV.write(joinpath(OUT_TABLES, "p1_flagged.csv"), flag)

# --- plot: IV against strike, mids as lines, bid to ask as a band -------------
plt = plot(size=(760, 440), xlabel="Strike", ylabel="Implied volatility (%)", ylims=(10, 60),
           title="XYZ implied volatility by strike, 35 days to expiry", titlefontsize=10,
           legend=:topright, legendfontsize=7, grid=true, gridalpha=0.3)
for (typ, col, mk) in (("Call", :royalblue, :circle), ("Put", :firebrick, :square))
    sub = sort(iv[iv.Type .== typ, :], :Strike)
    lo = 100 .* sub.IV_Bid
    hi = 100 .* sub.IV_Ask
    ok = .!isnan.(lo) .& .!isnan.(hi)
    plot!(plt, sub.Strike[ok], 100 .* sub.IV_Bid[ok], fillrange=100 .* sub.IV_Ask[ok], fillalpha=0.15,
          linealpha=0, color=col, label="$typ IV, bid to ask")
    plot!(plt, sub.Strike, 100 .* sub.IV_Mid, color=col, marker=mk, markersize=3, lw=1.4,
          label="$typ IV, mid")
end
vline!(plt, [S], color=:black, ls=:dash, lw=0.8, label="spot 182.40")
for k in flagged
    vline!(plt, [k], color=:grey, ls=:dot, lw=1, label="parity flag, K = $(Int(k))")
end
savefig(plt, joinpath(OUT_FIGURES, "p1_smile.pdf"))
savefig(plt, joinpath(OUT_FIGURES, "p1_smile.png"))

if abspath(PROGRAM_FILE) == @__FILE__
    @printf("T = %.6f years\n", T)
    show(parity, allrows=true); println()
    show(tbl, allrows=true); println()
    show(nope, allrows=true); println()
    show(flag, allcols=true); println()
end
