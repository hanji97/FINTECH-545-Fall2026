# Prices to returns.

"""
    return_calculate(prices; method="DISCRETE", dateColumn="date")

    DISCRETE    rₜ = Pₜ / Pₜ₋₁ - 1
    LOG         rₜ = ln(Pₜ / Pₜ₋₁)

`dateColumn` is carried through; every other column is treated as a price
series. The first row has no predecessor and is dropped, so the output is one
row shorter than the input.
"""
function return_calculate(prices::DataFrame; method::AbstractString="DISCRETE",
                          dateColumn::AbstractString="date")
    cols = names(prices)
    dateColumn in cols ||
        throw(ArgumentError("dateColumn \"$dateColumn\" is not in the DataFrame: $cols"))
    vars = Symbol.(filter(!=(dateColumn), cols))

    p = Matrix{Float64}(prices[!, vars])
    n = size(p, 1)
    ratio = p[2:n, :] ./ p[1:(n-1), :]

    m = uppercase(method)
    r = if m == "DISCRETE"
        ratio .- 1.0
    elseif m == "LOG"
        log.(ratio)
    else
        throw(ArgumentError("method \"$method\" must be one of (\"DISCRETE\", \"LOG\")"))
    end

    out = DataFrame(Symbol(dateColumn) => prices[2:n, Symbol(dateColumn)])
    for (j, v) in enumerate(vars)
        out[!, v] = r[:, j]
    end
    return out
end
