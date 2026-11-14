# Output helpers. Every table in the write-up is written from here, as a CSV and
# as a LaTeX tabular that the report \input's, so the PDF cannot drift from the
# code.

const OUT_TABLES = joinpath(@__DIR__, "..", "output", "tables")
const OUT_FIGURES = joinpath(@__DIR__, "..", "output", "figures")
mkpath(OUT_TABLES)
mkpath(OUT_FIGURES)

# Numbers go in math mode so a leading "-" prints as a minus sign. In text
# cells such as "-0.004 (0.028)" a "-" before a digit at the start or after a
# space or "(" is a minus too; dates like 2008-06-30 are left alone.
_cell(x::AbstractString, fmt) = replace(x, r"(^|[\s(])-(?=\d)" => s"\1$-$", r"(?<!\\)%" => "\\%")
_cell(x::Missing, fmt) = "--"
function _cell(x::Real, fmt)
    isnan(x) && return "--"
    s = Printf.format(Printf.Format(fmt), x)
    s = replace(s, r"^-(0(\.0+)?)$" => s"\1")      # a value that rounds to zero prints as 0, not -0
    return "\$" * s * "\$"
end

"""
    write_table(df, name; fmts=Dict(), header=nothing, align=nothing)

Write `output/tables/<name>.csv` and `<name>.tex` (a booktabs tabular, no
float, so the report sets the caption). `fmts` maps a column name to a printf
format; the default is "%.4f".
"""
function write_table(df::DataFrame, name::AbstractString; fmts=Dict{String,String}(),
                     header=nothing, align=nothing)
    CSV.write(joinpath(OUT_TABLES, name * ".csv"), df)
    cols = names(df)
    header = header === nothing ? replace.(cols, "_" => "\\_") : header
    align = align === nothing ? "l" * "r"^(length(cols) - 1) : align
    io = IOBuffer()
    println(io, "\\begin{tabular}{$align}")
    println(io, "\\toprule")
    println(io, join(header, " & "), " \\\\")
    println(io, "\\midrule")
    for row in eachrow(df)
        println(io, join([_cell(row[c], get(fmts, c, "%.4f")) for c in cols], " & "), " \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    write(joinpath(OUT_TABLES, name * ".tex"), String(take!(io)))
    return nothing
end
