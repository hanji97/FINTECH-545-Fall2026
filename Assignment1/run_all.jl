for f in ("problem1.jl", "problem2.jl", "problem3.jl", "problem4.jl", "problem5.jl")
    include(joinpath(@__DIR__, f))
    println()
end
