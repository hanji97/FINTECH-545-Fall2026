# Reproduces every number, table and figure in the write-up.
#
#     julia --project=. run_all.jl
#
# Runs the library checks first and stops if any fails, then each problem in
# turn. Outputs go to output/tables and output/figures. Problem 2 is the slow
# one (about 4 minutes) because of the convergence sweep over N.

cd(@__DIR__)
run(`$(Base.julia_cmd()) --project=. tests/run_course_tests.jl`)
for p in ("problem1.jl", "problem2.jl", "problem3.jl", "problem4.jl", "problem5.jl")
    println("\n==== ", p, " ====")
    t = @elapsed run(`$(Base.julia_cmd()) --project=. $p`)
    println(p, " finished in ", round(t, digits=1), " s")
end
