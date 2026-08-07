# Run once:  julia --project=. setup.jl
#
# All dependencies are added in a SINGLE Pkg.add call with auto-precompile
# switched off. Adding them one at a time makes Pkg try to precompile SubFit
# after every add, which fails on the next not-yet-added `using` and prints a
# stack trace each time -- lots of noise that looks like an infinite loop.
using Pkg
Pkg.activate(@__DIR__)

old = get(ENV, "JULIA_PKG_PRECOMPILE_AUTO", nothing)
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
try
    Pkg.add(["NLopt", "OrdinaryDiffEqTsit5", "SciMLBase",
             "DelimitedFiles", "LinearAlgebra", "Logging",
             "Printf", "Random", "Statistics", "Test"])
finally
    old === nothing ? delete!(ENV, "JULIA_PKG_PRECOMPILE_AUTO") :
                      (ENV["JULIA_PKG_PRECOMPILE_AUTO"] = old)
end

Pkg.precompile()
Pkg.status()
