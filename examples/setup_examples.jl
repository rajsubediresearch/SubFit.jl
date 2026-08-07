# Run once:  julia --project=examples examples/setup_examples.jl
#
# Plots is heavy and stays OUT of the package: the core remains low-dependency
# and installable anywhere, while plotting lives here in its own environment.
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
Pkg.develop(path = normpath(joinpath(@__DIR__, "..")))
Pkg.add(["Plots", "Statistics", "Printf"])
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "1"
Pkg.precompile()
Pkg.status()
