# =====================================================================
# Output artifacts, mirroring SubEpiPredict's output folder
# =====================================================================

_ensure(dir) = (isdir(dir) || mkpath(dir); dir)

"""
    output_dir(root, dataset, analysis)

`<root>/output/<dataset>/<analysis>`, created if needed. Dataset first, so a
new series never overwrites an old one.
"""
output_dir(root::AbstractString, dataset::AbstractString, analysis::AbstractString) =
    _ensure(joinpath(root, "output", dataset, analysis))

function _writecsv(path, header::Vector{String}, M::AbstractMatrix)
    open(path, "w") do io
        println(io, join(header, ","))
        writedlm(io, M, ',')
    end
    return path
end

"""
    save_forecast(dir, tag, t, point, curves)

Writes time, point forecast, and the 23 quantile levels used by computeWIS.m.
"""
function save_forecast(dir, tag, t::AbstractVector, point::AbstractVector,
                       curves::AbstractMatrix)
    _ensure(dir)
    qs = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400,
          0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900,
          0.950, 0.975, 0.990]
    Q = zeros(length(t), length(qs))
    for i in eachindex(t), (k, q) in enumerate(qs)
        Q[i, k] = quantile(@view(curves[i, :]), q)
    end
    header = vcat(["t", "point"], ["q" * string(q) for q in qs])
    return _writecsv(joinpath(dir, "forecast-$tag.csv"), header, hcat(t, point, Q))
end

"""
    save_parameters(dir, tag, boot, n)

Bootstrap parameter draws — the input to the parameter histograms.
"""
function save_parameters(dir, tag, boot::BootstrapResult, n::Integer)
    _ensure(dir)
    header = vcat(["r$i" for i in 1:n], ["p$i" for i in 1:n],
                  ["a$i" for i in 1:n], ["K$i" for i in 1:n],
                  ["gap$i" for i in 2:n])
    return _writecsv(joinpath(dir, "parameters-$tag.csv"), header, boot.thetas)
end

"""
    save_performance(dir, filename, rows)

`rows` is a vector of NamedTuples with identical fields, e.g. the output of
`performance` plus a model label.
"""
function save_performance(dir, filename, rows::Vector{<:NamedTuple})
    _ensure(dir)
    header = String.(collect(keys(first(rows))))
    M = permutedims(hcat([collect(values(r)) for r in rows]...))
    return _writecsv(joinpath(dir, filename), header, M)
end

"""
    save_fit(dir, tag, fit, boot)

Fitted mean, per-sub-epidemic profiles, and the 95% band over the calibration
period.
"""
function save_fit(dir, tag, fit::SubEpiFit, boot::BootstrapResult)
    _ensure(dir)
    nd = length(fit.t)
    lo = [quantile(@view(boot.fit_noisy[i, :]), 0.025) for i in 1:nd]
    hi = [quantile(@view(boot.fit_noisy[i, :]), 0.975) for i in 1:nd]
    header = vcat(["t", "y_obs", "fit", "lo95", "hi95"],
                  ["sub$i" for i in 1:fit.n])
    M = hcat(fit.t, fit.y, fit.fitted, lo, hi, fit.subcurves)
    return _writecsv(joinpath(dir, "fit-$tag.csv"), header, M)
end

"""
    save_rankings(dir, fits)

AICc table with relative likelihood and evidence ratio (paper Fig. 6).
"""
function save_rankings(dir, fits::Vector{SubEpiFit})
    _ensure(dir)
    a = [f.aicc for f in fits]
    amin = minimum(a)
    rel = exp.(-(a .- amin) ./ 2)
    M = hcat(collect(1:length(fits)), [float(f.n) for f in fits],
             [float(f.nparams) for f in fits], [f.fval for f in fits],
             a, rel, 1 ./ rel)
    return _writecsv(joinpath(dir, "AICc-topRanked.csv"),
                     ["rank", "nsub", "nparams", "objective", "AICc",
                      "rel_likelihood", "evidence_ratio"], M)
end

"""
    save_settings(dir, fit, cal; extra...)

Plain-text record of what produced a result.

The gate verdicts belong here as much as the parameters do: this project
produced two retracted findings from fits that had not converged, and a
metrics table with no record of whether the fit was gate-clean invites the
same mistake again.
"""
function save_settings(dir, fit::SubEpiFit, cal::CalibrationData; extra...)
    _ensure(dir)
    path = joinpath(dir, "settings.txt")
    open(path, "w") do io
        println(io, "SubFit run settings")
        println(io, "")
        println(io, "data")
        println(io, "  calibration points : ", length(cal.y))
        println(io, "  smoothing span     : ", cal.smoothfactor)
        println(io, "  window offset t0   : ", cal.t0,
                    "   # absolute onset = t0 + tau")
        println(io, "")
        println(io, "model")
        println(io, "  sub-epidemics n    : ", fit.n)
        println(io, "  growth kernel      : ", get(MODEL_NAMES, fit.flag, fit.flag))
        println(io, "  objective          : ", fit.method)
        println(io, "  incidence scale    : ", fit.incidence,
                    "   # :discrete matches the MATLAB toolbox and the paper's AICc")
        println(io, "  free parameters    : ", fit.nparams)
        println(io, "")
        println(io, "fit")
        println(io, "  objective value    : ", fit.fval)
        println(io, "  AICc               : ", fit.aicc)
        println(io, "  converged          : ", fit.converged)
        for i in 1:fit.n
            @printf(io, "  sub %d              : r=%.4f p=%.4f K=%.1f tau=%.2f (abs %.2f)\n",
                    i, fit.params.r[i], fit.params.p[i], fit.params.K[i],
                    fit.params.tau[i], cal.t0 + fit.params.tau[i])
        end
        println(io, "")
        println(io, "gates and options")
        for (k, v) in pairs(extra)
            println(io, "  ", rpad(string(k), 18), " : ", v)
        end
    end
    return path
end
