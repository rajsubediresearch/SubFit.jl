# =====================================================================
# Data loading and preprocessing, matched to SubEpiPredict conventions
# =====================================================================

"""
    movmean(x, k)

Centered moving average with SHRINKING windows at the endpoints.

This reproduces MATLAB's `smoothdata(x,'movmean',k)` exactly, including its
endpoint behaviour (MATLAB averages over the truncated window rather than
padding or returning NaN). A naive centred rolling mean will NOT match.
"""
function movmean(x::AbstractVector{<:Real}, k::Integer)
    n = length(x)
    out = Vector{Float64}(undef, n)
    if k <= 1
        out .= float.(x)
        return out
    end
    hl = (k - 1) ÷ 2
    hr = k ÷ 2
    @inbounds for i in 1:n
        lo = max(1, i - hl)
        hi = min(n, i + hr)
        s = 0.0
        for j in lo:hi
            s += x[j]
        end
        out[i] = s / (hi - lo + 1)
    end
    return out
end

"""
    load_series(path; column, cumulative=true)

Read a SubEpiPredict-format whitespace-delimited `.txt` file and return the
incidence series for `column` (1-based; e.g. 52 = USA national in the COVID
death files). When `cumulative=true` the series is differenced, keeping the
first value as-is — matching `getData.m`'s `[data1(1); diff(data1)]`.
"""
function load_series(path::AbstractString; column::Integer, cumulative::Bool=true)
    M = readdlm(path)
    y = Float64.(M[:, column])
    return cumulative ? vcat(y[1], diff(y)) : y
end

"""
    CalibrationData

`t` is 0-based time WITHIN THE WINDOW, `y_raw` the observed incidence over the
window, `y` the smoothed series used for fitting.

`t0` is the offset of the window's first point from the start of the full
series, and it exists because leaving it implicit caused a false result.

`calibration` keeps the LAST `calibration_period` points, so shortening the
window moves its START while the end stays put. Since `t` restarts at 0 each
time, a perfectly STATIONARY onset appears to shift by exactly one day per day
of window change. That produced an apparent slope of 1.0 and the conclusion
that the second sub-epidemic was a window-relative tail smoother. In absolute
time the same five fits put the onset at day 45.06, 45.23, 45.24, 45.32 and
45.81 -- mean 45.33, SD 0.28 days, one of the best-identified quantities in the
whole analysis.

ALWAYS compare onsets in absolute time: `t0 + tau`. Use [`absolute_onset`](@ref).
"""
struct CalibrationData
    t::Vector{Float64}
    y_raw::Vector{Float64}
    y::Vector{Float64}
    smoothfactor::Int
    t0::Float64
end

"""
    calibration(y_raw; smoothfactor=7, calibration_period=nothing)

Take the most recent `calibration_period` points (capped at the length of the
series, as `options.m` does) and apply the moving-average smoother.
"""
function calibration(y_raw::AbstractVector{<:Real};
                     smoothfactor::Integer=7,
                     calibration_period::Union{Nothing,Integer}=nothing)
    n = length(y_raw)
    m = calibration_period === nothing ? n : min(Int(calibration_period), n)
    yr = Float64.(y_raw[(n - m + 1):n])
    ys = movmean(yr, smoothfactor)
    return CalibrationData(collect(0.0:(m - 1)), yr, ys, Int(smoothfactor),
                           float(n - m))
end

"""
    absolute_onset(cal, fit, i)

Onset of sub-epidemic `i` on the FULL series' time axis: `cal.t0 + tau_i`.

Window-relative onsets from different calibration windows are not comparable.
Anything that varies the window -- boundary checks, window-shift gates,
rolling origins -- must convert first.
"""
absolute_onset(cal::CalibrationData, f, i::Integer) = cal.t0 + f.params.tau[i]
