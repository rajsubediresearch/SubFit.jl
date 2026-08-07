# =====================================================================
# Calibration / forecasting performance metrics
# =====================================================================

# Exactly the 11 levels used in computeWIS.m: [0.02 0.05 0.1:0.1:0.9]
const WIS_ALPHAS = vcat([0.02, 0.05], collect(0.1:0.1:0.9))

mae(yhat, y) = sum(abs, yhat .- y) / length(y)
mse(yhat, y) = sum(abs2, yhat .- y) / length(y)

"""
    coverage(curves, y; level=0.95)

Fraction of observations inside the pointwise prediction interval, where
`curves` is `ntime x nreplicates`.
"""
function coverage(curves::AbstractMatrix, y::AbstractVector; level::Float64=0.95)
    a = (1 - level) / 2
    c = 0
    for i in eachindex(y)
        row = @view curves[i, :]
        lo = quantile(row, a)
        hi = quantile(row, 1 - a)
        (y[i] >= lo && y[i] <= hi) && (c += 1)
    end
    return 100 * c / length(y)
end

"""
    wis(curves, y)

Weighted interval score (Bracher et al. 2021), averaged over time points.
`curves` is `ntime x nreplicates` of predictive samples.
"""
function wis(curves::AbstractMatrix, y::AbstractVector)
    K = length(WIS_ALPHAS)
    total = 0.0
    for i in eachindex(y)
        row = @view curves[i, :]
        med = quantile(row, 0.5)
        s = 0.5 * abs(y[i] - med)
        for a in WIS_ALPHAS
            l = quantile(row, a / 2)
            u = quantile(row, 1 - a / 2)
            IS = (u - l) + (2 / a) * (l - y[i]) * (y[i] < l) +
                 (2 / a) * (y[i] - u) * (y[i] > u)
            s += (a / 2) * IS
        end
        total += s / (K + 0.5)
    end
    return total / length(y)
end

"""
    performance(curves, point, y)

Bundle of MAE, MSE, 95% coverage and WIS. `point` is the mean/point forecast
(the model trajectory WITHOUT observation noise), `curves` the predictive
sample matrix used for the interval-based scores.
"""
function performance(curves::AbstractMatrix, point::AbstractVector, y::AbstractVector)
    return (MAE = mae(point, y), MSE = mse(point, y),
            Coverage95 = coverage(curves, y), WIS = wis(curves, y))
end
