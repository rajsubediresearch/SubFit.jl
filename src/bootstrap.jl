# =====================================================================
# Parametric bootstrap and forecasting
# =====================================================================

# Minimal Poisson sampler (Knuth for small mu, normal approx for large)
# so the package does not need a Distributions.jl dependency.
struct Poisson; lambda::Float64; end
function Base.rand(rng::Random.AbstractRNG, d::Poisson)
    l = d.lambda
    if l < 30
        L = exp(-l); k = 0; p = 1.0
        while true
            p *= rand(rng)
            p <= L && return k
            k += 1
            k > 10_000 && return k
        end
    else
        return max(0, round(Int, l + sqrt(l) * randn(rng)))
    end
end

"""
    add_error(mu, dist, factor, rng)

Generate one noisy realization of the mean curve `mu`.

Replicates `AddPoissonError.m`, including its quirk that the FIRST point is
never perturbed (`yirData(1)=yi(1)`). That quirk is preserved for
comparability, but it is a plausible contributor to under-nominal interval
coverage and is worth testing with `first_point_fixed=false`.
"""
function add_error(mu::AbstractVector, dist::Symbol, factor::Float64, rng;
                   first_point_fixed::Bool=true)
    out = Vector{Float64}(undef, length(mu))
    out[1] = mu[1]
    for i in eachindex(mu)
        (i == 1 && first_point_fixed) && continue
        m = abs(mu[i])
        if dist === :normal
            out[i] = m + factor * randn(rng)
        elseif dist === :poisson
            out[i] = m <= 0 ? 0.0 : float(rand(rng, Poisson(m)))
        else
            error("unsupported bootstrap distribution: $dist")
        end
    end
    return out
end

"""
    BootstrapResult

`thetas` is `B x nparams`; `fit_curves` and `forecast_curves` are the model
trajectories per replicate (no observation noise); `*_noisy` add it back.
"""
struct BootstrapResult
    thetas::Matrix{Float64}
    fit_curves::Matrix{Float64}
    fit_noisy::Matrix{Float64}
    t_full::Vector{Float64}
    nsuccess::Int
end

"""
    run_bootstrap(fit, cal; B=300, horizon=0, dist=:normal, seed=...)

Parametric bootstrap. Each replicate gets its own `Xoshiro` stream derived
from `seed` and the replicate index, so results depend on the seed and NOT on
the thread count.
"""
function run_bootstrap(fit::SubEpiFit; B::Integer=300, horizon::Integer=0,
                       dist::Symbol=:normal, seed::Integer=20260101,
                       y_obs::Union{Nothing,AbstractVector}=nothing,
                       nstarts::Integer=2, maxeval::Integer=2000,
                       first_point_fixed::Bool=true, threaded::Bool=true)

    t = fit.t
    nd = length(t)
    dt = nd > 1 ? t[2] - t[1] : 1.0
    t_full = horizon > 0 ? vcat(t, t[end] .+ dt .* collect(1.0:horizon)) : copy(t)

    # Noise scale, matching fittingModifiedLogisticFunctionPatchMultiple.m:218
    #     var1 = sum((bestfit-data).^2) ./ (length(bestfit)-numparams)
    # Two details that matter and are easy to get wrong:
    #   * `data` there is the RAW series, not the smoothed one the model was
    #     fitted to. Using smoothed residuals makes the noise ~4x too small on
    #     the COVID example (66 vs 250) and craters interval coverage.
    #   * the denominator is (nd - numparams), not nd.
    yref = y_obs === nothing ? fit.y : y_obs
    @assert length(yref) == nd "y_obs must have the same length as the calibration period"
    dof = max(nd - fit.nparams, 1)
    resid = fit.fitted .- yref
    factor = dist === :normal ? max(sqrt(sum(abs2, resid) / dof), 1e-8) : 1.0

    lb, ub = param_bounds(fit.n, fit.y, fit.flag;
                          Kmax_mult=10.0, min_gap=1.0, tend=float(t[end]))
    free = [i for i in eachindex(lb) if ub[i] > lb[i]]

    thetas = zeros(Float64, B, length(fit.theta))
    curves = zeros(Float64, length(t_full), B)
    noisy  = zeros(Float64, length(t_full), B)
    ok = falses(B)

    work = function (b::Int)
        rng = Xoshiro(hash((seed, b)))
        ysim = add_error(fit.fitted, dist, factor, rng;
                         first_point_fixed=first_point_fixed)
        fobj = th -> objective(th, fit.n, t, ysim, fit.flag, fit.I0, fit.method;
                               incidence=fit.incidence)
        th, fv, _ = _run_nlopt(:LN_COBYLA, fit.theta, lb, ub, free, fobj, maxeval)
        isfinite(fv) || return
        P = unpack(th, fit.n)
        yh, _ = simulate(P, t_full, fit.flag, fit.I0; incidence=fit.incidence)
        any(isnan, yh) && return
        thetas[b, :] .= th
        curves[:, b] .= yh
        noisy[:, b]  .= add_error(yh, dist, factor, Xoshiro(hash((seed, b, :obs)));
                                  first_point_fixed=first_point_fixed)
        ok[b] = true
        return
    end

    if threaded && Threads.nthreads() > 1
        Threads.@threads for b in 1:B
            work(b)
        end
    else
        for b in 1:B
            work(b)
        end
    end

    keep = findall(ok)
    return BootstrapResult(thetas[keep, :], curves[:, keep], noisy[:, keep],
                           t_full, length(keep))
end

"""
    point_forecast(fit, horizon)

Model trajectory over calibration + forecast horizon at the point estimate.
"""
function point_forecast(fit::SubEpiFit, horizon::Integer)
    dt = length(fit.t) > 1 ? fit.t[2] - fit.t[1] : 1.0
    t_full = horizon > 0 ? vcat(fit.t, fit.t[end] .+ dt .* collect(1.0:horizon)) : copy(fit.t)
    yh, sub = simulate(fit.params, t_full, fit.flag, fit.I0; incidence=fit.incidence)
    return t_full, yh, sub
end
