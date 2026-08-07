# =====================================================================
# Ranking across the number of sub-epidemics, and ensemble weighting
# =====================================================================

"""
    rank_models(t, y; nmax=2, topmodels=4, kwargs...)

Fit n = 1..nmax and return the fits sorted by AICc (best first), truncated to
`topmodels`.

Note the structural difference from SubEpiPredict: there, the ranked models
are *different C_thr values at possibly the same n*, so a "top 4" can contain
four two-sub-epidemic models that differ only in onset threshold. Here the
candidate set is indexed by n alone, so `topmodels` is capped at `nmax`. That
is a real reduction in the ensemble's diversity and should be reported
honestly in any comparison -- if diversity turns out to matter, the fix is to
ensemble over bootstrap-perturbed refits or over growth-model kernels, not to
reintroduce the grid.
"""
function rank_models(t::AbstractVector, y::AbstractVector;
                     nmax::Integer=2, topmodels::Integer=4, kwargs...)
    fits = SubEpiFit[]
    for n in 1:nmax
        push!(fits, fit_subepidemic(t, y, n; kwargs...))
    end
    sort!(fits, by = f -> f.aicc)
    return fits[1:min(topmodels, length(fits))]
end

"""
    ensemble_weights(aiccs; scheme=:akaike)

`:equal`       -> unweighted            (weight_type1 = -1)
`:aicc_recip`  -> proportional to 1/AICc (weight_type1 = 0, paper Eq. 19)
`:akaike`      -> relative likelihood    (weight_type1 = 1, paper Eq. 21)

`:aicc_recip` is provided only for reproducing the MATLAB behaviour. It is not
a defensible weighting rule -- weights proportional to the reciprocal of an
information criterion do not estimate anything, and the value is not even
invariant to adding a constant to the log-likelihood. Prefer `:akaike`, or
stack on out-of-sample WIS.
"""
function ensemble_weights(aiccs::AbstractVector; scheme::Symbol=:akaike)
    I = length(aiccs)
    if scheme === :equal
        return fill(1 / I, I)
    elseif scheme === :aicc_recip
        w = 1 ./ aiccs
        return w ./ sum(w)
    elseif scheme === :akaike
        amin = minimum(aiccs)
        l = exp.(-(aiccs .- amin) ./ 2)
        return l ./ sum(l)
    else
        error("unknown weighting scheme: $scheme")
    end
end

"""
    ensemble_curves(curveset, weights)

Weighted combination of per-model predictive sample matrices. Each element of
`curveset` is `ntime x B`; replicate b of the ensemble is the weighted average
of replicate b across models (the wild-bootstrap pairing used in the paper).
"""
function ensemble_curves(curveset::Vector{<:AbstractMatrix}, weights::AbstractVector)
    B = minimum(size(c, 2) for c in curveset)
    nt = size(first(curveset), 1)
    out = zeros(Float64, nt, B)
    for (w, c) in zip(weights, curveset)
        out .+= w .* c[:, 1:B]
    end
    return out
end

"""
    stack_weights(curveset, y_holdout; iters=2000, seed=1)

Simplex weights minimizing out-of-sample WIS on a holdout window, by random
search on the simplex. This is the principled replacement for AICc weights:
it optimizes the thing you actually report. Crude but adequate for I <= 6
models; swap in a proper solver if the model set grows.
"""
function stack_weights(curveset::Vector{<:AbstractMatrix}, y_holdout::AbstractVector;
                       iters::Integer=2000, seed::Integer=1)
    I = length(curveset)
    rng = Xoshiro(seed)
    best_w = fill(1 / I, I)
    best = wis(ensemble_curves(curveset, best_w), y_holdout)
    for _ in 1:iters
        w = rand(rng, I)
        w ./= sum(w)
        s = wis(ensemble_curves(curveset, w), y_holdout)
        if s < best
            best, best_w = s, w
        end
    end
    return best_w, best
end
