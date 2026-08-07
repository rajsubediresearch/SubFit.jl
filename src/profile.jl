# =====================================================================
# Profile likelihood over the onset, and the smoothness gate
#
# THE RULE: a fit does not get interpreted until its profile is smooth.
#
# Adopted 2026-08-02 after two results in a row turned out to be optimizer
# artifacts rather than findings:
#   * the national "overfitting" result -- the free fit sat at SSE 302,963
#     while pinning tau_2 = 22 reached 162,069
#   * the state replication -- 33 of 35 states pinned tau_2 at its lower
#     bound, and state 1 returned an identical objective at all 60 grid
#     points, meaning the optimizer never moved
#
# A true profile likelihood is smooth: each interior point is no lower than
# its neighbours suggest. A point sitting far BELOW both neighbours means the
# neighbours' optimizations failed, not that the objective has a spike there.
# `profile_dip` measures exactly that, and `is_smooth_profile` is the gate.
# =====================================================================

"""
    profile_tau2(t, y; grid, kwargs...)

Profile the objective over the second onset gap: pin tau_2 at each grid value,
optimize every other parameter, and record the best objective.

Runs three passes:
  1. COLD  -- multistart at every grid point, independently (threaded).
  2. FORWARD warm sweep  -- each point additionally optimized starting from the
     previous point's solution.
  3. BACKWARD warm sweep -- the same, right to left.

The warm sweeps matter. A profile point whose optimization fails is usually
rescued by starting from its neighbour's answer, because adjacent points have
adjacent optima. This both REPAIRS the profile and measures how unreliable the
cold search was: `warm_gain` is the largest relative improvement the sweeps
found over the cold pass. A large `warm_gain` means the plain fit cannot be
trusted on this data, which is precisely the situation that produced the
retracted national result (cold SSE 302,963 vs 162,069 reachable by pinning).

Returns `(tau, objective, theta, cold, warm_gain)`.
"""
function profile_tau2(t::AbstractVector, y::AbstractVector;
                      grid::AbstractVector=1.0:1.0:float(t[end]),
                      flag::Int=GLM, method::Symbol=:normal, I0::Real=y[1],
                      incidence::Symbol=:analytic, Kmax_mult::Float64=10.0,
                      min_gap::Float64=1.0, nstarts::Integer=6,
                      maxeval::Integer=3000, seed::Integer=20260101,
                      threaded::Bool=true, sweeps::Bool=true,
                      r_bounds=nothing, matlab_r_bounds::Bool=false)
    n = 2
    tend = float(t[end])
    lb0, ub0 = param_bounds(n, y, flag; Kmax_mult=Kmax_mult, min_gap=min_gap,
                             tend=tend, r_bounds=r_bounds, matlab_r_bounds=matlab_r_bounds)
    fobj = th -> objective(th, n, t, y, flag, I0, method; incidence=incidence)
    starts = initial_points(n, t, y, lb0, ub0; nstarts=nstarts, seed=seed)
    gidx = 4n + 1

    taus = collect(float.(grid))
    np = length(taus)
    vals = fill(Inf, np)
    thetas = zeros(Float64, np, length(lb0))

    # bounds with tau_2 pinned at grid point i
    pinned(i) = begin
        lb, ub = copy(lb0), copy(ub0)
        g = clamp(taus[i], lb0[gidx], ub0[gidx])
        lb[gidx] = g; ub[gidx] = g
        lb, ub, [k for k in eachindex(lb) if ub[k] > lb[k]], g
    end

    # ---- pass 1: cold multistart ----
    cold_work = function (i::Int)
        lb, ub, free, g = pinned(i)
        best_th, best_f = starts[1], Inf
        for x0 in starts
            z = copy(x0); z[gidx] = g
            th, f, _ = _run_nlopt(:LN_SBPLX, z, lb, ub, free, fobj, maxeval)
            f < best_f && ((best_th, best_f) = (th, f))
        end
        th, f, _ = _run_nlopt(:LN_COBYLA, best_th, lb, ub, free, fobj, maxeval)
        f < best_f && ((best_th, best_f) = (th, f))
        vals[i] = best_f
        thetas[i, :] .= best_th
        return
    end
    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:np
            cold_work(i)
        end
    else
        for i in 1:np
            cold_work(i)
        end
    end
    cold = copy(vals)

    # ---- passes 2 and 3: warm sweeps (sequential by construction) ----
    if sweeps && np > 1
        for order in (2:np, (np - 1):-1:1)
            for i in order
                lb, ub, free, g = pinned(i)
                nb = i == 1 ? 2 : (order === 2:np ? i - 1 : i + 1)
                x0 = vec(thetas[nb, :]); x0[gidx] = g
                for alg in (:LN_SBPLX, :LN_COBYLA)
                    th, f, _ = _run_nlopt(alg, x0, lb, ub, free, fobj, maxeval)
                    if f < vals[i]
                        vals[i] = f
                        thetas[i, :] .= th
                    end
                    x0 = th
                end
            end
        end
    end

    finite = [i for i in 1:np if isfinite(cold[i]) && cold[i] > 0]
    warm_gain = isempty(finite) ? 0.0 :
                maximum((cold[i] - vals[i]) / cold[i] for i in finite)

    return (tau = taus, objective = vals, theta = thetas,
            cold = cold, warm_gain = warm_gain)
end

"""
    profile_dip(objective)

Largest relative amount by which an interior point OTHER THAN THE GLOBAL
MINIMUM falls below both of its neighbours:
`max_i (min(f[i-1], f[i+1]) - f[i]) / f[i]`, skipping `argmin(f)`.

Excluding the global minimum is essential and was missed in the first version:
on any convex profile the true minimum sits below both neighbours by an amount
set purely by curvature and grid spacing, so including it flags every
well-behaved profile as broken. What a real profile does NOT have is a SECOND
deep local minimum -- that means the optimization failed somewhere.

Use together with `warm_gain` from `profile_tau2`: this catches leftover
multimodality, `warm_gain` catches an unreliable cold search.
"""
function profile_dip(objective::AbstractVector)
    n = length(objective)
    n < 3 && return 0.0
    gmin = argmin(objective)
    d = 0.0
    for i in 2:(n - 1)
        i == gmin && continue
        f = objective[i]
        (isfinite(f) && f > 0) || continue
        nb = min(objective[i - 1], objective[i + 1])
        isfinite(nb) || continue
        d = max(d, (nb - f) / f)
    end
    return d
end

"""
    is_smooth_profile(objective; tol=0.10)

The gate. `true` when no interior point sits more than `tol` (relative) below
both of its neighbours — i.e. the search is behaving consistently across the
profile and the fit is safe to interpret.
"""
is_smooth_profile(objective::AbstractVector; tol::Float64=0.10) =
    profile_dip(objective) <= tol

"""
    profile_interval(taus, objective, nd; level=0.95)

Profile-likelihood interval for tau_2 under Gaussian errors:
`nd * log(SSE/SSE_min) <= chi2(1, level)`, i.e. `SSE <= SSE_min*exp(chi2/nd)`.
Returns `(lo, hi, threshold, npoints)`.
"""
function profile_interval(taus::AbstractVector, objective::AbstractVector,
                          nd::Integer; level::Float64=0.95)
    chi2 = level == 0.95 ? 3.841 : level == 0.99 ? 6.635 : 3.841
    smin = minimum(objective)
    thr = smin * exp(chi2 / nd)
    sel = taus[objective .<= thr]
    return (lo = minimum(sel), hi = maximum(sel), threshold = thr,
            npoints = length(sel))
end

# =====================================================================
# The effort-stability gate
#
# `profile_dip` asks whether a profile is locally self-consistent. It cannot
# detect a profile that is UNIFORMLY stuck: no spikes, no second minimum, gate
# green, every point equally suboptimal. That is exactly what happened on the
# national COVID series -- dip 0.001, gate PASS, and the minimum had just moved
# 22 days and improved 28% relative to the previous search.
#
# The property every failure in this project shares is different: searching
# harder kept finding better answers. 302,963 -> 222,240 -> 162,069 -> 117,463,
# all on the same data and objective. So the gate that matters is whether the
# answer STOPS MOVING as effort increases.
# =====================================================================

"""
    fit_stability(t, y, n; levels, tol=0.02, kwargs...)

Refit at increasing search effort and report whether the answer settles.

`levels` is a vector of `(tau_grid, nstarts)` pairs in increasing order of
effort. Returns `(levels, objective, best, improvement, regression, fit, stable)`:

  * `objective`   — raw objective at each level
  * `best`        — running best objective up to each level
  * `improvement` — relative gain of the running best over the previous level
  * `regression`  — how far each level fell BACK above the running best before
                    it. A later level doing WORSE is instability, not
                    convergence: the first version of this gate accepted
                    `improvement = -1.3%` because -0.013 <= 0.02, which passed
                    a fit that had got worse with more effort.
  * `fit`         — the best fit found across all levels, not the last one

`stable` requires both that the running best has stopped improving and that no
level regressed by more than `tol`.
"""
function fit_stability(t::AbstractVector, y::AbstractVector, n::Integer;
                       levels=[(10, 6), (20, 12), (40, 24)],
                       tol::Float64=0.02, kwargs...)
    fits = [fit_subepidemic(t, y, n; tau_grid=g, nstarts=ns, kwargs...)
            for (g, ns) in levels]
    objs = [f.fval for f in fits]
    best = accumulate(min, objs)
    imp = [i == 1 ? NaN : (best[i - 1] - best[i]) / best[i - 1] for i in eachindex(objs)]
    reg = [i == 1 ? 0.0 : max(0.0, (objs[i] - best[i - 1]) / best[i - 1])
           for i in eachindex(objs)]
    stable = length(objs) >= 2 && isfinite(imp[end]) &&
             imp[end] <= tol && maximum(reg) <= tol
    return (levels = levels, objective = objs, best = best, improvement = imp,
            regression = reg, fit = fits[argmin(objs)], stable = stable)
end

"""
    is_stable_fit(stability; tol=0.02)

Gate over [`fit_stability`](@ref): the running best must have stopped improving
AND no level may have regressed above the running best by more than `tol`.
"""
function is_stable_fit(st; tol::Float64=0.02)
    length(st.objective) >= 2 || return false
    isfinite(st.improvement[end]) || return false
    return st.improvement[end] <= tol && maximum(st.regression) <= tol
end

"""
    bound_hits(fit; Kmax_mult=10.0, min_gap=1.0, rtol=1e-3)

Names of fitted parameters sitting on (or within `rtol` of the span of) their
bounds, e.g. `["gap2@lb"]`.

THE GATE THAT WAS MISSING. In the state replication tau_2 came back as exactly
1.00 -- the value of `min_gap` -- in 33 of 35 states, across two separate runs,
and neither the smoothness gate nor the stability gate flagged it: a search
that lands in the same wrong place at every effort level looks perfectly
settled. A parameter pinned to its bound means either the optimum lies outside
the feasible region or the search never left its start. Either way the fit is
not interpretable, and this is the cheapest possible way to detect it.

`p` at its upper bound of 1 is NOT flagged by default (see
`bound_gate_prefixes` in [`gated_fit`](@ref)): p = 1 is exponential early
growth, a meaningful value that happens to sit at the edge of the model.
"""
function bound_hits(f::SubEpiFit; Kmax_mult::Float64=10.0, min_gap::Float64=1.0,
                    rtol::Float64=1e-3)
    lb, ub = param_bounds(f.n, f.y, f.flag; Kmax_mult=Kmax_mult,
                          min_gap=min_gap, tend=f.t[end])
    names = vcat(["r$i" for i in 1:f.n], ["p$i" for i in 1:f.n],
                 ["a$i" for i in 1:f.n], ["K$i" for i in 1:f.n],
                 ["gap$i" for i in 2:f.n])
    hits = String[]
    for k in eachindex(f.theta)
        (k <= length(names) && ub[k] > lb[k]) || continue
        span = ub[k] - lb[k]
        if f.theta[k] <= lb[k] + rtol * span
            push!(hits, names[k] * "@lb")
        elseif f.theta[k] >= ub[k] - rtol * span
            push!(hits, names[k] * "@ub")
        end
    end
    return hits
end


"""
    gated_fit(t, y, n; do_profile=true, ...)

Fit, then apply both gates before handing the result back.

Returns `(fit, pass, dip, warm_gain, stability, reason)`. `pass` is `false`
whenever the fit should NOT be interpreted, with `reason` naming which gate
failed. Callers are expected to skip failing series and report how many were
skipped — a high skip rate is itself a finding, not an inconvenience.

`do_profile=false` runs the stability gate only. Profiling costs a full sweep
over the onset grid, which is too slow when fitting dozens of series; the
stability gate is the cheaper of the two and catches the failure mode that
matters most (a search that has not finished).

For `n == 1` there is no onset, so the profile gate is skipped automatically.
"""
function gated_fit(t::AbstractVector, y::AbstractVector, n::Integer;
                   do_profile::Bool=true,
                   profile_grid::AbstractVector=1.0:2.0:float(t[end]),
                   stability_levels=[(10, 6), (20, 12), (40, 24)],
                   tol_dip::Float64=0.10, tol_stab::Float64=0.02,
                   bound_gate_prefixes=["gap", "K", "r"],
                   Kmax_mult::Float64=10.0, min_gap::Float64=1.0,
                   profile_nstarts::Integer=6,
                   y_full=nothing, window_deltas=[0, -5, -10, -15],
                   slope_tol::Float64=0.5, kwargs...)

    st = fit_stability(t, y, n; levels=stability_levels, tol=tol_stab,
                       Kmax_mult=Kmax_mult, min_gap=min_gap, kwargs...)
    f = st.fit          # best across levels, NOT the last level
    hits = bound_hits(f; Kmax_mult=Kmax_mult, min_gap=min_gap)
    gated_hits = [h for h in hits
                  if any(startswith(h, pre) for pre in bound_gate_prefixes)]

    ws = nothing
    fail(reason) = (fit = f, pass = false, dip = NaN, warm_gain = NaN,
                    bounds = hits, stability = st, window = ws, reason = reason)

    is_stable_fit(st; tol=tol_stab) ||
        return fail("unstable: improvement " *
                    string(round(100 * st.improvement[end], digits=2)) *
                    "%, max regression " *
                    string(round(100 * maximum(st.regression), digits=2)) * "%")

    isempty(gated_hits) ||
        return fail("parameter on bound: " * join(gated_hits, ", "))

    f.converged || return fail("fit did not converge")

    # window-shift: only detectable by refitting across window lengths
    if y_full !== nothing && n >= 2
        ws = window_shift(y_full, n; deltas=window_deltas,
                          stability_levels=stability_levels, tol_stab=tol_stab,
                          slope_tol=slope_tol, bound_gate_prefixes=bound_gate_prefixes,
                          Kmax_mult=Kmax_mult, min_gap=min_gap, kwargs...)
        if ws.verdict === :tracking
            return fail("onset tracks the window end: slope " *
                        string(round(ws.slope, digits=2)) *
                        " over " * string(ws.nusable) * " windows")
        elseif ws.verdict === :insufficient
            return fail("window check inconclusive: only " *
                        string(ws.nusable) * " of " * string(length(ws.rows)) *
                        " windows gate-clean")
        end
    end

    dip, wg = NaN, NaN
    if do_profile && n >= 2
        pr = profile_tau2(t, y; grid=profile_grid, nstarts=profile_nstarts,
                          Kmax_mult=Kmax_mult, min_gap=min_gap, kwargs...)
        dip = profile_dip(pr.objective)
        wg  = pr.warm_gain
        is_smooth_profile(pr.objective; tol=tol_dip) ||
            return (fit = f, pass = false, dip = dip, warm_gain = wg,
                    bounds = hits, stability = st, window = ws,
                    reason = "profile not smooth: dip " * string(round(dip, digits=3)))
    end

    return (fit = f, pass = true, dip = dip, warm_gain = wg,
            bounds = hits, stability = st, window = ws, reason = "ok")
end

# =====================================================================
# The window-shift gate
#
# The failure it catches: on the national COVID series tau_2 came back as
# 24.81, 29.32, 34.24, 39.23, 44.06 for calibration windows of 55, 60, 65, 70
# and 75 days -- +4.9 days of onset per +5 days of data, slope ~1.0, and
# tau_2 = nd - 30.9 within a day every time. The second sub-epidemic was not
# locating a wave. It parks a fixed distance before wherever the data ends and
# absorbs the recent tail, behaving as an adaptive smoother rather than a
# transmission cluster.
#
# No other gate can see this. Smoothness, stability and bound-adjacency all
# examine ONE window, and a window-relative parameter looks perfectly healthy
# in any single window -- the whole profile simply moves with the data. Only
# refitting across window lengths exposes it.
#
# This also flatters short-horizon forecasts (day-75 MAE 374 vs 519 for n=1)
# with no mechanistic content whatsoever, which is exactly why it needs to be
# caught automatically rather than noticed by eye.
# =====================================================================

"""
    window_slope(nds, taus)

Least-squares slope of onset time on calibration length, plus the spread of
the implied offset `nd - tau`.

Returns `(slope, intercept, offset_mean, offset_sd, tau_sd)`.

`taus` MUST be ABSOLUTE onsets (`t0 + tau`), not window-relative ones. With
window-relative values and a moving window start, a stationary onset yields a
slope of exactly 1.0 -- which is how the retracted "tail smoother" result
arose. On absolute onsets, a slope near 0 means a fixed calendar time (a real
feature of the epidemic) and a slope near 1 means the onset really does follow
the data.
"""
function window_slope(nds::AbstractVector, taus::AbstractVector)
    n = length(nds)
    n < 2 && return (slope = NaN, intercept = NaN, offset_mean = NaN,
                     offset_sd = NaN, tau_sd = NaN)
    x = float.(collect(nds)); y = float.(collect(taus))
    mx, my = mean(x), mean(y)
    den = sum(abs2, x .- mx)
    slope = den <= 0 ? NaN : sum((x .- mx) .* (y .- my)) / den
    off = x .- y
    return (slope = slope, intercept = my - slope * mx,
            offset_mean = mean(off), offset_sd = n > 1 ? std(off) : 0.0,
            tau_sd = n > 1 ? std(y) : 0.0)
end

"""
    window_shift(y_full, n; deltas=[0,-5,-10], ...)

Refit at several calibration window lengths and test whether the onset tracks
the end of the data.

`y_full` is the raw incidence series; each window uses its final `nd + delta`
points. Windows whose fit fails the stability or bound gates are excluded and
counted, since a stalled fit says nothing about window dependence.

Returns `(rows, slope, offset_sd, tau_sd, nusable, tracking)`. `tracking` is
`true` when the onset is moving with the window — i.e. the component is an
artifact and must not be interpreted.
"""
function window_shift(y_full::AbstractVector, n::Integer;
                      deltas=[0, -5, -10, -15], smoothfactor::Integer=7,
                      calibration_period::Union{Nothing,Integer}=nothing,
                      anchor::Symbol=:start,
                      stability_levels=[(10, 6), (20, 12), (40, 24)],
                      tol_stab::Float64=0.02, slope_tol::Float64=0.5,
                      min_windows::Integer=3,
                      bound_gate_prefixes=["gap", "K", "r"],
                      Kmax_mult::Float64=10.0, min_gap::Float64=1.0, kwargs...)
    n < 2 && return (rows = NamedTuple[], slope = 0.0, offset_sd = NaN,
                     tau_sd = NaN, nusable = 0, tracking = false,
                     verdict = :not_applicable, anchor = anchor)

    base = calibration_period === nothing ? length(y_full) :
           min(Int(calibration_period), length(y_full))
    rows = NamedTuple[]
    for d in deltas
        nd = base + d
        nd < 20 && continue

        # `anchor` decides WHICH END of the window moves, and the two answer
        # different questions:
        #   :start -- fix the beginning, extend the end. Does new data move the
        #             onset? The forecasting-relevant version.
        #   :end   -- fix the end, drop early data. Is the onset supported by
        #             recent observations alone?
        # Either way the onset is converted to ABSOLUTE time before comparison.
        # Omitting that conversion is what produced a false slope of 1.0 and
        # the retracted "window-relative tail smoother" conclusion.
        cal = anchor === :start ?
              calibration(y_full[1:nd]; smoothfactor=smoothfactor) :
              calibration(y_full; smoothfactor=smoothfactor, calibration_period=nd)

        st = fit_stability(cal.t, cal.y, n; levels=stability_levels,
                           tol=tol_stab, Kmax_mult=Kmax_mult,
                           min_gap=min_gap, kwargs...)
        f = st.fit
        hits = bound_hits(f; Kmax_mult=Kmax_mult, min_gap=min_gap)
        gated = [h for h in hits if any(startswith(h, p) for p in bound_gate_prefixes)]
        ok = is_stable_fit(st; tol=tol_stab) && isempty(gated) && f.converged
        push!(rows, (nd = nd, t0 = cal.t0, tau2_rel = f.params.tau[2],
                     tau2_abs = cal.t0 + f.params.tau[2], SSE = f.fval,
                     K1 = f.params.K[1], K2 = f.params.K[2], usable = ok,
                     reason = ok ? "ok" : (isempty(gated) ? "unstable" :
                              "bound: " * join(gated, ","))))
    end

    use = [r for r in rows if r.usable]
    ws = window_slope([r.nd for r in use], [r.tau2_abs for r in use])

    verdict = length(use) < min_windows ? :insufficient :
              (isfinite(ws.slope) && abs(ws.slope) > slope_tol ? :tracking : :stable)

    return (rows = rows, slope = ws.slope, offset_sd = ws.offset_sd,
            tau_sd = ws.tau_sd, nusable = length(use),
            tracking = verdict === :tracking, verdict = verdict, anchor = anchor)
end


"""
    is_window_stable(ws; slope_tol=0.5, allow_insufficient=false)

Gate over [`window_shift`](@ref).

`:stable`        -> pass. The onset is a fixed calendar time.
`:tracking`      -> fail. The onset moves with the end of the data.
`:insufficient`  -> too few usable windows to tell. This is NOT the same as
                    passing, and by default it does not pass. Set
                    `allow_insufficient=true` only if you intend "cannot
                    assess" to be treated as acceptable, and report it
                    separately when you do.

Any unrecognized verdict is treated as `:insufficient` — the gate fails closed
rather than passing a value it does not understand.
"""
function is_window_stable(ws; slope_tol::Float64=0.5,
                          allow_insufficient::Bool=false)
    if haskey(ws, :verdict)
        v = ws.verdict
        v === :not_applicable && return true
        v === :stable         && return true
        v === :tracking       && return false
        # :insufficient, or ANY unrecognized verdict, fails closed. A gate that
        # passes a value it does not understand is worse than no gate.
        return allow_insufficient
    end
    # legacy (slope, nusable) input
    ws.nusable < 2 && return allow_insufficient
    return !isfinite(ws.slope) || abs(ws.slope) <= slope_tol
end
