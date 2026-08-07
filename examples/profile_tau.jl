# =====================================================================
# Profile likelihood over tau_2 (the second sub-epidemic's onset)
#
# Two things this settles at once:
#
#  1. IS THE SURFACE FLAT? The national fit put tau_2 at 27.00 under analytic
#     incidence and 8.16 under discrete, with nearly identical SSE, WIS and
#     forecast MAE. If a 19-day shift costs nothing, the onset is not
#     identified and AICc is free to trade wave size against wave timing for
#     free -- which is a MECHANISM for the overfitting, not just a symptom.
#
#  2. IS THE BOUND-PINNING REAL? In the state run, tau_2 hit its lower bound
#     of 1.0 in 33 of 35 states. Either that is a genuine optimum (the state
#     epidemics really are single-wave, and the national two-wave structure is
#     an aggregation artifact of asynchronous states) or it is an optimizer
#     artifact. A profile shows which: a genuine optimum has the objective
#     rising monotonically away from the bound.
#
# METHOD: fix tau_2 on a grid, optimize all remaining parameters at each grid
# point, and record the best objective. Uses SubFit internals directly so no
# src change is needed.
#
# The 95% profile-likelihood interval for Gaussian errors is the set where
#     nd * log(SSE / SSE_min)  <=  3.841        (chi-square, 1 df)
# i.e. SSE <= SSE_min * exp(3.841/nd).
#
# Run:  julia --project=. --threads=21 examples/profile_tau.jl
# =====================================================================

using SubFit, Statistics, Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT  = joinpath(ROOT, "output", "profile_tau")

HORIZON   = 30
NSTARTS   = 6
SEED      = 20260101
INCIDENCE = :discrete
TAU_GRID  = collect(1.0:1.0:60.0)

# column 52 = national; 36 and 48 did NOT pin at the bound, 1 did
COLUMNS = [(52, "national"), (36, "state 36 (unpinned)"),
           (48, "state 48 (unpinned)"), (1, "state 1 (pinned)")]

calib_file = joinpath(ROOT, "input", "cumulative-daily-coronavirus-deaths-USA-05-11-2020.txt")
eval_file  = joinpath(ROOT, "input", "cumulative-daily-coronavirus-deaths-USA-05-09-2022.txt")

"""Best objective with tau_2 held at `tau`, everything else free."""
function profile_point(cal, tau, tend, I0)
    lb, ub = param_bounds(2, cal.y, GLM; Kmax_mult=10.0, min_gap=1.0, tend=tend)
    tau_c = clamp(tau, lb[9], ub[9])
    lb[9] = tau_c; ub[9] = tau_c                       # pin the gap
    free = [i for i in eachindex(lb) if ub[i] > lb[i]]
    fobj = th -> objective(th, 2, cal.t, cal.y, GLM, I0, :normal;
                           incidence=INCIDENCE)
    starts = initial_points(2, cal.t, cal.y, lb, ub; nstarts=NSTARTS, seed=SEED)

    best_th, best_f = starts[1], Inf
    for x0 in starts
        th, f, _ = SubFit._run_nlopt(:LN_SBPLX, x0, lb, ub, free, fobj, 3000)
        f < best_f && ((best_th, best_f) = (th, f))
    end
    th, f, _ = SubFit._run_nlopt(:LN_COBYLA, best_th, lb, ub, free, fobj, 3000)
    f < best_f && ((best_th, best_f) = (th, f))
    return best_th, best_f
end

for (col, label) in COLUMNS
    y_raw = load_series(calib_file; column=col, cumulative=true)
    y_ev  = load_series(eval_file;  column=col, cumulative=true)
    cal   = calibration(y_raw; smoothfactor=7, calibration_period=90)
    truth = y_ev[(length(y_raw) + 1):(length(y_raw) + HORIZON)]
    nd    = length(cal.y)
    I0    = cal.y[1]
    tend  = cal.t[end]
    dt    = cal.t[2] - cal.t[1]
    t_full = vcat(cal.t, cal.t[end] .+ dt .* collect(1.0:HORIZON))

    println("\n", "="^72)
    @printf("%s  (column %d, %.0f deaths in calibration)\n", label, col, sum(cal.y_raw))
    println("="^72)

    taus, sses, maes, k1s, k2s = Float64[], Float64[], Float64[], Float64[], Float64[]
    for tau in TAU_GRID
        th, f = profile_point(cal, tau, tend, I0)
        P = unpack(th, 2)
        inc, _ = simulate(P, t_full, GLM, I0; incidence=INCIDENCE)
        m = any(isnan, inc) ? NaN : mae(inc[(nd + 1):end], truth)
        push!(taus, tau); push!(sses, f); push!(maes, m)
        push!(k1s, P.K[1]); push!(k2s, P.K[2])
    end

    smin = minimum(sses)
    thr  = smin * exp(3.841 / nd)
    inCI = taus[sses .<= thr]

    @printf("SSE_min = %.1f at tau_2 = %.1f\n", smin, taus[argmin(sses)])
    @printf("95%% profile interval for tau_2: %.1f to %.1f  (SSE <= %.1f, %.2f%% above min)\n",
            minimum(inCI), maximum(inCI), thr, 100 * (thr / smin - 1))
    @printf("  -> %d of %d grid points are statistically indistinguishable from the optimum\n",
            length(inCI), length(taus))
    @printf("forecast MAE across that interval: %.1f to %.1f (ratio %.1fx)\n",
            minimum(maes[sses .<= thr]), maximum(maes[sses .<= thr]),
            maximum(maes[sses .<= thr]) / max(minimum(maes[sses .<= thr]), 1e-9))

    println("\n  tau2    SSE        %above   fcstMAE      K1        K2    inCI")
    for i in eachindex(taus)
        (i % 3 == 1 || sses[i] <= thr) || continue
        bar = sses[i] <= thr ? "  <==" : ""
        @printf("  %4.0f  %10.1f  %7.2f%%  %8.1f  %8.0f  %8.0f%s\n",
                taus[i], sses[i], 100 * (sses[i] / smin - 1), maes[i],
                k1s[i], k2s[i], bar)
    end

    save_performance(OUT, "profile-tau-col$(col).csv",
        [(tau2 = taus[i], SSE = sses[i], pct_above_min = 100*(sses[i]/smin - 1),
          fcst_MAE = maes[i], K1 = k1s[i], K2 = k2s[i],
          in_CI = sses[i] <= thr ? 1 : 0) for i in eachindex(taus)])
end

println("""

How to read this
  * A WIDE 95% interval on tau_2 with a LARGE spread of forecast MAE inside it
    is the key result: the data cannot distinguish onset times that produce
    very different forecasts. That makes AICc-based selection unreliable here
    for a reason that has nothing to do with any optimizer.
  * For the pinned state, check whether SSE rises monotonically from tau_2=1.
    If it does, single-wave is the genuine optimum and the national two-wave
    structure is likely an aggregation artifact of asynchronous state
    epidemics -- which would be the framework's own motivation, confirmed.
    If SSE DIPS somewhere away from the bound, the state fits were stalling
    and the state replication needs rerunning before it means anything.""")
