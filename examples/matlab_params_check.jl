# =====================================================================
# Does MATLAB's better-forecasting fit have a WORSE calibration SSE?
#
# If yes, the story is clean overfitting: SubFit's optimizer finds a genuinely
# lower least-squares optimum that extrapolates badly, and MATLAB's
# under-convergence is acting as accidental regularization.
#
# If MATLAB's parameters give a SIMILAR or BETTER SSE than SubFit's own fit,
# the story is different and worse for the reformulation: the two are not even
# fitting the same surface, and the tau parameterization is losing something
# structural that C_thr provides.
#
# METHOD
# The paper's parameters come with a threshold C_thr, not an onset time. We
# map one to the other by integrating sub-epidemic 1 forward and finding the
# time at which its cumulative curve crosses C_thr -- which is exactly what
# the indicator A_i(t) in Eq. 3 does. That makes the published parameter set
# directly evaluable in SubFit's model.
#
# CAVEATS, both real:
#  1. Figure 8 is ambiguous about which model it reports. The caption says
#     "1st-ranked"; the legend inside every panel says "2nd Ranked Model".
#     So we try the C_thr values of BOTH ranks (Fig. 5: 39571.09 and 2262.15).
#  2. MATLAB computes incidence as a DISCRETE difference of the cumulative
#     solution ([y(1); diff(y)]), SubFit reads it off analytically as
#     growth_rhs(C(t)). On daily data during fast growth these differ. Both
#     are reported below so the SSE comparison is like-for-like.
#
# Run:  julia --project=. --threads=21 examples/matlab_params_check.jl
# =====================================================================

using SubFit, Statistics, Printf
using OrdinaryDiffEqTsit5, SciMLBase, Logging

const ROOT = normpath(joinpath(@__DIR__, ".."))
HORIZON = 30

# ---- data ---------------------------------------------------------------
y_raw = load_series(joinpath(ROOT, "input",
    "cumulative-daily-coronavirus-deaths-USA-05-11-2020.txt"); column=52)
cal = calibration(y_raw; smoothfactor=7, calibration_period=90)
nd  = length(cal.y)
I0  = cal.y[1]

y_eval = load_series(joinpath(ROOT, "input",
    "cumulative-daily-coronavirus-deaths-USA-05-09-2022.txt"); column=52)
truth = y_eval[(length(y_raw) + 1):(length(y_raw) + HORIZON)]
dt    = cal.t[2] - cal.t[1]
t_full = vcat(cal.t, cal.t[end] .+ dt .* collect(1.0:HORIZON))

# ---- cumulative solution of one sub-epidemic ----------------------------
function cumulative_curve(r, p, a, K, flag, C0, tspan)
    f(u, par, tt) = SubFit.growth_rhs(u, r, p, a, K, flag)
    prob = ODEProblem(f, C0, tspan)
    with_logger(NullLogger()) do
        solve(prob, Tsit5(); reltol=1e-8, abstol=1e-8)
    end
end

"""Time at which sub-epidemic 1's cumulative curve crosses C_thr."""
function onset_from_threshold(r, p, a, K, flag, I0, Cthr, tmax)
    Cthr >= K && return nothing          # Eq. 3 requires C_thr <= K_0i
    sol = cumulative_curve(r, p, a, K, flag, I0, (0.0, tmax))
    for tt in 0.0:0.01:tmax
        sol(tt) >= Cthr && return tt
    end
    return nothing
end

# ---- evaluate a parameter set in SubFit's model -------------------------
function evaluate(label, r, p, a, K, tau)
    P = SubEpiParams(collect(r), collect(p), collect(a), collect(K), collect(tau))
    inc, _ = simulate(P, t_full, GLM, I0)

    # discrete-difference incidence, MATLAB style
    n = length(r)
    Ctot = zeros(length(t_full))
    for i in 1:n
        tau[i] >= t_full[end] && continue
        C0 = i == 1 ? I0 : 1.0
        sol = cumulative_curve(r[i], p[i], a[i], K[i], GLM, C0, (tau[i], t_full[end]))
        for j in eachindex(t_full)
            t_full[j] >= tau[i] && (Ctot[j] += sol(t_full[j]))
        end
    end
    incd = vcat(Ctot[1], diff(Ctot))

    sse_a = sum(abs2, inc[1:nd]  .- cal.y)
    sse_d = sum(abs2, incd[1:nd] .- cal.y)
    m     = nparams(n, GLM, :normal)
    Ktot  = sum(K)
    obs   = sum(cal.y_raw)

    @printf("%-34s n=%d  tau2=%6.2f  SSE(analytic)=%10.1f  SSE(discrete)=%10.1f\n",
            label, n, n > 1 ? tau[2] : NaN, sse_a, sse_d)
    @printf("%34s  AICc(analytic)=%8.2f   Ktot=%9.1f   headroom=%+9.1f\n",
            "", aicc(sse_a, nd, m, :normal), Ktot, Ktot - obs)
    @printf("%34s  30d forecast: MAE=%8.2f  MSE=%11.1f   (discrete MAE=%8.2f)\n\n",
            "", mae(inc[(nd + 1):end], truth), mse(inc[(nd + 1):end], truth),
            mae(incd[(nd + 1):end], truth))
    return (sse_a = sse_a, mae = mae(inc[(nd + 1):end], truth))
end

println("observed cumulative by day $nd: ", Int(round(sum(cal.y_raw))))
println("truth over next $HORIZON days : ", Int(round(sum(truth))), "\n")

# ---- SubFit's own fit, as the reference --------------------------------
fit2 = fit_subepidemic(cal.t, cal.y, 2; flag=GLM, method=:normal,
                       nstarts=10, seed=20260101)
evaluate("SubFit free-tau fit (n=2)", fit2.params.r, fit2.params.p,
         fit2.params.a, fit2.params.K, fit2.params.tau)

fit1 = fit_subepidemic(cal.t, cal.y, 1; flag=GLM, method=:normal,
                       nstarts=10, seed=20260101)
evaluate("SubFit fit (n=1)", fit1.params.r, fit1.params.p,
         fit1.params.a, fit1.params.K, fit1.params.tau)

# ---- the paper's parameters (Fig. 8), under both candidate C_thr --------
# r, p, a, K per sub-epidemic; a=1 because the GLM does not use it
mr = [0.66, 6.7]
mp = [0.83, 0.56]
ma = [1.0, 1.0]
mK = [4.28e4, 8.33e4]

for (rank, Cthr) in (("rank 1", 39571.0856), ("rank 2", 2262.1477))
    tau2 = onset_from_threshold(mr[1], mp[1], ma[1], mK[1], GLM, I0, Cthr, t_full[end])
    if tau2 === nothing
        @printf("paper Fig.8 params, C_thr=%s (%.2f): sub-epidemic 1 never reaches the threshold\n\n",
                rank, Cthr)
        continue
    end
    evaluate("paper Fig.8 params, $rank C_thr", mr, mp, ma, mK, [0.0, tau2])
end

println("""
How to read this
  * If the paper's parameters give a HIGHER SSE than SubFit's free-tau fit but
    a LOWER forecast MAE, that is the overfitting result: a genuinely better
    least-squares optimum that extrapolates worse.
  * If the paper's parameters give a similar or LOWER SSE, then SubFit's
    optimizer is not finding the better basin and the free-tau surface is
    hiding it -- a problem with the reformulation, not with AICc.
  * Compare SSE(analytic) vs SSE(discrete) before drawing conclusions. If they
    differ materially, the paper's reported AICc of 1036.00 is on the discrete
    scale and only SSE(discrete) is comparable to it.""")
