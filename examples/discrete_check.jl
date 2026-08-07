# =====================================================================
# Does optimizing on DISCRETE incidence move the fit toward MATLAB's basin?
#
# The last remaining implementation difference between SubFit and
# SubEpiPredict: MATLAB reads incidence off the cumulative ODE solution as
# [y(1); diff(y)], SubFit reads it analytically from the RHS. On the COVID
# example these differ by ~19% in SSE, and the paper's AICc of 1036.00 is on
# the discrete scale — so only a discrete-incidence fit is strictly comparable
# to the published numbers.
#
# The question this answers: is the over-saturating n=2 fit an artifact of the
# analytic incidence definition, or does it survive the switch?
#
# Reference points (2026-08-02 runs, analytic):
#   SubFit n=2 free-tau : SSE(disc) 392,916  fcst MAE 610.0  tau2 27.00
#                         K = [58948, 39383]
#   MATLAB params       : SSE(disc) 689,865  fcst MAE 262.4  tau2 25.48
#                         K = [42800, 83300]
#
# Run:  julia --project=. --threads=21 examples/discrete_check.jl
# =====================================================================

using SubFit, Statistics, Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
HORIZON = 30
B       = 300
SEED    = 20260101

y_raw = load_series(joinpath(ROOT, "input",
    "cumulative-daily-coronavirus-deaths-USA-05-11-2020.txt"); column=52)
cal = calibration(y_raw; smoothfactor=7, calibration_period=90)
nd  = length(cal.y)

y_eval = load_series(joinpath(ROOT, "input",
    "cumulative-daily-coronavirus-deaths-USA-05-09-2022.txt"); column=52)
truth = y_eval[(length(y_raw) + 1):(length(y_raw) + HORIZON)]

@printf("observed cumulative by day %d: %.0f | truth over next %d days: %.0f\n\n",
        nd, sum(cal.y_raw), HORIZON, sum(truth))

for mode in (:analytic, :discrete)
    println("="^70)
    @printf("incidence = :%s\n", mode)
    println("="^70)
    for n in 1:2
        f = fit_subepidemic(cal.t, cal.y, n; flag=GLM, method=:normal,
                            nstarts=10, seed=SEED, incidence=mode)
        boot = run_bootstrap(f; B=B, horizon=HORIZON, dist=:normal,
                             seed=SEED, y_obs=cal.y_raw)
        _, point, _ = point_forecast(f, HORIZON)
        pc = performance(boot.fit_noisy[1:nd, :], point[1:nd], cal.y_raw)
        pf = performance(boot.fit_noisy[(nd + 1):end, :], point[(nd + 1):end], truth)
        Ktot = sum(f.params.K)

        @printf("  n=%d  SSE=%10.1f  AICc=%8.2f  Ktot=%9.1f  headroom=%+9.1f  tau2=%s\n",
                n, f.fval, f.aicc, Ktot, Ktot - sum(cal.y_raw),
                n > 1 ? @sprintf("%.2f", f.params.tau[2]) : "-")
        @printf("        K = [%s]\n",
                join([@sprintf("%.0f", k) for k in f.params.K], ", "))
        @printf("        calib : MAE=%7.2f  WIS=%7.2f  cov=%5.1f%%\n",
                pc.MAE, pc.WIS, pc.Coverage95)
        @printf("        fcst  : MAE=%7.2f  MSE=%10.1f  WIS=%7.2f  cov=%5.1f%%\n\n",
                pf.MAE, pf.MSE, pf.WIS, pf.Coverage95)
    end
end

println("""
What to look for
  * If the :discrete n=2 fit lands near K = [42800, 83300] with a forecast MAE
    in the 250-300 range, the over-saturation was an artifact of the analytic
    incidence definition and SubFit now agrees with MATLAB.
  * If it stays near K = [58948, 39383] with MAE ~600, the incidence
    definition is NOT the explanation, and the overfitting result stands:
    the least-squares optimum genuinely forecasts worse than the point
    MATLAB's fmincon stops at.
  * Either way, the :discrete AICc is the number comparable to the paper's
    1036.00 (1st ranked) and 1037.94 (2nd ranked).""")
