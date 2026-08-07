# =====================================================================
# Gated single-origin report: profile, gates, fit, forecast, score, save, plot.
#
#   julia --project=. --threads=21 examples/run_report.jl
#   julia --project=. --threads=21 examples/run_report.jl mpox_weekly_usa
#
# GATES FIRST, ALWAYS. Two findings in this project were retracted because
# they were read off fits that had not converged. Nothing here is interpreted
# until the profile is smooth, the answer has stopped moving with search
# effort, and no parameter sits on a bound.
# =====================================================================

using SubFit, Statistics, Printf
include(joinpath(@__DIR__, "datasets.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
D   = dataset(cli_dataset())
OUT = output_dir(ROOT, D.name, "report")
B, SEED, INCIDENCE = 300, 20260101, :discrete
H = D.horizon

cal   = calibration(D.y_cal; smoothfactor=D.smoothfactor,
                    calibration_period=D.calibration_period)
truth = D.y_full[(length(D.y_cal) + 1):(length(D.y_cal) + H)]
nd    = length(cal.y)
@printf("%s | %d %ss calibration | %d-%s horizon | t0=%.0f\n\n",
        D.name, nd, D.unit, H, D.unit, cal.t0)

# ---- gates ---------------------------------------------------------------
pr = profile_tau2(cal.t, cal.y; grid=1.0:1.0:float(cal.t[end]), incidence=INCIDENCE)
iv = profile_interval(pr.tau, pr.objective, nd)
dip = profile_dip(pr.objective)
@printf("profile : min %.1f at tau_2=%.1f (abs %.1f) | dip %.3f | warm_gain %.3f\n",
        minimum(pr.objective), pr.tau[argmin(pr.objective)],
        cal.t0 + pr.tau[argmin(pr.objective)], dip, pr.warm_gain)
@printf("          95%% interval %.0f-%.0f (%d of %d grid points)\n",
        iv.lo, iv.hi, iv.npoints, length(pr.tau))

ws = window_shift(D.y_cal, 2; deltas=[0, -5, -10, -15, -20], anchor=:end,
                  smoothfactor=D.smoothfactor, incidence=INCIDENCE, seed=SEED)
use = [r for r in ws.rows if r.usable]
if !isempty(use)
    a = [r.tau2_abs for r in use]
    @printf("window  : absolute onset mean %.2f SD %.3f over %d windows | slope %.3f | :%s\n",
            mean(a), length(a) > 1 ? std(a) : 0.0, length(a), ws.slope, ws.verdict)
end

rows, boots = NamedTuple[], Dict{Int,Any}()
for n in 1:D.nmax
    g = gated_fit(cal.t, cal.y, n; do_profile = n >= 2,
                  profile_grid = 1.0:2.0:float(cal.t[end]),
                  flag = GLM, incidence = INCIDENCE, seed = SEED)
    f = g.fit
    boot = run_bootstrap(f; B=B, horizon=H, dist=:normal, seed=SEED, y_obs=cal.y_raw)
    tfull, point, _ = point_forecast(f, H)
    pc = performance(boot.fit_noisy[1:nd, :], point[1:nd], cal.y_raw)
    pf = performance(boot.fit_noisy[(nd + 1):end, :], point[(nd + 1):end], truth)
    boots[n] = (f = f, boot = boot, tfull = tfull, point = point, gate = g)

    @printf("\nn=%d  %s  SSE=%.1f AICc=%.2f  K=[%s]%s\n", n,
            g.pass ? "GATE PASS" : "GATE FAIL ($(g.reason))", f.fval, f.aicc,
            join([@sprintf("%.0f", k) for k in f.params.K], ", "),
            n > 1 ? @sprintf("  tau2=%.2f (abs %.2f)", f.params.tau[2],
                             cal.t0 + f.params.tau[2]) : "")
    @printf("      calib MAE=%.1f WIS=%.1f cov=%.1f%% | fcst MAE=%.1f MSE=%.1f WIS=%.1f cov=%.1f%%\n",
            pc.MAE, pc.WIS, pc.Coverage95, pf.MAE, pf.MSE, pf.WIS, pf.Coverage95)

    push!(rows, (nsub = n, gate_pass = g.pass ? 1 : 0, reason = g.reason,
                 SSE = f.fval, AICc = f.aicc, nparams = f.nparams,
                 tau2_abs = n > 1 ? cal.t0 + f.params.tau[2] : NaN,
                 Ktot = sum(f.params.K), nboot = boot.nsuccess,
                 calib_MAE = pc.MAE, calib_WIS = pc.WIS, calib_cov = pc.Coverage95,
                 MAE = pf.MAE, MSE = pf.MSE, WIS = pf.WIS, Coverage95 = pf.Coverage95))
end

if D.paper !== nothing
    p = D.paper
    println("\npaper benchmarks (SubEpiPredict, 1st-ranked, onset_fixed=0):")
    @printf("  AICc %.2f | forecast MAE %.2f MSE %.2f WIS %.2f coverage %.2f%%\n",
            p.AICc, p.MAE, p.MSE, p.WIS, p.coverage)
    println("  NOTE: the paper's AICc is on the DISCRETE incidence scale, which")
    println("  is what INCIDENCE=:discrete above reproduces.")
end

# ---- save and plot -------------------------------------------------------
save_performance(OUT, "performance.csv", rows)
save_performance(OUT, "profile-tau2.csv",
    [(tau2_rel = pr.tau[i], tau2_abs = cal.t0 + pr.tau[i],
      objective = pr.objective[i], cold = pr.cold[i],
      in_CI = pr.objective[i] <= iv.threshold ? 1 : 0) for i in eachindex(pr.tau)])
save_performance(OUT, "window-shift.csv", ws.rows)

best = argmin([r.AICc for r in rows])
bf = boots[rows[best].nsub]
save_fit(OUT, "n$(rows[best].nsub)", bf.f, bf.boot)
save_parameters(OUT, "n$(rows[best].nsub)", bf.boot, rows[best].nsub)
save_forecast(OUT, "n$(rows[best].nsub)", bf.tfull, bf.point, bf.boot.fit_noisy)
save_rankings(OUT, [boots[n].f for n in 1:D.nmax])
save_settings(OUT, bf.f, cal; horizon = H, bootstrap_draws = B, seed = SEED,
              gate_profile_dip = bf.gate.dip, gate_window = ws.verdict,
              gate_pass = bf.gate.pass, gate_reason = bf.gate.reason,
              stability_improvement = bf.gate.stability.improvement[end])

savefig(plot_fit(bf.f, bf.boot, cal;
            title="Fit, n=$(rows[best].nsub) ($(D.name))"), joinpath(OUT, "fit.png"))
savefig(plot_forecast(bf.f, bf.tfull, bf.point, bf.boot.fit_noisy, truth;
            title="$H-$(D.unit) forecast, n=$(rows[best].nsub) ($(D.name))"),
        joinpath(OUT, "forecast.png"))
savefig(plot_profile(pr, nd; title="Profile over tau_2 ($(D.name))"),
        joinpath(OUT, "profile.png"))
savefig(plot_window(ws; title="Absolute onset vs window ($(D.name))"),
        joinpath(OUT, "window.png"))
savefig(plot_parameters(bf.boot, rows[best].nsub;
            title="Bootstrap parameters, n=$(rows[best].nsub) ($(D.name))"),
        joinpath(OUT, "parameters.png"))

println("\nwritten to $OUT")
