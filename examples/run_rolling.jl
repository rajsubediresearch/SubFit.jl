# =====================================================================
# Rolling-origin evaluation with gates, saving and plots.
#
#   julia --project=. --threads=21 examples/run_rolling.jl
#   julia --project=. --threads=21 examples/run_rolling.jl mpox_weekly_usa
#
# THE QUESTION: does AICc pick the model that actually forecasts best?
#
# Earlier versions of this analysis were invalidated twice -- once by leaky
# gates, once by a time-frame mix-up. Every fit here must clear the profile,
# stability and bound gates; origins where any model fails are skipped and
# counted rather than quietly included. Onsets are reported in ABSOLUTE time.
# =====================================================================

using SubFit, Statistics, Printf
include(joinpath(@__DIR__, "datasets.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
D   = dataset(cli_dataset())
OUT = output_dir(ROOT, D.name, "rolling")
B, SEED, INCIDENCE = 300, 20260101, :discrete
H = D.horizon

rows, ens_rows, summary, skipped = NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[]

for L in D.origins
    L + H <= length(D.y_full) || continue
    cal   = calibration(D.y_full[1:L]; smoothfactor=D.smoothfactor)
    truth = D.y_full[(L + 1):(L + H)]
    @info "origin $L"

    gated = [gated_fit(cal.t, cal.y, n; do_profile = n >= 2,
                       profile_grid = 1.0:2.0:float(cal.t[end]),
                       flag = GLM, incidence = INCIDENCE, seed = SEED)
             for n in 1:D.nmax]

    if !all(g -> g.pass, gated)
        for (n, g) in enumerate(gated)
            g.pass || (@printf("  n=%d GATE FAIL: %s\n", n, g.reason);
                       push!(skipped, (origin = L, nsub = n, reason = g.reason)))
        end
        continue
    end

    fits = sort([g.fit for g in gated], by = f -> f.aicc)
    curveset, origin_rows = Matrix{Float64}[], NamedTuple[]
    for f in fits
        boot = run_bootstrap(f; B=B, horizon=H, dist=:normal, seed=SEED, y_obs=cal.y_raw)
        _, point, _ = point_forecast(f, H)
        fc = boot.fit_noisy[(L + 1):end, :]
        p  = performance(fc, point[(L + 1):end], truth)
        g  = gated[f.n]
        row = (origin = L, nsub = f.n, SSE = f.fval, AICc = f.aicc,
               tau2_abs = f.n > 1 ? cal.t0 + f.params.tau[2] : NaN,
               Ktot = sum(f.params.K), dip = g.dip,
               stab_imp = g.stability.improvement[end], nboot = boot.nsuccess,
               MAE = p.MAE, MSE = p.MSE, Coverage95 = p.Coverage95, WIS = p.WIS)
        push!(rows, row); push!(origin_rows, row); push!(curveset, fc)
        @printf("  n=%d AICc=%8.2f SSE=%9.1f | fcst MAE=%7.1f WIS=%7.1f cov=%5.1f%%\n",
                f.n, f.aicc, f.fval, p.MAE, p.WIS, p.Coverage95)
    end

    aiccs = [f.aicc for f in fits]
    if length(fits) > 1
        for scheme in (:equal, :akaike)
            w  = ensemble_weights(aiccs; scheme=scheme)
            ec = ensemble_curves(curveset, w)
            p  = performance(ec, vec(mean(ec, dims=2)), truth)
            push!(ens_rows, (origin = L, scheme = String(scheme), w1 = w[1],
                             MAE = p.MAE, MSE = p.MSE,
                             Coverage95 = p.Coverage95, WIS = p.WIS))
            @printf("  Ensemble(%-6s) w1=%.3f WIS=%7.1f cov=%5.1f%%\n",
                    scheme, w[1], p.WIS, p.Coverage95)
        end
    end

    ap = fits[1].n
    wp = origin_rows[argmin([r.WIS for r in origin_rows])].nsub
    push!(summary, (origin = L, aicc_pick = ap, wis_pick = wp,
                    agree = ap == wp ? 1 : 0))
    @printf("  --> AICc n=%d, WIS n=%d %s\n", ap, wp, ap == wp ? "(agree)" : "(DISAGREE)")
end

save_performance(OUT, "rolling-models.csv", rows)
isempty(ens_rows) || save_performance(OUT, "rolling-ensembles.csv", ens_rows)
isempty(summary)  || save_performance(OUT, "rolling-summary.csv", summary)
isempty(skipped)  || save_performance(OUT, "rolling-skipped.csv", skipped)

println("\n", "="^70)
@printf("origins gate-clean: %d of %d", length(summary), length(D.origins))
isempty(skipped) || @printf(" | %d gate failures", length(skipped))
println()
if !isempty(summary)
    @printf("AICc and out-of-sample WIS agreed at %d of %d\n",
            count(r -> r.agree == 1, summary), length(summary))
    for n in 1:D.nmax
        sel = [r for r in rows if r.nsub == n]
        isempty(sel) && continue
        @printf("  n=%d : mean WIS %8.1f  mean coverage %5.1f%%  mean tau2_abs %s\n",
                n, mean(r.WIS for r in sel), mean(r.Coverage95 for r in sel),
                n > 1 ? @sprintf("%.2f", mean(r.tau2_abs for r in sel)) : "-")
    end
    for sc in unique(r.scheme for r in ens_rows)
        sel = [r for r in ens_rows if r.scheme == sc]
        @printf("  Ensemble(%-6s): mean WIS %8.1f  mean coverage %5.1f%%\n",
                sc, mean(r.WIS for r in sel), mean(r.Coverage95 for r in sel))
    end
    savefig(plot_rankings(rows; title="AICc vs forecast WIS by origin ($(D.name))"),
            joinpath(OUT, "aicc-vs-wis.png"))
end
println("="^70)
println("\nwritten to $OUT")
