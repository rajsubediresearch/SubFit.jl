# =====================================================================
# Sanity check: is the K upper bound causing the premature saturation?
#
# SubFit caps K at `Kmax_mult * sum(y)` (default 10x); SubEpiPredict uses a
# fixed 1e10. If the n=2 fit is saturating because it cannot reach a large
# enough final size, raising the cap will move it. If the fit is unchanged
# across three orders of magnitude, the cap is exonerated and the saturation
# is a property of the least-squares surface.
#
# This must be ruled out before building anything on top of the rolling-origin
# result.
#
# Run:  julia --project=. --threads=21 examples/kmax_check.jl
# =====================================================================

using SubFit, Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))

y_raw = load_series(joinpath(ROOT, "input",
    "cumulative-daily-coronavirus-deaths-USA-05-11-2020.txt");
    column=52, cumulative=true)
cal = calibration(y_raw; smoothfactor=7, calibration_period=90)
observed_cum = sum(cal.y_raw)
@printf("observed cumulative by day %d: %.0f\n\n", length(cal.y), observed_cum)

for mult in (10.0, 100.0, 1000.0)
    f = fit_subepidemic(cal.t, cal.y, 2; flag=GLM, method=:normal,
                        nstarts=10, seed=20260101, Kmax_mult=mult)
    Ktot = sum(f.params.K)
    @printf("Kmax_mult=%-7.0f (K bound %.3g)  SSE=%10.1f  AICc=%8.2f  Ktot=%9.1f  headroom=%+8.1f\n",
            mult, mult * sum(abs, cal.y), f.fval, f.aicc, Ktot, Ktot - observed_cum)
    for i in 1:f.n
        @printf("    sub %d: r=%.4f  p=%.4f  K=%9.1f  tau=%6.2f\n",
                i, f.params.r[i], f.params.p[i], f.params.K[i], f.params.tau[i])
    end
end

println("""
Reference points:
  SubFit at Kmax_mult=10 (2026-08-02 run): Ktot = 98,331  -> headroom ~14,200
  MATLAB top-ranked model (paper Fig. 8) : K1 = 4.28e4, K2 = 8.33e4
                                           Ktot ~ 126,100 -> headroom ~42,000
  Truth over the next 30 days            : ~30,000 deaths
If SSE and Ktot are flat across the three caps, the bound is not the cause.""")
