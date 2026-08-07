# =====================================================================
# Per-unit replication: the same analysis across every column of a
# multi-column input, with gates, saving and plots.
#
#   julia --project=. --threads=21 examples/run_units.jl
#
# WHY THIS EXISTS. Rolling origins on one series are heavily overlapping and
# therefore near-duplicate evidence. Fitting each state/unit separately turns
# "AICc disagrees with out-of-sample WIS" from an anecdote into a RATE.
#
# The first attempt at this was invalid: tau_2 pinned at its lower bound in 33
# of 35 states and the gates of the day did not catch it. The bound gate exists
# because of that failure, so watch the gate-failure count -- a high rate is a
# finding ("the two-wave model is not estimable on unit-sized series"), not a
# nuisance.
# =====================================================================

using SubFit, Statistics, Printf
include(joinpath(@__DIR__, "datasets.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
D   = dataset(cli_dataset())
OUT = output_dir(ROOT, D.name, "units")
B, SEED, INCIDENCE = 100, 20260101, :discrete
H = D.horizon
LEVELS = [(10, 6), (20, 12)]     # two levels; three is too slow across ~50 units

D.columns === nothing && error("dataset $(D.name) has no per-unit columns")

calib_file = joinpath(DATA_ROOT, "cumulative-daily-coronavirus-deaths-USA-05-11-2020.txt")
eval_file  = joinpath(DATA_ROOT, "cumulative-daily-coronavirus-deaths-USA-05-09-2022.txt")

function binom_p(k::Integer, n::Integer)
    n == 0 && return 1.0
    pk = [binomial(big(n), big(i)) / big(2)^n for i in 0:n]
    obs = pk[k + 1]
    return Float64(sum(x for x in pk if x <= obs * (1 + 1e-12)))
end

rows, summary = NamedTuple[], NamedTuple[]
skipped_screen, skipped_gate = Int[], NamedTuple[]

for st in D.columns
    local cal, truth
    try
        y_raw = load_series(calib_file; column=st, cumulative=true)
        y_ev  = load_series(eval_file;  column=st, cumulative=true)
        cal   = calibration(y_raw; smoothfactor=D.smoothfactor,
                            calibration_period=D.calibration_period)
        truth = y_ev[(length(y_raw) + 1):(length(y_raw) + H)]
    catch err
        @warn "unit $st: load failed" err; push!(skipped_screen, st); continue
    end

    if sum(cal.y_raw) < D.min_deaths || count(>(0), cal.y_raw) < D.min_nonzero ||
       sum(truth) < D.min_future
        push!(skipped_screen, st); continue
    end

    nd = length(cal.y)
    gated = [gated_fit(cal.t, cal.y, n; do_profile = false,
                       stability_levels = LEVELS, flag = GLM,
                       incidence = INCIDENCE, seed = SEED) for n in 1:D.nmax]

    if !all(g -> g.pass, gated)
        for (n, g) in enumerate(gated)
            g.pass || push!(skipped_gate, (unit = st, nsub = n, reason = g.reason))
        end
        @printf("unit %2d: GATE FAIL (%s)\n", st,
                join(unique([g.reason for g in gated if !g.pass]), "; "))
        continue
    end

    unit_rows = NamedTuple[]
    for g in gated
        f = g.fit
        boot = run_bootstrap(f; B=B, horizon=H, dist=:normal, seed=SEED, y_obs=cal.y_raw)
        _, point, _ = point_forecast(f, H)
        p = performance(boot.fit_noisy[(nd + 1):end, :], point[(nd + 1):end], truth)
        push!(unit_rows,
              (unit = st, nsub = f.n, SSE = f.fval, AICc = f.aicc,
               Ktot = sum(f.params.K), headroom = sum(f.params.K) - sum(cal.y_raw),
               future_true = sum(truth),
               tau2_abs = f.n > 1 ? cal.t0 + f.params.tau[2] : NaN,
               stab_imp = g.stability.improvement[end], nboot = boot.nsuccess,
               MAE = p.MAE, MSE = p.MSE, Coverage95 = p.Coverage95, WIS = p.WIS))
    end

    append!(rows, unit_rows)
    ap = unit_rows[argmin([r.AICc for r in unit_rows])].nsub
    wp = unit_rows[argmin([r.WIS  for r in unit_rows])].nsub
    r2 = unit_rows[findfirst(r -> r.nsub == D.nmax, unit_rows)]
    push!(summary, (unit = st, aicc_pick = ap, wis_pick = wp,
                    agree = ap == wp ? 1 : 0, tau2_abs = r2.tau2_abs,
                    undershoot = r2.headroom < r2.future_true ? 1 : 0))
    @printf("unit %2d: AICc n=%d, WIS n=%d  %s   tau2_abs=%6.2f\n",
            st, ap, wp, ap == wp ? "agree   " : "DISAGREE", r2.tau2_abs)
end

save_performance(OUT, "unit-models.csv", rows)
isempty(summary) || save_performance(OUT, "unit-summary.csv", summary)
isempty(skipped_gate) || save_performance(OUT, "unit-gatefail.csv", skipped_gate)

N = length(summary)
println("\n", "="^70)
@printf("units analysed: %d | screened out: %d | GATE FAILURES: %d\n",
        N, length(skipped_screen), length(unique(r.unit for r in skipped_gate)))
if N > 0
    ag = count(r -> r.agree == 1, summary)
    @printf("AICc picked n=%d in %d/%d | WIS picked n=1 in %d/%d\n", D.nmax,
            count(r -> r.aicc_pick == D.nmax, summary), N,
            count(r -> r.wis_pick == 1, summary), N)
    @printf("agreement %d/%d (%.0f%%)  exact binomial p = %.4f\n",
            ag, N, 100ag/N, binom_p(ag, N))
    ta = [r.tau2_abs for r in summary if isfinite(r.tau2_abs)]
    isempty(ta) || @printf("absolute onset: median %.1f  IQR %.1f-%.1f\n",
            median(ta), quantile(ta, 0.25), quantile(ta, 0.75))
    for n in 1:D.nmax
        sel = [r for r in rows if r.nsub == n]
        isempty(sel) && continue
        @printf("  n=%d : median WIS %8.2f  median coverage %5.1f%%\n",
                n, median(r.WIS for r in sel), median(r.Coverage95 for r in sel))
    end
    savefig(plot_rankings(rows; title="AICc vs forecast WIS by unit ($(D.name))"),
            joinpath(OUT, "aicc-vs-wis.png"))
    isempty(ta) || savefig(
        histogram(ta; bins=20, legend=false, color=RGB(0.35, 0.50, 0.70),
                  linealpha=0, xlabel="absolute onset", ylabel="units",
                  title=wrap_title("Fitted onset across units ($(D.name))")),
        joinpath(OUT, "onset-histogram.png"))
end
println("="^70)
println("\nwritten to $OUT")
