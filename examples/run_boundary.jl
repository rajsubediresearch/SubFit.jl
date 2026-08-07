# =====================================================================
# Is the second sub-epidemic's onset a real calendar feature, or does it
# follow the data?
#
# THE BUG THIS SCRIPT USED TO HAVE. `calibration()` keeps the LAST
# `calibration_period` points, so shortening the window moves its START while
# the end stays fixed. tau_2 is measured from t=0 at each window's start, so a
# perfectly STATIONARY onset shifts by exactly one day per day of window
# change. The earlier version compared those window-relative values directly,
# found a slope of 1.0, and concluded the second sub-epidemic was a
# window-relative tail smoother. It is not:
#
#   nd   window    tau_rel   ABSOLUTE onset
#   75   1-75       44.06     45.06
#   70   6-75       39.23     45.23
#   65  11-75       34.24     45.24
#   60  16-75       29.32     45.32
#   55  21-75       24.81     45.81
#                             mean 45.33, SD 0.28 days
#
# The onset is one of the best-identified quantities in the analysis.
#
# This version reports absolute onsets and runs BOTH anchors, which answer
# different questions: :end drops early data (is the onset supported by recent
# observations alone?), :start extends the window forward (does new data move
# it?).
#
# Run:  julia --project=. --threads=21 examples/boundary_check.jl
# =====================================================================

using SubFit, Statistics, Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
INCIDENCE = :discrete

y_raw = load_series(joinpath(ROOT, "input",
    "cumulative-daily-coronavirus-deaths-USA-05-11-2020.txt"); column=52)

for anchor in (:end, :start)
    println("\n", "="^72)
    @printf("anchor = :%s  (%s)\n", anchor,
            anchor === :end ? "fixed end, drop early data" :
                              "fixed start, extend the window")
    println("="^72)
    ws = window_shift(y_raw, 2; deltas=[0, -5, -10, -15, -20], anchor=anchor,
                      flag=GLM, incidence=INCIDENCE, seed=20260101)
    println("  nd    t0   tau_rel   tau_ABS       SSE        K1        K2   status")
    for r in ws.rows
        @printf("  %3d  %4.0f  %7.2f  %8.2f  %9.1f  %8.0f  %8.0f   %s\n",
                r.nd, r.t0, r.tau2_rel, r.tau2_abs, r.SSE, r.K1, r.K2,
                r.usable ? "ok" : r.reason)
    end
    use = [r for r in ws.rows if r.usable]
    if length(use) >= 2
        a = [r.tau2_abs for r in use]
        @printf("  absolute onset: mean %.2f, SD %.3f, range %.2f-%.2f over %d windows\n",
                mean(a), std(a), minimum(a), maximum(a), length(a))
    end
    @printf("  slope on ABSOLUTE onsets %.3f | verdict :%s\n", ws.slope, ws.verdict)
    println(ws.verdict === :stable ?
            "  -> the onset is a fixed calendar feature" :
            ws.verdict === :tracking ?
            "  -> the onset follows the data; the component is an artifact" :
            "  -> too few gate-clean windows to assess")
end

println("""

Read the tau_ABS column, never tau_rel. Window-relative onsets from different
calibration windows are not comparable, and comparing them anyway is what
produced a retracted result.""")
