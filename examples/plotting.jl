# Plot helpers. Included by the analysis scripts, so Plots stays out of
# SubFit's dependency list.

using Plots, Statistics

const BAND = RGBA(0.30, 0.45, 0.70, 0.25)
const LINE = RGB(0.15, 0.30, 0.55)
const SUBC = [RGB(0.75, 0.35, 0.30), RGB(0.30, 0.50, 0.65), RGB(0.35, 0.60, 0.45)]

Plots.default(size = (900, 520), titlefontsize = 10, guidefontsize = 9,
              tickfontsize = 8, legendfontsize = 8,
              left_margin = 6Plots.mm, right_margin = 8Plots.mm,
              top_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
              fg_legend = :transparent,
              background_color_legend = RGBA(1, 1, 1, 0.75))

"""Break long titles at word boundaries rather than letting them clip."""
function wrap_title(s::AbstractString; width::Integer=62)
    length(s) <= width && return String(s)
    lines, cur = String[], ""
    for w in split(s)
        if isempty(cur); cur = String(w)
        elseif length(cur) + 1 + length(w) <= width; cur *= " " * w
        else; push!(lines, cur); cur = String(w) end
    end
    isempty(cur) || push!(lines, cur)
    return join(lines, "\n")
end

"""Fit with 95% band, plus the individual sub-epidemic profiles."""
function plot_fit(fit, boot, cal; title="Model fit")
    nd = length(fit.t)
    lo = [quantile(@view(boot.fit_noisy[i, :]), 0.025) for i in 1:nd]
    hi = [quantile(@view(boot.fit_noisy[i, :]), 0.975) for i in 1:nd]
    p = plot(fit.t, hi; fillrange=lo, fillcolor=BAND, linealpha=0,
             label="95% PI", legend=:topleft, title=wrap_title(title),
             xlabel="time", ylabel="count")
    for i in 1:fit.n
        plot!(p, fit.t, fit.subcurves[:, i]; color=SUBC[mod1(i, 3)], lw=1.5,
              ls=:dash, label="sub-epidemic $i (tau=$(round(cal.t0 + fit.params.tau[i], digits=1)))")
    end
    plot!(p, fit.t, fit.fitted; color=LINE, lw=2.5, label="fitted total")
    scatter!(p, fit.t, cal.y_raw; color=:black, ms=2.5, alpha=0.7, label="observed")
    return p
end

"""Forecast fan against truth, with the calibration tail for context."""
function plot_forecast(fit, tfull, point, curves, truth; history=30,
                       title="Forecast")
    nd = length(fit.t)
    tf = tfull[(nd + 1):end]
    lo95 = [quantile(@view(curves[i, :]), 0.025) for i in (nd + 1):size(curves, 1)]
    hi95 = [quantile(@view(curves[i, :]), 0.975) for i in (nd + 1):size(curves, 1)]
    lo50 = [quantile(@view(curves[i, :]), 0.25) for i in (nd + 1):size(curves, 1)]
    hi50 = [quantile(@view(curves[i, :]), 0.75) for i in (nd + 1):size(curves, 1)]
    hs = max(1, nd - history + 1)
    p = plot(tf, hi95; fillrange=lo95, fillcolor=BAND, linealpha=0,
             label="95% PI", legend=:topleft, title=wrap_title(title),
             xlabel="time", ylabel="count")
    plot!(p, tf, hi50; fillrange=lo50, fillcolor=RGBA(0.30, 0.45, 0.70, 0.45),
          linealpha=0, label="50% PI")
    plot!(p, tf, point[(nd + 1):end]; color=LINE, lw=2, label="median forecast")
    scatter!(p, fit.t[hs:end], fit.y[hs:end]; color=:black, ms=2.5, alpha=0.7,
             label="observed")
    truth === nothing || scatter!(p, tf, truth; color=:firebrick, ms=3.5, label="truth")
    vline!(p, [fit.t[end] + 0.5]; color=:grey, ls=:dash, label="")
    return p
end

"""Profile likelihood over tau_2, with the 95% interval shaded."""
function plot_profile(pr, nd; title="Profile likelihood over tau_2")
    iv = profile_interval(pr.tau, pr.objective, nd)
    p = plot(pr.tau, pr.objective; color=LINE, lw=2, legend=:topright,
             title=wrap_title(title), xlabel="tau_2 (window-relative)",
             ylabel="objective", label="profile")
    plot!(pr.tau, pr.cold; color=:grey, lw=1.5, ls=:dot, label="cold pass only")
    hline!(p, [iv.threshold]; color=:firebrick, ls=:dash, label="95% threshold")
    vspan!(p, [iv.lo, iv.hi]; color=RGBA(0.75, 0.35, 0.30, 0.12), label="95% interval")
    return p
end

"""Absolute onset against window length: flat means a real calendar feature."""
function plot_window(ws; title="Onset vs calibration window")
    use = [r for r in ws.rows if r.usable]
    bad = [r for r in ws.rows if !r.usable]
    p = plot(title=wrap_title(title), xlabel="calibration length",
             ylabel="absolute onset", legend=:topleft)
    isempty(use) || scatter!(p, [r.nd for r in use], [r.tau2_abs for r in use];
                             color=LINE, ms=6, label="gate-clean")
    isempty(bad) || scatter!(p, [r.nd for r in bad], [r.tau2_abs for r in bad];
                             color=:grey, ms=5, alpha=0.5, marker=:xcross,
                             label="excluded")
    isempty(use) || hline!(p, [mean(r.tau2_abs for r in use)];
                           color=:firebrick, ls=:dash, label="mean")
    return p
end

"""Parameter histograms from the bootstrap draws."""
function plot_parameters(boot, n; title="Bootstrap parameter distributions")
    names = vcat(["r$i" for i in 1:n], ["p$i" for i in 1:n],
                 ["K$i" for i in 1:n], ["gap$i" for i in 2:n])
    cols = vcat(1:n, (n + 1):(2n), (3n + 1):(4n), (4n + 1):(4n + n - 1))
    ps = [histogram(boot.thetas[:, c]; bins=30, legend=false, title=names[k],
                    titlefontsize=8, color=RGB(0.35, 0.50, 0.70), linealpha=0)
          for (k, c) in enumerate(cols)]
    return plot(ps...; layout=(2, ceil(Int, length(ps) / 2)), size=(1000, 560),
                plot_title=wrap_title(title), plot_titlefontsize=10)
end

"""Model comparison by rank: AICc against out-of-sample WIS."""
function plot_rankings(rows; title="AICc vs out-of-sample WIS")
    p = plot(title=wrap_title(title), xlabel="AICc", ylabel="forecast WIS",
             legend=:topleft)
    for n in unique(r.nsub for r in rows)
        sel = [r for r in rows if r.nsub == n]
        scatter!(p, [r.AICc for r in sel], [r.WIS for r in sel];
                 ms=6, label="n = $n")
    end
    return p
end
