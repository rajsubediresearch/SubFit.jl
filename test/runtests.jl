using SubFit, Test, DelimitedFiles, Statistics, Random

const ROOT = normpath(joinpath(@__DIR__, ".."))

@testset "SubFit" begin

    @testset "MATLAB-parity preprocessing" begin
        fx = readdlm(joinpath(@__DIR__, "fixture_usa52_smoothed.csv"), ',', skipstart=1)
        y_raw = load_series(joinpath(ROOT, "input",
            "cumulative-daily-coronavirus-deaths-USA-05-11-2020.txt");
            column=52, cumulative=true)
        @test length(y_raw) == 75
        @test y_raw ≈ fx[:, 2]
        cal = calibration(y_raw; smoothfactor=7, calibration_period=90)
        @test length(cal.y) == 75          # capped at available data
        @test cal.t0 == 0.0                # no leading points dropped
        @test cal.y ≈ fx[:, 3]             # endpoint-shrinking movmean
        @test cal.t[1] == 0.0
    end

    @testset "movmean endpoints" begin
        @test movmean([1.0,2,3,4,5], 3) ≈ [1.5, 2.0, 3.0, 4.0, 4.5]
        @test movmean([1.0,2,3], 1) ≈ [1.0,2,3]
    end

    @testset "growth kernels" begin
        @test growth_rhs(0.0, 1, 1, 1, 100, GLM) == 0.0
        @test growth_rhs(100.0, 1, 1, 1, 100, GLM) ≈ 0.0 atol=1e-12
        @test growth_rhs(50.0, 0.2, 1.0, 1.0, 100.0, GLM) ≈ 0.2*50*0.5
        @test growth_rhs(50.0, 0.2, 1.0, 1.0, 100.0, LM) ≈ 0.2*50*0.5
    end

    @testset "onset ordering" begin
        P = unpack([0.2, 0.3, 0.9, 0.9, 1.0, 1.0, 500.0, 700.0, 12.0], 2)
        @test P.tau == [0.0, 12.0]
        @test issorted(P.tau)
    end

    @testset "simulate: second wave is silent before its onset" begin
        t = collect(0.0:1.0:40.0)
        P = unpack([0.3, 0.3, 0.9, 0.9, 1.0, 1.0, 400.0, 400.0, 20.0], 2)
        tot, sub = simulate(P, t, GLM, 1.0)
        @test all(sub[t .< 20.0, 2] .== 0.0)
        @test any(sub[t .>= 25.0, 2] .> 0.0)
        @test tot ≈ vec(sum(sub, dims=2))
    end

    @testset "n=1 recovers a single GLM wave" begin
        t = collect(0.0:1.0:50.0)
        truth = unpack([0.35, 0.95, 1.0, 3000.0], 1)
        y, _ = simulate(truth, t, GLM, 1.0)
        f = fit_subepidemic(t, y, 1; flag=GLM, method=:normal,
                            nstarts=8, seed=7, threaded=false)
        @test f.converged
        @test f.fval < 1e-2 * sum(abs2, y)   # loose: recovery to within 1% of scale
    end

    @testset "AICc bookkeeping" begin
        # GLM, n=2, normal: 3*2 + 1 gap + 1 variance = 8, same as MATLAB's
        # 3*2 + 1 (C_thr) + 1 (variance)
        @test nparams(2, GLM, :normal) == 8
        @test nparams(1, GLM, :normal) == 4
        @test aicc(100.0, 75, 8, :normal) > 0
        @test isinf(aicc(100.0, 5, 8, :normal))   # guard on tiny samples
    end

    @testset "metrics" begin
        @test length(WIS_ALPHAS) == 11
        @test WIS_ALPHAS[1] == 0.02
        y = [10.0, 20.0]
        curves = repeat(y, 1, 200) .+ randn(Xoshiro(1), 2, 200)
        @test coverage(curves, y) == 100.0
        @test wis(curves, y) > 0
        @test mae([1.0,2.0],[1.0,3.0]) ≈ 0.5
    end

    @testset "ensemble weights" begin
        a = [100.0, 104.0]
        @test sum(ensemble_weights(a; scheme=:akaike)) ≈ 1.0
        @test ensemble_weights(a; scheme=:akaike)[1] > 0.5
        @test ensemble_weights(a; scheme=:equal) ≈ [0.5, 0.5]
        @test sum(ensemble_weights(a; scheme=:aicc_recip)) ≈ 1.0
    end

    @testset "discrete vs analytic incidence" begin
        t = collect(0.0:1.0:40.0)
        P = unpack([0.3, 0.3, 0.9, 0.9, 1.0, 1.0, 400.0, 400.0, 20.0], 2)
        ta, _ = simulate(P, t, GLM, 1.0; incidence=:analytic)
        td, _ = simulate(P, t, GLM, 1.0; incidence=:discrete)
        @test length(ta) == length(td)
        @test td[1] ≈ 1.0                       # seed only, no spurious jump
        @test all(td[t .< 20.0] .>= 0.0)
        @test isapprox(sum(ta), sum(td); rtol=0.25)   # same scale, not equal
        @test_throws ErrorException simulate(P, t, GLM, 1.0; incidence=:bogus)
        f = fit_subepidemic(t, ta, 2; flag=GLM, nstarts=4, seed=3,
                            threaded=false, incidence=:discrete)
        @test f.incidence == :discrete
    end

    @testset "profile smoothness gate" begin
        # A steep, clearly unimodal profile: global minimum 100 at index 5.
        # Steepness matters -- the injected spike below has to be far below its
        # NEIGHBOURS while staying ABOVE the global minimum, which a shallow
        # parabola cannot express. Getting that wrong twice is what made the
        # earlier versions of this test misleading.
        f = [2000.0, 1200, 700, 300, 100, 300, 700, 1200, 2000, 3000]
        @test argmin(f) == 5
        @test profile_dip(f) ≈ 0.0 atol=1e-12
        @test is_smooth_profile(f)

        # A SECOND local minimum, deep relative to its neighbours (700, 2000)
        # but still above the global minimum -- the real failure signature.
        g = copy(f); g[8] = 150.0
        @test argmin(g) == 5           # global minimum unchanged
        @test profile_dip(g) > 0.5
        @test !is_smooth_profile(g)

        # A spike AT the global minimum is skipped by design: on any smooth
        # profile the true minimum sits below both neighbours. warm_gain, not
        # dip, is what catches an unreliable cold search there.
        h = copy(f); h[5] = 30.0
        @test profile_dip(h) ≈ 0.0 atol=1e-12

        taus = collect(1.0:10.0)
        iv = profile_interval(taus, f, 75)
        @test iv.lo <= 5.0 <= iv.hi
        @test iv.threshold > minimum(f)
        @test iv.npoints >= 1
    end

    @testset "profile_tau2 warm sweeps" begin
        t = collect(0.0:1.0:45.0)
        truth = unpack([0.35, 0.40, 0.95, 0.95, 1.0, 1.0, 2000.0, 4000.0, 20.0], 2)
        y, _ = simulate(truth, t, GLM, 1.0)
        pr = profile_tau2(t, y; grid=10.0:5.0:35.0, nstarts=3, threaded=false)
        @test length(pr.tau) == length(pr.objective) == 6
        @test all(pr.objective .<= pr.cold .+ 1e-9)   # sweeps never worsen
        @test pr.warm_gain >= 0.0
        @test isfinite(minimum(pr.objective))
    end

    @testset "gridded fit finds the onset" begin
        # two clearly separated waves; the free-gap search missed optima like
        # this, the grid phase should not
        t = collect(0.0:1.0:60.0)
        truth = unpack([0.35, 0.40, 0.95, 0.95, 1.0, 1.0, 3000.0, 5000.0, 25.0], 2)
        y, _ = simulate(truth, t, GLM, 1.0)
        f = fit_subepidemic(t, y, 2; flag=GLM, nstarts=6, seed=11,
                            tau_grid=15, threaded=false)
        @test f.converged
        @test f.fval < 5e-2 * sum(abs2, y)
        # the gridded fit should beat the ungridded one, or at worst tie
        f0 = fit_subepidemic(t, y, 2; flag=GLM, nstarts=6, seed=11,
                             tau_grid=0, threaded=false)
        @test f.fval <= f0.fval * 1.0001
    end

    @testset "effort stability gate" begin
        # a settled search
        st = (objective = [100.0, 99.5], best = [100.0, 99.5],
              improvement = [NaN, 0.005], regression = [0.0, 0.0])
        @test is_stable_fit(st)

        # still improving 28% when effort doubles
        st2 = (objective = [162069.0, 117463.0], best = [162069.0, 117463.0],
               improvement = [NaN, 0.2752], regression = [0.0, 0.0])
        @test !is_stable_fit(st2)

        # GOT WORSE with more effort. The first version passed this because
        # -0.013 <= 0.02. A negative improvement is instability.
        st3 = (objective = [100.0, 101.3], best = [100.0, 100.0],
               improvement = [NaN, 0.0], regression = [0.0, 0.013])
        @test !is_stable_fit(st3; tol=0.01)

        t = collect(0.0:1.0:45.0)
        truth = unpack([0.35, 0.95, 1.0, 3000.0], 1)
        y, _ = simulate(truth, t, GLM, 1.0)
        s1 = fit_stability(t, y, 1; levels=[(0, 4), (0, 8)], flag=GLM, threaded=false)
        @test length(s1.objective) == 2
        @test s1.best[2] <= s1.best[1] + 1e-9
        @test all(s1.regression .>= 0)
        @test s1.fit.fval == minimum(s1.objective)
    end

    @testset "bound-adjacency gate" begin
        # a fit whose second onset sits on min_gap is exactly the state-run
        # failure that both earlier gates missed
        t = collect(0.0:1.0:45.0)
        P = unpack([0.3, 0.3, 0.9, 0.9, 1.0, 1.0, 900.0, 900.0, 18.0], 2)
        y, _ = simulate(P, t, GLM, 1.0)
        f = fit_subepidemic(t, y, 2; flag=GLM, nstarts=4, seed=5,
                            tau_grid=8, threaded=false)
        h = bound_hits(f)
        @test h isa Vector{String}
        @test all(x -> occursin("@lb", x) || occursin("@ub", x), h)

        pinned = SubEpiFit(2, GLM, :normal,
                           [0.3, 0.3, 0.9, 0.9, 1.0, 1.0, 900.0, 900.0, 1.0],
                           unpack([0.3,0.3,0.9,0.9,1.0,1.0,900.0,900.0,1.0], 2),
                           1.0, 1.0, 8, 1.0, t, y, y, zeros(length(t), 2),
                           true, :analytic)
        @test any(startswith(x, "gap2@lb") for x in bound_hits(pinned))
    end

    @testset "absolute vs window-relative onsets" begin
        # calibration keeps the LAST m points, so a shorter window starts later
        y = collect(1.0:100.0)
        full = calibration(y; smoothfactor=1)
        short = calibration(y; smoothfactor=1, calibration_period=60)
        @test full.t0 == 0.0
        @test short.t0 == 40.0
        @test short.y_raw[1] == y[41]

        # a STATIONARY absolute onset looks like slope 1.0 in window-relative
        # terms -- the failure that produced the retracted "tail smoother"
        # result. On absolute onsets the same data gives slope ~0.
        nds  = [55.0, 60.0, 65.0, 70.0, 75.0]
        t0s  = [20.0, 15.0, 10.0, 5.0, 0.0]        # 75 - nd
        rel  = [24.81, 29.32, 34.24, 39.23, 44.06]
        abso = t0s .+ rel
        @test isapprox(window_slope(nds, rel).slope, 0.968; atol=0.02)
        @test abs(window_slope(nds, abso).slope) < 0.05
        @test std(abso) < 0.5                       # onset is stable
        @test !is_window_stable((verdict = :tracking, slope = 0.97, nusable = 5))
        @test is_window_stable((verdict = :stable, slope = 0.02, nusable = 5))
    end

    @testset "r bounds survive a series starting at zero" begin
        # MATLAB's getbounds.m derives rub from the FIRST TWO observations, so a
        # series beginning 0, 0 collapses the upper bound and r pins against it
        # before fitting starts. That failed 33 of 35 US state death series with
        # r1@ub and masqueraded as "the two-wave model is not estimable at state
        # scale". The fix uses the first two strictly POSITIVE values.
        y = vcat(zeros(8), [1.0, 3.0, 9.0, 25.0, 60.0, 120.0, 200.0, 260.0,
                            240.0, 180.0, 120.0, 70.0, 40.0, 20.0])
        lb, ub = param_bounds(2, y, GLM; tend=float(length(y) - 1))
        @test ub[1] >= 50.0                 # not collapsed by the leading zeros
        @test lb[1] < ub[1]

        lb0, ub0 = param_bounds(2, y, GLM; tend=float(length(y) - 1),
                                matlab_r_bounds=true)
        @test ub0[1] < 1e-3                 # the original behaviour, reproduced
        @test ub[1] > 1e4 * ub0[1]

        # explicit override wins over both
        lbx, ubx = param_bounds(2, y, GLM; tend=float(length(y) - 1),
                                r_bounds=(1e-4, 12.0))
        @test lbx[1] ≈ 1e-4 && ubx[1] ≈ 12.0

        # a fit on this series must not pin r at its bound
        f = fit_subepidemic(collect(0.0:(length(y) - 1)), y, 2; flag=GLM,
                            nstarts=6, seed=17, tau_grid=10, threaded=false)
        @test !any(startswith(h, "r1@ub") for h in bound_hits(f))
    end

    @testset "window-shift gate" begin
        # the observed national COVID pattern: tau_2 = nd - 30.9, slope ~1
        nds  = [55.0, 60.0, 65.0, 70.0, 75.0]
        taus = [24.81, 29.32, 34.24, 39.23, 44.06]
        w = window_slope(nds, taus)
        @test isapprox(w.slope, 0.968; atol=0.01)
        @test w.offset_sd < 1.0            # nd - tau is essentially constant

        # a genuine calendar-time onset: tau fixed while the window grows
        w2 = window_slope(nds, fill(22.0, 5))
        @test isapprox(w2.slope, 0.0; atol=1e-9)
        @test w2.tau_sd ≈ 0.0 atol=1e-12

        # three verdicts, deliberately distinct
        @test is_window_stable((verdict = :stable, slope = 0.0, nusable = 4))
        @test !is_window_stable((verdict = :tracking, slope = 0.97, nusable = 4))
        @test !is_window_stable((verdict = :not_enough, slope = NaN, nusable = 2))
        @test !is_window_stable((verdict = :insufficient, slope = NaN, nusable = 2))
        @test is_window_stable((verdict = :insufficient, slope = NaN, nusable = 2);
                               allow_insufficient=true)
        @test is_window_stable((verdict = :not_applicable, slope = 0.0, nusable = 0))

        # two usable windows must NOT be enough: a line through two points is
        # exact regardless of whether any trend exists
        t = collect(0.0:1.0:60.0)
        P = unpack([0.35, 0.40, 0.95, 0.95, 1.0, 1.0, 2500.0, 4000.0, 25.0], 2)
        y, _ = simulate(P, t, GLM, 1.0)
        ws = window_shift(y, 2; deltas=[0, -5], stability_levels=[(8, 4)],
                          flag=GLM, threaded=false)
        @test ws.verdict === :insufficient      # at most 2 windows available
        @test !is_window_stable(ws)

        ws3 = window_shift(y, 2; deltas=[0, -5, -10, -15],
                           stability_levels=[(8, 4)], flag=GLM, threaded=false)
        @test length(ws3.rows) == 4
        @test ws3.verdict in (:stable, :tracking, :insufficient)
        @test haskey(first(ws3.rows), :usable)
    end
end
