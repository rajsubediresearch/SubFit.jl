# =====================================================================
# Objective, bounds, initialization, and the fitting driver
# =====================================================================

"""
    objective(theta, n, t, y, flag, I0, method)

`:normal`  -> sum of squared errors (matches method1=0, dist1=0)
`:poisson` -> negative log-likelihood, dropping the constant log(y!) term
"""
function objective(theta::AbstractVector{<:Real}, n::Integer, t, y, flag::Int,
                   I0::Real, method::Symbol; incidence::Symbol=:analytic)
    yhat, _ = simulate(theta, n, t, flag, I0; incidence=incidence)
    (any(isnan, yhat) || any(isinf, yhat)) && return 1e12
    if method === :normal
        s = 0.0
        @inbounds for j in eachindex(y)
            d = yhat[j] - y[j]
            s += d * d
        end
        return s
    elseif method === :poisson
        s = 0.0
        @inbounds for j in eachindex(y)
            mu = max(yhat[j], 1e-10)
            s += mu - y[j] * log(mu)
        end
        return s
    else
        error("unsupported method: $method (use :normal or :poisson)")
    end
end

"""
    param_bounds(n, y, flag; Kmax_mult=10.0, min_gap=1.0, tend)

Bounds on `[r; p; a; K; gaps]`.

Deliberate deviation from `getbounds.m`: MATLAB uses a fixed `Kmax = 1e10`,
which spans nine orders of magnitude above any plausible value and wrecks the
conditioning of both the optimizer and any random start sampling. Here K is
capped at `Kmax_mult * sum(y)`. Set `Kmax_mult` large to recover MATLAB-like
behaviour when you want a like-for-like comparison.
"""
function param_bounds(n::Integer, y::AbstractVector, flag::Int;
                      Kmax_mult::Float64=10.0, min_gap::Float64=1.0,
                      tend::Float64=1.0, r_bounds=nothing,
                      matlab_r_bounds::Bool=false)
    # r bounds.
    #
    # INHERITED BUG, FIXED HERE. getbounds.m derives these from the FIRST TWO
    # observations: rub = max(abs(y[1:2])) * 50. That is fine for a series
    # starting at 1, 0 (the USA national curve gives rub = 50) and
    # catastrophic for one starting at 0, 0 -- rub collapses toward the floor
    # and r pins against its upper bound before fitting begins. On the 51 US
    # state death series this failed 33 of 35 units with r1@ub, which looked
    # like "the two-wave model is not estimable at state scale" and was
    # nothing of the kind.
    #
    # The fix uses the first two STRICTLY POSITIVE observations and floors the
    # upper bound at 50, so a series that starts with zeros is bounded by its
    # actual early growth rather than by its leading zeros. Pass `r_bounds` to
    # override, or `matlab_r_bounds=true` to reproduce the original behaviour.
    pos = [v for v in y if v > 0]
    head = isempty(pos) ? y[1:min(2, length(y))] : pos[1:min(2, length(pos))]
    rlb = max(sum(abs, head) / length(head) / 200, 1e-8)
    rub = max(maximum(abs, head) * 50, 50.0)
    if r_bounds !== nothing
        rlb, rub = float(r_bounds[1]), float(r_bounds[2])
    elseif matlab_r_bounds
        h2 = y[1:min(2, length(y))]
        rlb = max(sum(abs, h2) / length(h2) / 200, 1e-8)
        rub = max(maximum(abs, h2) * 50, 1000 * rlb)
    end
    Kmax = max(Kmax_mult * sum(abs, y), 100.0)

    plb, pub, alb, aub =
        flag == GGM  ? (0.0, 1.0, 1.0, 1.0) :
        flag == GLM  ? (0.0, 1.0, 1.0, 1.0) :
        flag == GRM  ? (0.0, 1.0, 0.0, 10.0) :
        flag == LM   ? (1.0, 1.0, 1.0, 1.0) :
        flag == RICH ? (1.0, 1.0, 0.0, 10.0) :
                       (1.0, 1.0, 0.0, 10.0)

    lb = vcat(fill(rlb, n), fill(plb, n), fill(alb, n), fill(20.0, n))
    ub = vcat(fill(rub, n), fill(pub, n), fill(aub, n), fill(Kmax, n))
    if n > 1
        lb = vcat(lb, fill(min_gap, n - 1))
        ub = vcat(ub, fill(max(tend, min_gap + 1.0), n - 1))
    end
    return lb, ub
end

"""
    initial_points(n, t, y, lb, ub; nstarts, seed)

One data-informed start plus `nstarts` random starts.

The data-informed start partitions the epidemic by equal cumulative mass:
onset times go at the mass boundaries, `K_i` at the mass of each block. Ten
structured starts beat a hundred blind ones on this surface.
"""
function initial_points(n::Integer, t::AbstractVector, y::AbstractVector,
                        lb::Vector{Float64}, ub::Vector{Float64};
                        nstarts::Integer=10, seed::Integer=20260101)
    yp = max.(y, 0.0)
    total = sum(yp)
    cs = cumsum(yp)
    tend = float(t[end])

    taus = zeros(Float64, n)
    for i in 2:n
        target = total * (i - 1) / n
        idx = findfirst(>=(target), cs)
        taus[i] = idx === nothing ? tend * (i - 1) / n : float(t[idx])
    end

    seedpt = Vector{Float64}(undef, length(lb))
    for i in 1:n
        seedpt[i]        = clamp(0.1, lb[i], ub[i])                     # r
        seedpt[n + i]    = clamp(0.9, lb[n + i], ub[n + i])             # p
        seedpt[2n + i]   = clamp(1.0, lb[2n + i], ub[2n + i])           # a
        seedpt[3n + i]   = clamp(total / n, lb[3n + i], ub[3n + i])     # K
    end
    for i in 2:n
        gap = max(taus[i] - taus[i - 1], lb[4n + i - 1])
        seedpt[4n + i - 1] = clamp(gap, lb[4n + i - 1], ub[4n + i - 1])
    end

    pts = [seedpt]
    rng = Xoshiro(seed)
    for _ in 1:nstarts
        push!(pts, lb .+ rand(rng, length(lb)) .* (ub .- lb))
    end
    return pts
end

"""
    nparams(n, flag, method)

Free-parameter count for AICc. Matches `get_nparams.m` except that the single
`C_thr` parameter is replaced by `n-1` onset gaps — so the count is identical
at n=2 and slightly more penalising at n>=3. That asymmetry is intentional and
honest: the reparameterization really does add parameters at higher n.
"""
function nparams(n::Integer, flag::Int, method::Symbol)
    per = flag == GGM ? 2 : flag == GLM ? 3 : flag == GRM ? 4 :
          flag == LM ? 2 : flag == RICH ? 3 : 2
    m = per * n + max(n - 1, 0)
    method === :normal && (m += 1)   # variance
    return m
end

"""
    aicc(fval, nd, m, method)

`method1=0` uses `nd*log(SSE) + penalty`, matching `getAICc.m` case 0.
For `:poisson`, `fval` is the negative log-likelihood so `-2logL = 2*fval`.
"""
function aicc(fval::Real, nd::Integer, m::Integer, method::Symbol)
    denom = nd - m - 1
    denom <= 0 && return Inf
    pen = 2m + (2m * (m + 1)) / denom
    return method === :normal ? nd * log(fval) + pen : 2 * fval + pen
end


"""
    gap_candidates(n, lb, ub, tau_grid, seed, seedpt)

Candidate onset-gap vectors for the outer grid search.

WHY THIS EXISTS: v0 estimated the onset gaps as free continuous parameters and
relied on multistart to find them. Profiling tau_2 on the national COVID series
showed that fails badly -- the free fit returned SSE 302,963 while pinning
tau_2 = 22 and re-optimizing everything else reached 162,069, a 1.87x better
optimum the unconstrained search never saw. Neighbouring grid points came back
40% worse, so the objective is fine and the SEARCH was the problem.

SubEpiPredict's brute-force grid over C_thr is therefore load-bearing, not just
slow: it is a global search over precisely the direction local optimizers fail
in. This restores that, but cheaply -- because the tau-parameterization
decouples the sub-epidemics into independent non-stiff scalar ODEs, ~20 grid
points here cost far less than MATLAB's ~90 over a coupled stiff system.
"""
function gap_candidates(n::Integer, lb::Vector{Float64}, ub::Vector{Float64},
                        tau_grid::Integer, seed::Integer, seedpt::Vector{Float64})
    n == 1 && return [Float64[]]
    gi = (4n + 1):(5n - 1)
    gmin, gmax = lb[gi[1]], ub[gi[1]]
    cands = Vector{Vector{Float64}}()
    push!(cands, seedpt[gi])                       # data-informed onset
    if n == 2
        for g in range(gmin, gmax; length=tau_grid)
            push!(cands, [g])
        end
    else
        rng = Xoshiro(seed)
        for _ in 1:tau_grid
            push!(cands, gmin .+ rand(rng, n - 1) .* (gmax - gmin))
        end
    end
    return cands
end

"""
    SubEpiFit

Result of fitting an n-sub-epidemic model.
"""
struct SubEpiFit
    n::Int
    flag::Int
    method::Symbol
    theta::Vector{Float64}
    params::SubEpiParams
    fval::Float64
    aicc::Float64
    nparams::Int
    I0::Float64
    t::Vector{Float64}
    y::Vector{Float64}
    fitted::Vector{Float64}
    subcurves::Matrix{Float64}
    converged::Bool
    incidence::Symbol
end

function _run_nlopt(alg::Symbol, x0::Vector{Float64}, lb::Vector{Float64},
                    ub::Vector{Float64}, free::Vector{Int}, fobj, maxeval::Int)
    isempty(free) && return (copy(x0), fobj(x0), :NO_FREE_PARAMS)
    opt = Opt(alg, length(free))
    lower_bounds!(opt, lb[free])
    upper_bounds!(opt, ub[free])
    maxeval!(opt, maxeval)
    xtol_rel!(opt, 1e-8)
    ftol_rel!(opt, 1e-10)
    full = copy(x0)
    function wrapped(xf::Vector, grad::Vector)
        isempty(grad) || fill!(grad, 0.0)
        full[free] .= xf
        return fobj(full)
    end
    min_objective!(opt, wrapped)
    (minf, minx, ret) = optimize(opt, clamp.(x0[free], lb[free], ub[free]))
    out = copy(x0)
    out[free] .= minx
    return out, minf, ret
end

"""
    fit_subepidemic(t, y, n; tau_grid=20, ...)

Two-phase fit:

  1. GRID PHASE -- the onset gaps are pinned to each candidate in turn and all
     remaining parameters optimized (SBPLX from `grid_starts` starts). This is
     the global search over the direction that defeats local methods.
  2. POLISH PHASE -- from the best grid point, the gaps are released and the
     whole vector is refined (SBPLX then COBYLA), keeping the lowest objective.

Set `tau_grid=0` to recover the v0 behaviour (free gaps, no grid). Do not
interpret any fit produced that way without checking `profile_tau2` first.
"""
function fit_subepidemic(t::AbstractVector, y::AbstractVector, n::Integer;
                         flag::Int=GLM, method::Symbol=:normal,
                         I0::Real=y[1], nstarts::Integer=10,
                         maxeval::Integer=3000, seed::Integer=20260101,
                         Kmax_mult::Float64=10.0, min_gap::Float64=1.0,
                         polish::Bool=true, threaded::Bool=true,
                         incidence::Symbol=:analytic, tau_grid::Integer=20,
                         r_bounds=nothing, matlab_r_bounds::Bool=false)

    tend = float(t[end])
    lb, ub = param_bounds(n, y, flag; Kmax_mult=Kmax_mult, min_gap=min_gap,
                             tend=tend, r_bounds=r_bounds, matlab_r_bounds=matlab_r_bounds)
    free_all = [i for i in eachindex(lb) if ub[i] > lb[i]]
    fobj = th -> objective(th, n, t, y, flag, I0, method; incidence=incidence)
    starts = initial_points(n, t, y, lb, ub; nstarts=nstarts, seed=seed)

    theta, fval = starts[1], Inf

    if n == 1 || tau_grid <= 0
        results = Vector{Tuple{Vector{Float64},Float64,Symbol}}(undef, length(starts))
        if threaded && Threads.nthreads() > 1
            Threads.@threads for k in eachindex(starts)
                results[k] = _run_nlopt(:LN_SBPLX, starts[k], lb, ub, free_all, fobj, maxeval)
            end
        else
            for k in eachindex(starts)
                results[k] = _run_nlopt(:LN_SBPLX, starts[k], lb, ub, free_all, fobj, maxeval)
            end
        end
        b = argmin([r[2] for r in results])
        theta, fval = results[b][1], results[b][2]
    else
        # ---- phase 1: grid over the onset gaps, with warm sweeps ----
        #
        # The cold pass alone is not enough. On the national COVID series a
        # 3-start cold grid reached SSE 222,240 while the same grid swept warm
        # (profile_tau2) reached 117,463. Adjacent grid points have adjacent
        # optima, so carrying each solution to its neighbour rescues the points
        # where the cold search fails.
        cands = gap_candidates(n, lb, ub, tau_grid, seed, starts[1])
        gi = (4n + 1):(5n - 1)
        ncand = length(cands)
        grid_starts = max(1, min(3, length(starts)))

        cand_th = [copy(starts[1]) for _ in 1:ncand]
        cand_f  = fill(Inf, ncand)

        pin = function (ci::Int)
            lb2, ub2 = copy(lb), copy(ub)
            for (k, gidx) in enumerate(gi)
                g = clamp(cands[ci][k], lb[gidx], ub[gidx])
                lb2[gidx] = g; ub2[gidx] = g
            end
            free2 = [i for i in eachindex(lb2) if ub2[i] > lb2[i]]
            return lb2, ub2, free2
        end

        jobs = [(c, s) for c in 1:ncand for s in 1:grid_starts]
        res = Vector{Tuple{Int,Vector{Float64},Float64}}(undef, length(jobs))
        run_job = function (idx::Int)
            (ci, si) = jobs[idx]
            lb2, ub2, free2 = pin(ci)
            x0 = copy(starts[si])
            for (k, gidx) in enumerate(gi)
                x0[gidx] = clamp(cands[ci][k], lb[gidx], ub[gidx])
            end
            th, f, _ = _run_nlopt(:LN_SBPLX, x0, lb2, ub2, free2, fobj, maxeval)
            res[idx] = (ci, th, f)
            return
        end
        if threaded && Threads.nthreads() > 1
            Threads.@threads for idx in eachindex(jobs)
                run_job(idx)
            end
        else
            for idx in eachindex(jobs)
                run_job(idx)
            end
        end
        for (ci, th, f) in res
            if f < cand_f[ci]
                cand_f[ci] = f
                cand_th[ci] = th
            end
        end

        # warm sweeps, forward then backward, in onset order
        if ncand > 1
            order = sortperm([sum(c) for c in cands])
            for pass in (order, reverse(order))
                prev = 0
                for ci in pass
                    if prev != 0
                        lb2, ub2, free2 = pin(ci)
                        x0 = copy(cand_th[prev])
                        for (k, gidx) in enumerate(gi)
                            x0[gidx] = clamp(cands[ci][k], lb[gidx], ub[gidx])
                        end
                        for alg in (:LN_SBPLX, :LN_COBYLA)
                            th, f, _ = _run_nlopt(alg, x0, lb2, ub2, free2, fobj, maxeval)
                            if f < cand_f[ci]
                                cand_f[ci] = f
                                cand_th[ci] = th
                            end
                            x0 = th
                        end
                    end
                    prev = ci
                end
            end
        end

        b = argmin(cand_f)
        theta, fval = cand_th[b], cand_f[b]
    end

    # ---- phase 2: release the gaps and refine ----
    if polish
        seeds = n == 1 || tau_grid <= 0 ? (theta, starts[1]) : (theta,)
        for x0 in seeds
            for alg in (:LN_SBPLX, :LN_COBYLA)
                th2, f2, _ = _run_nlopt(alg, x0, lb, ub, free_all, fobj, maxeval)
                f2 < fval && ((theta, fval) = (th2, f2))
            end
        end
        th3, f3, _ = _run_nlopt(:LN_COBYLA, theta, lb, ub, free_all, fobj, maxeval)
        f3 < fval && ((theta, fval) = (th3, f3))
    end

    P = unpack(theta, n)
    yhat, sub = simulate(P, t, flag, I0; incidence=incidence)
    m = nparams(n, flag, method)
    return SubEpiFit(n, flag, method, theta, P, fval, aicc(fval, length(y), m, method),
                     m, float(I0), collect(float.(t)), collect(float.(y)),
                     yhat, sub, isfinite(fval) && fval < 1e11, incidence)
end
