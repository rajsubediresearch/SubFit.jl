# =====================================================================
# The tau-parameterized n-sub-epidemic model
#
# Departure from SubEpiPredict: instead of a single threshold C_thr that
# triggers the (i+1)-th sub-epidemic when sub-epidemic i's cumulative curve
# crosses it, each sub-epidemic has its own ONSET TIME tau_i, parameterized
# through strictly positive gaps so that tau_1 < tau_2 < ... < tau_n by
# construction. Consequences:
#   * the sub-epidemics decouple completely -> n independent SCALAR
#     non-stiff ODEs instead of one coupled system with a latching switch
#   * the outer grid search over C_thr disappears
#   * the n! label-switching symmetry of the mixture is removed
# This is a DIFFERENT model, not a faster implementation of the same one.
# =====================================================================

# Growth-model codes match MATLAB's flag1
const GGM  = 0
const GLM  = 1
const GRM  = 2
const LM   = 3
const RICH = 4
const GOM  = 5

const MODEL_NAMES = Dict(GGM => "GGM", GLM => "GLM", GRM => "GRM",
                         LM => "LM", RICH => "Richards", GOM => "Gompertz")

"""
    growth_rhs(C, r, p, a, K, flag)

dC/dt for a single sub-epidemic building block. Mirrors
`modifiedLogisticGrowthPatch.m` minus the invasion indicator (which the
tau-parameterization makes unnecessary).
"""
@inline function growth_rhs(C::Real, r::Real, p::Real, a::Real, K::Real, flag::Int)
    C <= 0 && return 0.0
    if flag == GGM
        return r * C^p
    elseif flag == GLM
        return r * C^p * (1 - C / K)
    elseif flag == GRM
        b = 1 - C / K
        return b <= 0 ? 0.0 : r * C^p * b^a
    elseif flag == LM
        return r * C * (1 - C / K)
    elseif flag == RICH
        return r * C * (1 - (C / K)^a)
    elseif flag == GOM
        return C >= K ? 0.0 : r * C * log(K / C)
    else
        error("unknown growth model flag: $flag")
    end
end

"""
    SubEpiParams

Parameters of an n-sub-epidemic trajectory. `tau[1]` is always 0.
"""
struct SubEpiParams
    r::Vector{Float64}
    p::Vector{Float64}
    a::Vector{Float64}
    K::Vector{Float64}
    tau::Vector{Float64}
end

nsub(P::SubEpiParams) = length(P.r)

"""
    unpack(theta, n)

Parameter vector layout: `[r(1:n); p(1:n); a(1:n); K(1:n); gap(2:n)]`,
where `tau_i = tau_{i-1} + gap_i` and `gap_i > 0`.
"""
function unpack(theta::AbstractVector{<:Real}, n::Integer)
    r = Float64.(theta[1:n])
    p = Float64.(theta[(n + 1):(2n)])
    a = Float64.(theta[(2n + 1):(3n)])
    K = Float64.(theta[(3n + 1):(4n)])
    tau = zeros(Float64, n)
    for i in 2:n
        tau[i] = tau[i - 1] + theta[4n + i - 1]
    end
    return SubEpiParams(r, p, a, K, tau)
end

ntheta(n::Integer) = 4n + max(n - 1, 0)

"""
    simulate(P, t, flag, I0; reltol, abstol)

Return `(total_incidence, per_subepidemic_incidence)` on the time grid `t`.

Each sub-epidemic is integrated independently on `(tau_i, t_end)` with a
non-stiff solver; incidence is read off analytically as `growth_rhs(C_i(t))`
rather than by numerical differencing. Returns NaNs if any solve fails, which
the objective converts into a large finite penalty.
"""
function simulate(P::SubEpiParams, t::AbstractVector{<:Real}, flag::Int, I0::Real;
                  reltol::Float64=1e-8, abstol::Float64=1e-8,
                  incidence::Symbol=:analytic)
    n = nsub(P)
    nt = length(t)
    tend = float(t[end])
    inc = zeros(Float64, nt, n)

    for i in 1:n
        P.tau[i] >= tend && continue
        C0 = i == 1 ? max(float(I0), 1e-8) : 1.0
        ri, pi_, ai, Ki = P.r[i], P.p[i], P.a[i], P.K[i]
        f(u, par, tt) = growth_rhs(u, ri, pi_, ai, Ki, flag)
        prob = ODEProblem(f, C0, (P.tau[i], tend))
        sol = with_logger(NullLogger()) do
            solve(prob, Tsit5(); reltol=reltol, abstol=abstol)
        end
        SciMLBase.successful_retcode(sol) || return (fill(NaN, nt), fill(NaN, nt, n))

        if incidence === :analytic
            @inbounds for j in 1:nt
                tj = float(t[j])
                tj < P.tau[i] && continue
                inc[j, i] = growth_rhs(sol(tj), ri, pi_, ai, Ki, flag)
            end
        elseif incidence === :discrete
            # MATLAB reads incidence off the cumulative solution as
            # [y(1); diff(y)] (Run_Fit/plotFit), not from the RHS. On daily
            # data during fast growth the two differ by ~19% in SSE on the
            # COVID example, and the paper's AICc is on THIS scale.
            prev = NaN
            @inbounds for j in 1:nt
                tj = float(t[j])
                tj < P.tau[i] && continue
                Cj = sol(tj)
                if isnan(prev)
                    # first grid point for this sub-epidemic: drop the seed so
                    # it does not show up as spurious incidence. Mirrors
                    # MATLAB's totinc(1) = totinc(1) - (npatches-1).
                    inc[j, i] = i == 1 ? Cj : Cj - 1.0
                else
                    inc[j, i] = Cj - prev
                end
                prev = Cj
            end
        else
            error("incidence must be :analytic or :discrete, got :$incidence")
        end
    end

    return (vec(sum(inc, dims=2)), inc)
end

simulate(theta::AbstractVector, n::Integer, t, flag, I0; kw...) =
    simulate(unpack(theta, n), t, flag, I0; kw...)
