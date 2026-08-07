# SubFit.jl

A Julia reformulation of the ensemble *n*-sub-epidemic framework
(Chowell et al. 2024, *Infectious Disease Modelling* 9:411–436; MATLAB toolbox
`SubEpiPredict`), with the same output artifacts and a set of convergence gates
the original does not have.

Sibling to `GrowthFit.jl` (single-wave phenomenological), `MechFit.jl`
(mechanistic) and `RenewFit.jl` (renewal / R(t)).

## The reformulation

SubEpiPredict triggers sub-epidemic *i+1* when sub-epidemic *i*'s cumulative
curve crosses a threshold `C_thr`, estimated by grid search. SubFit gives each
wave its own **onset time** `τ_i`, parameterized through positive gaps so
`τ₁ < τ₂ < …` by construction.

| | SubEpiPredict | SubFit |
|---|---|---|
| onset | one threshold `C_thr`, grid-searched | per-wave onset times `τ_i` |
| system | *n* coupled states, latching switch, `ode15s` | *n* independent scalar ODEs, `Tsit5` |
| state | globals mutated inside the RHS | none |
| optimizer | `fmincon` sqp + MultiStart | grid phase with warm sweeps, then SBPLX + COBYLA |
| bootstrap | sequential `for real=1:M` | threaded, per-replicate `Xoshiro(hash(seed,b))` |

Parameter counts are identical at n=2 (3n + 1 `C_thr` versus 3n + (n−1) gaps,
both plus variance), so **AICc is directly comparable** there.

## Install and run

```
julia --project=. setup.jl                            # once
julia --project=. --threads=21 test/runtests.jl
julia --project=. --threads=21 examples/run_boundary.jl

julia --project=examples examples/setup_examples.jl   # plotting env, once
julia --project=examples --threads=21 examples/run_report.jl
julia --project=examples --threads=21 examples/run_rolling.jl
julia --project=examples --threads=21 examples/run_units.jl
```

Scripts take a dataset name, defaulting to `covid_deaths_usa`; configuration
lives in `examples/datasets.jl`. Output goes to
`output/<dataset>/{report,rolling,units}`.

## The gates — read this before trusting any number

**This project produced retracted findings from fits that had not converged.**
Four gates exist because of those failures, and each was added after the
previous set missed something:

| gate | catches | added after |
|---|---|---|
| `profile_dip` | a second deep local minimum in the τ₂ profile | a fit stuck 1.87× worse than reachable |
| `fit_stability` | an answer still moving as search effort rises | a *smooth* profile that was uniformly stuck |
| `bound_hits` | a parameter pinned to its bound | τ₂ = `min_gap` in 33 of 35 units, invisible to both gates above |
| `window_shift` | an onset that follows the end of the data | no single-window gate can see a window-relative parameter |

`gated_fit` runs all four and returns `pass=false` with a reason. Analysis
scripts skip failing series and **count them** — a high failure rate is a
finding, not a nuisance.

Smoothness alone is not sufficient, and neither is stability: a search that
lands in the same wrong place at every effort level looks perfectly settled.

## Results on US COVID-19 deaths

At origin 75 (the paper's calibration window), discrete incidence:

| | SSE | AICc | forecast MAE | WIS | coverage |
|---|---|---|---|---|---|
| n=1 | 2,041,389 | 1098.26 | 518.7 | 348.5 | 50.0% |
| **n=2** | **117,454** | **893.72** | **362.3** | **229.1** | 76.7% |
| paper, 1st ranked | ~782,000 | 1036.00 | 216.8 | 134.4 | 93.3% |

Calibration metrics reproduce the paper closely (WIS 118.6 against ~117–129,
coverage 93.3% against ~91–96%). SubFit fits substantially better and still
forecasts worse than the published MATLAB result — **the one unresolved
tension**, and much weaker than an earlier version of this claim (see
Corrections).

**Onset is well identified.** Across three gate-clean windows the absolute
onset is day 44.06, 44.23, 44.24 — mean 44.18, **SD 0.099 days** — and the
decomposition is stable to under 5% (K = [46817, 63614], [47446, 63374],
[49304, 61035]).

**Unit replication (14 gate-clean state series).** AICc picks n=2 in 14/14;
out-of-sample WIS picks n=1 in 5/14; agreement 9/14 (64%, binomial p = 0.42).
Median WIS marginally favours n=2 (7.56 vs 8.31), coverage favours n=1 (83.3%
vs 90.0%). **Underpowered and inconclusive** — at this agreement rate even all
35 units would not reach significance, because each unit contributes one binary
comparison. Units failing the stability gate are likely the noisier series, so
the 14 are not a random sample of the 35.

Absolute onsets across units range from 12.5 to 59.3 (median 42.9), with the
national onset near that median — consistent with asynchronous state epidemics,
though not yet a test of the aggregation hypothesis.

## Findings about the original framework

**The `C_thr` grid search is load-bearing, not merely slow.** Estimating onsets
as free continuous parameters and relying on multistart fails badly: the free
fit reached SSE 302,963 where pinning τ₂ and re-optimizing reached 117,463. The
brute-force grid is a global search over precisely the direction local
optimizers fail in. SubFit restores it — but over decoupled non-stiff scalar
ODEs, so ~20 grid points cost far less than MATLAB's ~90 over a coupled stiff
system.

**`getbounds.m` collapses on series that start at zero.** The r upper bound is
derived from the first two observations (`rub = max(abs(y[1:2]))*50`). Fine for
the national series (starts 1, 0 → rub = 50); catastrophic for state series
starting 0, 0, where `rub` collapses and r pins against it before fitting
begins. This failed 33 of 35 units and looked like a modelling limitation.
SubFit uses the first two strictly *positive* values and floors `rub` at 50;
`matlab_r_bounds=true` reproduces the original. **The published toolbox has the
same fragility.**

**The onset rule is mismatched to the framework's own motivation.** `C_thr`
encodes *endogenous* triggering — wave *i+1* fires when wave *i* reaches a
size. Spatial aggregation, which the framework cites as its motivation, is
*exogenous* asynchrony: places seed at different times for reasons unrelated to
each other's cumulative counts. Free onset times are the better-specified model
for aggregated data; `C_thr` is right for genuine resurgence in one well-mixed
population. This argument depends on no fit at all.

**Figure 8's caption and legend disagree.** The caption says 1st-ranked, every
panel legend says "2nd Ranked Model". Evaluating its parameters with the
rank-1 `C_thr` gives τ₂ = 67.78 and SSE 5×10⁷ (nonsense); with the rank-2
`C_thr` it gives τ₂ = 25.48 and reproduces the paper's 2nd-ranked metrics to
within ~5%. The legend is right.

**Akaike ensemble weights degenerate here.** ΔAICc between n=1 and n=2 is large
enough that `w₁ = 1.000` at every origin tested — the ensemble never ensembles,
and its metrics are identical to the top model alone. Equal weighting beat it.

## Corrections

Findings from this work that were **wrong and are retracted**. Recorded here
rather than deleted, because the failure modes are more instructive than the
conclusions were.

**"The second sub-epidemic is a window-relative tail smoother."** Retracted.
`calibration()` keeps the *last* `calibration_period` points, so shortening the
window moves its **start** while the end stays fixed, and τ is measured from
t=0 at each window's start. A perfectly stationary onset therefore appears to
shift by exactly one day per day of window change — the entire source of the
observed "slope 1.0". In absolute time the same five fits give onset 45.06,
45.23, 45.24, 45.32, 45.81. `CalibrationData` now carries `t0`, and
`absolute_onset(cal, fit, i)` exists so this cannot recur silently.

**"SubFit fits are not reproducible across runs."** Retracted. The two scripts
that disagreed were fitting different subsets of the data, for the same reason.
There is no thread-scheduling non-determinism.

**"Better optimization makes forecasts worse."** Retracted earlier in the same
work. The fit that appeared to demonstrate it was stuck in a local minimum
1.87× worse than reachable.

**"The two-wave model is not estimable at state scale."** Retracted. That was
the inherited `getbounds.m` failure above; with corrected r bounds the number
of gate-clean units went from 2 to 14.

## Known limitations

- **Only `:normal` and `:poisson` objectives.** Negative binomial (`method1` =
  3/4/5 in the paper) is not implemented.
- **`n ≥ 3` is untested.** `gap_candidates` falls back to random draws rather
  than a grid there, so multi-peak series beyond two waves are not really
  supported.
- **Forecast interval coverage is poor** away from the calibration window
  (down to 10% at some origins) even where calibration coverage is nominal. The
  parametric bootstrap propagates parameter uncertainty but nothing structural.
- **`incidence=:discrete` is the comparable scale.** MATLAB reads incidence off
  the cumulative solution as `[y(1); diff(y)]`; the analytic RHS reading differs
  by ~19% in SSE. The paper's AICc is on the discrete scale.
- **Doubling times and R(t)** from the MATLAB toolbox are not ported.

## Examples

| script | what it shows |
|---|---|
| `run_report.jl` | gated single-origin report: profile, gates, fit, forecast, CSVs, plots |
| `run_rolling.jl` | rolling origins with gates, AICc vs out-of-sample WIS |
| `run_units.jl` | per-unit replication across the 51 state columns |
| `run_boundary.jl` | absolute onset vs calibration window, both anchors |
| `kmax_check.jl` | rules out the K bound as a cause of saturation |
| `discrete_check.jl` | analytic vs discrete incidence |
| `matlab_params_check.jl` | the paper's published parameters evaluated in SubFit |
| `profile_tau.jl` | detailed τ₂ profile likelihood |

The last four are kept as the record of what was ruled out.
