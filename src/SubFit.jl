module SubFit

using DelimitedFiles
using LinearAlgebra
using Logging
using Printf
using Random
using Random: Xoshiro
using Statistics

using NLopt
using OrdinaryDiffEqTsit5
using SciMLBase
using SciMLBase: ODEProblem, solve

include("data.jl")
include("model.jl")
include("metrics.jl")
include("fit.jl")
include("profile.jl")
include("bootstrap.jl")
include("ensemble.jl")
include("report.jl")

# data
export movmean, load_series, calibration, CalibrationData, absolute_onset

# model
export GGM, GLM, GRM, LM, RICH, GOM, MODEL_NAMES
export growth_rhs, SubEpiParams, unpack, simulate, nsub

# fitting
export SubEpiFit, fit_subepidemic, param_bounds, initial_points,
       objective, nparams, aicc, gap_candidates

# profiling / the smoothness gate
export profile_tau2, profile_dip, is_smooth_profile, profile_interval,
       fit_stability, is_stable_fit, gated_fit, bound_hits,
       window_shift, window_slope, is_window_stable

# uncertainty
export BootstrapResult, run_bootstrap, point_forecast, add_error

# selection / ensembles
export rank_models, ensemble_weights, ensemble_curves, stack_weights

# metrics
export mae, mse, coverage, wis, performance, WIS_ALPHAS

# reporting
export output_dir, save_forecast, save_parameters, save_performance, save_fit,
       save_rankings, save_settings

end # module
