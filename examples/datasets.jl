# =====================================================================
# Dataset configurations for SubFit
#
# One entry per series. Adding a dataset means adding an entry here, not
# copying a script.
# =====================================================================

using SubFit

const DATA_ROOT = normpath(joinpath(@__DIR__, "..", "input"))

"""
    dataset(name)

  `y_cal`    calibration series (the file the paper calibrates on)
  `y_full`   longer series that extends it, supplying forecast truth
  `horizon`  forecast horizon
  `origins`  rolling-origin calibration lengths
  `columns`  per-unit columns for replication, or `nothing`
  `nmax`     largest number of sub-epidemics to consider
"""
_infile(x) = joinpath(DATA_ROOT, x)

function dataset(name::AbstractString)
    if name == "covid_deaths_usa"
        return (name = name, unit = "day", column = 52, nmax = 2,
                y_cal = load_series(_infile("cumulative-daily-coronavirus-deaths-USA-05-11-2020.txt");
                                    column=52),
                y_full = load_series(_infile("cumulative-daily-coronavirus-deaths-USA-05-09-2022.txt");
                                     column=52),
                horizon = 30, smoothfactor = 7, calibration_period = 90,
                origins = [50, 55, 60, 65, 70, 75],
                # 51 state/DC columns for replication
                columns = 1:51,
                min_deaths = 200, min_nonzero = 30, min_future = 30,
                # paper benchmarks, for the report script to print alongside
                paper = (AICc = 1036.00, MAE = 216.82, MSE = 69836.18,
                         coverage = 93.33, WIS = 134.36))

    elseif name == "mpox_weekly_usa"
        return (name = name, unit = "week", column = 1, nmax = 2,
                y_cal = load_series(_infile("cumulative-weekly-monkeypox-cases-USA-11-16-2022.txt")),
                y_full = load_series(_infile("cumulative-weekly-monkeypox-cases-USA-07-20-2023.txt")),
                horizon = 8, smoothfactor = 1, calibration_period = nothing,
                origins = [16, 18, 20, 22, 24],
                columns = nothing,
                min_deaths = 50, min_nonzero = 10, min_future = 10,
                paper = nothing)
    else
        error("unknown dataset '$name'. Known: covid_deaths_usa, mpox_weekly_usa")
    end
end

cli_dataset() = isempty(ARGS) ? "covid_deaths_usa" : String(ARGS[1])
