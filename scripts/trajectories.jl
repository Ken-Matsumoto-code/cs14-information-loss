#!/usr/bin/env julia

using CSV
using DataFrames
using LaTeXStrings
using Random

function env_bool(name::AbstractString, default::Bool)
    value = lowercase(strip(get(ENV, name, "")))
    return isempty(value) ? default : value in ("1", "true", "yes", "on")
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const MAKE_PLOTS = !env_bool("CS14_SKIP_PLOTS", false)
const DATA_DIR = joinpath(ROOT, "data")
const FIGURE_DIR = joinpath(ROOT, "figures")
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
MAKE_PLOTS && @eval using Plots

const State = NTuple{3,Int64}

Base.@kwdef struct CS14Params
    k1::Float64 = 0.2
    k2::Float64 = 1.0
    k3::Float64 = 1.0
    k4::Float64 = 1.0
    k5::Float64 = 1.0
    delta::Float64 = 0.02
    gamma_z::Float64 = 0.01
end

Base.@kwdef struct TauLeapConfig
    epsilon::Float64 = 0.02
    tau_max::Float64 = 0.005
    tau_min::Float64 = 1.0e-8
    ssa_switch_factor::Float64 = 10.0
end

env_float(name::AbstractString, default::Float64) =
    isempty(strip(get(ENV, name, ""))) ? default : parse(Float64, ENV[name])

const LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)

function loggamma_positive(z::Float64)
    y = z - 1
    a = LANCZOS[1]
    for i in 2:length(LANCZOS)
        a += LANCZOS[i] / (y + i - 1)
    end
    t = y + 7.5
    return 0.5log(2pi) + (y + 0.5) * log(t) - t + log(a)
end

function rand_poisson(rng::AbstractRNG, mean::Float64)
    mean >= 0 && isfinite(mean) || error("Invalid Poisson mean: $mean")
    mean == 0 && return 0
    if mean < 30
        threshold = exp(-mean)
        product = 1.0
        k = 0
        while product > threshold
            product *= rand(rng)
            k += 1
        end
        return k - 1
    end
    root = sqrt(mean)
    logmean = log(mean)
    b = 0.931 + 2.53root
    a = -0.059 + 0.02483b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2)
    while true
        u = rand(rng) - 0.5
        v = rand(rng)
        us = 0.5 - abs(u)
        us <= 0 && continue
        k = floor(Int, (2a / us + b) * u + mean + 0.43)
        (k < 0 || (us < 0.013 && v > us)) && continue
        (us >= 0.07 && v <= vr) && return k
        log(v * inv_alpha / (a / us^2 + b)) <=
            -mean + k * logmean - loggamma_positive(Float64(k + 1)) && return k
    end
end

function propensities(s::State, V::Int, p::CS14Params)
    X, Y, Z = s
    v = Float64(V)
    return (
        p.k1 * Y,
        X >= 2 ? p.k2 * X * (X - 1) / v : 0.0,
        p.k3 * X * Z / v,
        p.k4 * Z,
        p.k5 * X * Y / v,
        X >= 3 ? p.delta * X * (X - 1) * (X - 2) / v^2 : 0.0,
        p.gamma_z * v,
    )
end

function fire(s::State, r::Int)::State
    X, Y, Z = s
    r == 1 && return (X + 1, Y, Z)
    r == 2 && return (X + 1, Y, Z)
    r == 3 && return (X - 1, Y, Z + 1)
    r == 4 && return (X, Y + 1, Z - 1)
    r == 5 && return (X, Y - 1, Z)
    r == 6 && return (X - 1, Y, Z)
    r == 7 && return (X, Y, Z + 1)
    error("Unknown reaction channel: $r")
end

function choose_reaction(rng::AbstractRNG, rates, total::Float64)
    threshold = rand(rng) * total
    cumulative = 0.0
    for r in eachindex(rates)
        cumulative += rates[r]
        threshold < cumulative && return r
    end
    return length(rates)
end

function tau_candidate(s::State, rates, config::TauLeapConfig)
    X, Y, Z = s
    drift = (
        rates[1] + rates[2] - rates[3] - rates[6],
        rates[4] - rates[5],
        rates[3] - rates[4] + rates[7],
    )
    variance = (
        rates[1] + rates[2] + rates[3] + rates[6],
        rates[4] + rates[5],
        rates[3] + rates[4] + rates[7],
    )
    bounds = (
        max(config.epsilon * X / 3, 1.0),
        max(config.epsilon * Y / 2, 1.0),
        max(config.epsilon * Z / 2, 1.0),
    )
    tau = config.tau_max
    for i in 1:3
        drift[i] != 0 && (tau = min(tau, bounds[i] / abs(drift[i])))
        variance[i] > 0 && (tau = min(tau, bounds[i]^2 / variance[i]))
    end
    return tau
end

function step_to_limit(
    state::State,
    time::Float64,
    limit::Float64,
    V::Int,
    params::CS14Params,
    config::TauLeapConfig,
    rng::AbstractRNG,
)
    remaining = limit - time
    remaining <= 0 && return state, limit
    rates = propensities(state, V, params)
    total = sum(rates)
    (!(total > 0) || !isfinite(total)) && return state, limit
    tau = min(tau_candidate(state, rates, config), remaining)
    if tau < config.tau_min || tau < config.ssa_switch_factor / total
        wait = randexp(rng) / total
        time + wait > limit && return state, limit
        return fire(state, choose_reaction(rng, rates, total)), time + wait
    end
    for _ in 1:20
        counts = ntuple(r -> rand_poisson(rng, rates[r] * tau), 7)
        X, Y, Z = state
        candidate = (
            X + counts[1] + counts[2] - counts[3] - counts[6],
            Y + counts[4] - counts[5],
            Z + counts[3] - counts[4] + counts[7],
        )
        minimum(candidate) >= 0 && return candidate, time + tau
        tau *= 0.5
    end
    error("Tau-leap rejection limit reached at state $state")
end

function advance_to(
    state::State,
    duration::Float64,
    V::Int,
    params::CS14Params,
    config::TauLeapConfig,
    rng::AbstractRNG,
)
    time = 0.0
    while time < duration
        state, time = step_to_limit(state, time, duration, V, params, config, rng)
    end
    return state
end

function stochastic_trajectory(
    V::Int,
    times::Vector{Float64};
    initial::NTuple{3,<:Real}=(1.0, 2.0, 2.0),
    seed::Int=20260820,
    params::CS14Params=CS14Params(),
    config::TauLeapConfig=TauLeapConfig(),
)
    issorted(times) || error("times must be sorted")
    rng = Xoshiro(seed)
    state = Tuple(round.(Int64, V .* initial))
    result = Matrix{Int64}(undef, 3, length(times))
    previous = 0.0
    for (index, time) in enumerate(times)
        state = advance_to(state, time - previous, V, params, config, rng)
        result[:, index] .= state
        previous = time
    end
    return result
end

function trajectory_table(V::Int, times::Vector{Float64}, seed::Int)
    counts = stochastic_trajectory(V, times; seed=seed)
    return DataFrame(
        time=times,
        V=fill(V, length(times)),
        X=counts[1, :],
        Y=counts[2, :],
        Z=counts[3, :],
        x=counts[1, :] ./ V,
        y=counts[2, :] ./ V,
        z=counts[3, :] ./ V,
    )
end

function panel(data::DataFrame, V::Int; bottom::Bool)
    plot = Plots.plot(
        data.time,
        data.x;
        label=L"x",
        ylabel=L"\mathrm{concentration}",
        xlabel=bottom ? L"\mathrm{time}\quad t" : "",
        title=latexstring("V=10^{", round(Int, log10(V)), "}"),
        linewidth=2,
        legend=bottom ? false : :topright,
        left_margin=12Plots.mm,
    )
    Plots.plot!(plot, data.time, data.y; label=L"y", linewidth=2)
    Plots.plot!(plot, data.time, data.z; label=L"z", linewidth=2)
    return plot
end

function main()
    mkpath(DATA_DIR)
    mkpath(FIGURE_DIR)
    tmax = env_float("CS14_TRAJECTORY_TMAX", 100.0)
    save_dt = env_float("CS14_TRAJECTORY_SAVE_DT", 0.05)
    times = collect(0.0:save_dt:tmax)
    times[end] < tmax && push!(times, tmax)

    small = trajectory_table(100, times, 20260820)
    large = trajectory_table(10_000, times, 20260821)
    CSV.write(joinpath(DATA_DIR, "stochastic_trajectories.csv"), vcat(small, large))

    if MAKE_PLOTS
        gr()
        default(
            fontfamily="Computer Modern",
            framestyle=:box,
            grid=true,
            guidefontsize=20,
            tickfontsize=16,
            legendfontsize=15,
        )
        figure = Plots.plot(
            panel(small, 100; bottom=false),
            panel(large, 10_000; bottom=true);
            layout=(2, 1),
            link=:both,
            size=(1000, 760),
        )
        savefig(figure, joinpath(FIGURE_DIR, "stochastic_trajectories.pdf"))
    end
    println("Saved stochastic trajectories.")
end

main()
