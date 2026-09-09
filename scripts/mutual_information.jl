#!/usr/bin/env julia

using CSV
using DataFrames
using LaTeXStrings
using LinearAlgebra
using NearestNeighbors
using Printf
using Random
using Serialization
using Statistics
using Base.Threads: @threads

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
    tau_max::Float64 = 0.02
    tau_min::Float64 = 1.0e-8
    ssa_switch_factor::Float64 = 10.0
end

env_int(name::AbstractString, default::Int) =
    isempty(strip(get(ENV, name, ""))) ? default : parse(Int, ENV[name])

env_float(name::AbstractString, default::Float64) =
    isempty(strip(get(ENV, name, ""))) ? default : parse(Float64, ENV[name])

function env_int_list(name::AbstractString, default::Vector{Int})
    value = strip(get(ENV, name, ""))
    return isempty(value) ? default : parse.(Int, split(value, ','))
end

function mixed_seed(values::Integer...)
    state = UInt64(0x243f6a8885a308d3)
    for value in values
        state += UInt64(value) + UInt64(0x9e3779b97f4a7c15)
        state = (state ⊻ (state >> 30)) * UInt64(0xbf58476d1ce4e5b9)
        state = (state ⊻ (state >> 27)) * UInt64(0x94d049bb133111eb)
        state ⊻= state >> 31
    end
    return Int(state % UInt64(typemax(Int)))
end

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

function stationary_anchors(
    V::Int,
    n::Int;
    initial::NTuple{3,<:Real}=(1.0, 2.0, 1.0),
    burn::Float64,
    thin::Float64,
    seed::Int=20260722,
    params::CS14Params=CS14Params(),
    config::TauLeapConfig=TauLeapConfig(),
)
    rng = Xoshiro(mixed_seed(seed, V, 17))
    state = Tuple(round.(Int64, V .* initial))
    state = advance_to(state, burn, V, params, config, rng)
    anchors = Vector{State}(undef, n)
    for index in eachindex(anchors)
        state = advance_to(state, thin, V, params, config, rng)
        anchors[index] = state
    end
    return anchors
end

function simulate_ensemble(
    initial::State,
    V::Int,
    times::Vector{Float64},
    n::Int;
    seed::Int=20260722,
    initial_index::Int=1,
    params::CS14Params=CS14Params(),
    config::TauLeapConfig=TauLeapConfig(),
)
    issorted(times) || error("times must be sorted")
    samples = Array{Int64}(undef, 3, length(times), n)
    @threads for sample in 1:n
        rng = Xoshiro(mixed_seed(seed, V, initial_index, sample))
        state = initial
        previous = 0.0
        for (index, time) in enumerate(times)
            state = advance_to(state, time - previous, V, params, config, rng)
            samples[:, index, sample] .= state
            previous = time
        end
    end
    return samples
end

function whiten!(points::Matrix{Float64})
    points .-= vec(mean(points; dims=2))
    covariance = Symmetric(points * transpose(points) / (size(points, 2) - 1))
    scale = max(tr(covariance) / size(points, 1), 1.0)
    factor = cholesky(covariance + 1.0e-12scale * I)
    points .= factor.L \ points
    return sum(log, diag(factor.L))
end

function digamma_integer(n::Int)
    n > 0 || error("digamma_integer requires n > 0")
    x = Float64(n)
    value = 0.0
    while x < 8.0
        value -= 1.0 / x
        x += 1.0
    end
    inv = 1.0 / x
    inv2 = inv^2
    return value + log(x) - 0.5inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 / 252))
end

function knn_entropy(
    counts::AbstractMatrix{<:Integer};
    k::Int,
    seed::Int,
    query_block::Int=20_000,
)
    d, n = size(counts)
    1 <= k < n || error("k must satisfy 1 <= k < number of samples")
    rng = Xoshiro(seed)
    points = Float64.(counts)
    points .+= rand(rng, size(points))
    correction = whiten!(points)
    tree = KDTree(points)
    sum_log_radius = 0.0
    for lo in 1:query_block:n
        hi = min(lo + query_block - 1, n)
        _, distances = knn(tree, Matrix(@view points[:, lo:hi]), k + 1, true)
        for distance in distances
            radius = distance[k + 1]
            radius > 0 && isfinite(radius) || error("Invalid kNN radius")
            sum_log_radius += log(radius)
        end
    end
    log_ball = (d / 2) * log(pi) - loggamma_positive(d / 2 + 1)
    return digamma_integer(n) - digamma_integer(k) + log_ball +
           d * sum_log_radius / n + correction
end

function float_list(name::AbstractString, default::Vector{Float64})
    value = strip(get(ENV, name, ""))
    return isempty(value) ? default : parse.(Float64, split(value, ','))
end

function load_or_make_anchors(V::Int, n_initial::Int, burn::Float64, thin::Float64)
    path = joinpath(DATA_DIR, "initial_states_V$(V).csv")
    if isfile(path)
        table = CSV.read(path, DataFrame)
        nrow(table) == n_initial || error("$path was generated with a different number of initial states")
        return State[(row.X, row.Y, row.Z) for row in eachrow(table)]
    end
    anchors = stationary_anchors(V, n_initial; burn=burn, thin=thin)
    CSV.write(
        path,
        DataFrame(
            initial_index=1:n_initial,
            X=first.(anchors),
            Y=getindex.(anchors, 2),
            Z=last.(anchors),
            x=first.(anchors) ./ V,
            y=getindex.(anchors, 2) ./ V,
            z=last.(anchors) ./ V,
        ),
    )
    return anchors
end

function allocation(n_samples::Int, n_initial::Int, index::Int)
    count = div(n_samples, n_initial) + (index <= rem(n_samples, n_initial) ? 1 : 0)
    offset = (index - 1) * div(n_samples, n_initial) + min(index - 1, rem(n_samples, n_initial))
    return offset + 1, offset + count
end

function checkpoint_complete(raw::DataFrame, index::Int, n_times::Int, repeats::Int)
    isempty(raw) && return false
    rows = raw[raw.initial_index .== index, :]
    return nrow(rows) == n_times * repeats
end

function save_marginal_slice(
    path::AbstractString,
    samples::Array{Int64,3},
    lo::Int,
    hi::Int,
    V::Int,
    times::Vector{Float64},
)
    indices = round.(Int, range(1, size(samples, 3); length=hi - lo + 1))
    slice = samples[:, :, indices]
    temporary = path * ".tmp"
    open(temporary, "w") do io
        serialize(io, (V=V, times=times, lo=lo, hi=hi, samples=slice))
    end
    mv(temporary, path; force=true)
end

function conditional_entropy_for_v(
    V::Int,
    times::Vector{Float64},
    n_initial::Int,
    n_samples::Int,
    k::Int,
    repeats::Int,
    burn::Float64,
    thin::Float64,
)
    anchors = load_or_make_anchors(V, n_initial, burn, thin)
    raw_path = joinpath(DATA_DIR, "conditional_entropy_V$(V).csv")
    raw = isfile(raw_path) ? CSV.read(raw_path, DataFrame) : DataFrame()

    if !isempty(raw)
        all(raw.N_samples .== n_samples) || error("$raw_path uses a different sample count")
        all(raw.k .== k) || error("$raw_path uses a different k")
    end

    for (initial_index, initial) in enumerate(anchors)
        lo, hi = allocation(n_samples, n_initial, initial_index)
        marginal_path = joinpath(DATA_DIR, @sprintf("marginal_V%d_initial%03d.bin", V, initial_index))
        complete = checkpoint_complete(raw, initial_index, length(times), repeats)
        complete && isfile(marginal_path) && continue

        @printf("V=%d, initial=%d/%d: simulating %d trajectories\n", V, initial_index, n_initial, n_samples)
        samples = simulate_ensemble(
            initial,
            V,
            times,
            n_samples;
            initial_index=initial_index,
        )

        if !complete
            raw = isempty(raw) ? raw : raw[raw.initial_index .!= initial_index, :]
            rows = NamedTuple[]
            for (time_index, time) in enumerate(times), repeat in 1:repeats
                entropy = time == 0.0 ? 0.0 : knn_entropy(
                    @view(samples[:, time_index, :]);
                    k=k,
                    seed=mixed_seed(20260909, V, initial_index, time_index, repeat),
                )
                push!(rows, (
                    V=V,
                    time=time,
                    initial_index=initial_index,
                    repeat=repeat,
                    N_samples=n_samples,
                    k=k,
                    conditional_entropy=entropy,
                ))
            end
            append!(raw, DataFrame(rows); cols=:union)
            sort!(raw, [:initial_index, :time, :repeat])
            CSV.write(raw_path, raw)
        end

        save_marginal_slice(marginal_path, samples, lo, hi, V, times)
        samples = nothing
        GC.gc(false)
    end
    return raw
end

function load_marginal_pool(V::Int, times::Vector{Float64}, n_initial::Int, n_samples::Int)
    pooled = Array{Int64}(undef, 3, length(times), n_samples)
    for initial_index in 1:n_initial
        path = joinpath(DATA_DIR, @sprintf("marginal_V%d_initial%03d.bin", V, initial_index))
        isfile(path) || error("Missing marginal checkpoint: $path")
        data = open(deserialize, path)
        data.V == V || error("System-size mismatch in $path")
        data.times == times || error("Time-grid mismatch in $path")
        pooled[:, :, data.lo:data.hi] .= data.samples
    end
    return pooled
end

function summarize_mutual_information(
    V::Int,
    times::Vector{Float64},
    raw::DataFrame,
    n_initial::Int,
    n_samples::Int,
    k::Int,
    repeats::Int,
)
    by_initial = combine(
        groupby(raw, [:time, :initial_index]),
        :conditional_entropy => mean => :conditional_entropy,
    )
    conditional = combine(
        groupby(by_initial, :time),
        :conditional_entropy => mean => :conditional_entropy,
        :conditional_entropy => (x -> std(x; corrected=true) / sqrt(length(x))) => :conditional_entropy_se,
    )

    pooled = load_marginal_pool(V, times, n_initial, n_samples)
    positive_indices = findall(>(0), times)
    isempty(positive_indices) && error("At least one positive observation time is required")
    stationary_indices = positive_indices[max(1, length(positive_indices) - 2):end]
    marginal_rows = NamedTuple[]
    for time_index in stationary_indices, repeat in 1:repeats
        time = times[time_index]
        entropy = knn_entropy(
            @view(pooled[:, time_index, :]);
            k=k,
            seed=mixed_seed(20260910, V, time_index, repeat),
        )
        push!(marginal_rows, (time=time, repeat=repeat, marginal_entropy=entropy))
    end
    marginal_raw = DataFrame(marginal_rows)
    marginal_by_time = combine(
        groupby(marginal_raw, :time),
        :marginal_entropy => mean => :marginal_entropy,
        :marginal_entropy => (x -> length(x) > 1 ? std(x; corrected=true) / sqrt(length(x)) : 0.0) => :marginal_entropy_se,
    )
    stationary_entropy = mean(marginal_by_time.marginal_entropy)
    time_se = nrow(marginal_by_time) > 1 ? std(marginal_by_time.marginal_entropy; corrected=true) / sqrt(nrow(marginal_by_time)) : 0.0
    dequant_se = sqrt(sum(abs2, marginal_by_time.marginal_entropy_se)) / nrow(marginal_by_time)
    stationary_entropy_se = hypot(time_se, dequant_se)
    result = conditional
    sort!(result, :time)

    result.V = fill(V, nrow(result))
    result.N_initial = fill(n_initial, nrow(result))
    result.N_samples = fill(n_samples, nrow(result))
    result.k = fill(k, nrow(result))
    result.stationary_entropy = fill(stationary_entropy, nrow(result))
    result.stationary_entropy_se = fill(stationary_entropy_se, nrow(result))
    result.stationary_reference_times = fill(join(times[stationary_indices], ";"), nrow(result))
    result.mutual_information = stationary_entropy .- result.conditional_entropy
    result.mutual_information_se = hypot.(stationary_entropy_se, result.conditional_entropy_se)

    CSV.write(joinpath(DATA_DIR, "marginal_entropy_V$(V).csv"), marginal_raw)
    CSV.write(joinpath(DATA_DIR, "mutual_information_V$(V).csv"), result)
    return result
end

function plot_mutual_information(results::DataFrame)
    gr()
    default(
        fontfamily="Computer Modern",
        framestyle=:box,
        grid=false,
        guidefontsize=20,
        tickfontsize=16,
        legendfontsize=14,
    )
    figure = Plots.plot(
        xlabel=L"\mathrm{time}\quad t",
        ylabel=L"I(\mathbf{X}_t^{(V)};\mathbf{X}_0^{(V)})",
        legend=:topright,
        size=(1000, 650),
        left_margin=14Plots.mm,
    )
    for group in groupby(results, :V)
        exponent = round(Int, log10(first(group.V)))
        Plots.plot!(
            figure,
            group.time,
            group.mutual_information;
            yerror=group.mutual_information_se,
            marker=:circle,
            markersize=4,
            linewidth=2.5,
            label=latexstring("V=10^{", exponent, "}"),
        )
    end
    savefig(figure, joinpath(FIGURE_DIR, "mutual_information.pdf"))
end

function main()
    mkpath(DATA_DIR)
    mkpath(FIGURE_DIR)
    LinearAlgebra.BLAS.set_num_threads(1)

    V_list = env_int_list("CS14_V_LIST", [100, 1_000, 10_000, 100_000, 1_000_000])
    n_initial = env_int("CS14_N_INITIAL", 50)
    n_samples = env_int("CS14_N_SAMPLES", 1_000_000)
    k = env_int("CS14_K", 10)
    repeats = env_int("CS14_DEQUANT_REPEATS", 3)
    burn = env_float("CS14_STATIONARY_BURN", 300.0)
    thin = env_float("CS14_STATIONARY_THIN", 5.0)
    default_times = [
        0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0,
        50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
    ]
    times = float_list("CS14_TIMES", default_times)
    n_samples > k || error("N_samples must be larger than k")
    n_samples >= n_initial || error("N_samples must be at least N_initial")
    n_initial >= 2 || error("At least two initial concentrations are required")

    summaries = DataFrame()
    for V in V_list
        raw = conditional_entropy_for_v(V, times, n_initial, n_samples, k, repeats, burn, thin)
        result = summarize_mutual_information(V, times, raw, n_initial, n_samples, k, repeats)
        append!(summaries, result; cols=:union)
    end
    sort!(summaries, [:V, :time])
    CSV.write(joinpath(DATA_DIR, "mutual_information.csv"), summaries)
    MAKE_PLOTS && plot_mutual_information(summaries)
    println(MAKE_PLOTS ? "Saved mutual-information data and figure." : "Saved mutual-information data.")
end

main()
