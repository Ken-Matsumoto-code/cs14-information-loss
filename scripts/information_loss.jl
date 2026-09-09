#!/usr/bin/env julia

using CSV
using DataFrames
using LaTeXStrings
using LinearAlgebra
using Printf
using Random
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

Base.@kwdef struct CS14Params
    k1::Float64 = 0.2
    k2::Float64 = 1.0
    k3::Float64 = 1.0
    k4::Float64 = 1.0
    k5::Float64 = 1.0
    delta::Float64 = 0.02
    gamma_z::Float64 = 0.01
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

function drift(x::AbstractVector{<:Real}, p::CS14Params)
    X, Y, Z = x
    return Float64[
        p.k1 * Y + p.k2 * X^2 - p.k3 * X * Z - p.delta * X^3,
        p.k4 * Z - p.k5 * X * Y,
        p.k3 * X * Z - p.k4 * Z + p.gamma_z,
    ]
end

function jacobian(x::AbstractVector{<:Real}, p::CS14Params)
    X, Y, Z = x
    return Float64[
        2 * p.k2 * X - p.k3 * Z - 3 * p.delta * X^2  p.k1      -p.k3 * X;
        -p.k5 * Y                                      -p.k5 * X  p.k4;
        p.k3 * Z                                        0.0        p.k3 * X - p.k4
    ]
end

function diffusion_matrix(x::AbstractVector{<:Real}, p::CS14Params)
    X, Y, Z = x
    rates = (
        p.k1 * Y,
        p.k2 * X^2,
        p.k3 * X * Z,
        p.k4 * Z,
        p.k5 * X * Y,
        p.delta * X^3,
        p.gamma_z,
    )
    stoichiometry = (
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 1.0),
        (0.0, 1.0, -1.0),
        (0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    D = zeros(3, 3)
    for r in eachindex(rates)
        for i in 1:3, j in 1:3
            D[i, j] += max(Float64(rates[r]), 0.0) *
                       stoichiometry[r][i] * stoichiometry[r][j]
        end
    end
    return Symmetric(D)
end

function rk4_state(x::Vector{Float64}, dt::Float64, p::CS14Params)
    k1 = drift(x, p)
    k2 = drift(x .+ 0.5dt .* k1, p)
    k3 = drift(x .+ 0.5dt .* k2, p)
    k4 = drift(x .+ dt .* k3, p)
    return x .+ (dt / 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

function rk4_state_tangent(
    x::Vector{Float64},
    M::Matrix{Float64},
    dt::Float64,
    p::CS14Params,
)
    k1x = drift(x, p)
    k1M = jacobian(x, p) * M
    x2 = x .+ 0.5dt .* k1x
    M2 = M .+ 0.5dt .* k1M
    k2x = drift(x2, p)
    k2M = jacobian(x2, p) * M2
    x3 = x .+ 0.5dt .* k2x
    M3 = M .+ 0.5dt .* k2M
    k3x = drift(x3, p)
    k3M = jacobian(x3, p) * M3
    x4 = x .+ dt .* k3x
    M4 = M .+ dt .* k3M
    k4x = drift(x4, p)
    k4M = jacobian(x4, p) * M4
    return (
        x .+ (dt / 6) .* (k1x .+ 2k2x .+ 2k3x .+ k4x),
        M .+ (dt / 6) .* (k1M .+ 2k2M .+ 2k3M .+ k4M),
    )
end

function deterministic_anchors(
    n::Int;
    x0::AbstractVector{<:Real}=[1.0, 2.0, 1.0],
    burn::Float64,
    thin_min::Float64,
    thin_max::Float64,
    dt::Float64,
    seed::Int=20260908,
    params::CS14Params=CS14Params(),
)
    rng = Xoshiro(seed)
    x = Float64.(x0)
    for _ in 1:round(Int, burn / dt)
        x = rk4_state(x, dt, params)
    end
    anchors = Matrix{Float64}(undef, 3, n)
    min_steps = round(Int, thin_min / dt)
    max_steps = round(Int, thin_max / dt)
    for index in 1:n
        for _ in 1:rand(rng, min_steps:max_steps)
            x = rk4_state(x, dt, params)
        end
        anchors[:, index] = x
    end
    return anchors
end

function lyapunov_spectrum(
    x0::AbstractVector{<:Real};
    burn::Float64,
    duration::Float64,
    dt::Float64,
    qr_interval::Int=10,
    params::CS14Params=CS14Params(),
)
    x = Float64.(x0)
    for _ in 1:round(Int, burn / dt)
        x = rk4_state(x, dt, params)
    end
    total_steps = round(Int, duration / dt)
    Q = Matrix{Float64}(I, 3, 3)
    sums = zeros(3)
    completed = 0
    while completed < total_steps
        nsteps = min(qr_interval, total_steps - completed)
        M = Q
        for _ in 1:nsteps
            x, M = rk4_state_tangent(x, M, dt, params)
        end
        factor = qr(M)
        Q = Matrix(factor.Q)
        sums .+= log.(abs.(diag(factor.R)))
        completed += nsteps
    end
    return sort(sums ./ (total_steps * dt); rev=true)
end

function rk4_state_covariance(
    x::Vector{Float64},
    covariance::Matrix{Float64},
    dt::Float64,
    params::CS14Params,
)
    function rhs(xstage, covariance_stage)
        A = jacobian(xstage, params)
        D = Matrix(diffusion_matrix(xstage, params))
        return drift(xstage, params),
               A * covariance_stage + covariance_stage * transpose(A) + D
    end
    k1x, k1C = rhs(x, covariance)
    k2x, k2C = rhs(x .+ 0.5dt .* k1x, covariance .+ 0.5dt .* k1C)
    k3x, k3C = rhs(x .+ 0.5dt .* k2x, covariance .+ 0.5dt .* k2C)
    k4x, k4C = rhs(x .+ dt .* k3x, covariance .+ dt .* k3C)
    xnext = x .+ (dt / 6) .* (k1x .+ 2k2x .+ 2k3x .+ k4x)
    Cnext = covariance .+ (dt / 6) .* (k1C .+ 2k2C .+ 2k3C .+ k4C)
    return xnext, 0.5 .* (Cnext .+ transpose(Cnext))
end

function rk4_precision_logdet(
    x::Vector{Float64},
    precision::Matrix{Float64},
    logdet_covariance::Float64,
    dt::Float64,
    params::CS14Params,
)
    function rhs(xstage, precision_stage)
        A = jacobian(xstage, params)
        D = Matrix(diffusion_matrix(xstage, params))
        return drift(xstage, params),
               -precision_stage * A - transpose(A) * precision_stage -
               precision_stage * D * precision_stage,
               2tr(A) + tr(precision_stage * D)
    end
    k1x, k1P, k1e = rhs(x, precision)
    k2x, k2P, k2e = rhs(x .+ 0.5dt .* k1x, precision .+ 0.5dt .* k1P)
    k3x, k3P, k3e = rhs(x .+ 0.5dt .* k2x, precision .+ 0.5dt .* k2P)
    k4x, k4P, k4e = rhs(x .+ dt .* k3x, precision .+ dt .* k3P)
    xnext = x .+ (dt / 6) .* (k1x .+ 2k2x .+ 2k3x .+ k4x)
    Pnext = precision .+ (dt / 6) .* (k1P .+ 2k2P .+ 2k3P .+ k4P)
    ellnext = logdet_covariance + (dt / 6) * (k1e + 2k2e + 2k3e + k4e)
    return xnext, 0.5 .* (Pnext .+ transpose(Pnext)), ellnext
end

safe_logdet(covariance::Matrix{Float64}) =
    2sum(log, diag(cholesky(Symmetric(covariance)).L))

function gaussian_entropy_curve(
    anchors::Matrix{Float64},
    times::Vector{Float64},
    t_ref::Float64;
    dt::Float64,
    params::CS14Params=CS14Params(),
)
    issorted(times) || error("times must be sorted")
    ref_index = findfirst(t -> isapprox(t, t_ref; atol=1.0e-12), times)
    ref_index === nothing && error("t_ref must be included in times")
    steps = round.(Int, times ./ dt)
    all(isapprox.(steps .* dt, times; atol=1.0e-10)) ||
        error("Every output time must be an integer multiple of dt")
    ref_step = steps[ref_index]
    values = Matrix{Float64}(undef, size(anchors, 2), length(times))
    @threads for anchor in axes(anchors, 2)
        x = copy(anchors[:, anchor])
        covariance = zeros(3, 3)
        precision = zeros(3, 3)
        logdet_covariance = NaN
        target = 1
        for step in 0:maximum(steps)
            if step > 0
                if step <= ref_step
                    x, covariance = rk4_state_covariance(x, covariance, dt, params)
                    if step == ref_step
                        logdet_covariance = safe_logdet(covariance)
                        precision = Matrix(inv(Symmetric(covariance)))
                    end
                else
                    x, precision, logdet_covariance = rk4_precision_logdet(
                        x,
                        precision,
                        logdet_covariance,
                        dt,
                        params,
                    )
                end
            end
            while target <= length(steps) && steps[target] == step
                values[anchor, target] = step == 0 ? -Inf :
                    0.5 * (step <= ref_step ? safe_logdet(covariance) : logdet_covariance)
                target += 1
            end
        end
    end
    increments = values .- values[:, ref_index]
    means = vec(mean(increments; dims=1))
    sems = vec(std(increments; dims=1, corrected=true)) ./ sqrt(size(anchors, 2))
    means[ref_index] = 0.0
    sems[ref_index] = 0.0
    return means, sems
end

function finite_v_information_loss(V::Int, t_ref::Float64)
    path = joinpath(DATA_DIR, "conditional_entropy_V$(V).csv")
    isfile(path) || error("Run scripts/mutual_information.jl first; missing $path")
    raw = CSV.read(path, DataFrame)
    reference = raw[isapprox.(raw.time, t_ref; atol=1.0e-12), [:initial_index, :repeat, :conditional_entropy]]
    nrow(reference) > 0 || error("t_ref=$t_ref is absent from $path")
    rename!(reference, :conditional_entropy => :reference_entropy)
    paired = innerjoin(raw, reference; on=[:initial_index, :repeat])
    paired.information_loss = paired.conditional_entropy .- paired.reference_entropy
    by_initial = combine(
        groupby(paired, [:time, :initial_index]),
        :information_loss => mean => :information_loss,
    )
    summary = combine(
        groupby(by_initial, :time),
        :information_loss => mean => :information_loss,
        :information_loss => (x -> std(x; corrected=true) / sqrt(length(x))) => :information_loss_se,
    )
    sort!(summary, :time)
    summary.V = fill(V, nrow(summary))
    return summary
end

function fit_line(x::Vector{Float64}, y::Vector{Float64})
    x_mean = mean(x)
    y_mean = mean(y)
    slope = sum((x .- x_mean) .* (y .- y_mean)) / sum(abs2, x .- x_mean)
    return slope, y_mean - slope * x_mean
end

function save_figures(
    finite_v::DataFrame,
    gaussian::DataFrame,
    t_ref::Float64,
    h_ks::Float64,
    main_tmax::Float64,
)
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
        ylabel=L"\Delta I^{(V)}(t;t_{\mathrm{ref}})",
        legend=:topright,
        size=(1000, 650),
        left_margin=14Plots.mm,
        xlims=(t_ref, main_tmax),
        ylims=(-0.1,18)
    )
    for group in groupby(finite_v, :V)
        exponent = round(Int, log10(first(group.V)))
        Plots.plot!(
            figure,
            group.time,
            group.information_loss;
            yerror=group.information_loss_se,
            marker=:circle,
            markersize=4,
            linewidth=2.5,
            label=latexstring("V=10^{", exponent, "}"),
        )
    end
    gaussian_main = gaussian[gaussian.time .<= main_tmax, :]
    Plots.plot!(
        figure,
        gaussian_main.time,
        gaussian_main.gaussian_increment;
        linewidth=3,
        linestyle=:dash,
        seriescolor=:black,
        label=L"\Delta H_{\mathrm{G}}(t;t_{\mathrm{ref}})",
    )

    elapsed = gaussian.time .- t_ref
    rate = gaussian.gaussian_increment ./ elapsed
    residual = rate .- h_ks
    mask = (elapsed .> 0) .& (residual .> 0) .& isfinite.(residual)
    count(mask) >= 2 || error("The log-log inset requires at least two positive residuals")
    Plots.plot!(
        figure,
        elapsed[mask],
        residual[mask];
        inset=Plots.bbox(0.27, 0.05, 0.42, 0.38),
        subplot=2,
        linewidth=2.2,
        seriescolor=:black,
        label="",
        xlabel=L"t-t_{\mathrm{ref}}",
        ylabel=L"\frac{\Delta H_{\mathrm{G}}}{t-t_{\mathrm{ref}}}-h_{\mathrm{KS}}",
        xscale=:log10,
        yscale=:log10,
        xlims=(minimum(elapsed[mask]), maximum(elapsed[mask])),
        guidefontsize=14,
        tickfontsize=11,
        legend=false,
        framestyle=:box,
        background_color_subplot=:white,
        grid=false,
    )
    savefig(figure, joinpath(FIGURE_DIR, "information_loss.pdf"))

    log_elapsed = log.(elapsed[mask])
    scaled_residual = elapsed[mask] .* residual[mask]
    fit_mask = elapsed[mask] .>= 2.0
    slope, intercept = fit_line(log_elapsed[fit_mask], scaled_residual[fit_mask])
    fit_x = collect(range(minimum(log_elapsed), maximum(log_elapsed); length=400))
    diagnostic = Plots.plot(
        log_elapsed,
        scaled_residual;
        marker=:circle,
        markersize=4,
        linewidth=2.5,
        seriescolor=:black,
        label="Gaussian prediction",
        xlabel=L"\log(t-t_{\mathrm{ref}})",
        ylabel=L"(t-t_{\mathrm{ref}})\left[\frac{\Delta H_{\mathrm{G}}}{t-t_{\mathrm{ref}}}-h_{\mathrm{KS}}\right]",
        xlims=(minimum(log_elapsed), maximum(log_elapsed)),
        ylims=(-1,6.5),
        legend=:topleft,
        size=(1000, 650),
        left_margin=24Plots.mm,
        bottom_margin=8Plots.mm,
    )
    Plots.plot!(
        diagnostic,
        fit_x,
        intercept .+ slope .* fit_x;
        linestyle=:dash,
        linewidth=2.5,
        seriescolor=:red,
        label=latexstring(
            "\\mathrm{linear~fit}:~",
            @sprintf("%.4g", slope),
            "\\log(t-t_{\\mathrm{ref}})",
            @sprintf("%+.4g", intercept),
        ),
    )
    savefig(diagnostic, joinpath(FIGURE_DIR, "asymptotic_correction.pdf"))
end

function main()
    mkpath(DATA_DIR)
    mkpath(FIGURE_DIR)
    LinearAlgebra.BLAS.set_num_threads(1)

    V_list = env_int_list("CS14_LARGE_V_LIST", [10_000, 100_000, 1_000_000])
    t_ref = env_float("CS14_T_REF", 5.0)
    main_tmax = env_float("CS14_MAIN_TMAX", 120.0)
    gaussian_dt = env_float("CS14_GAUSSIAN_DT", 0.005)
    gaussian_tmax = env_float("CS14_GAUSSIAN_TMAX", 400.0)
    gaussian_output_dt = env_float("CS14_GAUSSIAN_OUTPUT_DT", 0.5)
    n_anchors = env_int("CS14_GAUSSIAN_ANCHORS", 20_000)
    t_ref < main_tmax || error("CS14_MAIN_TMAX must be larger than CS14_T_REF")
    main_tmax <= gaussian_tmax || error("CS14_MAIN_TMAX must not exceed CS14_GAUSSIAN_TMAX")

    finite_v = DataFrame()
    for V in V_list
        append!(finite_v, finite_v_information_loss(V, t_ref); cols=:union)
    end
    finite_v = finite_v[(finite_v.time .>= t_ref) .& (finite_v.time .<= main_tmax), :]
    CSV.write(joinpath(DATA_DIR, "information_loss.csv"), finite_v)

    spectrum_path = joinpath(DATA_DIR, "lyapunov_spectrum.csv")
    spectrum = if isfile(spectrum_path)
        CSV.read(spectrum_path, DataFrame).exponent
    else
        lyapunov_spectrum(
            [1.0, 2.0, 1.0];
            burn=2000.0,
            duration=10000.0,
            dt=gaussian_dt,
        )
    end
    h_ks = sum(filter(>(0), spectrum))

    dense_times = collect(t_ref:gaussian_output_dt:gaussian_tmax)
    finite_times = Float64.(unique(finite_v.time))
    times = sort(unique(vcat(dense_times, finite_times, [t_ref, gaussian_tmax])))
    anchors = deterministic_anchors(
        n_anchors;
        burn=2000.0,
        thin_min=20.0,
        thin_max=33.0,
        dt=gaussian_dt,
    )
    increment, increment_se = gaussian_entropy_curve(anchors, times, t_ref; dt=gaussian_dt)
    gaussian = DataFrame(
        time=times,
        t_ref=fill(t_ref, length(times)),
        gaussian_increment=increment,
        gaussian_increment_se=increment_se,
        h_KS=fill(h_ks, length(times)),
        N_anchors=fill(n_anchors, length(times)),
    )
    CSV.write(joinpath(DATA_DIR, "gaussian_information_loss.csv"), gaussian)
    MAKE_PLOTS && save_figures(finite_v, gaussian, t_ref, h_ks, main_tmax)
    @printf("h_KS = %.10g\n", h_ks)
    println(MAKE_PLOTS ? "Saved information-loss data and figures." : "Saved information-loss data.")
end

main()
