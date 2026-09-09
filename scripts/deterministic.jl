#!/usr/bin/env julia

using CSV
using DataFrames
using LaTeXStrings
using LinearAlgebra
using Printf

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

function deterministic_trajectory(
    x0::AbstractVector{<:Real};
    transient::Float64,
    duration::Float64,
    dt::Float64,
    save_dt::Float64,
    params::CS14Params=CS14Params(),
)
    x = Float64.(x0)
    for _ in 1:round(Int, transient / dt)
        x = rk4_state(x, dt, params)
    end
    save_every = round(Int, save_dt / dt)
    save_every > 0 || error("save_dt must be at least dt")
    times = Float64[0.0]
    values = Vector{Float64}[copy(x)]
    nsteps = round(Int, duration / dt)
    for step in 1:nsteps
        x = rk4_state(x, dt, params)
        if step % save_every == 0 || step == nsteps
            push!(times, step * dt)
            push!(values, copy(x))
        end
    end
    return times, reduce(hcat, values)
end

function lyapunov_spectrum(
    x0::AbstractVector{<:Real};
    burn::Float64,
    duration::Float64,
    dt::Float64,
    qr_interval::Int,
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

function main()
    mkpath(DATA_DIR)
    mkpath(FIGURE_DIR)

    dt = env_float("CS14_DET_DT", 0.005)
    transient = env_float("CS14_DET_TRANSIENT", 300.0)
    duration = env_float("CS14_DET_DURATION", 300.0)
    lyapunov_burn = env_float("CS14_LYAPUNOV_BURN", 2000.0)
    lyapunov_duration = env_float("CS14_LYAPUNOV_DURATION", 10000.0)
    qr_interval = env_int("CS14_LYAPUNOV_QR_INTERVAL", 10)
    initial = [1.0, 2.0, 1.0]

    times, trajectory = deterministic_trajectory(
        initial;
        transient=transient,
        duration=duration,
        dt=dt,
        save_dt=max(0.05, dt),
    )
    CSV.write(
        joinpath(DATA_DIR, "deterministic_trajectory.csv"),
        DataFrame(time=times, x=trajectory[1, :], y=trajectory[2, :], z=trajectory[3, :]),
    )

    spectrum = lyapunov_spectrum(
        initial;
        burn=lyapunov_burn,
        duration=lyapunov_duration,
        dt=dt,
        qr_interval=qr_interval,
    )
    CSV.write(
        joinpath(DATA_DIR, "lyapunov_spectrum.csv"),
        DataFrame(
            index=1:3,
            exponent=spectrum,
            dt=fill(dt, 3),
            burn_time=fill(lyapunov_burn, 3),
            measurement_time=fill(lyapunov_duration, 3),
            qr_interval=fill(qr_interval, 3),
        ),
    )

    if MAKE_PLOTS
        gr()
        default(fontfamily="Computer Modern", framestyle=:box, grid=false)
        plot = Plots.plot(
            trajectory[1, :],
            trajectory[2, :],
            trajectory[3, :];
            xlabel=L"x",
            ylabel=L"y",
            zlabel=L"z",
            linewidth=0.8,
            legend=false,
            size=(800, 650),
        )
        savefig(plot, joinpath(FIGURE_DIR, "deterministic_trajectory.pdf"))
    end

    @printf("Lyapunov spectrum: (%.8f, %.8f, %.8f)\n", spectrum...)
    println("Saved deterministic data to: ", DATA_DIR)
    MAKE_PLOTS && println("Saved deterministic trajectory to: ", FIGURE_DIR)
end

main()
