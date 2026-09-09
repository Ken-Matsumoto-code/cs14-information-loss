# Mutual-information loss in the modified CS14 model

This repository contains four self-contained Julia scripts used to generate the numerical results for the modified CS14 chemical reaction network. Each script contains the model and numerical routines needed for its calculation and does not require a separate source file to be executed first.

Julia 1.10 or later is required.

## Calculations

Run the scripts from the repository root in the following order:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/deterministic.jl
julia --project=. scripts/trajectories.jl
julia --project=. -t auto scripts/mutual_information.jl
julia --project=. scripts/information_loss.jl
```

The scripts perform the following tasks:

1. `deterministic.jl` contains the deterministic model, RK4 integrator, and QR algorithm used to compute a trajectory and the Lyapunov spectrum.
2. `trajectories.jl` contains the stochastic reaction model and adaptive tau-leap algorithm used to generate trajectories at two system sizes.
3. `mutual_information.jl` contains the stochastic simulation and k-nearest-neighbor estimator used to compute conditional entropies and two-time mutual information for `V = 10^2, ..., 10^6`.
4. `information_loss.jl` contains the Gaussian covariance analysis used to compute the relative information loss for `V = 10^4, 10^5, 10^6` and create the main and asymptotic figures.

Numerical data are written to `data/`, and the figures are written to `figures/`. Generated files are ignored by Git.

## Numerical settings

The defaults in `scripts/mutual_information.jl` use 50 initial concentrations and `10^6` conditional samples per initial concentration. This is a large calculation. It is checkpointed after every initial concentration and should be run with multiple Julia threads.

The principal settings can also be overridden with environment variables. For example:

```bash
CS14_N_INITIAL=10 CS14_N_SAMPLES=100000 CS14_K=10 \
  julia --project=. -t auto scripts/mutual_information.jl
```

For `scripts/information_loss.jl`, `CS14_MAIN_TMAX` controls the maximum time in the main panel, while `CS14_GAUSSIAN_TMAX` sets the common maximum time for the inset and `asymptotic_correction.pdf`.

## Model

The reaction channels are

```text
Y       -> X + Y
2X      -> 3X
X + Z   -> 2Z
Z       -> Y
X + Y   -> X
3X      -> 2X
0       -> Z
```

with rate constants `k1 = 0.2`, `k2 = ... = k5 = 1`, `delta = 0.02`, and `gamma_z = 0.01`.

Random seeds and all numerical settings are recorded in the output CSV files.

## Output files

The scripts create the following figures under `figures/`:

- `deterministic_trajectory.pdf`: deterministic attractor trajectory.
- `stochastic_trajectories.pdf`: tau-leap trajectories at small and large system size.
- `mutual_information.pdf`: two-time mutual information for all five system sizes.
- `information_loss.pdf`: relative information loss with the Gaussian prediction and its rate residual in the inset.
- `asymptotic_correction.pdf`: transformed residual used to examine the logarithmic finite-time correction.
