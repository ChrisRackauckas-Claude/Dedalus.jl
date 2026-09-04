# Tutorial 4: Analysis and Output

This tutorial covers Dedalus.jl's analysis framework for saving simulation
data and monitoring flow properties during time-stepping.

## File Handlers

The analysis system uses file handlers to save field data to HDF5 files
at specified intervals during an IVP simulation.

```julia
using Dedalus

# After building an IVP solver:
solver = build_solver(problem, RK222)
solver.stop_sim_time = 100.0

# Create a file handler that saves every 0.25 time units
snapshots = add_file_handler(
    solver.evaluator, "snapshots";
    sim_dt=0.25,
    max_writes=50
)

# Add fields or expressions to save
add_task!(snapshots, b; name="buoyancy")
add_task!(snapshots, -divergence(skew(u)); name="vorticity")
```

File handlers support several triggering modes:
- `sim_dt`: save every N simulation time units
- `wall_dt`: save every N wall-clock seconds
- `iteration`: save every N solver iterations

The `max_writes` parameter limits the number of snapshots per HDF5 file
before a new file is started.

## CFL Condition

The [`CFL`](@ref) object adaptively computes the timestep for IVP solvers
based on the Courant-Friedrichs-Lewy condition.

```julia
cfl = CFL(
    solver;
    initial_dt=0.1,
    cadence=10,
    safety=0.5,
    threshold=0.05,
    max_change=1.5,
    min_change=0.5,
    max_dt=0.125
)
add_velocity!(cfl, u)

# In the time-stepping loop:
while solver.proceed
    dt = compute_timestep(cfl)
    step!(solver, dt)
end
```

Parameters:
- `initial_dt`: starting timestep
- `cadence`: how often to recompute (every N iterations)
- `safety`: safety factor applied to the CFL limit
- `threshold`: minimum grid-crossing fraction to trigger dt reduction
- `max_change` / `min_change`: maximum allowed timestep change ratio
- `max_dt`: hard upper bound on timestep

## Global Flow Properties

[`GlobalFlowProperty`](@ref) computes global statistics of field expressions
during the simulation, useful for monitoring convergence and physical behavior.

```julia
flow = GlobalFlowProperty(solver; cadence=10)
add_property!(flow, sqrt(DotProduct(u, u)) / nu; name="Re")

# In the time-stepping loop:
while solver.proceed
    dt = compute_timestep(cfl)
    step!(solver, dt)
    if (solver.iteration - 1) % 10 == 0
        max_Re = flow_max(flow, "Re")
        @info "Iteration=$(solver.iteration), max(Re)=$(max_Re)"
    end
end
```

Available reduction functions:
- `flow_max(flow, name)`: maximum over all grid points
- `flow_min(flow, name)`: minimum over all grid points
- `flow_volume_integral(flow, name)`: volume integral

## Gathering Data

To collect distributed data for post-processing or plotting:

```julia
# Gather a field's grid data across all MPI ranks
ug = allgather_data(u, "g")

# Get global grid coordinates
x = global_grid(xbasis, dist; scale=1)
y = global_grid(ybasis, dist; scale=1)
```

## Complete IVP Loop Pattern

A typical IVP simulation combines all the above:

```julia
# Solver setup
solver = build_solver(problem, RK222)
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = add_file_handler(solver.evaluator, "snapshots"; sim_dt=0.25)
add_task!(snapshots, b; name="buoyancy")

# CFL
cfl = CFL(solver; initial_dt=max_timestep, cadence=10, safety=0.5,
          max_change=1.5, min_change=0.5, max_dt=max_timestep)
add_velocity!(cfl, u)

# Flow monitoring
flow = GlobalFlowProperty(solver; cadence=10)
add_property!(flow, sqrt(DotProduct(u, u)) / nu; name="Re")

# Main loop
try
    @info "Starting main loop"
    while solver.proceed
        timestep = compute_timestep(cfl)
        step!(solver, timestep)
        if (solver.iteration - 1) % 10 == 0
            max_Re = flow_max(flow, "Re")
            @info "Iteration=$(solver.iteration), Time=$(solver.sim_time), " *
                  "dt=$(timestep), max(Re)=$(max_Re)"
        end
    end
catch e
    @error "Exception raised, triggering end of main loop."
    rethrow(e)
finally
    log_stats(solver)
end
```

This pattern — file handler for snapshots, CFL for adaptive time-stepping,
flow properties for monitoring — is used across all IVP examples in the
Dedalus.jl repository.
