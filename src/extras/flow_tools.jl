"""
Extra tools useful in hydrodynamical problems.

Translated from dedalus/extras/flow_tools.py.  Provides:

- `GlobalArrayReducer`: parallel reduction of distributed array data
- `GlobalFlowProperty`: global flow property computation on the grid
- `CFL`: CFL-limited timestep computation

When MPI is not available (serial mode), all reductions operate locally.
"""

using Logging

const _flow_tools_logger = Logging.current_logger()

# ---------------------------------------------------------------------------
# GlobalArrayReducer
# ---------------------------------------------------------------------------

"""
    GlobalArrayReducer

Directs parallelised reduction of distributed array data.

# Fields
- `comm`: MPI communicator (or `nothing` for serial mode)
- `dtype`: Element type used for the internal scalar buffer
- `_scalar_buffer`: Single-element vector used as reduction scratch space

# Constructor
    GlobalArrayReducer(comm=nothing; dtype=Float64)
"""
mutable struct GlobalArrayReducer{T<:AbstractFloat}
    comm::Any               # MPI communicator or `nothing`
    _scalar_buffer::Vector{T}

    function GlobalArrayReducer(comm=nothing; dtype::Type{T}=Float64) where {T<:AbstractFloat}
        new{T}(comm, zeros(T, 1))
    end
end

"""
    reduce_scalar(reducer, local_scalar, mpi_reduce_op) -> scalar

Compute global reduction of a scalar from each process.

In serial mode (comm === nothing), the local value is returned directly.
When MPI is available, `MPI.Allreduce!` is used with the specified operation.
"""
function reduce_scalar(reducer::GlobalArrayReducer, local_scalar, mpi_reduce_op)
    reducer._scalar_buffer[1] = local_scalar
    if reducer.comm === nothing
        return reducer._scalar_buffer[1]
    end
    # MPI path
    MPI = Main.MPI
    Base.invokelatest(MPI.Allreduce!, reducer._scalar_buffer, mpi_reduce_op, reducer.comm)
    return reducer._scalar_buffer[1]
end

"""
    global_min(reducer, data; empty=Inf) -> scalar

Compute global min of all array data. When `data` is empty, `empty` is used as
the local contribution (defaults to `Inf` so it does not affect the result).
"""
function global_min(reducer::GlobalArrayReducer, data; empty=Inf)
    if length(data) > 0
        local_min = minimum(data)
    else
        local_min = empty
    end
    if reducer.comm === nothing
        return local_min
    end
    MPI = Main.MPI
    return reduce_scalar(reducer, local_min, Base.invokelatest(getproperty, MPI, :MIN))
end

"""
    global_max(reducer, data; empty=-Inf) -> scalar

Compute global max of all array data. When `data` is empty, `empty` is used as
the local contribution (defaults to `-Inf`).
"""
function global_max(reducer::GlobalArrayReducer, data; empty=-Inf)
    if length(data) > 0
        local_max = maximum(data)
    else
        local_max = empty
    end
    if reducer.comm === nothing
        return local_max
    end
    MPI = Main.MPI
    return reduce_scalar(reducer, local_max, Base.invokelatest(getproperty, MPI, :MAX))
end

"""
    global_mean(reducer, data) -> scalar

Compute global mean of all array data (total sum / total count across all
processes).
"""
function global_mean(reducer::GlobalArrayReducer, data)
    local_sum = sum(data)
    local_size = length(data)
    if reducer.comm === nothing
        return local_sum / local_size
    end
    MPI = Main.MPI
    mpi_sum = Base.invokelatest(getproperty, MPI, :SUM)
    global_sum = reduce_scalar(reducer, local_sum, mpi_sum)
    global_size = reduce_scalar(reducer, Float64(local_size), mpi_sum)
    return global_sum / global_size
end

# ---------------------------------------------------------------------------
# GlobalFlowProperty
# ---------------------------------------------------------------------------

"""
    GlobalFlowProperty

Directs parallelised determination of a global flow property on the grid.

# Fields
- `solver`: The problem solver
- `cadence`: Iteration cadence for property evaluation
- `reducer`: `GlobalArrayReducer` used for parallel reductions
- `properties`: Dictionary handler from the solver's evaluator

# Constructor
    GlobalFlowProperty(solver; cadence=1)
"""
mutable struct GlobalFlowProperty
    solver::Any
    cadence::Int
    reducer::GlobalArrayReducer
    properties::Any  # dictionary handler from solver.evaluator

    function GlobalFlowProperty(solver; cadence::Int=1)
        comm = _get_solver_comm(solver)
        reducer = GlobalArrayReducer(comm)
        properties = add_dictionary_handler(solver.evaluator; iter=cadence)
        new(solver, cadence, reducer, properties)
    end
end

"""
    _get_solver_comm(solver)

Retrieve the MPI communicator from a solver's distributor.
Returns `nothing` when the solver has no `dist` field or no `comm_cart`.
"""
function _get_solver_comm(solver)
    try
        return solver.dist.comm_cart
    catch
        return nothing
    end
end

"""
    add_dictionary_handler(evaluator; iter=1)

Placeholder that delegates to the evaluator's `add_dictionary_handler` method.
This will call the real implementation once the evaluator module is translated.
"""
function add_dictionary_handler(evaluator; iter=1)
    return evaluator.add_dictionary_handler(; iter=iter)
end

"""
    add_property!(flow::GlobalFlowProperty, property, name::AbstractString;
                  precompute_integral::Bool=false)

Add a property to be evaluated. If `precompute_integral` is `true`, a companion
integral task is registered under the name `_<name>_integral`.
"""
function add_property!(flow::GlobalFlowProperty, property, name::AbstractString;
                       precompute_integral::Bool=false)
    add_task!(flow.properties, property; layout=:g, name=name)
    if precompute_integral
        task_op = flow.properties.tasks[end]["operator"]
        integral_op = Integrate(task_op)
        integral_name = "_$(name)_integral"
        add_task!(flow.properties, integral_op; layout=:g, name=integral_name)
    end
end

"""
    add_task!(handler, task; kwargs...)

Placeholder that delegates to the handler's `add_task` method. This will call
the real implementation once the evaluator/handler modules are translated.
"""
function add_task!(handler, task; kwargs...)
    handler.add_task(task; kwargs...)
end

"""
    flow_min(flow::GlobalFlowProperty, name::AbstractString) -> scalar

Compute global minimum of a property on the grid.
"""
function flow_min(flow::GlobalFlowProperty, name::AbstractString)
    gdata = flow.properties[name]["g"]
    return global_min(flow.reducer, gdata)
end

"""
    flow_max(flow::GlobalFlowProperty, name::AbstractString) -> scalar

Compute global maximum of a property on the grid.
"""
function flow_max(flow::GlobalFlowProperty, name::AbstractString)
    gdata = flow.properties[name]["g"]
    return global_max(flow.reducer, gdata)
end

"""
    grid_average(flow::GlobalFlowProperty, name::AbstractString) -> scalar

Compute global mean of a property on the grid.
"""
function grid_average(flow::GlobalFlowProperty, name::AbstractString)
    gdata = flow.properties[name]["g"]
    return global_mean(flow.reducer, gdata)
end

"""
    volume_integral(flow::GlobalFlowProperty, name::AbstractString) -> scalar

Compute volume integral of a property. Uses a precomputed integral task if one
was registered via `add_property!(...; precompute_integral=true)`, otherwise
computes the integral on the fly.
"""
function volume_integral(flow::GlobalFlowProperty, name::AbstractString)
    integral_name = "_$(name)_integral"
    integral_field = nothing
    try
        integral_field = flow.properties[integral_name]
    catch e
        if isa(e, KeyError)
            # Compute volume integral on the fly
            field = flow.properties[name]
            integral_op = Integrate(field)
            integral_field = evaluate_future(integral_op)
        else
            rethrow(e)
        end
    end
    # Communicate integral value to all processes (max gives the single value)
    return global_max(flow.reducer, integral_field["g"])
end

"""
    volume_average(flow::GlobalFlowProperty, name::AbstractString) -> scalar

Compute volume average of a property. Currently raises an error because the
hypervolume definition is not yet implemented (matches the Python version).
"""
function volume_average(flow::GlobalFlowProperty, name::AbstractString)
    error("volume_average is not implemented: missing definition of hypervolume")
    # When hypervolume is available:
    # return volume_integral(flow, name) / flow.solver.domain.hypervolume
end

# ---------------------------------------------------------------------------
# Integrate / AdvectiveCFL operator placeholders
# ---------------------------------------------------------------------------

# Integrate, AdvectiveCFL, and evaluate_future are defined in core/operators.jl
# and core/future.jl respectively (included before this file).

# ---------------------------------------------------------------------------
# CFL
# ---------------------------------------------------------------------------

"""
    CFL

Computes CFL-limited timestep from a set of frequencies/velocities.

The new timestep is computed by summing across the provided frequencies
for each grid point, and then reciprocating the maximum "total" frequency
from the entire grid.

# Fields
- `solver`: Problem solver
- `stored_dt`: Current stored timestep
- `cadence`: Iteration cadence for computing new timestep
- `safety`: Safety factor for scaling computed timestep
- `max_dt`: Maximum allowable timestep
- `min_dt`: Minimum allowable timestep
- `max_change`: Maximum fractional change between timesteps
- `min_change`: Minimum fractional change between timesteps
- `threshold`: Fractional change threshold for updating timestep
- `reducer`: `GlobalArrayReducer` for parallel reductions
- `frequencies`: Dictionary handler for frequency tasks

# Constructor
    CFL(solver, initial_dt;
        cadence=1, safety=1.0, max_dt=Inf, min_dt=0.0,
        max_change=Inf, min_change=0.0, threshold=0.0)
"""
mutable struct CFL
    solver::Any
    stored_dt::Float64
    cadence::Int
    safety::Float64
    max_dt::Float64
    min_dt::Float64
    max_change::Float64
    min_change::Float64
    threshold::Float64
    reducer::GlobalArrayReducer
    frequencies::Any  # dictionary handler

    function CFL(solver, initial_dt::Real;
                 cadence::Int=1,
                 safety::Real=1.0,
                 max_dt::Real=Inf,
                 min_dt::Real=0.0,
                 max_change::Real=Inf,
                 min_change::Real=0.0,
                 threshold::Real=0.0)
        comm = _get_solver_comm(solver)
        reducer = GlobalArrayReducer(comm)
        frequencies = add_dictionary_handler(solver.evaluator; iter=cadence)
        new(solver, Float64(initial_dt), cadence, Float64(safety),
            Float64(max_dt), Float64(min_dt), Float64(max_change),
            Float64(min_change), Float64(threshold),
            reducer, frequencies)
    end
end

"""
    compute_dt(cfl::CFL) -> Float64

Deprecated. Use `compute_timestep` instead.
"""
function compute_dt(cfl::CFL)
    @warn "'compute_dt' is deprecated. Use 'compute_timestep' instead."
    return compute_timestep(cfl)
end

"""
    compute_timestep(cfl::CFL) -> Float64

Compute the CFL-limited timestep. The timestep is recomputed only when the
cadence divides the previous iteration (i.e., when the frequency dictionary
handler is freshly updated). On the very first evaluation the stored initial
timestep is returned.
"""
function compute_timestep(cfl::CFL)
    iteration = cfl.solver.iteration
    # Compute new timestep when cadence divides previous iteration
    # (this is when the frequency dicthandler is freshly updated)
    if (iteration - 1) % cfl.cadence == 0
        # Return initial dt on first evaluation
        if (iteration - 1) <= cfl.solver.initial_iteration
            return cfl.stored_dt
        end
        # Sum across frequencies for each local grid point
        local_freqs = sum(abs.(field["g"]) for field in values(cfl.frequencies.fields))
        # Compute new timestep from max frequency across all grid points
        max_global_freq = global_max(cfl.reducer, local_freqs)
        if max_global_freq == 0.0
            dt = Inf
        else
            dt = 1.0 / max_global_freq
        end
        # Apply restrictions
        dt *= cfl.safety
        dt = min(dt, cfl.max_dt, cfl.max_change * cfl.stored_dt)
        dt = max(dt, cfl.min_dt, cfl.min_change * cfl.stored_dt)
        if abs(dt - cfl.stored_dt) > cfl.threshold * cfl.stored_dt
            cfl.stored_dt = dt
        end
    end
    return cfl.stored_dt
end

"""
    add_frequency!(cfl::CFL, freq)

Add an on-grid frequency to the CFL computation.
"""
function add_frequency!(cfl::CFL, freq)
    add_task!(cfl.frequencies, freq; layout=:g, scales=freq.domain.dealias)
end

"""
    add_velocity!(cfl::CFL, velocity)

Add grid-crossing frequency from a velocity vector.

The velocity must be a vector (its `tensorsig` must have exactly one element).
An `AdvectiveCFL` operator is constructed and added as a frequency.
"""
function add_velocity!(cfl::CFL, velocity)
    coords = velocity.tensorsig
    if length(coords) != 1
        throw(ArgumentError("Velocity must be a vector (tensorsig length 1), " *
                            "got length $(length(coords))"))
    end
    cfl_operator = AdvectiveCFL(velocity, coords[1])
    add_frequency!(cfl, cfl_operator)
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export GlobalArrayReducer,
       reduce_scalar,
       global_min,
       global_max,
       global_mean,
       GlobalFlowProperty,
       add_property!,
       flow_min,
       flow_max,
       grid_average,
       volume_integral,
       volume_average,
       CFL,
       compute_dt,
       compute_timestep,
       add_frequency!,
       add_velocity!
