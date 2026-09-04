"""
    Evaluator and output handler types for Dedalus.jl

Julia translation of `dedalus/core/evaluator.py`. Manages centralized
evaluation of operator expression trees and directs the results to various
output handlers (dictionary, system, HDF5 file).

## Type hierarchy

    AbstractHandler
    +-- DictionaryHandler  (stores outputs in a Dict)
    +-- SystemHandler       (collects fields for a FieldSystem)
    +-- H5FileHandlerBase  (HDF5 file output -- concrete, mode-dispatched)
        Modes (selected via `_parallel_mode` field):
        +-- :gather   (H5GatherFileHandler -- root gathers all data via MPI)
        +-- :virtual  (H5VirtualFileHandler -- per-process files + virtual dataset)
        +-- :mpio     (H5ParallelFileHandler -- parallel HDF5 via MPI-IO)

    Evaluator  (coordinates evaluation and dispatches to handlers)

## Key translation choices

- Python `h5py`            -> Julia `HDF5.jl` (`h5open`, `create_dataset`, etc.)
- Python `defaultdict`     -> Julia `Dict{String,Vector}` with `get!`
- Python `uuid.uuid4()`    -> Julia `UUIDs.uuid4()`
- Python `shutil.rmtree`   -> Julia `rm(...; recursive=true)`
- Python `pathlib.Path`    -> Julia stdlib `Base.Filesystem` paths (strings)
- Python `collections.OrderedSet` -> local `OrderedSet` from tools/general.jl
- Python `Sync` context    -> `sync` do-block / `_handler_sync` from tools/parallel.jl
- Python class inheritance -> Julia abstract types + composition + mode dispatch
- Python `prod` from math  -> Julia `prod`
- HDF5 dimension scales    -> omitted (HDF5.jl support is limited; metadata
  is written as attributes instead)
- Virtual datasets         -> low-level HDF5 API with graceful fallback
- Parallel HDF5 (MPIO)     -> `h5open(path, mode, comm, info)` via HDF5.jl
"""

using HDF5
using UUIDs: uuid4
using SHA: sha1

# ============================================================================
# Configuration defaults
# ============================================================================

const FILEHANDLER_MODE_DEFAULT = get(
    get(config, "analysis", Dict{String,Any}()),
    "FILEHANDLER_MODE_DEFAULT", "overwrite")

const FILEHANDLER_PARALLEL_DEFAULT = get(
    get(config, "analysis", Dict{String,Any}()),
    "FILEHANDLER_PARALLEL_DEFAULT", "gather")

const FILEHANDLER_TOUCH_TMPFILE = let
    val = get(get(config, "analysis", Dict{String,Any}()),
              "FILEHANDLER_TOUCH_TMPFILE", false)
    if val isa Bool
        val
    elseif val isa AbstractString
        lowercase(val) in ("true", "1", "yes")
    else
        false
    end
end

# ============================================================================
# Evaluator
# ============================================================================

"""
    Evaluator

Coordinates evaluation of operator trees through various handlers.

# Fields
- `dist`      -- Distributor object
- `vars`      -- namespace dictionary for parsing task expressions
- `handlers`  -- list of all registered handlers
- `groups`    -- dictionary mapping group names to handler lists
"""
mutable struct Evaluator
    dist::Any
    vars::Dict{String, Any}
    handlers::Vector{Any}
    groups::Dict{String,Vector{Any}}
    # Pre-allocated buffers to avoid per-evaluation allocations
    _scheduled_buf::Vector{Any}      # reused by evaluate_scheduled
    _tasks_buf::Vector{Dict{String,Any}}  # reused by evaluate_handlers
    _unfinished_buf::Vector{Dict{String,Any}}  # reused by attempt_tasks

    function Evaluator(dist, vars::Dict{String, Any})
        return new(dist, vars, Any[], Dict{String,Vector{Any}}(),
                   Any[], Dict{String,Any}[], Dict{String,Any}[])
    end
end

"""
    add_dictionary_handler(ev::Evaluator; kw...) -> DictionaryHandler

Create a DictionaryHandler and register it with the evaluator.
"""
function add_dictionary_handler(ev::Evaluator; kw...)
    dh = DictionaryHandler(ev.dist, ev.vars; kw...)
    return add_handler(ev, dh)
end

"""
    add_system_handler(ev::Evaluator; kw...) -> SystemHandler

Create a SystemHandler and register it with the evaluator.
"""
function add_system_handler(ev::Evaluator; kw...)
    sh = SystemHandler(ev.dist, ev.vars; kw...)
    return add_handler(ev, sh)
end

"""
    add_file_handler(ev::Evaluator, filename; parallel=nothing, kw...) -> Handler

Create a file handler and register it with the evaluator.

`parallel` selects the parallel writing strategy:
- `"gather"` -- root process gathers all data (default in serial mode)
- `"virtual"` -- virtual datasets (H5VirtualFileHandler)
- `"mpio"` -- parallel HDF5 (H5ParallelFileHandler)
"""
function add_file_handler(ev::Evaluator, filename; parallel=nothing, kw...)
    if parallel === nothing
        if _handler_comm_size(ev.dist.comm_cart) == 1
            parallel = "gather"
        else
            parallel = FILEHANDLER_PARALLEL_DEFAULT
        end
    end
    if parallel == "gather"
        FH = H5GatherFileHandler
    elseif parallel == "virtual"
        FH = H5VirtualFileHandler
    elseif parallel == "mpio"
        FH = H5ParallelFileHandler
    else
        throw(ArgumentError("Parallel method '$(parallel)' not recognized."))
    end
    return add_handler(ev, FH(filename, ev.dist, ev.vars; kw...))
end

"""
    add_handler(ev::Evaluator, handler) -> handler

Register a handler with the evaluator. If the handler has a non-nothing
`group`, also register it under that group name.
"""
function add_handler(ev::Evaluator, handler)
    push!(ev.handlers, handler)
    if handler.group !== nothing
        handlers_list = get!(Vector{Any}, ev.groups, handler.group)
        push!(handlers_list, handler)
    end
    return handler
end

"""
    evaluate_group(ev::Evaluator, group; kw...)

Evaluate all handlers belonging to a named group.
"""
function evaluate_group(ev::Evaluator, group; kw...)::Nothing
    handlers = get(ev.groups, group, Any[])
    evaluate_handlers(ev, handlers; kw...)
    return nothing
end

"""
    evaluate_scheduled(ev::Evaluator; kw...)

Evaluate all handlers whose schedules indicate they are due.

Uses a reusable buffer (`_scheduled_buf`) to avoid allocating a new handler
vector on every evaluation cycle.  The buffer is resized in-place via
`empty!` / `push!`.
"""
function evaluate_scheduled(ev::Evaluator; kw...)::Nothing
    buf = ev._scheduled_buf
    empty!(buf)
    for h in ev.handlers
        if check_schedule(h; kw...)
            push!(buf, h)
        end
    end
    evaluate_handlers(ev, buf; kw...)
    return nothing
end

"""Alias: callers in solvers.jl and timesteppers.jl use the bang name."""
const evaluate_scheduled! = evaluate_scheduled

"""
    evaluate_handlers(ev::Evaluator, handlers; id=nothing, kw...)

Evaluate a collection of handlers: attempt all tasks, transform fields
through layouts until every task is resolved, then process outputs.

Reuses pre-allocated buffers on the Evaluator (`_tasks_buf`, `_unfinished_buf`)
to avoid creating temporary vectors on every evaluation cycle.
"""
function evaluate_handlers(ev::Evaluator, handlers; id=nothing, kw...)
    # Default to uuid to cache within evaluation but not across evaluations
    if id === nothing
        id = uuid4()
    end

    # Collect tasks into reusable buffer and clear previous outputs
    tasks_buf = ev._tasks_buf
    empty!(tasks_buf)
    for h in handlers
        for t in h.tasks
            push!(tasks_buf, t)
        end
    end
    for task in tasks_buf
        task["out"] = nothing
    end

    # Attempt initial evaluation (attempt_tasks! filters in-place)
    attempt_tasks!(ev, tasks_buf; id=id)

    # Move all fields to coefficient layout
    fields = get_task_fields(tasks_buf)
    require_coeff_space(ev, fields)
    attempt_tasks!(ev, tasks_buf; id=id)

    # Oscillate through layouts until all tasks are evaluated
    # Limit to 10 passes to break on potential infinite loops
    n_layouts = length(ev.dist.layouts)
    osc = oscillate(0:(n_layouts - 1); max_passes=10)
    osc_state = iterate(osc)
    if osc_state === nothing
        return
    end
    current_index, osc_st = osc_state

    while !isempty(tasks_buf)
        next_result = iterate(osc, osc_st)
        if next_result === nothing
            break
        end
        next_index, osc_st = next_result
        # Transform fields
        fields = get_task_fields(tasks_buf)
        if current_index < next_index
            path = ev.dist.paths[current_index + 1]  # 1-based indexing
            increment(path, fields)
        else
            path = ev.dist.paths[next_index + 1]      # 1-based indexing
            decrement(path, fields)
        end
        current_index = next_index
        # Attempt evaluation
        attempt_tasks!(ev, tasks_buf; id=id)
    end

    # Transform all outputs to coefficient layout to dealias
    outputs = OrderedSet{Any}()
    for h in handlers
        for t in h.tasks
            out = t["out"]
            if out !== nothing && !isa(out, LockedField)
                push!(outputs, out)
            end
        end
    end
    require_coeff_space(ev, outputs)

    # Copy redundant outputs so processing is independent
    seen = Set{Any}()
    for handler in handlers
        for task in handler.tasks
            if task["out"] in seen
                task["out"] = copy(task["out"])
            else
                push!(seen, task["out"])
            end
        end
    end

    # Process
    for handler in handlers
        process(handler; kw...)
    end
end

"""
    require_coeff_space(ev::Evaluator, fields)

Move all fields in `fields` to the coefficient layout.
"""
function require_coeff_space(ev::Evaluator, fields)::Nothing
    coeff_layout = ev.dist.coeff_layout
    # Quick return if all fields are already in coeff layout
    if all(f -> f.layout === coeff_layout, fields)
        return
    end
    # Build dictionary of starting layout indices
    layouts = Dict{Int,Vector{Any}}()
    for f in fields
        if f.layout !== coeff_layout
            idx = f.layout.index
            push!(get!(Vector{Any}, layouts, idx), f)
        end
    end
    isempty(layouts) && return
    # Decrement all fields down to coeff layout
    current_fields = Any[]
    for index in sort(collect(keys(layouts)); rev=true)
        index <= coeff_layout.index && continue
        append!(current_fields, layouts[index])
        path = ev.dist.paths[index]  # path at index connects index-1 <-> index
        decrement(path, current_fields)
    end
end

"""
    require_grid_space(ev::Evaluator, fields)

Move all fields in `fields` to the grid layout.
"""
function require_grid_space(ev::Evaluator, fields)
    grid_layout = ev.dist.grid_layout
    # Quick return if all fields are already in grid layout
    if all(f -> f.layout === grid_layout, fields)
        return
    end
    # Build dictionary of starting layout indices
    layouts = Dict{Int,Vector{Any}}()
    for f in fields
        if f.layout !== grid_layout
            idx = f.layout.index
            push!(get!(Vector{Any}, layouts, idx), f)
        end
    end
    isempty(layouts) && return
    # Increment all fields up to grid layout
    current_fields = Any[]
    for index in sort(collect(keys(layouts)))
        index >= grid_layout.index && continue
        append!(current_fields, layouts[index])
        path = ev.dist.paths[index + 1]  # path at index+1 connects index <-> index+1
        increment(path, current_fields)
    end
end

"""
    get_task_fields(tasks) -> OrderedSet

Collect all `Field` atoms from a list of tasks, excluding `LockedField`.
"""
function get_task_fields(tasks)
    fields = OrderedSet{Any}()
    for task in tasks
        for atom in atoms(task["operator"], Field)
            # Skip LockedField atoms directly instead of collecting and
            # removing them in a second pass, avoiding a temporary vector.
            isa(atom, LockedField) && continue
            push!(fields, atom)
        end
    end
    return fields
end

"""
    attempt_tasks(tasks; kw...) -> Vector

Attempt each task's operator and return only the unfinished tasks.
"""
function attempt_tasks(tasks; kw...)
    unfinished = similar(tasks, 0)
    for task in tasks
        output = attempt(task["operator"]; kw...)
        if output === nothing
            push!(unfinished, task)
        else
            task["out"] = output
        end
    end
    return unfinished
end

"""
    attempt_tasks!(ev::Evaluator, tasks::Vector{Dict{String,Any}}; kw...)

In-place variant of `attempt_tasks`: evaluates each task and removes
finished entries from `tasks`, reusing the Evaluator's `_unfinished_buf`
to avoid allocating a new vector per call.
"""
function attempt_tasks!(ev::Evaluator, tasks::Vector{Dict{String,Any}}; kw...)
    buf = ev._unfinished_buf
    empty!(buf)
    for task in tasks
        output = attempt(task["operator"]; kw...)
        if output === nothing
            push!(buf, task)
        else
            task["out"] = output
        end
    end
    # Swap contents back into tasks
    empty!(tasks)
    append!(tasks, buf)
    return nothing
end

# ============================================================================
# AbstractHandler
# ============================================================================

"""
    AbstractHandler

Abstract base type for all output handlers.
"""
abstract type AbstractHandler end

# ============================================================================
# Handler (concrete base with scheduling)
# ============================================================================

"""
    Handler <: AbstractHandler

Concrete handler with scheduling cadence support. Serves as base for
`DictionaryHandler`, `SystemHandler`, and `H5FileHandlerBase`.

# Fields
- `dist`       -- Distributor
- `vars`       -- namespace dictionary
- `group`      -- optional group name
- `wall_dt`    -- wall-time cadence (seconds), or nothing
- `sim_dt`     -- simulation-time cadence, or nothing
- `iter`       -- iteration cadence, or nothing
- `custom_schedule` -- optional custom scheduling function
- `tasks`      -- vector of task dictionaries
- `last_wall_div` -- last wall-time division index
- `last_sim_div`  -- last sim-time division index
- `last_iter_div` -- last iteration division index
"""
mutable struct Handler <: AbstractHandler
    dist::Any
    vars::Dict
    group::Any
    wall_dt::Union{Nothing,Real}
    sim_dt::Union{Nothing,Real}
    iter::Union{Nothing,Integer}
    custom_schedule::Any
    tasks::Vector{Dict{String,Any}}
    last_wall_div::Int
    last_sim_div::Int
    last_iter_div::Int

    function Handler(dist, vars::Dict;
                     group=nothing, wall_dt=nothing, sim_dt=nothing,
                     iter=nothing, custom_schedule=nothing)
        return new(dist, vars, group, wall_dt, sim_dt, iter, custom_schedule,
                   Dict{String,Any}[], -1, -1, -1)
    end
end

"""
    check_schedule(h::AbstractHandler; kw...) -> Bool

Check whether the handler should be triggered based on wall_time,
sim_time, iteration, and/or custom schedule.
"""
function check_schedule(h::AbstractHandler; kw...)
    scheduled = false
    # Wall time
    if h.wall_dt !== nothing && h.wall_dt > 0
        wall_time = kw[:wall_time]
        wall_div = floor(Int, wall_time / h.wall_dt)
        if wall_div > h.last_wall_div
            scheduled = true
            h.last_wall_div = wall_div
        end
    end
    # Sim time
    if h.sim_dt !== nothing && h.sim_dt > 0
        t = kw[:sim_time]
        dt = kw[:timestep]
        closest_sim_div = round(Int, t / h.sim_dt)
        if closest_sim_div > h.last_sim_div
            closest_sim_time = closest_sim_div * h.sim_dt
            if abs(t - closest_sim_time) < abs(t + dt - closest_sim_time)
                scheduled = true
                h.last_sim_div = closest_sim_div
            end
        end
    end
    # Iteration
    if h.iter !== nothing && h.iter > 0
        iteration = kw[:iteration]
        iter_div = fld(iteration, h.iter)
        if iter_div > h.last_iter_div
            scheduled = true
            h.last_iter_div = iter_div
        end
    end
    # Custom
    if h.custom_schedule !== nothing
        if h.custom_schedule(; kw...)
            scheduled = true
        end
    end
    return scheduled
end

"""
    add_task!(h::AbstractHandler, task; layout="g", name=nothing, scales=nothing)

Register an output task with the handler.

`task` may be a string (parsed to an operator), a `Field` (wrapped in a
copy operator), or an already-built operator.
"""
function add_task!(h::AbstractHandler, task; layout="g", name=nothing, scales=nothing)
    # Default name
    if name === nothing
        name = string(task)
    end
    # Create operator
    if task isa AbstractString
        op = parse_future_field(task, h.vars, h.dist)
    elseif isa(task, Field)
        op = _field_copy(task)
    else
        op = task
    end
    # Check scales
    if isa(op, LockedField) || isa(op, FutureLockedField)
        if scales === nothing
            scales = op.domain.dealias
        else
            scales = remedy_scales(h.dist, scales)
            if scales != op.domain.dealias
                scales = op.domain.dealias
                @warn "Cannot specify non-dealias scales for LockedFields"
            end
        end
    else
        scales = remedy_scales(h.dist, scales)
    end
    # Build task dictionary
    td = Dict{String,Any}()
    td["operator"] = op
    td["layout"] = get_layout_object(h.dist, layout)
    td["name"] = name
    td["scales"] = scales
    td["dtype"] = op.dtype
    push!(h.tasks, td)
    return nothing
end

"""
    add_tasks!(h::AbstractHandler, tasks; name="", kw...)

Register multiple output tasks.
"""
function add_tasks!(h::AbstractHandler, tasks; name::AbstractString="", kw...)
    for task in tasks
        tname = name * string(task)
        add_task!(h, task; name=tname, kw...)
    end
end

"""
    add_system!(h::AbstractHandler, system; kw...)

Add fields from a FieldSystem.
"""
function add_system!(h::AbstractHandler, system; kw...)
    add_tasks!(h, system.fields; kw...)
end

"""
    process(h::AbstractHandler; kw...)

Process handler outputs. Must be implemented by concrete subtypes.
"""
function process(h::AbstractHandler; kw...)
    error("process() not implemented for $(typeof(h))")
end

# ============================================================================
# DictionaryHandler
# ============================================================================

"""
    DictionaryHandler <: AbstractHandler

Handler that stores task outputs in a dictionary, keyed by task name.

# Fields
Inherits all Handler fields plus:
- `fields` -- Dict{String,Any} mapping task names to field objects
"""
mutable struct DictionaryHandler <: AbstractHandler
    dist::Any
    vars::Dict
    group::Any
    wall_dt::Union{Nothing,Real}
    sim_dt::Union{Nothing,Real}
    iter::Union{Nothing,Integer}
    custom_schedule::Any
    tasks::Vector{Dict{String,Any}}
    last_wall_div::Int
    last_sim_div::Int
    last_iter_div::Int
    fields::Dict{String,Any}

    function DictionaryHandler(dist, vars::Dict; kw...)
        h = new()
        base = Handler(dist, vars; kw...)
        h.dist = base.dist
        h.vars = base.vars
        h.group = base.group
        h.wall_dt = base.wall_dt
        h.sim_dt = base.sim_dt
        h.iter = base.iter
        h.custom_schedule = base.custom_schedule
        h.tasks = base.tasks
        h.last_wall_div = base.last_wall_div
        h.last_sim_div = base.last_sim_div
        h.last_iter_div = base.last_iter_div
        h.fields = Dict{String,Any}()
        return h
    end
end

"""
    Base.getindex(dh::DictionaryHandler, name::AbstractString)

Access a stored field by name.
"""
Base.getindex(dh::DictionaryHandler, name::AbstractString) = dh.fields[name]

"""
    process(dh::DictionaryHandler; kw...)

Reference task outputs from the dictionary handler's fields dict.
"""
function process(dh::DictionaryHandler; kw...)
    for task in dh.tasks
        out = task["out"]
        change_scales!(out, task["scales"])
        change_layout!(out, task["layout"])
        dh.fields[task["name"]] = out
    end
end

# ============================================================================
# SystemHandler
# ============================================================================

"""
    SystemHandler <: AbstractHandler

Handler that collects fields for system-level operations.
"""
mutable struct SystemHandler <: AbstractHandler
    dist::Any
    vars::Dict
    group::Any
    wall_dt::Union{Nothing,Real}
    sim_dt::Union{Nothing,Real}
    iter::Union{Nothing,Integer}
    custom_schedule::Any
    tasks::Vector{Dict{String,Any}}
    last_wall_div::Int
    last_sim_div::Int
    last_iter_div::Int
    fields::Vector{Any}

    function SystemHandler(dist, vars::Dict; kw...)
        h = new()
        base = Handler(dist, vars; kw...)
        h.dist = base.dist
        h.vars = base.vars
        h.group = base.group
        h.wall_dt = base.wall_dt
        h.sim_dt = base.sim_dt
        h.iter = base.iter
        h.custom_schedule = base.custom_schedule
        h.tasks = base.tasks
        h.last_wall_div = base.last_wall_div
        h.last_sim_div = base.last_sim_div
        h.last_iter_div = base.last_iter_div
        h.fields = Any[]
        return h
    end
end

"""
    build_system!(sh::SystemHandler)

Build field list and set task outputs.
"""
function build_system!(sh::SystemHandler)
    empty!(sh.fields)
    for task in sh.tasks
        op = task["operator"]
        if isa(op, AbstractFuture)
            op.out = build_out(op)
            push!(sh.fields, op.out)
        else
            push!(sh.fields, op)
        end
    end
end

"""
    process(sh::SystemHandler; kw...)

Process system handler -- currently a no-op matching the Python implementation.
"""
function process(sh::SystemHandler; kw...)
    # No-op, matching Python
    return nothing
end

# ============================================================================
# Handler comm helpers (handle both MPI and SerialCommCart transparently)
# ============================================================================

"""
    _handler_comm_rank(comm) -> Int

Return rank for a handler communicator. Returns 0 for SerialCommCart or nothing.
"""
function _handler_comm_rank(comm)
    if comm === nothing || comm isa SerialCommCart
        return 0
    end
    if MPI_ENABLED[]
        mpi = get_mpi()
        return mpi.Comm_rank(comm)
    end
    return 0
end

"""
    _handler_comm_size(comm) -> Int

Return size for a handler communicator. Returns 1 for SerialCommCart or nothing.
"""
function _handler_comm_size(comm)
    if comm === nothing || comm isa SerialCommCart
        return 1
    end
    if MPI_ENABLED[]
        mpi = get_mpi()
        return mpi.Comm_size(comm)
    end
    return 1
end

"""
    _handler_barrier(comm)

MPI barrier on a handler communicator. No-op for SerialCommCart or serial mode.
"""
function _handler_barrier(comm)
    if comm === nothing || comm isa SerialCommCart || !MPI_ENABLED[]
        return nothing
    end
    mpi = get_mpi()
    mpi.Barrier(comm)
    return nothing
end

"""
    _handler_sync(f::Function, comm)

Execute f() with entry+exit barriers on a handler communicator.
No-ops for SerialCommCart or serial mode.
"""
function _handler_sync(f::Function, comm)
    _handler_barrier(comm)
    try
        f()
    finally
        _handler_barrier(comm)
    end
end

# ============================================================================
# H5FileHandlerBase (abstract base for HDF5 handlers)
# ============================================================================

"""
    H5FileHandlerBase <: AbstractHandler

Abstract base for handlers that write task outputs to HDF5 files.

Manages file/set numbering, directory creation, and the HDF5 file
structure (scales group, tasks group, metadata).

The `_parallel_mode` field selects the writing strategy:
- `:gather`  -- root process gathers all data via MPI, then writes (default)
- `:virtual` -- each rank writes its own file, a virtual dataset links them
- `:mpio`    -- all ranks write to a single file via MPI-IO parallel HDF5
"""
mutable struct H5FileHandlerBase <: AbstractHandler
    dist::Any
    vars::Dict
    group::Any
    wall_dt::Union{Nothing,Real}
    sim_dt::Union{Nothing,Real}
    iter::Union{Nothing,Integer}
    custom_schedule::Any
    tasks::Vector{Dict{String,Any}}
    last_wall_div::Int
    last_sim_div::Int
    last_iter_div::Int
    # File handler specific fields
    base_path::String
    name::String
    max_writes::Union{Nothing,Int}
    comm::Any           # Cartesian communicator (MPI or SerialCommCart)
    set_num::Int
    total_write_num::Int
    file_write_num::Int
    _parallel_mode::Symbol  # :gather, :virtual, or :mpio
    _empty::Bool            # true if all tasks have zero local size (used by virtual mode)

    function H5FileHandlerBase(base_path::AbstractString, dist, vars::Dict;
                               max_writes=nothing, mode=nothing,
                               _parallel_mode::Symbol=:gather, kw...)
        h = new()
        base = Handler(dist, vars; kw...)
        h.dist = base.dist
        h.vars = base.vars
        h.group = base.group
        h.wall_dt = base.wall_dt
        h.sim_dt = base.sim_dt
        h.iter = base.iter
        h.custom_schedule = base.custom_schedule
        h.tasks = base.tasks
        h.last_wall_div = base.last_wall_div
        h.last_sim_div = base.last_sim_div
        h.last_iter_div = base.last_iter_div
        h._parallel_mode = _parallel_mode
        h._empty = false

        if mode === nothing
            mode = FILEHANDLER_MODE_DEFAULT
        end

        # Resolve base_path
        bp = abspath(base_path)
        if isfile(bp)
            throw(ArgumentError("base_path should indicate a folder for storing HDF5 files."))
        end
        h.base_path = bp
        h.name = basename(bp)
        h.max_writes = max_writes

        # Resolve mode
        mode = lowercase(mode)
        if mode ∉ ("overwrite", "append")
            throw(ArgumentError("Write mode '$(mode)' not defined."))
        end

        # Use the Cartesian communicator, matching Python's self.comm = self.dist.comm_cart
        h.comm = dist.comm_cart
        _rank = _handler_comm_rank(h.comm)
        _sz = _handler_comm_size(h.comm)

        # Resolve file mode: only rank 0 touches filesystem, then broadcast
        if _rank == 0
            set_num, total_write_num = _resolve_file_mode(bp, h.name, mode)
        else
            set_num = 0
            total_write_num = 0
        end

        # Broadcast from rank 0 to all processes
        if _sz > 1
            mpi = get_mpi()
            set_num = mpi.bcast(set_num, 0, h.comm)
            total_write_num = mpi.bcast(total_write_num, 0, h.comm)
        end

        h.set_num = set_num
        h.total_write_num = total_write_num
        h.file_write_num = 0

        # Create output folder (rank 0 creates, barrier ensures visibility)
        if _sz > 1
            sync(; comm=h.comm) do
                if _rank == 0
                    mkpath(bp)
                end
            end
        else
            mkpath(bp)
        end

        return h
    end
end

"""
    _resolve_file_mode(bp, name, mode) -> (set_num, total_write_num)

Determine the starting set number and total write number based on mode.
"""
function _resolve_file_mode(bp::String, name::String, mode::String)
    set_pattern = Regex("^$(name)_s(\\d+)")
    if isdir(bp)
        entries = readdir(bp)
    else
        entries = String[]
    end

    if mode == "overwrite"
        # Remove existing sets
        for entry in entries
            m = match(set_pattern, entry)
            if m !== nothing
                full = joinpath(bp, entry)
                if isdir(full)
                    rm(full; recursive=true)
                elseif isfile(full)
                    rm(full)
                end
            end
        end
        return (1, 0)
    else  # append
        set_nums = Int[]
        for entry in entries
            m = match(set_pattern, entry)
            if m !== nothing
                push!(set_nums, parse(Int, m.captures[1]))
            end
        end
        if isempty(set_nums)
            return (1, 0)
        end
        max_set = maximum(set_nums)
        # Try to read last write number from existing files
        joined_file = joinpath(bp, "$(name)_s$(max_set).h5")
        p0_file = joinpath(bp, "$(name)_s$(max_set)", "$(name)_s$(max_set)_p0.h5")
        last_write_num = 0
        if isfile(joined_file)
            try
                h5open(joined_file, "r") do testfile
                    if haskey(testfile, "scales/write_number")
                        wn = read(testfile["scales/write_number"])
                        last_write_num = wn[end]
                    end
                end
            catch
                @warn "Cannot determine write num from files. Restarting count."
            end
        elseif isfile(p0_file)
            try
                h5open(p0_file, "r") do testfile
                    if haskey(testfile, "scales/write_number")
                        wn = read(testfile["scales/write_number"])
                        last_write_num = wn[end]
                    end
                end
            catch
                @warn "Cannot determine write num from files. Restarting count."
            end
        else
            @warn "Cannot determine write num from files. Restarting count."
        end
        return (max_set + 1, last_write_num)
    end
end

"""
    current_path(h::H5FileHandlerBase) -> String

Return the path for the current set directory.
"""
function current_path(h::H5FileHandlerBase)
    set_name = "$(h.name)_s$(h.set_num)"
    return joinpath(h.base_path, set_name)
end

"""
    current_file(h::H5FileHandlerBase) -> String

Return the path for the current HDF5 file.
"""
function current_file(h::H5FileHandlerBase)
    return current_path(h) * ".h5"
end

"""
    add_task!(h::H5FileHandlerBase, task; kw...)

Add a task to the file handler. Extends the base `add_task!` to compute
data distribution information (global shape, local start, local shape).
"""
function add_task!(h::H5FileHandlerBase, task; kw...)
    # Call base add_task! via Handler method
    _handler_add_task!(h, task; kw...)
    # Add data distribution information
    td = h.tasks[end]
    global_shape, local_start, local_shape = get_data_distribution(h, td)
    td["global_shape"] = global_shape
    td["local_start"] = local_start
    td["local_shape"] = local_shape
    td["local_size"] = prod(local_shape)
    td["local_slices"] = Tuple(s:s+sz-1 for (s, sz) in zip(local_start, local_shape))
    return nothing
end

"""
    _handler_add_task!(h::AbstractHandler, task; layout="g", name=nothing, scales=nothing)

Internal: the base add_task! logic, callable from H5FileHandlerBase without
dispatch ambiguity.
"""
function _handler_add_task!(h::AbstractHandler, task; layout="g", name=nothing, scales=nothing)
    # Default name
    if name === nothing
        name = string(task)
    end
    # Create operator
    if task isa AbstractString
        op = parse_future_field(task, h.vars, h.dist)
    elseif isa(task, Field)
        op = _field_copy(task)
    else
        op = task
    end
    # Check scales
    if isa(op, LockedField) || isa(op, FutureLockedField)
        if scales === nothing
            scales = op.domain.dealias
        else
            scales = remedy_scales(h.dist, scales)
            if scales != op.domain.dealias
                scales = op.domain.dealias
                @warn "Cannot specify non-dealias scales for LockedFields"
            end
        end
    else
        scales = remedy_scales(h.dist, scales)
    end
    # Build task dictionary
    td = Dict{String,Any}()
    td["operator"] = op
    td["layout"] = get_layout_object(h.dist, layout)
    td["name"] = name
    td["scales"] = scales
    td["dtype"] = op.dtype
    push!(h.tasks, td)
    return nothing
end

"""
    get_data_distribution(h::H5FileHandlerBase, task; rank=nothing)

Determine write parameters (global_shape, local_start, local_shape) for a task.
"""
function get_data_distribution(h::H5FileHandlerBase, task; rank=nothing)
    layout = task["layout"]
    scales = task["scales"]
    domain = task["operator"].domain
    tensorsig = task["operator"].tensorsig
    # Domain shapes
    gs = global_shape(layout, domain, scales)
    ls = local_shape(layout, domain, scales; rank=rank)
    # Local start
    le = local_elements(layout, domain, scales; rank=rank)
    local_start_vec = Int[]
    for (axis, lei) in enumerate(le)
        if length(lei) == 0
            push!(local_start_vec, gs[axis])
        else
            push!(local_start_vec, first(lei))
        end
    end
    local_start_tup = Tuple(local_start_vec)
    # Field shapes with tensor axes
    tensor_shape = Tuple(get_dim(cs) for cs in tensorsig)
    full_global = (tensor_shape..., gs...)
    full_local = (tensor_shape..., ls...)
    full_start = (ntuple(_ -> 0, length(tensor_shape))..., local_start_tup...)
    return full_global, full_start, full_local
end

"""
    setup_file(h::H5FileHandlerBase, file)

Prepare a new HDF5 file with metadata, scales group, and task datasets.
"""
function setup_file(h::H5FileHandlerBase, file)
    # Metadata
    attrs(file)["set_number"] = h.set_num
    attrs(file)["handler_name"] = h.name
    attrs(file)["writes"] = h.file_write_num

    # Scales group
    g_scales = create_group(file, "scales")

    # Constant scale
    write_dataset(g_scales, "constant", [0.0])

    # Time scales (Float64, resizable)
    for sn in ("sim_time", "timestep", "wall_time")
        if h.max_writes !== nothing
            d = create_dataset(g_scales, sn, Float64, ((0,), (h.max_writes,));
                               chunk=(1,))
        else
            d = create_dataset(g_scales, sn, Float64, ((0,), (-1,));
                               chunk=(1,))
        end
    end
    # Integer time scales
    for sn in ("iteration", "write_number")
        if h.max_writes !== nothing
            d = create_dataset(g_scales, sn, Int64, ((0,), (h.max_writes,));
                               chunk=(1,))
        else
            d = create_dataset(g_scales, sn, Int64, ((0,), (-1,));
                               chunk=(1,))
        end
    end

    # Tasks group
    g_tasks = create_group(file, "tasks")
    for task in h.tasks
        op = task["operator"]
        layout = task["layout"]
        scales = task["scales"]
        dset = create_task_dataset(h, file, task)
        # Metadata attributes
        attrs(dset)["grid_space"] = Int.(layout.grid_space)
        attrs(dset)["scales"] = collect(Float64, scales)
        # Spatial scale metadata as attributes
        _rank = length(op.tensorsig)
        for axis in 1:h.dist.dim
            basis = op.domain.full_bases[axis]
            if basis === nothing
                attrs(dset)["scale_name_$(axis)"] = "constant"
            else
                subaxis = axis - get_basis_axis(h.dist, basis) + 1
                if layout.grid_space[axis]
                    sn = get_coord_name(basis, subaxis)
                else
                    sn = "k" * get_coord_name(basis, subaxis)
                end
                attrs(dset)["scale_name_$(axis)"] = sn
            end
        end
    end
end

"""
    create_task_dataset(h::H5FileHandlerBase, file, task)

Create a resizable HDF5 dataset for a task.
"""
function create_task_dataset(h::H5FileHandlerBase, file, task)
    if h._parallel_mode == :mpio
        return _mpio_create_task_dataset(h, file, task)
    end
    # Default: standard resizable dataset (used by gather and virtual joint file)
    g_shape = task["global_shape"]
    shape = (1, g_shape...)
    if h.max_writes !== nothing
        maxshape = (h.max_writes, g_shape...)
    else
        maxshape = (-1, (s for s in g_shape)...)
    end
    jl_dtype = task["dtype"]
    chunk_dims = (1, g_shape...)
    dset = create_dataset(file["tasks"], task["name"], jl_dtype,
                          (shape, maxshape); chunk=chunk_dims)
    return dset
end

"""
    process(h::H5FileHandlerBase; iteration=0, wall_time=0.0, sim_time=0.0, timestep=0.0, kw...)

Save task outputs to HDF5 file.
"""
function process(h::H5FileHandlerBase;
                 iteration=0, wall_time=0.0, sim_time=0.0, timestep=0.0, kw...)
    # Update write counts
    h.total_write_num += 1
    h.file_write_num += 1
    # Move to next set if necessary
    if h.max_writes !== nothing
        if h.file_write_num > h.max_writes
            h.set_num += 1
            h.file_write_num = 1
        end
    end
    # Write file metadata
    file = get_file(h)
    write_file_metadata(h, file;
                        write_number=h.total_write_num,
                        iteration=iteration,
                        wall_time=wall_time,
                        sim_time=sim_time,
                        timestep=timestep)
    # Write tasks
    for task in h.tasks
        out = task["out"]
        change_scales!(out, task["scales"])
        change_layout!(out, task["layout"])
        write_task(h, file, task)
    end
    # Finalize
    close_h5file(h, file)
end

"""
    write_file_metadata(h::H5FileHandlerBase, file; kw...)

Write file metadata and time scale data.
Dispatches based on `_parallel_mode` to handle MPI-aware writing.
"""
function write_file_metadata(h::H5FileHandlerBase, file; kw...)
    if h._parallel_mode == :gather
        _write_file_metadata_gather(h, file; kw...)
    elseif h._parallel_mode == :virtual
        _write_file_metadata_virtual(h, file; kw...)
    elseif h._parallel_mode == :mpio
        _write_file_metadata_base(h, file; kw...)
    else
        error("Unrecognized _parallel_mode: $(h._parallel_mode)")
    end
end

"""
    _write_file_metadata_base(h, file; kw...)

Core metadata writing logic (used by all modes when they have a valid file handle).
"""
function _write_file_metadata_base(h::H5FileHandlerBase, file; kw...)
    attrs(file)["writes"] = h.file_write_num
    for sn in ("sim_time", "wall_time", "timestep", "iteration", "write_number")
        dset = file["scales"][sn]
        HDF5.set_extent_dims(dset, (h.file_write_num,))
        dset[h.file_write_num] = kw[Symbol(sn)]
    end
end

"""
    _is_handler_empty(h::H5FileHandlerBase) -> Bool

Check if all tasks have zero local data on this process.
"""
function _is_handler_empty(h::H5FileHandlerBase)
    return !any(t -> t["local_size"] > 0, h.tasks)
end

# ============================================================================
# H5GatherFileHandler
# ============================================================================

"""
    H5GatherFileHandler

H5FileHandler that gathers global data to write from the root process.
In serial mode (single process), data is written directly.
In parallel mode, all data is gathered to rank 0 via MPI, then rank 0 writes.
"""
struct H5GatherFileHandler end

"""
    H5GatherFileHandler(filename, dist, vars; kw...) -> H5FileHandlerBase

Construct an H5FileHandlerBase configured for gather-mode writing.
"""
function H5GatherFileHandler(filename::AbstractString, dist, vars::Dict; kw...)
    return H5FileHandlerBase(filename, dist, vars; _parallel_mode=:gather, kw...)
end

"""
    _gather_create_current_file(h::H5FileHandlerBase)

Create a new HDF5 file for gather mode. Root process creates and sets up,
all others wait via barrier.
"""
function _gather_create_current_file(h::H5FileHandlerBase)
    fp = current_file(h)
    _handler_sync(h.comm) do
        if _handler_comm_rank(h.comm) == 0
            h5open(fp, "w") do file
                setup_file(h, file)
            end
        end
    end
end

"""
    _gather_open_file(h::H5FileHandlerBase; mode="r+")

Open the current HDF5 file for gather-mode processing.
Only root process opens the file; other processes receive `nothing`.
"""
function _gather_open_file(h::H5FileHandlerBase; mode="r+")
    if _handler_comm_rank(h.comm) == 0
        return h5open(current_file(h), mode)
    end
    return nothing
end

"""
    _gather_close_file(h::H5FileHandlerBase, file)

Close the HDF5 file on root process. No-op on other processes.
"""
function _gather_close_file(h::H5FileHandlerBase, file)
    if _handler_comm_rank(h.comm) == 0
        close(file)
    end
end

"""
    _write_file_metadata_gather(h, file; kw...)

Write file metadata from root process only.
"""
function _write_file_metadata_gather(h::H5FileHandlerBase, file; kw...)
    if _handler_comm_rank(h.comm) == 0
        _write_file_metadata_base(h, file; kw...)
    end
end

"""
    _gather_write_task(h, file, task)

Write task data in gather mode. Data is gathered from all processes to root,
then root writes the global data.
"""
function _gather_write_task(h::H5FileHandlerBase, file, task)
    out = task["out"]
    # gather_data collects all data to root (returns full array on root, nothing elsewhere)
    data = gather_data(out)
    # Write global data from root process
    if _handler_comm_rank(h.comm) == 0
        dset = file["tasks"][task["name"]]
        HDF5.set_extent_dims(dset, (h.file_write_num, size(data)...))
        _write_data_to_dset(dset, data, h.file_write_num)
    end
end

"""
    _write_data_to_dset(dset, data, write_num)

Write data into an HDF5 dataset at the given write index (time slice).
"""
function _write_data_to_dset(dset, data, write_num)
    if ndims(data) == 0
        dset[write_num] = data[]
    elseif ndims(data) == 1
        dset[write_num, :] = data
    elseif ndims(data) == 2
        dset[write_num, :, :] = data
    elseif ndims(data) == 3
        dset[write_num, :, :, :] = data
    else
        idx = (write_num, ntuple(_ -> Colon(), ndims(data))...)
        dset[idx...] = data
    end
end

# ============================================================================
# H5ParallelFileHandler
# ============================================================================

"""
    H5ParallelFileHandler

H5FileHandler using parallel HDF5 writes via MPI-IO.
All processes collectively write to a single shared HDF5 file.
Requires HDF5.jl built with MPI support (parallel HDF5).
"""
struct H5ParallelFileHandler end

"""
    H5ParallelFileHandler(filename, dist, vars; kw...) -> H5FileHandlerBase

Construct an H5FileHandlerBase configured for parallel MPIO writing.
Throws an error if parallel HDF5 is not available.
"""
function H5ParallelFileHandler(filename::AbstractString, dist, vars::Dict; kw...)
    # Check that HDF5 has MPI-IO support
    if !_hdf5_has_parallel()
        error("H5ParallelFileHandler requires HDF5.jl built with MPI/parallel support. " *
              "Use parallel=\"gather\" or parallel=\"virtual\" instead.")
    end
    return H5FileHandlerBase(filename, dist, vars; _parallel_mode=:mpio, kw...)
end

"""
    _hdf5_has_parallel() -> Bool

Check if HDF5.jl was built with parallel (MPI-IO) support.
"""
function _hdf5_has_parallel()
    try
        # HDF5.has_parallel() is available in recent HDF5.jl versions
        if isdefined(HDF5, :has_parallel)
            return HDF5.has_parallel()
        end
        # Fallback: check if h5open accepts a communicator argument
        # by looking for the method signature
        return MPI_ENABLED[]
    catch
        return false
    end
end

"""
    _mpio_create_current_file(h::H5FileHandlerBase)

Create a new HDF5 file for MPIO mode. All processes collectively create
and set up the file using the MPI-IO driver.
"""
function _mpio_create_current_file(h::H5FileHandlerBase)
    fp = current_file(h)
    if _handler_comm_size(h.comm) > 1 && MPI_ENABLED[]
        mpi = get_mpi()
        h5open(fp, "w", h.comm, mpi.Info()) do file
            setup_file(h, file)
        end
    else
        # Fallback for serial mode
        h5open(fp, "w") do file
            setup_file(h, file)
        end
    end
end

"""
    _mpio_create_task_dataset(h, file, task) -> HDF5.Dataset

Create a dataset for a task in MPIO mode. Uses chunk sizes based on local
data shape from rank 0 to optimize parallel I/O.
"""
function _mpio_create_task_dataset(h::H5FileHandlerBase, file, task)
    g_shape = task["global_shape"]
    shape = (1, g_shape...)
    if h.max_writes !== nothing
        maxshape = (h.max_writes, g_shape...)
    else
        maxshape = (-1, (s for s in g_shape)...)
    end
    jl_dtype = task["dtype"]
    # Use local shape for chunk sizes to optimize parallel writes.
    # Broadcast rank 0's local shape as the chunk template.
    l_shape = task["local_shape"]
    if _handler_comm_size(h.comm) > 1 && MPI_ENABLED[]
        mpi = get_mpi()
        l_shape = mpi.bcast(l_shape, 0, h.comm)
    end
    chunk_dims = (1, l_shape...)
    # Ensure chunk dims are valid (all > 0)
    chunk_dims = Tuple(max(c, 1) for c in chunk_dims)
    dset = create_dataset(file["tasks"], task["name"], jl_dtype,
                          (shape, maxshape); chunk=chunk_dims)
    return dset
end

"""
    _mpio_open_file(h::H5FileHandlerBase; mode="r+")

Open the current HDF5 file with the MPI-IO driver.
"""
function _mpio_open_file(h::H5FileHandlerBase; mode="r+")
    if _handler_comm_size(h.comm) > 1 && MPI_ENABLED[]
        mpi = get_mpi()
        return h5open(current_file(h), mode, h.comm, mpi.Info())
    else
        return h5open(current_file(h), mode)
    end
end

"""
    _mpio_close_file(h::H5FileHandlerBase, file)

Close the MPI-IO HDF5 file (all processes).
"""
function _mpio_close_file(h::H5FileHandlerBase, file)
    close(file)
end

"""
    _mpio_write_task(h, file, task)

Write task data in MPIO mode. All processes collectively write their
local portions to the shared dataset using hyperslab selection.
"""
function _mpio_write_task(h::H5FileHandlerBase, file, task)
    out = task["out"]
    dset = file["tasks"][task["name"]]
    # Collectively resize
    HDF5.set_extent_dims(dset, (h.file_write_num, task["global_shape"]...))
    index = h.file_write_num  # 1-based write index

    # Build hyperslab selection for this process's portion
    local_shape = task["local_shape"]
    local_start = task["local_start"]

    if prod(local_shape) > 0
        # Construct index ranges: time dimension + spatial dimensions
        # The time dimension is a single index; spatial dims use ranges
        idx = Any[index]
        for (start, sz) in zip(local_start, local_shape)
            # local_start is 0-based from get_data_distribution, convert to 1-based
            push!(idx, (start + 1):(start + sz))
        end
        dset[idx...] = out.data
    end
    # Note: processes with empty local data simply don't write.
    # For truly collective I/O, we would use low-level HDF5 API with
    # memory/file dataspaces. The high-level indexing approach works
    # for non-collective (independent) I/O mode.
end

# ============================================================================
# H5VirtualFileHandler
# ============================================================================

"""
    H5VirtualFileHandler

H5FileHandler using process files and virtual joint files.
Each process writes its own HDF5 file with local data. A joint file
on rank 0 uses HDF5 virtual datasets to present a unified view
linking to the per-process files.
"""
struct H5VirtualFileHandler end

"""
    H5VirtualFileHandler(filename, dist, vars; kw...) -> H5FileHandlerBase

Construct an H5FileHandlerBase configured for virtual-dataset mode.
"""
function H5VirtualFileHandler(filename::AbstractString, dist, vars::Dict; kw...)
    return H5FileHandlerBase(filename, dist, vars; _parallel_mode=:virtual, kw...)
end

"""
    _virtual_current_process_file(h::H5FileHandlerBase, rank::Int) -> String

Return the path for the process file of the given rank.
"""
function _virtual_current_process_file(h::H5FileHandlerBase, rank::Int)
    file_name = "$(h.name)_s$(h.set_num)_p$(rank).h5"
    return joinpath(current_path(h), file_name)
end

"""
    _virtual_current_process_file(h::H5FileHandlerBase) -> String

Return the path for this process's file.
"""
function _virtual_current_process_file(h::H5FileHandlerBase)
    return _virtual_current_process_file(h, _handler_comm_rank(h.comm))
end

"""
    _virtual_create_current_file(h::H5FileHandlerBase)

Create new files for virtual mode:
1. Root creates the joint file with virtual datasets (via base create_current_file).
2. Root creates the set directory.
3. Each nonempty process creates its own process file.
"""
function _virtual_create_current_file(h::H5FileHandlerBase)
    is_empty = _is_handler_empty(h)
    _rank = _handler_comm_rank(h.comm)

    # Create joint file from root process
    _handler_sync(h.comm) do
        if _rank == 0
            fp = current_file(h)
            h5open(fp, "w") do file
                _virtual_setup_joint_file(h, file)
            end
        end
    end

    # Create set folder from root process
    _handler_sync(h.comm) do
        if _rank == 0
            mkpath(current_path(h))
        end
    end

    # Create process files on nonempty processes
    if !is_empty
        # Touch temp files to update filesystem cache (if configured)
        tmpfile = nothing
        if FILEHANDLER_TOUCH_TMPFILE
            tmpfile = joinpath(current_path(h), "tmpfile_p$(_rank)")
            touch(tmpfile)
        end
        # Create and setup process file
        proc_path = _virtual_current_process_file(h)
        h5open(proc_path, "w") do file
            _virtual_setup_process_file(h, file)
        end
        # Remove temp file
        if FILEHANDLER_TOUCH_TMPFILE && tmpfile !== nothing
            rm(tmpfile; force=true)
        end
    end
end

"""
    _virtual_setup_joint_file(h, file)

Set up the joint virtual dataset file. This calls the standard setup_file
but overrides create_task_dataset to produce virtual datasets.
"""
function _virtual_setup_joint_file(h::H5FileHandlerBase, file)
    # Metadata
    attrs(file)["set_number"] = h.set_num
    attrs(file)["handler_name"] = h.name
    attrs(file)["writes"] = h.file_write_num

    # Scales group
    g_scales = create_group(file, "scales")
    write_dataset(g_scales, "constant", [0.0])

    for sn in ("sim_time", "timestep", "wall_time")
        if h.max_writes !== nothing
            create_dataset(g_scales, sn, Float64, ((0,), (h.max_writes,)); chunk=(1,))
        else
            create_dataset(g_scales, sn, Float64, ((0,), (-1,)); chunk=(1,))
        end
    end
    for sn in ("iteration", "write_number")
        if h.max_writes !== nothing
            create_dataset(g_scales, sn, Int64, ((0,), (h.max_writes,)); chunk=(1,))
        else
            create_dataset(g_scales, sn, Int64, ((0,), (-1,)); chunk=(1,))
        end
    end

    # Tasks group with virtual datasets
    g_tasks = create_group(file, "tasks")
    for task in h.tasks
        op = task["operator"]
        layout = task["layout"]
        scales = task["scales"]
        dset = _virtual_create_task_dataset(h, file, task)
        # Metadata attributes
        attrs(dset)["grid_space"] = Int.(layout.grid_space)
        attrs(dset)["scales"] = collect(Float64, scales)
        _rank = length(op.tensorsig)
        for axis in 1:h.dist.dim
            basis = op.domain.full_bases[axis]
            if basis === nothing
                attrs(dset)["scale_name_$(axis)"] = "constant"
            else
                subaxis = axis - get_basis_axis(h.dist, basis) + 1
                if layout.grid_space[axis]
                    sn = get_coord_name(basis, subaxis)
                else
                    sn = "k" * get_coord_name(basis, subaxis)
                end
                attrs(dset)["scale_name_$(axis)"] = sn
            end
        end
    end
end

"""
    _virtual_create_task_dataset(h, file, task) -> HDF5.Dataset

Create a virtual dataset that maps to per-process files.

Note: HDF5.jl virtual dataset support varies. If VDS API is not available,
falls back to a regular dataset with a warning (data must then be merged
manually post-run).
"""
function _virtual_create_task_dataset(h::H5FileHandlerBase, file, task)
    g_shape = task["global_shape"]
    shape = (1, g_shape...)
    if h.max_writes !== nothing
        maxshape = (h.max_writes, g_shape...)
    else
        maxshape = (-1, (s for s in g_shape)...)
    end
    jl_dtype = task["dtype"]
    comm_sz = _handler_comm_size(h.comm)

    # Try to create a virtual dataset linking per-process files.
    # HDF5.jl virtual dataset support requires the low-level API.
    if _hdf5_has_vds_support() && comm_sz > 1
        dset = _create_vds_dataset(h, file, task, shape, maxshape, jl_dtype, comm_sz)
    else
        # Fallback: create a regular resizable dataset.
        # In serial mode this is exactly what we want.
        # In parallel mode without VDS, data will be in separate process files.
        chunk_dims = (1, g_shape...)
        dset = create_dataset(file["tasks"], task["name"], jl_dtype,
                              (shape, maxshape); chunk=chunk_dims)
    end
    return dset
end

"""
    _hdf5_has_vds_support() -> Bool

Check if HDF5.jl supports the Virtual Dataset (VDS) API.
"""
function _hdf5_has_vds_support()
    # Check for the necessary low-level API functions
    return isdefined(HDF5, :VirtualLayout) ||
           isdefined(HDF5.API, :h5p_set_virtual)
end

"""
    _create_vds_dataset(h, file, task, shape, maxshape, dtype, comm_sz)

Create a virtual dataset using HDF5 low-level API that maps to per-process
files. Each process file contains the local portion of the data.
"""
function _create_vds_dataset(h::H5FileHandlerBase, file, task, shape, maxshape, dtype, comm_sz)
    # Use the low-level HDF5 API for virtual dataset creation
    dset_name = task["name"]
    g_shape = task["global_shape"]

    # Create the dataset creation property list
    dcpl = HDF5.API.h5p_create(HDF5.API.H5P_DATASET_CREATE)
    try
        # Set up virtual mappings for each nonempty process
        for rank in 0:(comm_sz - 1)
            rank_global_shape, rank_local_start, rank_local_shape = get_data_distribution(h, task; rank=rank)
            if prod(rank_local_shape) == 0
                continue
            end

            # Source file path (relative to base_path)
            proc_filename = "$(h.name)_s$(h.set_num)_p$(rank).h5"
            src_path = joinpath("$(h.name)_s$(h.set_num)", proc_filename)
            src_dset_name = "tasks/$(dset_name)"

            # Source shape = (unlimited, local_shape...)
            src_shape = (1, rank_local_shape...)
            src_maxshape = maxshape[1:1]
            for ls in rank_local_shape
                src_maxshape = (src_maxshape..., ls)
            end

            # Virtual (file) space: hyperslab in the joint dataset
            # Offset: (0, local_start...), count: (UNLIMITED, local_shape...)
            vspace = HDF5.API.h5s_create_simple(collect(Int, reverse(shape)), collect(Int, reverse(maxshape)))
            file_start = (0, rank_local_start...)
            file_count = (1, rank_local_shape...)
            # HDF5 C API uses reversed dimensions (row-major <-> column-major)
            HDF5.API.h5s_select_hyperslab(vspace, HDF5.API.H5S_SELECT_SET,
                collect(UInt64, reverse(file_start)),
                C_NULL,
                collect(UInt64, reverse(file_count)),
                C_NULL)

            # Source space
            src_space = HDF5.API.h5s_create_simple(collect(Int, reverse(src_shape)), collect(Int, reverse(src_maxshape)))
            src_start = ntuple(_ -> UInt64(0), length(src_shape))
            src_count = Tuple(UInt64(s) for s in src_shape)
            HDF5.API.h5s_select_hyperslab(src_space, HDF5.API.H5S_SELECT_SET,
                collect(UInt64, reverse(src_start)),
                C_NULL,
                collect(UInt64, reverse(src_count)),
                C_NULL)

            # Add virtual mapping
            HDF5.API.h5p_set_virtual(dcpl, vspace, src_path, src_dset_name, src_space)

            HDF5.API.h5s_close(vspace)
            HDF5.API.h5s_close(src_space)
        end

        # Create the virtual dataset
        hdf5_dtype = HDF5.datatype(dtype)
        total_space = HDF5.API.h5s_create_simple(collect(Int, reverse(shape)), collect(Int, reverse(maxshape)))
        dset_id = HDF5.API.h5d_create(file["tasks"], dset_name,
                                       hdf5_dtype, total_space, HDF5.API.H5P_DEFAULT,
                                       dcpl, HDF5.API.H5P_DEFAULT)
        HDF5.API.h5s_close(total_space)
        return HDF5.Dataset(dset_id, file["tasks"])
    catch e
        # VDS creation failed; fall back to regular dataset
        @warn "Virtual dataset creation failed: $e. Falling back to regular dataset."
        g_shape = task["global_shape"]
        chunk_dims = (1, g_shape...)
        dset = create_dataset(file["tasks"], task["name"], dtype,
                              (shape, maxshape); chunk=chunk_dims)
        return dset
    finally
        HDF5.API.h5p_close(dcpl)
    end
end

"""
    _virtual_setup_process_file(h, file)

Set up a per-process HDF5 file with local task datasets.
"""
function _virtual_setup_process_file(h::H5FileHandlerBase, file)
    g_tasks = create_group(file, "tasks")
    for task in h.tasks
        if task["local_size"] > 0
            l_shape = task["local_shape"]
            shape = (1, l_shape...)
            if h.max_writes !== nothing
                maxshape = (h.max_writes, l_shape...)
            else
                maxshape = (-1, (s for s in l_shape)...)
            end
            jl_dtype = task["dtype"]
            chunk_dims = (1, l_shape...)
            dset = create_dataset(g_tasks, task["name"], jl_dtype,
                                  (shape, maxshape); chunk=chunk_dims)
            # Save write attributes for merging
            layout = task["layout"]
            attrs(dset)["ext_mesh"] = collect(Int, layout.ext_mesh)
            attrs(dset)["ext_coords"] = collect(Int, layout.ext_coords)
            attrs(dset)["global_shape"] = collect(Int, task["global_shape"])
            attrs(dset)["local_start"] = collect(Int, task["local_start"])
            attrs(dset)["local_shape"] = collect(Int, task["local_shape"])
        end
    end
end

"""
    _virtual_open_file(h::H5FileHandlerBase; mode="r+")

Open files for virtual mode. Returns a tuple (joint_file, proc_file).
Joint file opened on root only; process file opened on nonempty processes.
"""
function _virtual_open_file(h::H5FileHandlerBase; mode="r+")
    is_empty = _is_handler_empty(h)
    _rank = _handler_comm_rank(h.comm)

    # Open joint file on root
    if _rank == 0
        joint_file = h5open(current_file(h), mode)
    else
        joint_file = nothing
    end

    # Open process file on nonempty processes
    if !is_empty
        proc_file = h5open(_virtual_current_process_file(h), mode)
    else
        proc_file = nothing
    end

    return (joint_file, proc_file)
end

"""
    _virtual_close_file(h::H5FileHandlerBase, file)

Close virtual mode files (joint on root, process on nonempty processes).
"""
function _virtual_close_file(h::H5FileHandlerBase, file)
    joint_file, proc_file = file
    if _handler_comm_rank(h.comm) == 0 && joint_file !== nothing
        close(joint_file)
    end
    if proc_file !== nothing
        close(proc_file)
    end
end

"""
    _write_file_metadata_virtual(h, file; kw...)

Write file metadata for virtual mode. Only root writes to the joint file.
"""
function _write_file_metadata_virtual(h::H5FileHandlerBase, file; kw...)
    joint_file, proc_file = file
    if _handler_comm_rank(h.comm) == 0 && joint_file !== nothing
        _write_file_metadata_base(h, joint_file; kw...)
    end
end

"""
    _virtual_write_task(h, file, task)

Write task data for virtual mode. Each nonempty process writes its local
data to its own process file.
"""
function _virtual_write_task(h::H5FileHandlerBase, file, task)
    joint_file, proc_file = file
    out = task["out"]
    # Write local data to process file from nonempty processes
    if task["local_size"] > 0 && proc_file !== nothing
        dset = proc_file["tasks"][task["name"]]
        HDF5.set_extent_dims(dset, (h.file_write_num, task["local_shape"]...))
        _write_data_to_dset(dset, out.data, h.file_write_num)
    end
end

"""
    merge_virtual_task(file, task_name; overwrite=false)

Merge a virtual dataset into a regular dataset. This reads the virtual
dataset and copies all data into a new contiguous dataset.

# Arguments
- `file`: HDF5 file handle (the joint file)
- `task_name`: name of the task dataset under "tasks/"
- `overwrite`: if true, replace the virtual dataset with the merged one
"""
function merge_virtual_task(file, task_name::AbstractString; overwrite::Bool=false)
    old_dset = file["tasks"][task_name]
    old_shape = size(old_dset)
    old_dtype = eltype(old_dset)

    new_name = "$(task_name)_merged"
    # Create new dataset with shape[1]=1 for initial chunking, then resize
    new_shape = (1, old_shape[2:end]...)
    maxshape = (-1, old_shape[2:end]...)
    new_dset = create_dataset(file["tasks"], new_name, old_dtype,
                              (new_shape, maxshape); chunk=new_shape)
    HDF5.set_extent_dims(new_dset, old_shape)

    # Copy data chunk by chunk (one time slice at a time)
    for t in 1:old_shape[1]
        if length(old_shape) == 1
            new_dset[t] = old_dset[t]
        elseif length(old_shape) == 2
            new_dset[t, :] = old_dset[t, :]
        elseif length(old_shape) == 3
            new_dset[t, :, :] = old_dset[t, :, :]
        else
            idx = (t, ntuple(_ -> Colon(), length(old_shape) - 1)...)
            new_dset[idx...] = old_dset[idx...]
        end
    end

    # Overwrite old dataset if requested
    if overwrite
        HDF5.delete_object(file["tasks"], task_name)
        # HDF5 doesn't support rename directly; we'd need low-level API
        # For now, the merged dataset is available as task_name_merged
        try
            HDF5.API.h5l_move(file["tasks"], new_name, file["tasks"], task_name,
                              HDF5.API.H5P_DEFAULT, HDF5.API.H5P_DEFAULT)
        catch
            @warn "Could not rename merged dataset. Available as '$(new_name)'."
        end
    end
end

# ============================================================================
# Dispatch: get_file, create_current_file, open_h5file, close_h5file, write_task
# ============================================================================

"""
    get_file(h::H5FileHandlerBase; kw...)

Return the current HDF5 file(s), creating if necessary.
Dispatches based on _parallel_mode.
"""
function get_file(h::H5FileHandlerBase; kw...)
    if h._parallel_mode == :virtual
        # For virtual mode, check both the joint file and process file existence
        fp = current_file(h)
        _rank = _handler_comm_rank(h.comm)
        need_create = false
        if _rank == 0
            need_create = !isfile(fp)
        end
        # Broadcast the need_create flag
        if _handler_comm_size(h.comm) > 1
            mpi = get_mpi()
            need_create = mpi.bcast(need_create, 0, h.comm)
        end
        if need_create
            _virtual_create_current_file(h)
        end
        return _virtual_open_file(h; kw...)
    elseif h._parallel_mode == :mpio
        fp = current_file(h)
        # All processes need to agree on whether to create
        _rank = _handler_comm_rank(h.comm)
        need_create = false
        if _rank == 0
            need_create = !isfile(fp)
        end
        if _handler_comm_size(h.comm) > 1
            mpi = get_mpi()
            need_create = mpi.bcast(need_create, 0, h.comm)
        end
        if need_create
            _mpio_create_current_file(h)
        end
        return _mpio_open_file(h; kw...)
    else  # :gather
        fp = current_file(h)
        _rank = _handler_comm_rank(h.comm)
        need_create = false
        if _rank == 0
            need_create = !isfile(fp)
        end
        if _handler_comm_size(h.comm) > 1
            mpi = get_mpi()
            need_create = mpi.bcast(need_create, 0, h.comm)
        end
        if need_create
            _gather_create_current_file(h)
        end
        return _gather_open_file(h; kw...)
    end
end

"""
    open_h5file(h::H5FileHandlerBase; mode="r+")

Open the current HDF5 file. Dispatches based on _parallel_mode.
"""
function open_h5file(h::H5FileHandlerBase; mode="r+")
    if h._parallel_mode == :gather
        return _gather_open_file(h; mode=mode)
    elseif h._parallel_mode == :virtual
        return _virtual_open_file(h; mode=mode)
    elseif h._parallel_mode == :mpio
        return _mpio_open_file(h; mode=mode)
    else
        error("Unrecognized _parallel_mode: $(h._parallel_mode)")
    end
end

"""
    close_h5file(h::H5FileHandlerBase, file)

Close the current HDF5 file. Dispatches based on _parallel_mode.
"""
function close_h5file(h::H5FileHandlerBase, file)
    if h._parallel_mode == :gather
        _gather_close_file(h, file)
    elseif h._parallel_mode == :virtual
        _virtual_close_file(h, file)
    elseif h._parallel_mode == :mpio
        _mpio_close_file(h, file)
    else
        error("Unrecognized _parallel_mode: $(h._parallel_mode)")
    end
end

"""
    write_task(h::H5FileHandlerBase, file, task)

Write task data. Dispatches based on _parallel_mode.
"""
function write_task(h::H5FileHandlerBase, file, task)
    if h._parallel_mode == :gather
        _gather_write_task(h, file, task)
    elseif h._parallel_mode == :virtual
        _virtual_write_task(h, file, task)
    elseif h._parallel_mode == :mpio
        _mpio_write_task(h, file, task)
    else
        error("Unrecognized _parallel_mode: $(h._parallel_mode)")
    end
end

"""
    create_current_file(h::H5FileHandlerBase)

Generate and set up a new HDF5 file. Dispatches based on _parallel_mode.
"""
function create_current_file(h::H5FileHandlerBase)
    if h._parallel_mode == :gather
        _gather_create_current_file(h)
    elseif h._parallel_mode == :virtual
        _virtual_create_current_file(h)
    elseif h._parallel_mode == :mpio
        _mpio_create_current_file(h)
    else
        error("Unrecognized _parallel_mode: $(h._parallel_mode)")
    end
end

# ============================================================================
# Helper: parse_future_field
# ============================================================================

"""
    parse_future_field(expr_str, vars, dist)

Parse a string expression into a FutureField operator using the given
namespace dictionary. Falls back to `FutureField.parse` if available,
otherwise evaluates in a constructed namespace.
"""
function parse_future_field(expr_str::AbstractString, vars::Dict, dist)
    # Try to use FutureField's parse method if available
    if isdefined(@__MODULE__, :FutureField) && hasmethod(parse, Tuple{Type{FutureField}, AbstractString, Dict, Any})
        return parse(FutureField, expr_str, vars, dist)
    end
    # Fallback: evaluate the expression string in the vars namespace
    # This is a simplified version; full implementation requires the
    # expression parser from the operators module
    mod = Module()
    for (k, v) in vars
        Core.eval(mod, :($(Symbol(k)) = $v))
    end
    return Core.eval(mod, Meta.parse(expr_str))
end

# ============================================================================
# Helper: get_coord_name
# ============================================================================

"""
    get_coord_name(basis, subaxis) -> String

Get the coordinate name for a given basis and sub-axis index.
"""
function get_coord_name(basis, subaxis)
    cs = basis.coordsys
    coords = get_coords(cs)
    if subaxis isa Integer && 1 <= subaxis <= length(coords)
        return string(coords[subaxis].name)
    else
        return "x$(subaxis)"
    end
end

# ============================================================================
# Helper: get_basis_axis (forward reference)
# ============================================================================

"""
    get_basis_axis(dist, basis) -> Int

Get the starting axis index for a basis within the distributor.
Delegates to the distributor's method.
"""
function get_basis_axis(dist, basis)
    if hasmethod(Main.get_basis_axis, Tuple{typeof(dist), typeof(basis)})
        return Main.get_basis_axis(dist, basis)
    end
    # Fallback: search for the basis in the distributor's coordinate systems
    axis = 1
    for cs in dist.coordsystems
        for coord in get_coords(cs)
            # Check if this coordinate's basis matches
            if hasfield(typeof(basis), :coordsys) && basis.coordsys === cs
                return axis
            end
            axis += 1
        end
    end
    return 1  # default
end

const add_system_handler! = add_system_handler
const evaluate_group! = evaluate_group
const FileHandler = H5FileHandlerBase

export Evaluator,
       FileHandler,
       H5FileHandlerBase,
       H5GatherFileHandler,
       H5ParallelFileHandler,
       H5VirtualFileHandler,
       AbstractHandler,
       merge_virtual_task


# Evaluator-specific dispatches for require_coeff/grid_space!
require_coeff_space!(ev::Evaluator, fields)::Nothing = require_coeff_space(ev, fields)
require_grid_space!(ev::Evaluator, fields)::Nothing = require_grid_space(ev, fields)
