"""
    Evaluator and output handler types for Dedalus.jl

Julia translation of `dedalus/core/evaluator.py`. Manages centralized
evaluation of operator expression trees and directs the results to various
output handlers (dictionary, system, HDF5 file).

## Type hierarchy

    AbstractHandler
    +-- DictionaryHandler  (stores outputs in a Dict)
    +-- SystemHandler       (collects fields for a FieldSystem)
    +-- H5FileHandlerBase  (HDF5 file output -- abstract)
        +-- H5GatherFileHandler  (root gathers and writes)
        +-- H5ParallelFileHandler (stub -- parallel HDF5)
        +-- H5VirtualFileHandler  (stub -- virtual datasets)

    Evaluator  (coordinates evaluation and dispatches to handlers)

## Key translation choices

- Python `h5py`            -> Julia `HDF5.jl` (`h5open`, `create_dataset`, etc.)
- Python `defaultdict`     -> Julia `Dict{String,Vector}` with `get!`
- Python `uuid.uuid4()`    -> Julia `UUIDs.uuid4()`
- Python `shutil.rmtree`   -> Julia `rm(...; recursive=true)`
- Python `pathlib.Path`    -> Julia stdlib `Base.Filesystem` paths (strings)
- Python `collections.OrderedSet` -> local `OrderedSet` from tools/general.jl
- Python `Sync` context    -> `sync` do-block from tools/parallel.jl
- Python class inheritance -> Julia abstract types + composition
- Python `prod` from math  -> Julia `prod`
- HDF5 dimension scales    -> omitted (HDF5.jl support is limited; metadata
  is written as attributes instead)
- Virtual datasets / MPIO  -> stubbed for milestone 3
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
    vars::Dict
    handlers::Vector{Any}
    groups::Dict{Any,Vector{Any}}

    function Evaluator(dist, vars::Dict)
        return new(dist, vars, Any[], Dict{Any,Vector{Any}}())
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
- `"virtual"` -- virtual datasets (stub)
- `"mpio"` -- parallel HDF5 (stub)
"""
function add_file_handler(ev::Evaluator, filename; parallel=nothing, kw...)
    if parallel === nothing
        if ev.dist.comm.size == 1
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
function evaluate_group(ev::Evaluator, group; kw...)
    handlers = get(ev.groups, group, Any[])
    evaluate_handlers(ev, handlers; kw...)
end

"""
    evaluate_scheduled(ev::Evaluator; kw...)

Evaluate all handlers whose schedules indicate they are due.
"""
function evaluate_scheduled(ev::Evaluator; kw...)
    handlers = [h for h in ev.handlers if check_schedule(h; kw...)]
    evaluate_handlers(ev, handlers; kw...)
end

"""Alias: callers in solvers.jl and timesteppers.jl use the bang name."""
const evaluate_scheduled! = evaluate_scheduled

"""
    evaluate_handlers(ev::Evaluator, handlers; id=nothing, kw...)

Evaluate a collection of handlers: attempt all tasks, transform fields
through layouts until every task is resolved, then process outputs.
"""
function evaluate_handlers(ev::Evaluator, handlers; id=nothing, kw...)
    # Default to uuid to cache within evaluation but not across evaluations
    if id === nothing
        id = uuid4()
    end

    # Collect tasks and clear previous outputs
    tasks = [t for h in handlers for t in h.tasks]
    for task in tasks
        task["out"] = nothing
    end

    # Attempt initial evaluation
    tasks = attempt_tasks(tasks; id=id)

    # Move all fields to coefficient layout
    fields = get_task_fields(tasks)
    require_coeff_space(ev, fields)
    tasks = attempt_tasks(tasks; id=id)

    # Oscillate through layouts until all tasks are evaluated
    # Limit to 10 passes to break on potential infinite loops
    n_layouts = length(ev.dist.layouts)
    osc = oscillate(0:(n_layouts - 1); max_passes=10)
    osc_state = iterate(osc)
    if osc_state === nothing
        return
    end
    current_index, osc_st = osc_state

    while !isempty(tasks)
        next_result = iterate(osc, osc_st)
        if next_result === nothing
            break
        end
        next_index, osc_st = next_result
        # Transform fields
        fields = get_task_fields(tasks)
        if current_index < next_index
            path = ev.dist.paths[current_index + 1]  # 1-based indexing
            increment(path, fields)
        else
            path = ev.dist.paths[next_index + 1]      # 1-based indexing
            decrement(path, fields)
        end
        current_index = next_index
        # Attempt evaluation
        tasks = attempt_tasks(tasks; id=id)
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
function require_coeff_space(ev::Evaluator, fields)
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
            push!(fields, atom)
        end
    end
    # Drop locked fields
    locked = [f for f in fields if isa(f, LockedField)]
    for f in locked
        delete!(fields, f)
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
# H5FileHandlerBase (abstract base for HDF5 handlers)
# ============================================================================

"""
    H5FileHandlerBase <: AbstractHandler

Abstract base for handlers that write task outputs to HDF5 files.

Manages file/set numbering, directory creation, and the HDF5 file
structure (scales group, tasks group, metadata).
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
    comm::Any
    set_num::Int
    total_write_num::Int
    file_write_num::Int

    function H5FileHandlerBase(base_path::AbstractString, dist, vars::Dict;
                               max_writes=nothing, mode=nothing, kw...)
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

        h.comm = dist.comm

        # Serial mode: rank 0 logic always applies
        set_num, total_write_num = _resolve_file_mode(bp, h.name, mode)

        h.set_num = set_num
        h.total_write_num = total_write_num
        h.file_write_num = 0

        # Create output folder
        mkpath(bp)

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
    get_file(h::H5FileHandlerBase; kw...)

Return the current HDF5 file, creating it if necessary.
"""
function get_file(h::H5FileHandlerBase; kw...)
    fp = current_file(h)
    if !isfile(fp)
        create_current_file(h)
    end
    return open_h5file(h; kw...)
end

"""
    create_current_file(h::H5FileHandlerBase)

Generate and set up a new HDF5 file from the root process.
"""
function create_current_file(h::H5FileHandlerBase)
    # In serial mode, always root
    fp = current_file(h)
    h5open(fp, "w") do file
        setup_file(h, file)
    end
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
                subaxis = axis - get_basis_axis(h.dist, basis)
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
    g_shape = task["global_shape"]
    shape = (1, g_shape...)
    if h.max_writes !== nothing
        maxshape = (h.max_writes, g_shape...)
    else
        maxshape = (-1, (s for s in g_shape)...)
    end
    # Determine Julia dtype
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
"""
function write_file_metadata(h::H5FileHandlerBase, file; kw...)
    # Update file metadata
    attrs(file)["writes"] = h.file_write_num
    # Update time scales
    for sn in ("sim_time", "wall_time", "timestep", "iteration", "write_number")
        dset = file["scales"][sn]
        HDF5.set_extent_dims(dset, (h.file_write_num,))
        dset[h.file_write_num] = kw[Symbol(sn)]
    end
end

# open_h5file, close_h5file, and write_task for H5FileHandlerBase are
# defined below with the H5GatherFileHandler implementation (serial mode).

# ============================================================================
# H5GatherFileHandler
# ============================================================================

"""
    H5GatherFileHandler

H5FileHandler that gathers global data to write from the root process.
In serial mode (single process), this simply writes data directly.

This is the primary file handler for milestone 1 (serial mode).
"""
struct H5GatherFileHandler end

"""
    H5GatherFileHandler(filename, dist, vars; kw...) -> H5FileHandlerBase

Construct an H5FileHandlerBase configured for gather-mode writing.
The returned object is an H5FileHandlerBase with gather-specific methods
dispatched via the `_gather_handler` tag.
"""
function H5GatherFileHandler(filename::AbstractString, dist, vars::Dict; kw...)
    h = H5FileHandlerBase(filename, dist, vars; kw...)
    # Tag this handler as gather mode
    h.tasks  # ensure initialized
    return h
end

# For serial mode, H5GatherFileHandler methods operate on H5FileHandlerBase
# since it IS the concrete type. We make the methods work for H5FileHandlerBase
# directly, as serial gather is the primary (and currently only) mode.

"""
    open_h5file(h::H5FileHandlerBase; mode="r+")

Open the current HDF5 file for gather-mode processing (root process only).
"""
function open_h5file(h::H5FileHandlerBase; mode="r+")
    # In serial mode, always root process
    return h5open(current_file(h), mode)
end

"""
    close_h5file(h::H5FileHandlerBase, file)

Close the current HDF5 file after gather-mode processing.
"""
function close_h5file(h::H5FileHandlerBase, file)
    close(file)
end

"""
    write_task(h::H5FileHandlerBase, file, task)

Write task data in gather mode. In serial mode, data is written directly
since all data is local.
"""
function write_task(h::H5FileHandlerBase, file, task)
    out = task["out"]
    data = gather_data(out)
    # Write global data
    dset = file["tasks"][task["name"]]
    HDF5.set_extent_dims(dset, (h.file_write_num, size(data)...))
    # Write into the last time slice
    if ndims(data) == 0
        dset[h.file_write_num] = data[]
    elseif ndims(data) == 1
        dset[h.file_write_num, :] = data
    elseif ndims(data) == 2
        dset[h.file_write_num, :, :] = data
    elseif ndims(data) == 3
        dset[h.file_write_num, :, :, :] = data
    else
        # General N-dimensional case
        idx = (h.file_write_num, ntuple(_ -> Colon(), ndims(data))...)
        dset[idx...] = data
    end
end

# ============================================================================
# H5ParallelFileHandler (stub)
# ============================================================================

"""
    H5ParallelFileHandler

Stub for parallel HDF5 file handler using MPIO driver.
Will be fully implemented in milestone 3.
"""
function H5ParallelFileHandler(filename::AbstractString, dist, vars::Dict; kw...)
    error("H5ParallelFileHandler is not yet implemented. " *
          "Use parallel=\"gather\" for serial mode.")
end

# ============================================================================
# H5VirtualFileHandler (stub)
# ============================================================================

"""
    H5VirtualFileHandler

Stub for virtual-dataset HDF5 file handler.
Will be fully implemented in milestone 3.
"""
function H5VirtualFileHandler(filename::AbstractString, dist, vars::Dict; kw...)
    error("H5VirtualFileHandler is not yet implemented. " *
          "Use parallel=\"gather\" for serial mode.")
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




# Evaluator-specific dispatches for require_coeff/grid_space!
require_coeff_space!(ev::Evaluator, fields) = require_coeff_space(ev, fields)
require_grid_space!(ev::Evaluator, fields) = require_grid_space(ev, fields)
