"""
Parallel execution utilities for the Dedalus framework.

Translated from dedalus/tools/parallel.py. Provides synchronisation primitives
and parallel-safe filesystem helpers. When MPI.jl is not loaded the module
operates in serial mode where all barriers are no-ops, rank is 0, and size is 1.
"""

# ---------------------------------------------------------------------------
# MPI availability flag and helpers
# ---------------------------------------------------------------------------

"""
    MPI_ENABLED

`Ref{Bool}` indicating whether MPI.jl has been loaded and initialised.
Defaults to `false` (serial mode).
"""
const MPI_ENABLED = Ref(false)

"""
    _mpi_comm

Reference to the MPI communicator object. Only meaningful when `MPI_ENABLED[]`
is `true`.
"""
const _mpi_comm = Ref{Any}(nothing)

"""
    mpi_rank(; comm=nothing) -> Int

Return the MPI rank of the current process. Returns `0` in serial mode.
"""
function mpi_rank(; comm=nothing)
    MPI_ENABLED[] || return 0
    c = comm === nothing ? _mpi_comm[] : comm
    return c.rank
end

"""
    mpi_size(; comm=nothing) -> Int

Return the total number of MPI processes. Returns `1` in serial mode.
"""
function mpi_size(; comm=nothing)
    MPI_ENABLED[] || return 1
    c = comm === nothing ? _mpi_comm[] : comm
    return c.size
end

"""
    mpi_barrier(; comm=nothing)

Execute an MPI barrier. A no-op in serial mode.
"""
function mpi_barrier(; comm=nothing)
    MPI_ENABLED[] || return nothing
    c = comm === nothing ? _mpi_comm[] : comm
    # Assumes MPI.jl's Barrier function
    Base.invokelatest(Main.MPI.Barrier, c)
    return nothing
end

# ---------------------------------------------------------------------------
# sync  (Python: Sync context manager)
# ---------------------------------------------------------------------------

"""
    sync(f::Function; comm=nothing, enter_barrier=true, exit_barrier=true)

Execute `f()` with optional MPI barriers on entry and exit. The barriers are
no-ops when MPI is not available.

This is the Julia equivalent of the Python `Sync` context manager; the
do-block syntax provides the same scoping:

```julia
sync(; enter_barrier=false, exit_barrier=true) do
    # ... work that needs post-synchronisation ...
end
```
"""
function sync(f::Function; comm=nothing, enter_barrier::Bool=true, exit_barrier::Bool=true)
    if enter_barrier
        mpi_barrier(; comm=comm)
    end
    try
        f()
    finally
        if exit_barrier
            mpi_barrier(; comm=comm)
        end
    end
end

# ---------------------------------------------------------------------------
# rotate_processes  (Python: RotateProcesses context manager)
# ---------------------------------------------------------------------------

"""
    rotate_processes(f::Function; comm=nothing)

Execute `f()` in a rotating fashion across MPI processes. Each rank waits for
all lower-ranked processes to finish before executing, and all higher-ranked
processes wait after. In serial mode, `f()` is simply called.

```julia
rotate_processes() do
    println("rank ", mpi_rank(), " executing")
end
```
"""
function rotate_processes(f::Function; comm=nothing)
    rank = mpi_rank(; comm=comm)
    size = mpi_size(; comm=comm)
    # Wait for all lower-ranked processes
    for _ in 1:rank
        mpi_barrier(; comm=comm)
    end
    try
        f()
    finally
        # Wait for all higher-ranked processes
        for _ in 1:(size - rank)
            mpi_barrier(; comm=comm)
        end
    end
end

# ---------------------------------------------------------------------------
# parallel_mkdir
# ---------------------------------------------------------------------------

"""
    parallel_mkdir(path; comm=nothing)

Create a directory from the root process (rank 0) only, with an exit barrier
to ensure the directory exists on all processes afterwards.

In serial mode, simply creates the directory if it does not exist.
"""
function parallel_mkdir(path; comm=nothing)
    sync(; comm=comm, enter_barrier=false, exit_barrier=true) do
        if mpi_rank(; comm=comm) == 0
            if !isdir(path)
                mkpath(path)
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export sync,
       rotate_processes,
       parallel_mkdir
