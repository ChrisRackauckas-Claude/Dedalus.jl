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
    _mpi_module

Reference to the MPI module itself, set by `init_mpi!()`.
"""
const _mpi_module = Ref{Module}(Main)

"""
    init_mpi!(; comm=nothing)

Initialise MPI support. Must be called after `using MPI` in the main module.
If `comm` is not provided, uses `MPI.COMM_WORLD`.

```julia
using MPI
MPI.Init()
Dedalus.init_mpi!()
```
"""
function init_mpi!(; comm=nothing)
    mpi_mod = nothing
    try
        mpi_mod = Base.require(Main, :MPI)
    catch
        @warn "MPI.jl could not be loaded; staying in serial mode."
        return nothing
    end
    _mpi_module[] = mpi_mod
    # Ensure MPI is initialized
    if !mpi_mod.Initialized()
        mpi_mod.Init()
    end
    if comm === nothing
        comm = mpi_mod.COMM_WORLD
    end
    _mpi_comm[] = comm
    MPI_ENABLED[] = true
    return comm
end

"""
    get_mpi()

Return the MPI module. Throws if MPI is not enabled.
"""
function get_mpi()
    if !MPI_ENABLED[]
        error("MPI is not initialised. Call init_mpi!() first.")
    end
    return _mpi_module[]
end

"""
    mpi_rank(; comm=nothing) -> Int

Return the MPI rank of the current process. Returns `0` in serial mode.
"""
function mpi_rank(; comm=nothing)
    MPI_ENABLED[] || return 0
    c = comm === nothing ? _mpi_comm[] : comm
    mpi = _mpi_module[]
    return mpi.Comm_rank(c)
end

"""
    mpi_size(; comm=nothing) -> Int

Return the total number of MPI processes. Returns `1` in serial mode.
"""
function mpi_size(; comm=nothing)
    MPI_ENABLED[] || return 1
    c = comm === nothing ? _mpi_comm[] : comm
    mpi = _mpi_module[]
    return mpi.Comm_size(c)
end

"""
    mpi_barrier(; comm=nothing)

Execute an MPI barrier. A no-op in serial mode.
"""
function mpi_barrier(; comm=nothing)
    MPI_ENABLED[] || return nothing
    c = comm === nothing ? _mpi_comm[] : comm
    mpi = _mpi_module[]
    mpi.Barrier(c)
    return nothing
end

# ---------------------------------------------------------------------------
# Cart communicator helpers
# ---------------------------------------------------------------------------

"""
    create_cart_comm(comm, dims; periods=nothing, reorder=false)

Create a Cartesian communicator. Returns `(comm_cart, coords)`.
In serial mode, returns `(nothing, Int[])`.

`dims` is a vector of process counts per dimension (only entries > 1 are used).
`periods` defaults to all non-periodic.
"""
function create_cart_comm(comm, dims; periods=nothing, reorder::Bool=false)
    if !MPI_ENABLED[]
        return (nothing, Int[])
    end
    mpi = _mpi_module[]
    ndims = length(dims)
    if periods === nothing
        periods = zeros(Bool, ndims)
    end
    comm_cart = mpi.Cart_create(comm, dims; periodic=periods, reorder=reorder)
    coords = mpi.Cart_coords(comm_cart)
    return (comm_cart, coords)
end

"""
    cart_sub(comm_cart, remain_dims)

Create a sub-communicator from a Cartesian communicator by selecting
dimensions where `remain_dims[i] == true`.

In serial mode, returns `nothing`.
"""
function cart_sub(comm_cart, remain_dims)
    if !MPI_ENABLED[] || comm_cart === nothing
        return nothing
    end
    mpi = _mpi_module[]
    return mpi.Cart_sub(comm_cart, remain_dims)
end

"""
    cart_coords_for_rank(comm_cart, rank) -> Vector{Int}

Return the Cartesian coordinates for a given rank in `comm_cart`.
In serial mode, returns an empty vector.
"""
function cart_coords_for_rank(comm_cart, rank::Integer)
    if !MPI_ENABLED[] || comm_cart === nothing
        return Int[]
    end
    mpi = _mpi_module[]
    return mpi.Cart_coords(comm_cart, rank)
end

"""
    comm_size(comm) -> Int

Return the size of a communicator. In serial mode returns 1.
"""
function comm_size(comm)
    if !MPI_ENABLED[] || comm === nothing
        return 1
    end
    mpi = _mpi_module[]
    return mpi.Comm_size(comm)
end

"""
    comm_rank(comm) -> Int

Return the rank within a communicator. In serial mode returns 0.
"""
function comm_rank(comm)
    if !MPI_ENABLED[] || comm === nothing
        return 0
    end
    mpi = _mpi_module[]
    return mpi.Comm_rank(comm)
end

# ---------------------------------------------------------------------------
# Alltoallv wrapper
# ---------------------------------------------------------------------------

"""
    alltoallv!(sendbuf, sendcounts, sdispls,
               recvbuf, recvcounts, rdispls, comm)

Perform MPI_Alltoallv. All counts and displacements are in number of elements,
not bytes. In serial mode, copies sendbuf to recvbuf.
"""
function alltoallv!(sendbuf::AbstractVector, sendcounts::AbstractVector{<:Integer},
                    sdispls::AbstractVector{<:Integer},
                    recvbuf::AbstractVector, recvcounts::AbstractVector{<:Integer},
                    rdispls::AbstractVector{<:Integer}, comm)
    if !MPI_ENABLED[] || comm === nothing
        # Serial fallback: direct copy of the relevant segment
        copyto!(recvbuf, 1, sendbuf, 1, min(length(sendbuf), length(recvbuf)))
        return nothing
    end
    mpi = _mpi_module[]
    # MPI.jl's VBuffer for Alltoallv
    vbuf_send = mpi.VBuffer(sendbuf, sendcounts, sdispls)
    vbuf_recv = mpi.VBuffer(recvbuf, recvcounts, rdispls)
    mpi.Alltoallv!(vbuf_send, vbuf_recv, comm)
    return nothing
end

# ---------------------------------------------------------------------------
# Allgatherv wrapper
# ---------------------------------------------------------------------------

"""
    allgatherv!(sendbuf, sendcount::Integer,
                recvbuf, recvcounts, rdispls, comm)

Perform MPI_Allgatherv. In serial mode, copies sendbuf into recvbuf.
"""
function allgatherv!(sendbuf::AbstractVector, sendcount::Integer,
                     recvbuf::AbstractVector, recvcounts::AbstractVector{<:Integer},
                     rdispls::AbstractVector{<:Integer}, comm)
    if !MPI_ENABLED[] || comm === nothing
        copyto!(recvbuf, 1, sendbuf, 1, sendcount)
        return nothing
    end
    mpi = _mpi_module[]
    vbuf_recv = mpi.VBuffer(recvbuf, recvcounts, rdispls)
    mpi.Allgatherv!(mpi.Buffer(sendbuf, sendcount), vbuf_recv, comm)
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
       parallel_mkdir,
       init_mpi!,
       get_mpi,
       mpi_rank,
       mpi_size,
       mpi_barrier,
       create_cart_comm,
       cart_sub,
       cart_coords_for_rank,
       comm_size,
       comm_rank,
       alltoallv!,
       allgatherv!,
       MPI_ENABLED
