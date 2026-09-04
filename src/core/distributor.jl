"""
    Distributor, Layout, Transform, and Transpose types for Dedalus.jl

Julia translation of `dedalus/core/distributor.py`. Manages data distribution
layouts (coefficient space vs grid space) and coordinates basis transforms.

## Type hierarchy

    AbstractDistributor (defined in domain.jl)
    `-- Distributor

    Layout
    Transform
    Transpose

    AbstractTransposePlanner
    `-- AlltoallvTranspose
        `-- ColDistributor
        `-- RowDistributor

## Key translation choices

- Python's MPI communicator -> `nothing` for serial mode; MPI.jl communicators
  for parallel mode.
- Python 0-based layout indices -> kept 0-based conceptually, but stored in
  1-based Julia arrays (layout at conceptual index `i` lives at `layouts[i+1]`).
- Python `np.array` of bools -> Julia `Vector{Bool}` / `NTuple{N, Bool}`.
- Python `WeakSet` -> Julia `WeakKeyDict` (keys are fields, values are `nothing`).
- Python `@CachedAttribute` / `@CachedMethod` -> accessor functions with
  Dict-based memoization.
- Python `isinstance` checks -> Julia multiple dispatch / `isa`.
- Python `prod` from `math` -> Julia `prod`.
- Python `numbers.Number` -> Julia `Number`.
- Julia arrays are column-major; buffer packing loops account for this.

## Serial-mode simplifications

In serial (single-process) mode:
- `comm` is `nothing` -- no real MPI needed.
- `mesh` is `[1]` -- all data is local.
- All layouts have `local_flags = (true, true, ...)` -- everything is local.
- Transposes never occur (no distributed axes).
- Transforms are just basis forward/backward transform calls.
"""

using LinearAlgebra
using OrderedCollections: OrderedDict

# ============================================================================
# Serial Communicator (placeholder for MPI)
# ============================================================================

"""
    SerialComm

Placeholder for an MPI communicator in serial (single-process) mode.
Provides `size = 1` and `coords = ()`.
"""
struct SerialComm
    size::Int
end

SerialComm() = SerialComm(1)

"""
    SerialCommCart

Placeholder for a Cartesian MPI communicator in serial mode.
"""
struct SerialCommCart
    dims::Vector{Int}
    coords::Vector{Int}
end

SerialCommCart() = SerialCommCart(Int[], Int[])

Base.getproperty(c::SerialCommCart, s::Symbol) = begin
    if s === :dim
        return length(getfield(c, :dims))
    else
        return getfield(c, s)
    end
end

# ============================================================================
# Abstract transpose planner type
# ============================================================================

"""
    AbstractTransposePlanner

Abstract supertype for MPI transpose planners. Subtypes implement
`localize_rows(planner, CL, RL)` and `localize_columns(planner, RL, CL)`.
"""
abstract type AbstractTransposePlanner end

# ============================================================================
# AlltoallvTranspose
# ============================================================================

"""
    AlltoallvTranspose <: AbstractTransposePlanner

MPI Alltoallv-based distributed array transpose, for redistributing a
block-distributed multidimensional array across adjacent axes.

Translates `AlltoallvTranspose` from `dedalus/core/transposes.pyx`.

# Fields
- `comm_sub` -- sub-communicator for the transpose axis.
- `datasize::Int` -- number of doubles per element (1 for Float64, 2 for ComplexF64).
- `axis::Int` -- column axis of transposition plan (1-based).
- `N0::Int`, `N1::Int`, `N2::Int`, `N3::Int` -- reduced 4D global shape.
- `global_shape::Vector{Int32}` -- global array shape.
- `col_starts::Vector{Int32}`, `col_ends::Vector{Int32}` -- per-rank column start/end.
- `row_starts::Vector{Int32}`, `row_ends::Vector{Int32}` -- per-rank row start/end.
- `col_counts::Vector{Int32}`, `row_counts::Vector{Int32}` -- per-rank column/row counts.
- `CL_reduced_shape::Vector{Int32}`, `RL_reduced_shape::Vector{Int32}` -- local reduced shapes.
- `CL_displs::Vector{Int32}`, `RL_displs::Vector{Int32}` -- Alltoallv displacements.
- `CL_counts::Vector{Int32}`, `RL_counts::Vector{Int32}` -- Alltoallv counts.
- `CL_buffer::Vector{Float64}`, `RL_buffer::Vector{Float64}` -- communication buffers.
- `local_col_count::Int`, `local_row_count::Int` -- local counts.
"""
mutable struct AlltoallvTranspose <: AbstractTransposePlanner
    comm_sub::Any        # MPI sub-communicator
    datasize::Int
    axis::Int            # 1-based
    N0::Int
    N1::Int
    N2::Int
    N3::Int
    global_shape::Vector{Int32}
    col_starts::Vector{Int32}
    col_ends::Vector{Int32}
    row_starts::Vector{Int32}
    row_ends::Vector{Int32}
    col_counts::Vector{Int32}
    row_counts::Vector{Int32}
    CL_reduced_shape::Vector{Int32}
    RL_reduced_shape::Vector{Int32}
    CL_displs::Vector{Int32}
    RL_displs::Vector{Int32}
    CL_counts::Vector{Int32}
    RL_counts::Vector{Int32}
    CL_buffer::Vector{Float64}
    RL_buffer::Vector{Float64}
    local_col_count::Int
    local_row_count::Int
end

"""
    AlltoallvTranspose(global_shape, dtype, axis, comm_sub)

Construct an Alltoallv transpose planner.

# Arguments
- `global_shape` -- global array shape (vector or tuple of integers).
- `dtype` -- data type (`Float64` or `ComplexF64`).
- `axis` -- column axis (1-based in Julia; Python's 0-based axis is shifted).
- `comm_sub` -- MPI sub-communicator for this transpose axis.
"""
function AlltoallvTranspose(global_shape_in, dtype, axis::Integer, comm_sub)
    gs = Int32.(collect(global_shape_in))
    datasize = dtype == ComplexF64 ? 2 : 1

    # Get communicator size and rank
    nprocs = comm_size(comm_sub)
    myrank = comm_rank(comm_sub)

    # Reduced global shape (4D array):
    # Dimensions before the transpose axis, the two transpose axes, and after.
    # axis is 1-based: axes 1..axis-1 are "before", axis and axis+1 are transposed,
    # axis+2..end are "after".
    N0 = axis > 1 ? prod(gs[1:axis-1]) : 1
    N1 = gs[axis]
    N2 = gs[axis+1]
    N3_raw = axis + 2 <= length(gs) ? prod(gs[axis+2:end]) : 1
    N3 = N3_raw * datasize

    # Block sizes (ceiling division)
    B1 = cld(Int(N1), nprocs)
    B2 = cld(Int(N2), nprocs)

    # Per-rank start/end/count arrays
    ranks = Int32.(collect(0:nprocs-1))

    col_starts = Int32.(min.(B2 .* ranks, Int(N2)))
    row_starts = Int32.(min.(B1 .* ranks, Int(N1)))
    col_ends   = Int32.(min.(B2 .* (ranks .+ 1), Int(N2)))
    row_ends   = Int32.(min.(B1 .* (ranks .+ 1), Int(N1)))

    col_counts = col_ends .- col_starts
    row_counts = row_ends .- row_starts

    local_col_count = Int(col_counts[myrank + 1])
    local_row_count = Int(row_counts[myrank + 1])

    # Local reduced shapes
    CL_reduced_shape = Int32[N0, N1, local_col_count, N3]
    RL_reduced_shape = Int32[N0, local_row_count, N2, N3]

    # Alltoallv displacements and counts (in doubles, matching Python)
    CL_displs = Int32.((Int(N0) * local_col_count * Int(N3)) .* row_starts)
    RL_displs = Int32.((Int(N0) * local_row_count * Int(N3)) .* col_starts)

    CL_counts = Int32.((Int(N0) * local_col_count * Int(N3)) .* row_counts)
    RL_counts = Int32.((Int(N0) * local_row_count * Int(N3)) .* col_counts)

    # Communication buffers
    CL_size = Int(N0) * Int(N1) * local_col_count * Int(N3)
    RL_size = Int(N0) * local_row_count * Int(N2) * Int(N3)
    CL_buffer = zeros(Float64, max(CL_size, 1))
    RL_buffer = zeros(Float64, max(RL_size, 1))

    return AlltoallvTranspose(
        comm_sub, datasize, Int(axis),
        Int(N0), Int(N1), Int(N2), Int(N3),
        gs,
        col_starts, col_ends, row_starts, row_ends,
        col_counts, row_counts,
        CL_reduced_shape, RL_reduced_shape,
        CL_displs, RL_displs,
        CL_counts, RL_counts,
        CL_buffer, RL_buffer,
        local_col_count, local_row_count
    )
end

# ---------------------------------------------------------------------------
# AlltoallvTranspose buffer packing/unpacking
# ---------------------------------------------------------------------------

"""
    split_rows!(plan::AlltoallvTranspose, A, B)

Reorder column-local dataset `A` (4D view: [N0, N1, local_col_count, N3])
into the sending buffer `B` (flat), grouped by destination process (row blocks).
"""
function split_rows!(plan::AlltoallvTranspose, A::AbstractArray{Float64, 4}, B::Vector{Float64})
    N0 = plan.N0
    col_count = plan.local_col_count
    N3 = plan.N3
    nprocs = length(plan.row_starts)
    i = 1
    @inbounds for proc in 1:nprocs
        row_start = Int(plan.row_starts[proc]) + 1  # 1-based
        row_end   = Int(plan.row_ends[proc])         # inclusive
        for n0 in 1:N0
            for n1 in row_start:row_end
                for n2 in 1:col_count
                    for n3 in 1:N3
                        B[i] = A[n0, n1, n2, n3]
                        i += 1
                    end
                end
            end
        end
    end
end

"""
    combine_rows!(plan::AlltoallvTranspose, B, A)

Reorder receiving buffer `B` (flat) into column-local dataset `A`
(4D view: [N0, N1, local_col_count, N3]), grouped by source process (row blocks).
"""
function combine_rows!(plan::AlltoallvTranspose, B::Vector{Float64}, A::AbstractArray{Float64, 4})
    N0 = plan.N0
    col_count = plan.local_col_count
    N3 = plan.N3
    nprocs = length(plan.row_starts)
    i = 1
    @inbounds for proc in 1:nprocs
        row_start = Int(plan.row_starts[proc]) + 1
        row_end   = Int(plan.row_ends[proc])
        for n0 in 1:N0
            for n1 in row_start:row_end
                for n2 in 1:col_count
                    for n3 in 1:N3
                        A[n0, n1, n2, n3] = B[i]
                        i += 1
                    end
                end
            end
        end
    end
end

"""
    split_columns!(plan::AlltoallvTranspose, A, B)

Reorder row-local dataset `A` (4D view: [N0, local_row_count, N2, N3])
into the sending buffer `B` (flat), grouped by destination process (column blocks).
"""
function split_columns!(plan::AlltoallvTranspose, A::AbstractArray{Float64, 4}, B::Vector{Float64})
    N0 = plan.N0
    row_count = plan.local_row_count
    N3 = plan.N3
    nprocs = length(plan.col_starts)
    i = 1
    @inbounds for proc in 1:nprocs
        col_start = Int(plan.col_starts[proc]) + 1  # 1-based
        col_end   = Int(plan.col_ends[proc])         # inclusive
        for n0 in 1:N0
            for n1 in 1:row_count
                for n2 in col_start:col_end
                    for n3 in 1:N3
                        B[i] = A[n0, n1, n2, n3]
                        i += 1
                    end
                end
            end
        end
    end
end

"""
    combine_columns!(plan::AlltoallvTranspose, B, A)

Reorder receiving buffer `B` (flat) into row-local dataset `A`
(4D view: [N0, local_row_count, N2, N3]), grouped by source process (column blocks).
"""
function combine_columns!(plan::AlltoallvTranspose, B::Vector{Float64}, A::AbstractArray{Float64, 4})
    N0 = plan.N0
    row_count = plan.local_row_count
    N3 = plan.N3
    nprocs = length(plan.col_starts)
    i = 1
    @inbounds for proc in 1:nprocs
        col_start = Int(plan.col_starts[proc]) + 1
        col_end   = Int(plan.col_ends[proc])
        for n0 in 1:N0
            for n1 in 1:row_count
                for n2 in col_start:col_end
                    for n3 in 1:N3
                        A[n0, n1, n2, n3] = B[i]
                        i += 1
                    end
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------
# AlltoallvTranspose localize_rows / localize_columns
# ---------------------------------------------------------------------------

"""
    _make_reduced_view(data, shape::Vector{Int32})

Reshape `data` (treated as a flat Float64 buffer via `reinterpret`) into
a 4D array view of the given shape. If `data` is a complex array, it is
first reinterpreted as Float64 (doubling the last dimension).
"""
function _make_reduced_view(data::AbstractArray, shape::Vector{Int32})
    s = Tuple(Int.(shape))
    if eltype(data) <: Complex
        flat = reinterpret(Float64, vec(data))
    else
        flat = reinterpret(Float64, vec(data))
    end
    return reshape(flat, s)
end

"""
    localize_rows(plan::AlltoallvTranspose, CL, RL)

Transpose from column-local to row-local data distribution.
`CL` and `RL` are the source and destination arrays (arbitrary-dimensional;
will be reshaped internally to the plan's 4D reduced shapes).
"""
function localize_rows(plan::AlltoallvTranspose, CL::AbstractArray, RL::AbstractArray)
    CL_reduced = _make_reduced_view(CL, plan.CL_reduced_shape)
    RL_reduced = _make_reduced_view(RL, plan.RL_reduced_shape)

    # Pack column-local data into send buffer, split by row blocks
    if plan.local_col_count > 0
        split_rows!(plan, CL_reduced, plan.CL_buffer)
    end

    # MPI Alltoallv
    alltoallv!(plan.CL_buffer, Vector{Int}(plan.CL_counts), Vector{Int}(plan.CL_displs),
               plan.RL_buffer, Vector{Int}(plan.RL_counts), Vector{Int}(plan.RL_displs),
               plan.comm_sub)

    # Unpack receiving buffer into row-local dataset
    if plan.local_row_count > 0
        combine_columns!(plan, plan.RL_buffer, RL_reduced)
    end
end

"""
    localize_columns(plan::AlltoallvTranspose, RL, CL)

Transpose from row-local to column-local data distribution.
`RL` and `CL` are the source and destination arrays (arbitrary-dimensional;
will be reshaped internally to the plan's 4D reduced shapes).
"""
function localize_columns(plan::AlltoallvTranspose, RL::AbstractArray, CL::AbstractArray)
    CL_reduced = _make_reduced_view(CL, plan.CL_reduced_shape)
    RL_reduced = _make_reduced_view(RL, plan.RL_reduced_shape)

    # Pack row-local data into send buffer, split by column blocks
    if plan.local_row_count > 0
        split_columns!(plan, RL_reduced, plan.RL_buffer)
    end

    # MPI Alltoallv
    alltoallv!(plan.RL_buffer, Vector{Int}(plan.RL_counts), Vector{Int}(plan.RL_displs),
               plan.CL_buffer, Vector{Int}(plan.CL_counts), Vector{Int}(plan.CL_displs),
               plan.comm_sub)

    # Unpack receiving buffer into column-local dataset
    if plan.local_col_count > 0
        combine_rows!(plan, plan.CL_buffer, CL_reduced)
    end
end

# ============================================================================
# ColDistributor (1D column distribution via Allgatherv/Scatterv)
# ============================================================================

"""
    ColDistributor <: AbstractTransposePlanner

MPI Allgatherv-based distributed array duplication and condensation for
column distribution. Translates `ColDistributor` from transposes.pyx.

In localize_rows: restricts to local rows (simple slice, no communication).
In localize_columns: gathers data across all ranks via Allgatherv.
"""
mutable struct ColDistributor <: AbstractTransposePlanner
    comm_sub::Any
    datasize::Int
    axis::Int
    N0::Int
    N1::Int
    N2::Int
    N3::Int
    global_shape::Vector{Int32}
    col_starts::Vector{Int32}
    col_ends::Vector{Int32}
    row_starts::Vector{Int32}
    row_ends::Vector{Int32}
    col_counts::Vector{Int32}
    row_counts::Vector{Int32}
    CL_reduced_shape::Vector{Int32}
    RL_reduced_shape::Vector{Int32}
    CL_displs::Vector{Int32}
    RL_displs::Vector{Int32}
    CL_counts::Vector{Int32}
    RL_counts::Vector{Int32}
    CL_buffer::Vector{Float64}
    RL_buffer::Vector{Float64}
    local_col_count::Int
    local_row_count::Int
    local_row_start::Int
    local_row_end::Int
    local_RL_count::Int
end

"""
    ColDistributor(global_shape, dtype, axis, comm_sub)

Construct a column distributor.

Unlike `AlltoallvTranspose`, columns span the full N2 dimension on every rank
(B2 = N2). Rows are block-distributed across ranks.
"""
function ColDistributor(global_shape_in, dtype, axis::Integer, comm_sub)
    gs = Int32.(collect(global_shape_in))
    datasize = dtype == ComplexF64 ? 2 : 1

    nprocs = comm_size(comm_sub)
    myrank = comm_rank(comm_sub)

    N0 = axis > 1 ? prod(gs[1:axis-1]) : 1
    N1 = gs[axis]
    N2 = gs[axis+1]
    N3_raw = axis + 2 <= length(gs) ? prod(gs[axis+2:end]) : 1
    N3 = N3_raw * datasize

    # Blocks: rows are block-distributed, columns span full N2
    B1 = cld(Int(N1), nprocs)
    B2 = Int(N2)

    ranks = Int32.(collect(0:nprocs-1))

    col_starts = Int32.(0 .* ranks)                       # always 0
    row_starts = Int32.(min.(B1 .* ranks, Int(N1)))
    col_ends   = Int32.(0 .* ranks .+ B2)                 # always B2
    row_ends   = Int32.(min.(B1 .* (ranks .+ 1), Int(N1)))

    local_row_start = Int(row_starts[myrank + 1])
    local_row_end   = Int(row_ends[myrank + 1])

    col_counts = col_ends .- col_starts
    row_counts = row_ends .- row_starts

    local_row_count = Int(row_counts[myrank + 1])
    local_col_count = Int(col_counts[myrank + 1])

    CL_reduced_shape = Int32[N0, N1, local_col_count, N3]
    RL_reduced_shape = Int32[N0, local_row_count, N2, N3]

    CL_displs = Int32.((Int(N0) * local_col_count * Int(N3)) .* row_starts)
    RL_displs = Int32.((Int(N0) * local_row_count * Int(N3)) .* col_starts)

    CL_counts = Int32.((Int(N0) * local_col_count * Int(N3)) .* row_counts)
    RL_counts = Int32.((Int(N0) * local_row_count * Int(N3)) .* col_counts)

    local_RL_count = Int(RL_counts[myrank + 1])

    CL_size = Int(N0) * Int(N1) * local_col_count * Int(N3)
    RL_size = Int(N0) * local_row_count * Int(N2) * Int(N3)
    CL_buffer = zeros(Float64, max(CL_size, 1))
    RL_buffer = zeros(Float64, max(RL_size, 1))

    return ColDistributor(
        comm_sub, datasize, Int(axis),
        Int(N0), Int(N1), Int(N2), Int(N3),
        gs,
        col_starts, col_ends, row_starts, row_ends,
        col_counts, row_counts,
        CL_reduced_shape, RL_reduced_shape,
        CL_displs, RL_displs,
        CL_counts, RL_counts,
        CL_buffer, RL_buffer,
        local_col_count, local_row_count,
        local_row_start, local_row_end,
        local_RL_count
    )
end

"""
    localize_rows(plan::ColDistributor, CL, RL)

Column-local to row-local: simply restricts to local rows (no communication).
"""
function localize_rows(plan::ColDistributor, CL::AbstractArray, RL::AbstractArray)
    CL_reduced = _make_reduced_view(CL, plan.CL_reduced_shape)
    RL_reduced = _make_reduced_view(RL, plan.RL_reduced_shape)
    # Restrict to local rows (1-based slicing)
    start = plan.local_row_start + 1
    stop  = plan.local_row_end
    @views copyto!(RL_reduced, CL_reduced[:, start:stop, :, :])
end

"""
    localize_columns(plan::ColDistributor, RL, CL)

Row-local to column-local: gathers data across all ranks via Allgatherv,
then unpacks with `combine_rows!`.
"""
function localize_columns(plan::ColDistributor, RL::AbstractArray, CL::AbstractArray)
    CL_reduced = _make_reduced_view(CL, plan.CL_reduced_shape)
    RL_reduced = _make_reduced_view(RL, plan.RL_reduced_shape)

    # Copy from input array to RL buffer
    flat_rl = vec(RL_reduced)
    copyto!(plan.RL_buffer, 1, flat_rl, 1, min(length(flat_rl), length(plan.RL_buffer)))

    # Allgatherv
    allgatherv!(plan.RL_buffer, plan.local_RL_count,
                plan.CL_buffer, Vector{Int}(plan.CL_counts), Vector{Int}(plan.CL_displs),
                plan.comm_sub)

    # Unpack buffer into column-local dataset
    combine_rows!(plan, plan.CL_buffer, CL_reduced)
end

# Share the packing functions from AlltoallvTranspose:
# combine_rows! and split_columns! work on the same 4D layout.
# We define a dispatcher so the ColDistributor can use them.
function combine_rows!(plan::ColDistributor, B::Vector{Float64}, A::AbstractArray{Float64, 4})
    N0 = plan.N0
    col_count = plan.local_col_count
    N3 = plan.N3
    nprocs = length(plan.row_starts)
    i = 1
    @inbounds for proc in 1:nprocs
        row_start = Int(plan.row_starts[proc]) + 1
        row_end   = Int(plan.row_ends[proc])
        for n0 in 1:N0
            for n1 in row_start:row_end
                for n2 in 1:col_count
                    for n3 in 1:N3
                        A[n0, n1, n2, n3] = B[i]
                        i += 1
                    end
                end
            end
        end
    end
end

# ============================================================================
# RowDistributor (1D row distribution via Allgatherv/Scatterv)
# ============================================================================

"""
    RowDistributor <: AbstractTransposePlanner

MPI Allgatherv-based distributed array duplication and condensation for
row distribution. Translates `RowDistributor` from transposes.pyx.

In localize_rows: gathers data across all ranks via Allgatherv.
In localize_columns: restricts to local columns (simple slice, no communication).
"""
mutable struct RowDistributor <: AbstractTransposePlanner
    comm_sub::Any
    datasize::Int
    axis::Int
    N0::Int
    N1::Int
    N2::Int
    N3::Int
    global_shape::Vector{Int32}
    col_starts::Vector{Int32}
    col_ends::Vector{Int32}
    row_starts::Vector{Int32}
    row_ends::Vector{Int32}
    col_counts::Vector{Int32}
    row_counts::Vector{Int32}
    CL_reduced_shape::Vector{Int32}
    RL_reduced_shape::Vector{Int32}
    CL_displs::Vector{Int32}
    RL_displs::Vector{Int32}
    CL_counts::Vector{Int32}
    RL_counts::Vector{Int32}
    CL_buffer::Vector{Float64}
    RL_buffer::Vector{Float64}
    local_col_count::Int
    local_row_count::Int
    local_col_start::Int
    local_col_end::Int
    local_CL_count::Int
end

"""
    RowDistributor(global_shape, dtype, axis, comm_sub)

Construct a row distributor.

Unlike `AlltoallvTranspose`, rows span the full N1 dimension on every rank
(B1 = N1). Columns are block-distributed across ranks.
"""
function RowDistributor(global_shape_in, dtype, axis::Integer, comm_sub)
    gs = Int32.(collect(global_shape_in))
    datasize = dtype == ComplexF64 ? 2 : 1

    nprocs = comm_size(comm_sub)
    myrank = comm_rank(comm_sub)

    N0 = axis > 1 ? prod(gs[1:axis-1]) : 1
    N1 = gs[axis]
    N2 = gs[axis+1]
    N3_raw = axis + 2 <= length(gs) ? prod(gs[axis+2:end]) : 1
    N3 = N3_raw * datasize

    # Blocks: rows span full N1, columns are block-distributed
    B1 = Int(N1)
    B2 = cld(Int(N2), nprocs)

    ranks = Int32.(collect(0:nprocs-1))

    col_starts = Int32.(min.(B2 .* ranks, Int(N2)))
    row_starts = Int32.(0 .* ranks)                       # always 0
    col_ends   = Int32.(min.(B2 .* (ranks .+ 1), Int(N2)))
    row_ends   = Int32.(0 .* ranks .+ B1)                 # always B1

    local_col_start = Int(col_starts[myrank + 1])
    local_col_end   = Int(col_ends[myrank + 1])

    col_counts = col_ends .- col_starts
    row_counts = row_ends .- row_starts

    local_col_count = Int(col_counts[myrank + 1])
    local_row_count = Int(row_counts[myrank + 1])

    CL_reduced_shape = Int32[N0, N1, local_col_count, N3]
    RL_reduced_shape = Int32[N0, local_row_count, N2, N3]

    CL_displs = Int32.((Int(N0) * local_col_count * Int(N3)) .* row_starts)
    RL_displs = Int32.((Int(N0) * local_row_count * Int(N3)) .* col_starts)

    CL_counts = Int32.((Int(N0) * local_col_count * Int(N3)) .* row_counts)
    RL_counts = Int32.((Int(N0) * local_row_count * Int(N3)) .* col_counts)

    local_CL_count = Int(CL_counts[myrank + 1])

    CL_size = Int(N0) * Int(N1) * local_col_count * Int(N3)
    RL_size = Int(N0) * local_row_count * Int(N2) * Int(N3)
    CL_buffer = zeros(Float64, max(CL_size, 1))
    RL_buffer = zeros(Float64, max(RL_size, 1))

    return RowDistributor(
        comm_sub, datasize, Int(axis),
        Int(N0), Int(N1), Int(N2), Int(N3),
        gs,
        col_starts, col_ends, row_starts, row_ends,
        col_counts, row_counts,
        CL_reduced_shape, RL_reduced_shape,
        CL_displs, RL_displs,
        CL_counts, RL_counts,
        CL_buffer, RL_buffer,
        local_col_count, local_row_count,
        local_col_start, local_col_end,
        local_CL_count
    )
end

"""
    localize_rows(plan::RowDistributor, CL, RL)

Column-local to row-local: gathers data across all ranks via Allgatherv,
then unpacks with `combine_columns!`.
"""
function localize_rows(plan::RowDistributor, CL::AbstractArray, RL::AbstractArray)
    CL_reduced = _make_reduced_view(CL, plan.CL_reduced_shape)
    RL_reduced = _make_reduced_view(RL, plan.RL_reduced_shape)

    # Copy from input array to CL buffer
    flat_cl = vec(CL_reduced)
    copyto!(plan.CL_buffer, 1, flat_cl, 1, min(length(flat_cl), length(plan.CL_buffer)))

    # Allgatherv
    allgatherv!(plan.CL_buffer, plan.local_CL_count,
                plan.RL_buffer, Vector{Int}(plan.RL_counts), Vector{Int}(plan.RL_displs),
                plan.comm_sub)

    # Unpack buffer into row-local dataset
    combine_columns!(plan, plan.RL_buffer, RL_reduced)
end

function combine_columns!(plan::RowDistributor, B::Vector{Float64}, A::AbstractArray{Float64, 4})
    N0 = plan.N0
    row_count = plan.local_row_count
    N3 = plan.N3
    nprocs = length(plan.col_starts)
    i = 1
    @inbounds for proc in 1:nprocs
        col_start = Int(plan.col_starts[proc]) + 1
        col_end   = Int(plan.col_ends[proc])
        for n0 in 1:N0
            for n1 in 1:row_count
                for n2 in col_start:col_end
                    for n3 in 1:N3
                        A[n0, n1, n2, n3] = B[i]
                        i += 1
                    end
                end
            end
        end
    end
end

"""
    localize_columns(plan::RowDistributor, RL, CL)

Row-local to column-local: simply restricts to local columns (no communication).
"""
function localize_columns(plan::RowDistributor, RL::AbstractArray, CL::AbstractArray)
    CL_reduced = _make_reduced_view(CL, plan.CL_reduced_shape)
    RL_reduced = _make_reduced_view(RL, plan.RL_reduced_shape)
    # Restrict to local columns (1-based slicing)
    start = plan.local_col_start + 1
    stop  = plan.local_col_end
    @views copyto!(CL_reduced, RL_reduced[:, :, start:stop, :])
end

# ============================================================================
# Distributor
# ============================================================================

"""
    Distributor <: AbstractDistributor

Directs parallelized distribution and transformation of fields.

Supports both serial mode (single process, `comm=nothing`) and MPI-parallel
mode (with a real MPI communicator).

# Fields
- `coordsystems::Tuple` -- coordinate systems managed by this distributor.
- `coords::Tuple` -- flattened tuple of all coordinates.
- `dim::Int` -- total dimensionality.
- `dtype` -- default data type (e.g. `Float64`, `ComplexF64`).
- `comm` -- MPI communicator or `nothing` for serial mode.
- `comm_cart` -- Cartesian communicator or `SerialCommCart`.
- `comm_coords::Vector{Int}` -- coordinates in the Cartesian communicator.
- `mesh::Vector{Int}` -- process mesh.
- `single_coordsys` -- the single coordinate system if only one, else `false`.
- `layouts::Vector{Any}` -- available Layout objects.
- `paths::Vector{Any}` -- Path objects connecting adjacent layouts.
- `transforms::Vector{Any}` -- Transform objects (one per axis).
- `coeff_layout` -- coefficient-space layout (first layout).
- `grid_layout` -- grid-space layout (last layout).
- `layout_references::Dict{String, Any}` -- string references to layouts.
- `fields` -- weak set of field references.
- `_cs_by_axis::Union{Nothing, Dict{Int, Any}}` -- cached coord-system-by-axis map.
- `_default_nonconst_groups::Union{Nothing, Tuple}` -- cached default non-const groups.

# Constructor

    Distributor(coordsystems, dtype; mesh=nothing, comm=nothing)

- `coordsystems` -- a single `AbstractCoordinateSystem` or a tuple/vector of them.
- `dtype` -- numeric element type.
- `mesh` -- process mesh (default: serial, i.e. empty).
- `comm` -- MPI communicator (default: `nothing` for serial).

# Examples
```julia
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords, Float64)
```
"""
mutable struct Distributor <: AbstractDistributor
    coordsystems::Tuple
    coords::Tuple
    dim::Int
    dtype::Any
    comm::Any
    comm_cart::Any
    comm_coords::Vector{Int}
    mesh::Vector{Int}
    single_coordsys::Any  # AbstractCoordinateSystem or false
    layouts::Vector{Any}
    paths::Vector{Any}
    transforms::Vector{Any}
    coeff_layout::Any     # Layout (forward ref)
    grid_layout::Any      # Layout (forward ref)
    layout_references::Dict{String, Any}
    fields::WeakKeyDict{Any, Nothing}
    _cs_by_axis::Union{Nothing, Dict{Int, Any}}
    _default_nonconst_groups::Union{Nothing, Tuple}

    function Distributor(coordsystems, dtype; mesh=nothing, comm=nothing)
        # Accept single coordsys in place of tuple/list
        if !(coordsystems isa Tuple || coordsystems isa AbstractVector)
            coordsystems = (coordsystems,)
        else
            coordsystems = Tuple(coordsystems)
        end

        # Note if only a single coordsys for simplicity
        single_cs = length(coordsystems) == 1 ? coordsystems[1] : false

        # Get coords: flatten all coordinate system coords into a single tuple
        all_coords = ()
        for cs in coordsystems
            all_coords = (all_coords..., get_coords(cs)...)
        end

        dim = length(all_coords)

        # Handle comm -- determine if we are in MPI or serial mode
        using_mpi = false
        mpi_comm_size = 1
        if comm === nothing
            if MPI_ENABLED[]
                # Use MPI.COMM_WORLD
                mpi = get_mpi()
                comm = mpi.COMM_WORLD
                mpi_comm_size = mpi.Comm_size(comm)
                using_mpi = true
            else
                # Serial mode
                comm = nothing
                mpi_comm_size = 1
            end
        else
            # Caller provided a communicator -- assume MPI
            if MPI_ENABLED[]
                mpi = get_mpi()
                mpi_comm_size = mpi.Comm_size(comm)
                using_mpi = true
            else
                # comm provided but MPI not initialized; treat as serial
                @warn "Communicator provided but MPI not initialised; running in serial mode."
                comm = nothing
                mpi_comm_size = 1
            end
        end

        # Handle mesh
        if mesh === nothing
            mesh_arr = Int[mpi_comm_size]
        elseif mesh isa Tuple || mesh isa AbstractVector
            mesh_arr = Int[m for m in mesh]
        else
            mesh_arr = Int[mesh]
        end

        # Trim trailing ones (equivalent to np.trim_zeros(mesh-1, 'b') + 1)
        while length(mesh_arr) > 1 && mesh_arr[end] == 1
            pop!(mesh_arr)
        end

        # Check mesh compatibility
        if length(mesh_arr) >= dim
            throw(ArgumentError(
                "Mesh ($(mesh_arr)) must have lower dimension than distributor ($dim)"))
        end
        if prod(mesh_arr) != mpi_comm_size
            throw(ArgumentError(
                "Wrong number of processes ($mpi_comm_size) for specified mesh ($(mesh_arr))"))
        end

        # Create cartesian communicator
        reduced_mesh = [m for m in mesh_arr if m > 1]
        if using_mpi && length(reduced_mesh) > 0
            mpi = get_mpi()
            # Create Cartesian communicator via MPI.jl
            comm_cart = mpi.Cart_create(comm, reduced_mesh;
                                         periodic=zeros(Bool, length(reduced_mesh)),
                                         reorder=false)
            comm_coords_arr = Int.(mpi.Cart_coords(comm_cart))
        elseif using_mpi
            # MPI but no distributed axes (single process or mesh = [1])
            comm_cart = SerialCommCart(reduced_mesh, zeros(Int, length(reduced_mesh)))
            comm_coords_arr = zeros(Int, length(reduced_mesh))
        else
            # Pure serial
            comm_cart = SerialCommCart(reduced_mesh, zeros(Int, length(reduced_mesh)))
            comm_coords_arr = zeros(Int, length(reduced_mesh))
        end

        dist = new(
            coordsystems,
            all_coords,
            dim,
            dtype,
            comm,
            comm_cart,
            comm_coords_arr,
            mesh_arr,
            single_cs,
            Any[],    # layouts
            Any[],    # paths
            Any[],    # transforms
            nothing,  # coeff_layout (set by _build_layouts!)
            nothing,  # grid_layout (set by _build_layouts!)
            Dict{String, Any}(),
            WeakKeyDict{Any, Nothing}(),
            nothing,  # _cs_by_axis
            nothing   # _default_nonconst_groups
        )

        # Build layout objects
        _build_layouts!(dist)

        return dist
    end
end

"""
    _is_mpi_distributor(dist::Distributor) -> Bool

Return `true` if the distributor is using real MPI (not serial mode).
"""
function _is_mpi_distributor(dist::Distributor)
    return dist.comm !== nothing && !(dist.comm_cart isa SerialCommCart && all(m -> m <= 1, dist.mesh))
end

# ============================================================================
# Distributor display
# ============================================================================

function Base.show(io::IO, d::Distributor)
    print(io, "Distributor(dim=$(d.dim), mesh=$(d.mesh))")
end

# ============================================================================
# Distributor - cs_by_axis (cached property)
# ============================================================================

"""
    cs_by_axis(dist::Distributor) -> Dict{Int, Any}

Return a mapping from axis index (1-based) to the coordinate system that
covers that axis.
"""
function cs_by_axis(dist::Distributor)
    if dist._cs_by_axis === nothing
        cs_dict = Dict{Int, Any}()
        for cs in dist.coordsystems
            for subaxis in 0:(get_dim(cs) - 1)
                ax = get_axis(dist, cs)
                cs_dict[ax + subaxis] = cs
            end
        end
        dist._cs_by_axis = cs_dict
    end
    return dist._cs_by_axis
end

"""
    get_coordsystem(dist::Distributor, axis::Integer)

Return the coordinate system covering the given axis (1-based).
"""
function get_coordsystem(dist::Distributor, axis::Integer)
    return cs_by_axis(dist)[axis]
end

# ============================================================================
# Distributor - AbstractDistributor interface
# ============================================================================

"""
    get_dim(dist::Distributor) -> Int

Return the total number of axes.
"""
get_dim(dist::Distributor) = dist.dim

"""
    get_axis(dist::Distributor, coord) -> Int

Return the 1-based axis index for a coordinate or coordinate system.

For a coordinate system, returns the axis of its first coordinate.
"""
function get_axis(dist::Distributor, coord)
    if coord isa AbstractCoordinateSystem
        coord = get_coords(coord)[1]
    end
    idx = findfirst(c -> c === coord, dist.coords)
    if idx === nothing
        throw(ArgumentError("Coordinate not found in distributor."))
    end
    return idx
end

"""
    get_basis_axis(dist::Distributor, basis) -> Int

Return the 1-based first axis index of `basis` within `dist`.
"""
function get_basis_axis(dist::Distributor, basis)
    return get_axis(dist, get_coords(coordsys(basis))[1])
end

"""
    first_axis(dist::Distributor, basis) -> Int

Return the first axis (1-based) of `basis`.
"""
function first_axis(dist::Distributor, basis)
    return get_basis_axis(dist, basis)
end

"""
    last_axis(dist::Distributor, basis) -> Int

Return the last axis (1-based) of `basis`.
"""
function last_axis(dist::Distributor, basis)
    return first_axis(dist, basis) + get_dim(basis) - 1
end

"""
    get_coords(dist::Distributor) -> Tuple

Return all coordinates managed by this distributor.
"""
get_coords(dist::Distributor) = dist.coords

"""
    coeff_layout(dist::Distributor)

Return the coefficient-space layout.
"""
coeff_layout(dist::Distributor) = dist.coeff_layout

"""
    grid_layout(dist::Distributor)

Return the grid-space layout.
"""
grid_layout(dist::Distributor) = dist.grid_layout

# ============================================================================
# Distributor - get_layout_object
# ============================================================================

"""
    get_layout_object(dist::Distributor, input)

Dereference layout identifiers.

- If `input` is already a `Layout`, return it directly.
- If `input` is a string (`"c"` or `"g"`), return the corresponding layout.
"""
function get_layout_object(dist::Distributor, input)
    if input isa Layout
        return input
    else
        return dist.layout_references[string(input)]
    end
end

# ============================================================================
# Distributor - get_transform_object
# ============================================================================

"""
    get_transform_object(dist::Distributor, axis::Integer)

Return the Transform object for the given axis (1-based).
"""
function get_transform_object(dist::Distributor, axis::Integer)
    return dist.transforms[axis]
end

# ============================================================================
# Distributor - remedy_scales
# ============================================================================

"""
    remedy_scales(dist::Distributor, scales)

Canonicalize scale inputs into a tuple of length `dist.dim`.

- `nothing` -> all ones
- A single number -> replicated to all axes
- A tuple/vector -> converted to tuple
- Zero scales are rejected.
"""
function remedy_scales(dist::Distributor, scales)
    if scales === nothing
        scales = 1
    end
    if scales isa Number
        scales = ntuple(_ -> scales, dist.dim)
    end
    scales = Tuple(scales)
    if 0 in scales
        throw(ArgumentError("Scales must be nonzero."))
    end
    return scales
end

# ============================================================================
# Distributor - buffer_size
# ============================================================================

"""
    buffer_size(dist::Distributor, domain, scales, dtype)

Compute the necessary buffer size (bytes) for all layouts.
"""
function buffer_size(dist::Distributor, domain, scales, dtype)
    return maximum(buffer_size(layout, domain, scales, dtype)
                   for layout in dist.layouts)
end

# ============================================================================
# Distributor - default_nonconst_groups
# ============================================================================

"""
    default_nonconst_groups(dist::Distributor) -> Tuple

Concatenation of default non-constant groups from all coordinate systems.
"""
function default_nonconst_groups(dist::Distributor)
    if dist._default_nonconst_groups === nothing
        groups = ()
        for cs in dist.coordsystems
            groups = (groups..., default_nonconst_groups(cs)...)
        end
        dist._default_nonconst_groups = groups
    end
    return dist._default_nonconst_groups
end

# ============================================================================
# Distributor - local_grid / local_grids / local_modes
# ============================================================================

"""
    local_grid(dist::Distributor, basis; scale=nothing)

Return the local grid for a 1D basis.
"""
function local_grid(dist::Distributor, basis; scale=nothing)
    if scale === nothing
        scale = 1
    end
    if get_dim(basis) == 1
        return local_grid(basis, dist; scale=scale)
    else
        throw(ArgumentError("Use `local_grids` for multidimensional bases."))
    end
end

"""
    local_grids(dist::Distributor, bases...; scales=nothing)

Return local grids for one or more bases.
"""
function local_grids(dist::Distributor, bases...; scales=nothing)
    scales = remedy_scales(dist, scales)
    grids = Any[]
    for basis in bases
        fa = first_axis(dist, basis)
        la = last_axis(dist, basis)
        basis_scales = scales[fa:la]
        append!(grids, local_grids(basis, dist; scales=basis_scales))
    end
    return grids
end

"""
    local_modes(dist::Distributor, basis)

Return local modes for a basis.
"""
function local_modes(dist::Distributor, basis)
    return local_modes(basis, dist)
end

# ============================================================================
# Distributor - Field constructors (forward references)
# ============================================================================

# These will be defined properly once the Field module is translated.
# For now, they serve as forward declarations that can be called through
# the distributor.

# function Field(dist::Distributor, args...; kw...)
#     # from .field import Field
#     error("Field module not yet loaded")
# end

# ============================================================================
# Layout
# ============================================================================

"""
    Layout

Describes the data distribution for a given transform and distribution state.

Supports both serial and MPI-parallel modes.

# Fields
- `dist::Distributor` -- parent distributor.
- `grid_space::Vector{Bool}` -- per-axis grid-space flags.
- `local_flags::Vector{Bool}` -- per-axis locality flags (all true in serial).
- `index::Int` -- layout index (0-based conceptually, matching Python).
- `ext_mesh::Vector{Int}` -- extended mesh (1 for local axes, mesh size otherwise).
- `ext_coords::Vector{Int}` -- extended coordinates in mesh.
- `_local_shape_cache::Dict{Any, Tuple}` -- cache for local_shape.
- `_valid_elements_cache::Dict{Any, Any}` -- cache for valid_elements.
- `_local_group_arrays_cache::Dict{Any, Any}` -- cache for local_group_arrays.
- `_global_group_arrays_cache::Dict{Any, Any}` -- cache for global_group_arrays.
- `_local_groupsets_cache::Dict{Any, Any}` -- cache for local_groupsets.
- `_local_groupset_slices_cache::Dict{Any, Any}` -- cache for local_groupset_slices.
"""
mutable struct Layout
    dist::Any  # Distributor (using Any to avoid circular reference issues)
    grid_space::Vector{Bool}
    local_flags::Vector{Bool}
    index::Int
    ext_mesh::Vector{Int}
    ext_coords::Vector{Int}
    _local_shape_cache::Dict{Any, Tuple}
    _valid_elements_cache::Dict{Any, Any}
    _local_group_arrays_cache::Dict{Any, Any}
    _global_group_arrays_cache::Dict{Any, Any}
    _local_groupsets_cache::Dict{Any, Any}
    _local_groupset_slices_cache::Dict{Any, Any}

    function Layout(dist, local_flags, grid_space)
        dim = dist.dim

        # Freeze into copies
        gs = Bool[grid_space[i] for i in 1:dim]
        lf = Bool[local_flags[i] for i in 1:dim]

        # Extended mesh: 1 for local axes, mesh element for distributed axes
        ext_m = ones(Int, dim)
        reduced_mesh = [m for m in dist.mesh if m > 1]
        dist_idx = 1
        for i in 1:dim
            if !lf[i]
                if dist_idx <= length(reduced_mesh)
                    ext_m[i] = reduced_mesh[dist_idx]
                    dist_idx += 1
                end
            end
        end

        # Extended coords
        ext_c = zeros(Int, dim)
        dist_idx = 1
        for i in 1:dim
            if !lf[i]
                if dist_idx <= length(dist.comm_coords)
                    ext_c[i] = dist.comm_coords[dist_idx]
                    dist_idx += 1
                end
            end
        end

        return new(dist, gs, lf, -1,  # index set later
                   ext_m, ext_c,
                   Dict{Any, Tuple}(),
                   Dict{Any, Any}(),
                   Dict{Any, Any}(),
                   Dict{Any, Any}(),
                   Dict{Any, Any}(),
                   Dict{Any, Any}())
    end
end

function Base.show(io::IO, l::Layout)
    print(io, "Layout(index=$(l.index), grid_space=$(l.grid_space))")
end

# ============================================================================
# Layout - Shape and element methods
# ============================================================================

"""
    global_shape(layout::Layout, domain, scales) -> Tuple

Global data shape for this layout, domain, and scales.
"""
function global_shape(layout::Layout, domain, scales)
    scales = remedy_scales(layout.dist, scales)
    return global_shape(domain, layout, scales)
end

"""
    chunk_shape(layout::Layout, domain) -> Tuple

Chunk shape for this layout and domain.
"""
function chunk_shape(layout::Layout, domain)
    return chunk_shape(domain, layout)
end

"""
    group_shape(layout::Layout, domain) -> Tuple

Group shape for this layout and domain.
"""
function group_shape(layout::Layout, domain)
    return group_shape(domain, layout)
end

"""
    local_chunks(layout::Layout, domain, scales; rank=nothing, broadcast=false) -> Tuple

Local chunk indices by axis.
"""
function local_chunks(layout::Layout, domain, scales;
                      rank=nothing, broadcast::Bool=false)
    gs = global_shape(layout, domain, scales)
    cs = chunk_shape(layout, domain)

    # Ceiling division for chunk counts
    chunk_nums = [cld(g, c) for (g, c) in zip(gs, cs)]

    lc = Vector{Vector{Int}}()

    # Get coordinates
    if rank === nothing
        ext_c = layout.ext_coords
    else
        ext_c = zeros(Int, layout.dist.dim)
        # In MPI mode, compute coords for the given rank
        if _is_mpi_distributor(layout.dist) && !(layout.dist.comm_cart isa SerialCommCart)
            mpi = get_mpi()
            remote_coords = Int.(mpi.Cart_coords(layout.dist.comm_cart, rank))
            # Fill in ext_c for distributed axes
            dist_idx = 1
            for i in 1:layout.dist.dim
                if !layout.local_flags[i]
                    if dist_idx <= length(remote_coords)
                        ext_c[i] = remote_coords[dist_idx]
                        dist_idx += 1
                    end
                end
            end
        end
    end

    # Get chunks axis by axis
    fb = full_bases(domain)
    for ax in 1:length(fb)
        basis = fb[ax]
        if layout.local_flags[ax]
            # All chunks for local dimensions (0-based chunk indices)
            push!(lc, collect(0:(chunk_nums[ax] - 1)))
        else
            # Block distribution
            m = layout.ext_mesh[ax]
            if broadcast && basis === nothing
                coord = 0
            else
                coord = ext_c[ax]
            end
            block = cld(chunk_nums[ax], m)
            start_idx = min(chunk_nums[ax], block * coord)
            end_idx = min(chunk_nums[ax], block * (coord + 1))
            push!(lc, collect(start_idx:(end_idx - 1)))
        end
    end

    return Tuple(lc)
end

"""
    global_elements(layout::Layout, domain, scales) -> Tuple

Global element indices by axis (0-based, matching Python convention).
"""
function global_elements(layout::Layout, domain, scales)
    gs = global_shape(layout, domain, scales)
    indices = [collect(0:(n - 1)) for n in gs]
    return Tuple(indices)
end

"""
    local_elements(layout::Layout, domain, scales;
                   rank=nothing, broadcast=false) -> Tuple

Local element indices by axis (0-based, matching Python convention).
"""
function local_elements(layout::Layout, domain, scales;
                        rank=nothing, broadcast::Bool=false)
    cs = chunk_shape(layout, domain)
    lc = local_chunks(layout, domain, scales; rank=rank, broadcast=broadcast)

    indices = Vector{Vector{Int}}()
    for (chunk_size, chunks) in zip(cs, lc)
        # For each chunk index, generate element indices within that chunk
        ax_indices = Int[]
        for c in chunks
            for offset in 0:(chunk_size - 1)
                push!(ax_indices, chunk_size * c + offset)
            end
        end
        push!(indices, ax_indices)
    end

    return Tuple(indices)
end

"""
    valid_elements(layout::Layout, tensorsig, domain, scales;
                   rank=nothing, broadcast=false)

Make dense array of mode inclusion. Returns a boolean array indicating
which elements are valid.
"""
function valid_elements(layout::Layout, tensorsig, domain, scales;
                        rank=nothing, broadcast::Bool=false)
    cache_key = (objectid(tensorsig), objectid(domain), scales, rank, broadcast)
    cached = get(layout._valid_elements_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end

    elements = local_elements(layout, domain, scales; rank=rank, broadcast=broadcast)

    # Create meshgrid-like dense array of elements
    dim = length(elements)
    if dim == 0
        result = trues()
        layout._valid_elements_cache[cache_key] = result
        return result
    end

    # Build shape for the meshgrid
    grid_shape_sizes = [length(e) for e in elements]
    tensor_shape = [get_dim(cs) for cs in tensorsig]
    vshape = (tensor_shape..., grid_shape_sizes...)
    valid = trues(vshape...)

    # Check validity basis-by-basis
    gs = layout.grid_space
    for basis in domain_bases(domain)
        fa = first_axis(layout.dist, basis)
        la = last_axis(layout.dist, basis)
        basis_axes = fa:la
        # Get grid_space and elements for this basis's axes
        basis_gs = gs[basis_axes]
        basis_elements = [elements[ax] for ax in basis_axes]
        # Create meshgrid of basis elements
        basis_valid = valid_elements(basis, tensorsig, basis_gs, basis_elements)
        valid .&= basis_valid
    end

    layout._valid_elements_cache[cache_key] = valid
    return valid
end

"""
    slices(layout::Layout, domain, scales) -> Tuple

Local element slices by axis.
"""
function slices(layout::Layout, domain, scales)
    le = local_elements(layout, domain, scales)
    result = UnitRange{Int}[]
    for elem_indices in le
        if length(elem_indices) > 0
            # +1 for Julia 1-based indexing
            push!(result, (minimum(elem_indices) + 1):(maximum(elem_indices) + 1))
        else
            push!(result, 1:0)  # empty range
        end
    end
    return Tuple(result)
end

"""
    local_shape(layout::Layout, domain, scales; rank=nothing) -> Tuple

Local data shape.
"""
function local_shape(layout::Layout, domain, scales; rank=nothing)
    cache_key = (objectid(domain), Tuple(scales), rank)
    cached = get(layout._local_shape_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    le = local_elements(layout, domain, scales; rank=rank)
    shape = Tuple(length(e) for e in le)
    layout._local_shape_cache[cache_key] = shape
    return shape
end

"""
    buffer_size(layout::Layout, domain, scales, dtype) -> Int

Local buffer size (bytes).
"""
function buffer_size(layout::Layout, domain, scales, dtype)
    ls = local_shape(layout, domain, scales)
    return prod(ls) * sizeof(dtype)
end

# ============================================================================
# Layout - Group methods (for advanced operator assembly)
# ============================================================================

"""
    _group_arrays(layout::Layout, elements, domain)

Convert element arrays to group arrays basis-by-basis.
Internal helper.
"""
function _group_arrays(layout::Layout, elements, domain)
    gs = layout.grid_space
    groups = copy(elements)
    for basis in domain_bases(domain)
        fa = first_axis(layout.dist, basis)
        la = last_axis(layout.dist, basis)
        basis_axes = fa:la
        basis_gs = gs[basis_axes]
        # elements_to_groups is a basis method that will be implemented
        # when basis types are translated
        groups[basis_axes] .= elements_to_groups(basis, basis_gs, elements[basis_axes])
    end
    return groups
end

"""
    local_group_arrays(layout::Layout, domain, scales;
                       rank=nothing, broadcast=false)

Dense array of local groups (first axis).
"""
function local_group_arrays(layout::Layout, domain, scales;
                            rank=nothing, broadcast::Bool=false)
    cache_key = (objectid(domain), Tuple(scales), rank, broadcast)
    cached = get(layout._local_group_arrays_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end

    elements = local_elements(layout, domain, scales; rank=rank, broadcast=broadcast)
    # Build meshgrid-like dense array of elements
    # This is a simplified version for serial mode
    result = _build_group_arrays_from_elements(layout, elements, domain)
    layout._local_group_arrays_cache[cache_key] = result
    return result
end

"""
    global_group_arrays(layout::Layout, domain, scales)

Dense array of global groups.
"""
function global_group_arrays(layout::Layout, domain, scales)
    cache_key = (objectid(domain), Tuple(scales))
    cached = get(layout._global_group_arrays_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end

    elements = global_elements(layout, domain, scales)
    result = _build_group_arrays_from_elements(layout, elements, domain)
    layout._global_group_arrays_cache[cache_key] = result
    return result
end

"""
    _build_group_arrays_from_elements(layout, elements, domain)

Build dense group arrays from element index vectors.
Internal helper shared by `local_group_arrays` and `global_group_arrays`.
"""
function _build_group_arrays_from_elements(layout::Layout, elements, domain)
    dim = length(elements)
    if dim == 0
        return Array{Int}(undef, 0)
    end

    # Create meshgrid of elements (ij indexing)
    grid_sizes = [length(e) for e in elements]
    element_grids = Array{Any}(undef, dim)
    for d in 1:dim
        shape = ones(Int, dim)
        shape[d] = grid_sizes[d]
        base = reshape(elements[d], shape...)
        reps = copy(grid_sizes)
        reps[d] = 1
        element_grids[d] = repeat(base; outer=reps)
    end

    # Convert to groups
    gs = layout.grid_space
    groups = copy(element_grids)
    for basis in domain_bases(domain)
        fa = first_axis(layout.dist, basis)
        la = last_axis(layout.dist, basis)
        basis_axes = fa:la
        basis_gs = gs[basis_axes]
        for ax in basis_axes
            groups[ax] = elements_to_groups(basis, basis_gs, element_grids[ax])
        end
    end

    return groups
end

"""
    local_groupsets(layout::Layout, group_coupling, domain, scales;
                    rank=nothing, broadcast=false)

Compute unique local groupsets.
"""
function local_groupsets(layout::Layout, group_coupling, domain, scales;
                         rank=nothing, broadcast::Bool=false)
    cache_key = (Tuple(group_coupling), objectid(domain), Tuple(scales), rank, broadcast)
    cached = get(layout._local_groupsets_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end

    lga = local_group_arrays(layout, domain, scales; rank=rank, broadcast=broadcast)
    dim = length(lga)

    # Replace non-enumerated axes (coupled axes) with nothing
    local_gs = Vector{Any}(undef, dim)
    for ax in 1:dim
        if group_coupling[ax]
            local_gs[ax] = fill(nothing, size(lga[ax]))
        else
            local_gs[ax] = lga[ax]
        end
    end

    # Flatten and collect unique groupsets
    if dim == 0
        result = OrderedSet{Tuple}()
        layout._local_groupsets_cache[cache_key] = result
        return result
    end

    total_elements = prod(size(local_gs[1]))
    groupset_list = Vector{Tuple}()
    for idx in 1:total_elements
        gs_tuple = Tuple(local_gs[d][idx] for d in 1:dim)
        if !(gs_tuple in groupset_list)
            push!(groupset_list, gs_tuple)
        end
    end

    result = OrderedSet{Tuple}(groupset_list)
    layout._local_groupsets_cache[cache_key] = result
    return result
end

# ============================================================================
# Build layouts
# ============================================================================

"""
    _build_layouts!(dist::Distributor; dry_run=false)

Construct Layout objects for all transform/distribution states, and
the Transform/Transpose paths between them.

With mesh producing R distributed axes, we get D+R+1 layouts
connected by D transforms and R transposes.

In serial mode with mesh=[1], R=0 (no distributed axes), so we get
D+1 layouts (one per transform axis plus the initial coeff layout)
connected by D transforms (one per axis).
"""
function _build_layouts!(dist::Distributor; dry_run::Bool=false)
    D = dist.dim
    # R = number of mesh dimensions > 1
    R = count(m -> m > 1, dist.mesh)

    # First layout: full coefficient space
    local_flags = fill(true, D)
    # In parallel mode, axes covered by mesh entries > 1 are non-local
    mesh_idx = 1
    for i in 1:length(dist.mesh)
        if i <= D && dist.mesh[i] > 1
            local_flags[i] = false
        end
    end
    grid_space = fill(false, D)

    layout_0 = Layout(dist, local_flags, grid_space)
    layout_0.index = 0

    dist.layouts = Any[layout_0]
    dist.paths = Any[]
    dist.transforms = Any[]

    # Subsequent layouts
    # Total number of additional layouts = R + D
    for i in 1:(R + D)
        # Iterate backwards over axes to find last coefficient-space axis
        found = false
        local layout_i, path_i
        layout_i = nothing
        path_i = nothing
        for d in D:-1:1
            if !grid_space[d]
                if local_flags[d]
                    # Transform: this axis is local and in coeff space
                    grid_space[d] = true
                    layout_i = Layout(dist, local_flags, grid_space)
                    if !dry_run
                        path_i = DistTransform(dist.layouts[end], layout_i, d)
                        # Insert at beginning of transforms list
                        # (transforms are indexed by axis, built in reverse)
                        pushfirst!(dist.transforms, path_i)
                    end
                    found = true
                    break
                else
                    # Transpose: this axis is distributed
                    local_flags[d] = true
                    if d < D
                        local_flags[d + 1] = false
                    end
                    layout_i = Layout(dist, local_flags, grid_space)
                    if !dry_run
                        path_i = Transpose(dist.layouts[end], layout_i, d, dist.comm_cart)
                    end
                    found = true
                    break
                end
            end
        end

        if !found
            break
        end

        layout_i.index = i
        push!(dist.layouts, layout_i)
        if !dry_run
            push!(dist.paths, path_i)
        end
    end

    # Reference coefficient and grid space layouts
    dist.coeff_layout = dist.layouts[1]
    dist.grid_layout = dist.layouts[end]

    # String references
    dist.layout_references = Dict{String, Any}(
        "c" => dist.coeff_layout,
        "g" => dist.grid_layout
    )

    return nothing
end

# ============================================================================
# Transform
# ============================================================================

"""
    Transform

Directs spectral transforms between two adjacent layouts along a single axis.

In serial mode, transforms are simple basis forward/backward calls.

# Fields
- `layout0` -- coefficient-space side layout.
- `layout1` -- grid-space side layout.
- `axis::Int` -- axis being transformed (1-based).
"""
struct DistTransform
    layout0::Layout
    layout1::Layout
    axis::Int
end

function Base.show(io::IO, t::DistTransform)
    print(io, "DistTransform(axis=$(t.axis), layout0=$(t.layout0.index) -> layout1=$(t.layout1.index))")
end

"""
    increment(transform::DistTransform, fields)

Backward transform (coeff -> grid) a list of fields along the transform axis.
"""
function increment(transform::DistTransform, fields)
    if length(fields) == 1
        increment_single(transform, fields[1])
    else
        for field in fields
            increment_single(transform, field)
        end
    end
end

"""
    decrement(transform::DistTransform, fields)

Forward transform (grid -> coeff) a list of fields along the transform axis.
"""
function decrement(transform::DistTransform, fields)
    if length(fields) == 1
        decrement_single(transform, fields[1])
    else
        for field in fields
            decrement_single(transform, field)
        end
    end
end

"""
    increment_single(transform::DistTransform, field)

Backward transform a single field (coeff -> grid).
"""
function increment_single(transform::DistTransform, field)
    ax = transform.axis
    basis = full_bases(field.domain)[ax]

    # Reference view from coefficient layout
    cdata = field.data

    # Switch to grid layout
    preset_layout!(field, transform.layout1)
    gdata = field.data

    # Transform non-constant bases with data
    if basis !== nothing && prod(size(cdata)) > 0
        backward_transform(basis, field, ax, cdata, gdata)
    end
end

"""
    decrement_single(transform::DistTransform, field)

Forward transform a single field (grid -> coeff).
"""
function decrement_single(transform::DistTransform, field)
    ax = transform.axis
    basis = full_bases(field.domain)[ax]

    # Reference view from grid layout
    gdata = field.data

    # Switch to coefficient layout
    preset_layout!(field, transform.layout0)
    cdata = field.data

    # Transform non-constant bases with data
    if basis !== nothing && prod(size(gdata)) > 0
        forward_transform(basis, field, ax, gdata, cdata)
    end
end

# ============================================================================
# Transpose (supports both serial no-op and MPI transpose)
# ============================================================================

"""
    Transpose

Directs distributed transposes between two layouts.

In serial mode, transposes are no-ops since all data is local.
In MPI mode, transposes use AlltoallvTranspose planners for actual
data redistribution.

# Fields
- `layout0` -- source layout.
- `layout1` -- destination layout.
- `axis::Int` -- axis being transposed (1-based).
- `comm_cart` -- Cartesian communicator.
- `comm_sub` -- Sub-communicator for the transpose axis (MPI mode) or `nothing`.
- `_plan_cache::Dict` -- cached transpose plans.
"""
mutable struct Transpose
    layout0::Layout
    layout1::Layout
    axis::Int
    comm_cart::Any
    comm_sub::Any
    _plan_cache::Dict{Any, Any}

    function Transpose(layout0, layout1, axis, comm_cart)
        # Create subgrid communicator along the moving mesh axis
        comm_sub = nothing
        if !(comm_cart isa SerialCommCart) && MPI_ENABLED[]
            mesh = layout0.dist.mesh
            ndims_cart = count(m -> m > 1, mesh)
            remain_dims = zeros(Bool, ndims_cart)
            # Find the cart dimension corresponding to this axis
            # comm_cart_axis = axis - count(mesh[1:axis] .== 1) when mesh entries
            # before this axis are == 1 (those do not appear in the reduced cart)
            n_trivial_before = 0
            for i in 1:min(axis, length(mesh))
                if mesh[i] == 1
                    n_trivial_before += 1
                end
            end
            comm_cart_axis = axis - n_trivial_before
            if comm_cart_axis >= 1 && comm_cart_axis <= ndims_cart
                remain_dims[comm_cart_axis] = true
            end
            comm_sub = cart_sub(comm_cart, remain_dims)
        end
        return new(layout0, layout1, axis, comm_cart, comm_sub, Dict{Any, Any}())
    end
end

function Base.show(io::IO, t::Transpose)
    print(io, "Transpose(axis=$(t.axis), layout0=$(t.layout0.index) -> layout1=$(t.layout1.index))")
end

"""
    _sub_shape(transpose_obj::Transpose, domain, scales) -> Tuple

Build global shape of data assigned to sub-communicator.
Local shape along non-transposing axes, global shape along transposing axes.
"""
function _sub_shape(transpose_obj::Transpose, domain, scales)
    ls = local_shape(transpose_obj.layout0, domain, scales)
    gs = global_shape(transpose_obj.layout0, domain, scales)
    ax = transpose_obj.axis
    sub = collect(ls)
    sub[ax] = gs[ax]
    if ax + 1 <= length(sub)
        sub[ax + 1] = gs[ax + 1]
    end
    return Tuple(sub)
end

"""
    _get_plan(transpose_obj::Transpose, ncomp, sub_shape, chunk_shape_val, dtype)

Build or retrieve a cached transpose plan. Returns `nothing` if no
communication is needed (serial mode or identity shapes).
"""
function _get_plan(transpose_obj::Transpose, ncomp::Integer, sub_shape::Tuple,
                   chunk_shape_val::Tuple, dtype)
    cache_key = (ncomp, sub_shape, chunk_shape_val, dtype)
    cached = get(transpose_obj._plan_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end

    ax = transpose_obj.axis
    if prod(sub_shape) == 0
        plan = nothing  # no data
    elseif sub_shape[ax] == chunk_shape_val[ax] && (ax + 1 <= length(sub_shape) ? sub_shape[ax + 1] == chunk_shape_val[ax + 1] : true)
        plan = nothing  # no change needed
    elseif transpose_obj.comm_sub === nothing
        plan = nothing  # serial mode
    else
        # Build AlltoallvTranspose plan
        # Prepend ncomp to sub_shape (matching Python's (ncomp,) + sub_shape)
        full_sub_shape = (ncomp, sub_shape...)
        full_chunk_shape = (ncomp, chunk_shape_val...)
        # axis+1 because we prepended ncomp dimension
        plan = AlltoallvTranspose(full_sub_shape, dtype, ax + 1, transpose_obj.comm_sub)
    end

    transpose_obj._plan_cache[cache_key] = plan
    return plan
end

"""
    _single_plan(transpose_obj::Transpose, field)

Build a transpose plan for a single field.
"""
function _single_plan(transpose_obj::Transpose, field)
    ncomp = prod([get_dim(cs) for cs in field.tensorsig])
    sub_shape_val = _sub_shape(transpose_obj, field.domain, field.scales)
    chunk_shape_val = Tuple(chunk_shape(field.domain, transpose_obj.layout0))
    return _get_plan(transpose_obj, ncomp, sub_shape_val, chunk_shape_val, field.dtype)
end

"""
    _group_plans(transpose_obj::Transpose, fields)

Build group transpose plans. Segments fields by sub_shapes and chunk_shapes,
computing a combined plan for each group.
"""
function _group_plans(transpose_obj::Transpose, fields)
    field_groups = OrderedDict{Any, Vector{Any}}()
    for field in fields
        sub_shape_val = _sub_shape(transpose_obj, field.domain, field.scales)
        chunk_shape_val = Tuple(chunk_shape(field.domain, transpose_obj.layout0))
        key = (sub_shape_val, chunk_shape_val)
        if haskey(field_groups, key)
            push!(field_groups[key], field)
        else
            field_groups[key] = Any[field]
        end
    end
    plans = []
    for ((sub_shape_val, chunk_shape_val), grp_fields) in field_groups
        ncomp = 0
        for f in grp_fields
            ncomp += prod([get_dim(cs) for cs in f.tensorsig])
        end
        plan = _get_plan(transpose_obj, ncomp, sub_shape_val, chunk_shape_val,
                         grp_fields[end].dtype)
        push!(plans, (grp_fields, plan))
    end
    return plans
end

"""
    increment(transpose_obj::Transpose, fields)

Backward transpose a list of fields (coeff-side to grid-side).
"""
function increment(transpose_obj::Transpose, fields)
    if length(fields) == 1
        increment_single(transpose_obj, fields[1])
    else
        for field in fields
            increment_single(transpose_obj, field)
        end
    end
end

"""
    decrement(transpose_obj::Transpose, fields)

Forward transpose a list of fields (grid-side to coeff-side).
"""
function decrement(transpose_obj::Transpose, fields)
    if length(fields) == 1
        decrement_single(transpose_obj, fields[1])
    else
        for field in fields
            decrement_single(transpose_obj, field)
        end
    end
end

"""
    increment_single(transpose_obj::Transpose, field)

Backward transpose a single field.

In serial mode (plan == nothing), just updates the field layout.
In MPI mode, uses `localize_columns` to redistribute data.
"""
function increment_single(transpose_obj::Transpose, field)
    plan = _single_plan(transpose_obj, field)
    if plan !== nothing
        # Reference views from both layouts
        data0 = field.data
        preset_layout!(field, transpose_obj.layout1)
        data1 = field.data
        # Transpose between data views (localize_columns: RL -> CL direction)
        localize_columns(plan, data0, data1)
    else
        # No communication: just update field layout
        preset_layout!(field, transpose_obj.layout1)
    end
end

"""
    decrement_single(transpose_obj::Transpose, field)

Forward transpose a single field.

In serial mode (plan == nothing), just updates the field layout.
In MPI mode, uses `localize_rows` to redistribute data.
"""
function decrement_single(transpose_obj::Transpose, field)
    plan = _single_plan(transpose_obj, field)
    if plan !== nothing
        # Reference views from both layouts
        data1 = field.data
        preset_layout!(field, transpose_obj.layout0)
        data0 = field.data
        # Transpose between data views (localize_rows: CL -> RL direction)
        localize_rows(plan, data1, data0)
    else
        # No communication: just update field layout
        preset_layout!(field, transpose_obj.layout0)
    end
end

# backward_transform, forward_transform, elements_to_groups are defined in basis.jl
# domain_bases is defined in domain.jl
# preset_layout! is defined in field.jl

# ============================================================================
# Exports
# ============================================================================

export Distributor,
       Layout,
       DistTransform,
       Transpose,
       SerialComm,
       SerialCommCart,
       AbstractTransposePlanner,
       AlltoallvTranspose,
       ColDistributor,
       RowDistributor,
       get_layout_object,
       get_transform_object,
       get_coordsystem,
       cs_by_axis,
       first_axis,
       last_axis,
       local_grid,
       local_grids,
       local_modes,
       global_shape,
       chunk_shape,
       group_shape,
       local_chunks,
       global_elements,
       local_elements,
       valid_elements,
       slices,
       local_shape,
       buffer_size,
       local_group_arrays,
       global_group_arrays,
       local_groupsets,
       increment,
       decrement,
       increment_single,
       decrement_single,
       backward_transform,
       forward_transform,
       elements_to_groups,
       localize_rows,
       localize_columns
