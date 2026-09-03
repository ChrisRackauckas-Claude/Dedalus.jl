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

## Key translation choices

- Python's MPI communicator -> `nothing` for serial mode; a comm wrapper for
  future MPI support.
- Python 0-based layout indices -> kept 0-based conceptually, but stored in
  1-based Julia arrays (layout at conceptual index `i` lives at `layouts[i+1]`).
- Python `np.array` of bools -> Julia `Vector{Bool}` / `NTuple{N, Bool}`.
- Python `WeakSet` -> Julia `WeakKeyDict` (keys are fields, values are `nothing`).
- Python `@CachedAttribute` / `@CachedMethod` -> accessor functions with
  Dict-based memoization.
- Python `isinstance` checks -> Julia multiple dispatch / `isa`.
- Python `prod` from `math` -> Julia `prod`.
- Python `numbers.Number` -> Julia `Number`.

## Serial-mode simplifications

In serial (single-process) mode:
- `comm` is `nothing` -- no real MPI needed.
- `mesh` is an empty tuple `()` -- all data is local.
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
# Distributor
# ============================================================================

"""
    Distributor <: AbstractDistributor

Directs parallelized distribution and transformation of fields.

In serial mode (the only mode supported in this milestone), all data is local
and transforms are simple basis forward/backward calls.

# Fields
- `coordsystems::Tuple` -- coordinate systems managed by this distributor.
- `coords::Tuple` -- flattened tuple of all coordinates.
- `dim::Int` -- total dimensionality.
- `dtype` -- default data type (e.g. `Float64`, `ComplexF64`).
- `comm` -- MPI communicator or `nothing` for serial mode.
- `comm_cart` -- Cartesian communicator or `SerialCommCart`.
- `comm_coords::Vector{Int}` -- coordinates in the Cartesian communicator.
- `mesh::Vector{Int}` -- process mesh (empty in serial mode).
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

        # Handle comm
        if comm === nothing
            comm = SerialComm()
        end

        # Handle mesh
        if mesh === nothing
            # Serial: single process
            mesh_arr = Int[comm.size]
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
        if prod(mesh_arr) != comm.size
            throw(ArgumentError(
                "Wrong number of processes ($(comm.size)) for specified mesh ($(mesh_arr))"))
        end

        # Create cartesian communicator
        # In serial mode, this is a placeholder
        reduced_mesh = [m for m in mesh_arr if m > 1]
        if comm isa SerialComm
            comm_cart = SerialCommCart(reduced_mesh, zeros(Int, length(reduced_mesh)))
            comm_coords_arr = zeros(Int, length(reduced_mesh))
        else
            # Future MPI support would create a real Cartesian communicator here
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
        return basis.local_grid(dist; scale=scale)
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
        append!(grids, basis.local_grids(dist; scales=basis_scales))
    end
    return grids
end

"""
    local_modes(dist::Distributor, basis)

Return local modes for a basis.
"""
function local_modes(dist::Distributor, basis)
    return basis.local_modes(dist)
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

In serial mode, all axes are local, so distribution logic simplifies
significantly.

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
        # In serial mode, rank is always 0, coords are all 0
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
    # In parallel mode, axes covered by mesh entries > 1 would be non-local
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
# Transpose (placeholder for parallel mode)
# ============================================================================

"""
    Transpose

Directs distributed transposes between two layouts.

In serial mode, transposes are no-ops since all data is local.
This type exists for API completeness and future MPI support.

# Fields
- `layout0` -- source layout.
- `layout1` -- destination layout.
- `axis::Int` -- axis being transposed (1-based).
- `comm_cart` -- Cartesian communicator.
"""
struct Transpose
    layout0::Layout
    layout1::Layout
    axis::Int
    comm_cart::Any
end

function Base.show(io::IO, t::Transpose)
    print(io, "Transpose(axis=$(t.axis), layout0=$(t.layout0.index) -> layout1=$(t.layout1.index))")
end

"""
    increment(transpose_obj::Transpose, fields)

Backward transpose a list of fields.

In serial mode, this is a no-op (just updates layout).
"""
function increment(transpose_obj::Transpose, fields)
    for field in fields
        increment_single(transpose_obj, field)
    end
end

"""
    decrement(transpose_obj::Transpose, fields)

Forward transpose a list of fields.

In serial mode, this is a no-op (just updates layout).
"""
function decrement(transpose_obj::Transpose, fields)
    for field in fields
        decrement_single(transpose_obj, field)
    end
end

"""
    increment_single(transpose_obj::Transpose, field)

Backward transpose a single field.

In serial mode, just updates the field layout.
"""
function increment_single(transpose_obj::Transpose, field)
    preset_layout!(field, transpose_obj.layout1)
end

"""
    decrement_single(transpose_obj::Transpose, field)

Forward transpose a single field.

In serial mode, just updates the field layout.
"""
function decrement_single(transpose_obj::Transpose, field)
    preset_layout!(field, transpose_obj.layout0)
end

# ============================================================================
# Forward-reference stubs for Field interface
# ============================================================================

# These functions define the interface that Field types must implement.
# They will be properly defined when the Field module is translated.

# preset_layout!(field, layout) is defined in field.jl (loaded before this file)

"""
    backward_transform(basis, field, axis, cdata, gdata)

Perform a backward (coeff -> grid) transform.
Must be implemented by basis types.
"""
function backward_transform end

"""
    forward_transform(basis, field, axis, gdata, cdata)

Perform a forward (grid -> coeff) transform.
Must be implemented by basis types.
"""
function forward_transform end

"""
    elements_to_groups(basis, grid_space, elements)

Convert element indices to group indices.
Must be implemented by basis types.
"""
function elements_to_groups end

"""
    domain_bases(domain)

Return the bases of a domain. Alias for the function defined in domain.jl.
This stub is provided so that distributor.jl can be loaded independently.
"""
function domain_bases end

# ============================================================================
# Exports
# ============================================================================

export Distributor,
       Layout,
       Transform,
       Transpose,
       SerialComm,
       SerialCommCart,
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
       preset_layout,
       backward_transform,
       forward_transform,
       elements_to_groups
