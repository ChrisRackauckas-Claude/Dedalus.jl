"""
    Domain type for Dedalus.jl

Julia translation of `dedalus/core/domain.py`. A `Domain` represents the
direct product of a set of spectral bases that define the computational
domain for a field or operand.

## Key translation choices

- Python's `CachedClass` metaclass → `CachedClass` wrapper from `tools/cache.jl`.
- Python `@CachedAttribute` → lazily-computed fields with accessor functions.
- Python `@CachedMethod` → `CachedMethod` wrappers or Julia memoization.
- Python `OrderedDict` → `OrderedDict` from `OrderedCollections`.
- `numpy` arrays → Julia arrays.
- 0-based axis indexing → 1-based (Julia convention).

## Forward references

Basis types do not exist yet.  We introduce `AbstractBasis` as an abstract
type here so that Domain can reference it.  Concrete basis implementations
will subtype `AbstractBasis` when they are created.

Similarly, the `Distributor` type is forward-referenced via `AbstractDistributor`.
"""

using OrderedCollections: OrderedDict
using LinearAlgebra

# ============================================================================
# Forward-reference abstract types
# ============================================================================

"""
    AbstractBasis

Abstract supertype for all spectral basis types.  Concrete bases (Fourier,
Chebyshev, SphericalBasis, etc.) will subtype this.

Expected interface (to be implemented by concrete subtypes):
- `get_dim(b)::Int`
- `coordsys(b)` — the coordinate or coordinate system the basis spans
- `get_coords(b)` — same as `coordsys` (alias)
- `dealias_tuple(b)` — tuple of dealias factors per sub-axis
- `volume(b)` — domain volume for this basis
- `global_shape(b, grid_space, scales)` — global data shape
- `chunk_shape(b, grid_space)` — chunk shape for distribution
- `group_shape_val(b)` — group shape as a tuple
- `grid_shape(b, scales)` — grid-space shape
- `constant_flags(b)` — tuple of booleans
- `subaxis_dependence_flags(b)` — tuple of booleans
- `axis(b)::Int` — first axis index in the distributor layout
"""
abstract type AbstractBasis end

"""
    AbstractDistributor

Abstract supertype for the Distributor type, which manages data distribution
across processes.  Forward-referenced here so `Domain` can hold a reference
without importing the full distributor module.

Expected interface (to be implemented):
- `get_dim(dist)::Int` — total number of axes
- `get_basis_axis(dist, basis)::Int` — first axis of a basis (1-based)
- `get_axis(dist, coords)::Int` — axis for a coordinate/coord-system
- `get_coords(dist)` — all coordinates in the distributor
- `coeff_layout(dist)` — the coefficient-space layout
- `remedy_scales(dist, scales)` — canonicalise scale factors
"""
abstract type AbstractDistributor end

# ============================================================================
# Domain
# ============================================================================

"""
    Domain

The direct product of a set of spectral bases defining a computational
domain.  Domains are constructed from a distributor and a collection of
bases.

# Fields
- `dist::AbstractDistributor` — the data distributor.
- `bases::Tuple` — ordered tuple of `AbstractBasis` objects (sorted by axis,
  with `nothing` and duplicates removed during preprocessing).
- `_dim::Int` — total dimensionality (sum of basis dimensions).
- `_cache::Dict{Symbol, Any}` — lazy-attribute cache.

# Construction

Use the factory function [`make_domain`](@ref) which applies preprocessing
(deduplication, overlap checking, axis sorting) matching Python's
`Domain._preprocess_args`.

Direct construction via `Domain(dist, bases)` assumes `bases` is already
preprocessed.

# Examples
```julia
dom = make_domain(dist, (basis_x, basis_y))
get_dim(dom)          # total dimensionality
domain_bases(dom)     # ordered tuple of bases
domain_volume(dom)    # product of basis volumes
```
"""
mutable struct Domain
    dist::AbstractDistributor
    bases::Tuple
    _dim::Int
    _cache::Dict{Symbol, Any}

    function Domain(dist::AbstractDistributor, bases::Tuple)
        dim_val = sum(get_dim(b) for b in bases; init=0)
        return new(dist, bases, dim_val, Dict{Symbol, Any}())
    end
end

# ============================================================================
# Preprocessing (mirrors Python Domain._preprocess_args)
# ============================================================================

"""
    preprocess_domain_args(dist, bases)

Canonicalise the arguments for `Domain` construction:
1. Drop `nothing` entries from `bases`.
2. Remove duplicate bases (preserving order).
3. Verify that no two bases share the same coordinate system.
4. Sort by first axis index.

Returns `(dist, bases)` ready for `Domain` construction.
"""
function preprocess_domain_args(dist::AbstractDistributor, bases)
    # Drop Nothings
    filtered = [b for b in bases if b !== nothing]

    # Drop duplicates (preserving order)
    seen = Set{Any}()
    unique_bases = []
    for b in filtered
        oid = objectid(b)
        if oid in seen
            continue
        end
        push!(seen, oid)
        push!(unique_bases, b)
    end

    # Check for overlapping coordinate systems
    cs_list = [coordsys(b) for b in unique_bases]
    if length(Set(cs_list)) < length(cs_list)
        throw(ArgumentError("Overlapping bases specified."))
    end

    # Sort by first axis
    sorted_bases = sort(unique_bases; by = b -> get_basis_axis(dist, b))
    return (dist, Tuple(sorted_bases))
end

"""
    make_domain(dist, bases) -> Domain

Construct a `Domain` with full preprocessing: drop `nothing`, deduplicate,
check for overlapping coordinate systems, and sort by axis.

This is the recommended entry point and mirrors Python's `Domain(dist, bases)`
with `CachedClass` + `_preprocess_args`.
"""
function make_domain(dist::AbstractDistributor, bases)
    d, b = preprocess_domain_args(dist, bases)
    return Domain(d, b)
end

# ============================================================================
# Accessor functions (properties)
# ============================================================================

"""
    domain_dist(dom::Domain) -> AbstractDistributor

Return the distributor associated with this domain.
"""
domain_dist(dom::Domain) = dom.dist

"""
    domain_bases(dom::Domain) -> Tuple

Return the ordered tuple of bases in this domain.
"""
domain_bases(dom::Domain) = dom.bases

"""
    get_dim(dom::Domain) -> Int

Return the total dimensionality (sum of basis dimensions).
"""
get_dim(dom::Domain) = dom._dim

# ============================================================================
# Cached attribute helpers
# ============================================================================

"""
    _get_cached!(dom::Domain, key::Symbol, compute::Function)

Retrieve a cached attribute, computing it on first access.
"""
function _get_cached!(dom::Domain, key::Symbol, compute::Function)
    cached = get(dom._cache, key, nothing)
    if cached !== nothing
        return cached
    end
    val = compute()
    dom._cache[key] = val
    return val
end

# ============================================================================
# Volume
# ============================================================================

"""
    domain_volume(dom::Domain)

Return the product of the volumes of all bases in the domain.
"""
function domain_volume(dom::Domain)
    return _get_cached!(dom, :volume, () -> begin
        prod(volume(b) for b in dom.bases)
    end)
end

# ============================================================================
# bases_by_axis
# ============================================================================

"""
    bases_by_axis(dom::Domain) -> OrderedDict{Int, AbstractBasis}

Return an ordered mapping from axis index (1-based) to the basis that
covers that axis.
"""
function bases_by_axis(dom::Domain)
    return _get_cached!(dom, :bases_by_axis, () -> begin
        result = OrderedDict{Int, Any}()
        for basis in dom.bases
            first_ax = get_basis_axis(dom.dist, basis)
            d = get_dim(basis)
            for ax in first_ax:(first_ax + d - 1)
                result[ax] = basis
            end
        end
        result
    end)
end

# ============================================================================
# full_bases
# ============================================================================

"""
    full_bases(dom::Domain) -> Tuple

Return a tuple of length `dist.dim` where each slot is the basis covering
that axis, or `nothing` if no basis covers it.
"""
function full_bases(dom::Domain)
    return _get_cached!(dom, :full_bases, () -> begin
        dist_dim = get_dim(dom.dist)
        fb = Vector{Any}(nothing, dist_dim)
        for basis in dom.bases
            first_ax = get_basis_axis(dom.dist, basis)
            d = get_dim(basis)
            for ax in first_ax:(first_ax + d - 1)
                fb[ax] = basis
            end
        end
        Tuple(fb)
    end)
end

# ============================================================================
# bases_by_coord
# ============================================================================

"""
    bases_by_coord(dom::Domain) -> OrderedDict

Return an ordered mapping from coordinate (or coordinate system) to the
basis that covers it, or `nothing`.
"""
function bases_by_coord(dom::Domain)
    return _get_cached!(dom, :bases_by_coord, () -> begin
        result = OrderedDict{Any, Any}()
        # Initialise with all coords from the distributor
        for coord in get_coords(dom.dist)
            cs = coord.cs
            if cs === nothing || cs isa CartesianCoordinates
                result[coord] = nothing
            else
                result[cs] = nothing
            end
        end
        # Fill in with bases
        for basis in dom.bases
            result[get_coords(basis)] = basis
        end
        result
    end)
end

# ============================================================================
# dealias
# ============================================================================

"""
    domain_dealias(dom::Domain) -> Tuple

Return a tuple of dealias factors of length `dist.dim`.  Axes not covered
by any basis get a factor of `1`.
"""
function domain_dealias(dom::Domain)
    return _get_cached!(dom, :dealias, () -> begin
        dist_dim = get_dim(dom.dist)
        da = ones(dist_dim)
        for basis in dom.bases
            first_ax = get_basis_axis(dom.dist, basis)
            dt = dealias_tuple(basis)
            d = get_dim(basis)
            for sub in 1:d
                da[first_ax + sub - 1] = dt[sub]
            end
        end
        Tuple(da)
    end)
end

# ============================================================================
# substitute_basis
# ============================================================================

"""
    substitute_basis(dom::Domain, old_basis, new_basis) -> Domain

Return a new `Domain` where `old_basis` is replaced by `new_basis`.
If `old_basis` is not present, `new_basis` is simply appended.
"""
function substitute_basis(dom::Domain, old_basis, new_basis)
    new_bases_list = collect(Any, dom.bases)
    idx = findfirst(b -> b === old_basis, new_bases_list)
    if idx !== nothing
        deleteat!(new_bases_list, idx)
    end
    push!(new_bases_list, new_basis)
    return make_domain(dom.dist, new_bases_list)
end

# ============================================================================
# get_basis
# ============================================================================

"""
    get_basis(dom::Domain, coords_or_axis)

Retrieve the basis covering a coordinate, coordinate system, or axis index
(1-based integer).
"""
function get_basis(dom::Domain, axis::Integer)
    fb = full_bases(dom)
    return fb[axis]
end

function get_basis(dom::Domain, coords)
    ax = get_axis(dom.dist, coords)
    return full_bases(dom)[ax]
end

# ============================================================================
# get_basis_subaxis
# ============================================================================

"""
    get_basis_subaxis(dom::Domain, coord) -> Union{Int, Nothing}

Return the sub-axis index (1-based, within the basis) that corresponds to
`coord`.  Returns `nothing` if no basis covers that coordinate.
"""
function get_basis_subaxis(dom::Domain, coord)
    ax = get_axis(dom.dist, coord)
    for basis in dom.bases
        basis_ax = get_basis_axis(dom.dist, basis)
        d = get_dim(basis)
        if basis_ax <= ax < basis_ax + d
            return ax - basis_ax + 1  # 1-based sub-axis
        end
    end
    return nothing
end

# ============================================================================
# get_coord
# ============================================================================

"""
    get_coord(dom::Domain, name::AbstractString)

Retrieve a `Coordinate` from the domain by name.  Searches all bases.
"""
function get_coord(dom::Domain, name::AbstractString)
    for basis in dom.bases
        bc = get_coords(basis)
        # If the basis coords is a single Coordinate
        if bc isa CoordinateOrAzimuthal
            if bc.name == name
                return bc
            end
        else
            # It's a coordinate system — iterate its coords
            for basis_coord in get_coords(bc)
                if basis_coord.name == name
                    return basis_coord
                end
            end
        end
    end
    throw(ArgumentError("Coordinate name '$name' not in domain."))
end

# ============================================================================
# enumerate_unique_bases
# ============================================================================

"""
    enumerate_unique_bases(dom::Domain)

Return an iterator of `(axis, basis)` pairs for each unique basis (or
`nothing` slot) in `full_bases`.  Uses 1-based axis indexing.
"""
function enumerate_unique_bases(dom::Domain)
    fb = full_bases(dom)
    axes = Int[]
    unique_b = []
    for (ax, basis) in enumerate(fb)
        if basis === nothing || !(basis in unique_b)
            push!(axes, ax)
            push!(unique_b, basis)
        end
    end
    return zip(axes, unique_b)
end

# ============================================================================
# constant / nonconstant
# ============================================================================

"""
    domain_constant(dom::Domain) -> Tuple{Vararg{Bool}}

Return a tuple of boolean flags indicating whether each axis is constant
(i.e. has no basis variation).  Axes not covered by a basis are constant.
"""
function domain_constant(dom::Domain)
    return _get_cached!(dom, :constant, () -> begin
        dist_dim = get_dim(dom.dist)
        c = trues(dist_dim)
        for basis in dom.bases
            first_ax = get_basis_axis(dom.dist, basis)
            cf = constant_flags(basis)
            d = get_dim(basis)
            for sub in 1:d
                c[first_ax + sub - 1] = cf[sub]
            end
        end
        Tuple(c)
    end)
end

"""
    domain_nonconstant(dom::Domain) -> Tuple{Vararg{Bool}}

Return a tuple of boolean flags — the logical negation of
[`domain_constant`](@ref).
"""
function domain_nonconstant(dom::Domain)
    return _get_cached!(dom, :nonconstant, () -> begin
        Tuple(.!collect(domain_constant(dom)))
    end)
end

# ============================================================================
# mode_dependence
# ============================================================================

"""
    mode_dependence(dom::Domain) -> Tuple{Vararg{Bool}}

Return a tuple of dependence flags indicating whether each axis carries
mode dependence.
"""
function mode_dependence(dom::Domain)
    return _get_cached!(dom, :mode_dependence, () -> begin
        dist_dim = get_dim(dom.dist)
        dep = falses(dist_dim)
        for basis in dom.bases
            first_ax = get_basis_axis(dom.dist, basis)
            sd = subaxis_dependence_flags(basis)
            d = get_dim(basis)
            for sub in 1:d
                dep[first_ax + sub - 1] = sd[sub]
            end
        end
        Tuple(dep)
    end)
end

# ============================================================================
# coeff_shape / grid_shape / global_shape / chunk_shape / group_shape
# ============================================================================

"""
    coeff_shape(dom::Domain) -> Tuple

Compute the coefficient-space shape of the domain.
"""
function coeff_shape(dom::Domain)
    return _get_cached!(dom, :coeff_shape, () -> begin
        dist_dim = get_dim(dom.dist)
        scales = ntuple(_ -> 1, dist_dim)
        global_shape(dom, coeff_layout(dom.dist), scales)
    end)
end

"""
    grid_shape(dom::Domain, scales) -> Tuple

Compute the grid-space shape for the given scale factors.
Applies `remedy_scales` from the distributor before computation.
"""
function grid_shape(dom::Domain, scales)
    scales = remedy_scales(dom.dist, scales)
    return _grid_shape_cached(dom, scales)
end

"""
    _grid_shape_cached(dom::Domain, scales) -> Tuple

Internal cached grid-shape computation.
"""
function _grid_shape_cached(dom::Domain, scales)
    dist_dim = get_dim(dom.dist)
    shape = ones(Int, dist_dim)
    for basis in dom.bases
        ax = axis(basis)
        d = get_dim(basis)
        subscales = scales[ax:ax+d-1]
        subshape = grid_shape(basis, subscales)
        shape[ax:ax+d-1] .= subshape
    end
    return Tuple(shape)
end

"""
    global_shape(dom::Domain, layout, scales) -> Tuple

Compute the global data shape for a given layout and scale factors.
"""
function global_shape(dom::Domain, layout, scales)
    dist_dim = get_dim(dom.dist)
    shape = ones(Int, dist_dim)
    for basis in dom.bases
        first_ax = get_basis_axis(dom.dist, basis)
        d = get_dim(basis)
        basis_axes = first_ax:(first_ax + d - 1)
        gs = layout.grid_space[basis_axes]
        sc = scales[basis_axes]  # note: may be a tuple slice or array view
        shape[basis_axes] .= global_shape(basis, gs, sc)
    end
    return Tuple(shape)
end

"""
    chunk_shape(dom::Domain, layout) -> Tuple

Compute the chunk shape for a given layout.
"""
function chunk_shape(dom::Domain, layout)
    dist_dim = get_dim(dom.dist)
    shape = ones(Int, dist_dim)
    for basis in dom.bases
        first_ax = get_basis_axis(dom.dist, basis)
        d = get_dim(basis)
        basis_axes = first_ax:(first_ax + d - 1)
        gs = layout.grid_space[basis_axes]
        shape[basis_axes] .= chunk_shape(basis, gs)
    end
    return Tuple(shape)
end

"""
    group_shape(dom::Domain, layout) -> Tuple

Compute the group shape for a given layout.  Grid-space axes get size 1.
"""
function group_shape(dom::Domain, layout)
    dist_dim = get_dim(dom.dist)
    gs = ones(Int, dist_dim)
    for basis in dom.bases
        first_ax = get_basis_axis(dom.dist, basis)
        d = get_dim(basis)
        basis_axes = first_ax:(first_ax + d - 1)
        gs[basis_axes] .= group_shape_val(basis)
    end
    # Zero out group shape for grid-space axes
    for ax in 1:dist_dim
        if layout.grid_space[ax]
            gs[ax] = 1
        end
    end
    return Tuple(gs)
end

# ============================================================================
# Stub interface functions
#
# These functions define the interface that AbstractBasis and
# AbstractDistributor subtypes must implement.  They allow this module
# to compile and be tested structurally before concrete basis/distributor
# types exist.
# ============================================================================

# --- AbstractBasis interface stubs ---

"""
    coordsys(basis::AbstractBasis)

Return the coordinate or coordinate system that the basis spans.
"""
function coordsys end

"""
    dealias_tuple(basis::AbstractBasis) -> Tuple

Return the per-subaxis dealias factors as a tuple.
"""
function dealias_tuple end

"""
    volume(basis::AbstractBasis)

Return the domain volume covered by this basis.
"""
function volume end

"""
    constant_flags(basis::AbstractBasis) -> Tuple{Vararg{Bool}}

Return a tuple of booleans indicating constant sub-axes.
"""
function constant_flags end

"""
    subaxis_dependence_flags(basis::AbstractBasis) -> Tuple{Vararg{Bool}}

Return a tuple of booleans indicating sub-axis mode dependence.
"""
function subaxis_dependence_flags end

"""
    group_shape_val(basis::AbstractBasis) -> Tuple

Return the group shape for this basis.
"""
function group_shape_val end

"""
    axis(basis::AbstractBasis) -> Int

Return the first axis index (1-based) of this basis in the distributor.
"""
function axis end

# --- AbstractDistributor interface stubs ---

"""
    get_basis_axis(dist::AbstractDistributor, basis) -> Int

Return the first axis index (1-based) of `basis` within `dist`.
"""
function get_basis_axis end

"""
    get_axis(dist::AbstractDistributor, coords) -> Int

Return the axis index (1-based) for a coordinate or coordinate system
within `dist`.
"""
function get_axis end

"""
    coeff_layout(dist::AbstractDistributor)

Return the coefficient-space layout object for `dist`.
"""
function coeff_layout end

"""
    remedy_scales(dist::AbstractDistributor, scales)

Canonicalise scale factors (e.g. expand scalars to tuples).
"""
function remedy_scales end

# ============================================================================
# Exports
# ============================================================================

export AbstractBasis,
       AbstractDistributor,
       Domain,
       make_domain,
       preprocess_domain_args,
       domain_dist,
       domain_bases,
       domain_volume,
       bases_by_axis,
       full_bases,
       bases_by_coord,
       domain_dealias,
       substitute_basis,
       get_basis,
       get_basis_subaxis,
       get_coord,
       enumerate_unique_bases,
       domain_constant,
       domain_nonconstant,
       mode_dependence,
       coeff_shape,
       grid_shape,
       global_shape,
       chunk_shape,
       group_shape,
       coordsys,
       dealias_tuple,
       volume,
       constant_flags,
       subaxis_dependence_flags,
       group_shape_val,
       axis,
       get_basis_axis,
       get_axis,
       coeff_layout,
       remedy_scales
