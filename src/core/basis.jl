"""
    Spectral basis types for Dedalus.jl (1D interval bases)

Julia translation of the 1D interval-basis portions of `dedalus/core/basis.py`.
Provides the abstract `Basis` and `IntervalBasis` types plus concrete bases:
`Jacobi`, `ChebyshevT`, `ChebyshevU`, `ChebyshevV`, `Legendre`,
`Ultraspherical`, `RealFourier`, `ComplexFourier`, `Fourier`, and
`CardinalBasis`.

## Type hierarchy

    AbstractBasis (from domain.jl)
    +-- Basis (abstract; root for all spectral bases in this file)
        +-- CardinalBasis (cardinal function basis, no transforms)
        +-- IntervalBasis (abstract; 1D bases on intervals)
            +-- Jacobi (general Jacobi polynomial basis)
            +-- FourierBase (abstract; base for Fourier bases)
                +-- ComplexFourier (complex exponential Fourier basis)
                +-- RealFourier (real sine/cosine Fourier basis)

## Key translation choices

- Python `CachedClass` metaclass --> constructor-level `Dict`-based caching
  via `_basis_cache` dictionaries (one per concrete type).
- Python `@CachedAttribute` --> mutable `_cache::Dict{Symbol,Any}` field
  with accessor helpers.
- Python `@CachedMethod` --> `Dict`-keyed memoisation inside each struct.
- Python `@property` --> Julia accessor functions.
- Python `isinstance(x, SomeClass)` --> Julia `isa(x, SomeType)`.
- Python `NotImplemented` (from binary ops) --> `nothing` (caller must check).
- Python `np` arrays --> Julia arrays.
- Python `scipy.sparse` --> `SparseArrays`.
- 0-based indexing --> 1-based where appropriate (grid arrays, etc.).
- Python `__add__`, `__mul__`, `__matmul__` --> Julia `basis_add`, `basis_mul`,
  `basis_matmul` (non-operator functions) plus overloaded `+`, `*` on Basis.

## Dependencies

Uses: tools/jacobi.jl, tools/clenshaw.jl, tools/array.jl, tools/cache.jl,
      tools/general.jl, core/coords.jl, core/domain.jl
"""

using LinearAlgebra
using SparseArrays
using FFTW

# ============================================================================
# Import tools (assumed already included in the module; we reference the
# functions directly).
# ============================================================================

# From tools/jacobi.jl:  mass, build_grid, build_weights, build_polynomials,
#                         conversion_matrix, differentiation_matrix,
#                         jacobi_matrix, integration_vector
# From tools/clenshaw.jl: matrix_clenshaw, jacobi_recursion
# From tools/array.jl:    reshape_vector, axslice, permute_axis, apply_matrix,
#                          interleave_matrices
# From tools/cache.jl:    CachedClass, cached_construct
# From tools/general.jl:  DeferredTuple, unify
# From core/coords.jl:    Coordinate, AzimuthalCoordinate, CoordinateOrAzimuthal,
#                          CartesianCoordinates, check_bounds
# From core/domain.jl:    AbstractBasis, AbstractDistributor, Domain, make_domain,
#                          get_dim, get_basis_axis, coordsys, volume

import Logging
const _basis_logger = Logging.current_logger()

# ============================================================================
# AffineCOV  (Affine Change of Variables)
# ============================================================================

"""
    AffineCOV

Affine change-of-variables for remapping coordinate bounds.

Maps between *native* bounds (e.g. [-1,1] for Jacobi, [0,2pi] for Fourier)
and *problem* bounds (user-specified interval).

# Fields
- `native_bounds::Tuple{Float64,Float64}`
- `problem_bounds::Tuple{Float64,Float64}`
- `native_left::Float64`
- `native_right::Float64`
- `native_length::Float64`
- `native_center::Float64`
- `problem_left::Float64`
- `problem_right::Float64`
- `problem_length::Float64`
- `problem_center::Float64`
- `stretch::Float64`
"""
struct AffineCOV
    native_bounds::Tuple{Float64,Float64}
    problem_bounds::Tuple{Float64,Float64}
    native_left::Float64
    native_right::Float64
    native_length::Float64
    native_center::Float64
    problem_left::Float64
    problem_right::Float64
    problem_length::Float64
    problem_center::Float64
    stretch::Float64

    function AffineCOV(native_bounds::Tuple, problem_bounds::Tuple)
        nl = Float64(native_bounds[1])
        nr = Float64(native_bounds[2])
        n_len = nr - nl
        n_cen = (nl + nr) / 2
        pl = Float64(problem_bounds[1])
        pr = Float64(problem_bounds[2])
        p_len = pr - pl
        p_cen = (pl + pr) / 2
        stretch = p_len / n_len
        new((nl, nr), (pl, pr), nl, nr, n_len, n_cen, pl, pr, p_len, p_cen, stretch)
    end
end

"""
    problem_coord(cov::AffineCOV, native_coord)

Convert native coordinates to problem coordinates.
`native_coord` can be a number, vector, or a string ("left"/"right"/"center").
"""
function problem_coord(cov::AffineCOV, native_coord::AbstractString)
    if native_coord in ("left", "lower")
        return cov.problem_left
    elseif native_coord in ("right", "upper")
        return cov.problem_right
    elseif native_coord in ("center", "middle")
        return cov.problem_center
    else
        throw(ArgumentError("String coordinate '$(native_coord)' not recognized."))
    end
end

function problem_coord(cov::AffineCOV, native_coord::Number)
    neutral = (native_coord - cov.native_left) / cov.native_length
    return cov.problem_left + neutral * cov.problem_length
end

function problem_coord(cov::AffineCOV, native_coord::AbstractVector)
    neutral = (native_coord .- cov.native_left) ./ cov.native_length
    return cov.problem_left .+ neutral .* cov.problem_length
end

"""
    native_coord(cov::AffineCOV, prob_coord)

Convert problem coordinates to native coordinates.
`prob_coord` can be a number, vector, or a string ("left"/"right"/"center").
"""
function native_coord(cov::AffineCOV, prob_coord::AbstractString)
    if prob_coord in ("left", "lower")
        return cov.native_left
    elseif prob_coord in ("right", "upper")
        return cov.native_right
    elseif prob_coord in ("center", "middle")
        return cov.native_center
    else
        throw(ArgumentError("String coordinate '$(prob_coord)' not recognized."))
    end
end

function native_coord(cov::AffineCOV, prob_coord::Number)
    neutral = (prob_coord - cov.problem_left) / cov.problem_length
    return cov.native_left + neutral * cov.native_length
end

function native_coord(cov::AffineCOV, prob_coord::AbstractVector)
    neutral = (prob_coord .- cov.problem_left) ./ cov.problem_length
    return cov.native_left .+ neutral .* cov.native_length
end

# ============================================================================
# Basis  (abstract base)
# ============================================================================

"""
    Basis <: AbstractBasis

Abstract base type for all spectral bases.  Defines the interface that every
basis must implement.

## Required fields for concrete subtypes
- `coords` — the coordinate(s) this basis spans (a `Coordinate` or coord system)

## Interface methods (to be implemented by subtypes)
- `basis_coord(b)` → the Coordinate object
- `basis_coordsys(b)` → coordinate or coordinate system
- `basis_size(b)` → number of modes
- `basis_shape(b)` → tuple of sizes per sub-axis
- `basis_dealias(b)` → tuple of dealias factors
- `basis_dim(b)` → dimensionality (number of sub-axes)
- `basis_constant(b)` → tuple of bools (constant per sub-axis)
- `basis_group_shape(b)` → group shape tuple
- `basis_subaxis_dependence(b)` → tuple of bools
- `global_shape(b, grid_space, scales)` → global data shape
- `chunk_shape(b, grid_space)` → chunk shape
- `grid_shape(b, scales)` → grid-space shape
- `forward_transform(b, field, axis, gdata, cdata)` → grid-to-coeff
- `backward_transform(b, field, axis, cdata, gdata)` → coeff-to-grid
- `basis_add(b1, b2)` → output basis for addition
- `basis_mul(b1, b2)` → output basis for multiplication
- `basis_rmatmul(ncc, self)` → output basis for NCC multiplication
"""
abstract type Basis <: AbstractBasis end

# -- Domain construction helper --

"""
    basis_domain(b::Basis, dist)

Construct (or retrieve cached) a Domain for a single basis.
"""
function basis_domain(b::Basis, dist)
    return make_domain(dist, (b,))
end

# -- Default implementations of AbstractBasis interface used by Domain --

get_dim(b::Basis) = basis_dim(b)
coordsys(b::Basis) = basis_coordsys(b)
dealias_tuple(b::Basis) = basis_dealias(b)
constant_flags(b::Basis) = basis_constant(b)
subaxis_dependence_flags(b::Basis) = basis_subaxis_dependence(b)
group_shape_val(b::Basis) = basis_group_shape(b)

"""
    grid_shape(b::Basis, scales)

Grid-space shape for the basis at the given scale factors.
"""
function grid_shape(b::Basis, scales)
    shape_arr = [Int(ceil(s * n)) for (s, n) in zip(scales, basis_shape(b))]
    orig = collect(basis_shape(b))
    for i in eachindex(orig)
        if orig[i] == 1
            shape_arr[i] = 1
        end
    end
    return Tuple(shape_arr)
end

# -- Basis algebra: default dispatching --

"""
    basis_add(a::Basis, b)

Return the output basis for `a + b`.  Returns `nothing` if incompatible.
`b` can be `nothing` (constant) or another Basis.
"""
function basis_add(a::Basis, b)
    if b === nothing || b === a
        return a
    end
    return nothing  # incompatible by default
end

"""
    basis_radd(a::Basis, b)

Reverse addition: `b + a`.  Delegates to `basis_add(a, b)`.
"""
basis_radd(a::Basis, b) = basis_add(a, b)

"""
    basis_mul(a::Basis, b)

Return the output basis for `a * b`.  Returns `nothing` if incompatible.
"""
function basis_mul(a::Basis, b)
    if b === nothing || b === a
        return a
    end
    return nothing
end

"""
    basis_rmul(a::Basis, b)

Reverse multiplication: `b * a`.  Delegates to `basis_mul(a, b)`.
"""
basis_rmul(a::Basis, b) = basis_mul(a, b)

"""
    basis_matmul(ncc::Basis, operand)

Return the output basis for NCC (ncc) @ operand multiplication.
If `operand` is `nothing`, returns `ncc`.  Otherwise delegates to
`basis_rmatmul(operand, ncc)`.
"""
function basis_matmul(ncc::Basis, operand)
    if operand === nothing
        return ncc
    else
        return basis_rmatmul(operand, ncc)
    end
end

"""
    basis_rmatmul(operand::Basis, ncc)

Reverse NCC multiplication: determines output basis when `ncc @ operand`.
Default returns `nothing` (incompatible).
"""
function basis_rmatmul(operand::Basis, ncc)
    if ncc === nothing || ncc === operand
        return operand
    end
    return nothing
end

# -- NCC matrix building --

"""
    ncc_matrix(ncc_basis, arg_basis, out_basis, coeffs; cutoff=1e-6)

Build NCC matrix via direct summation over coefficient modes.
"""
function ncc_matrix(ncc_basis::Basis, arg_basis, out_basis, coeffs::AbstractVector; cutoff::Float64=1e-6)
    N = length(coeffs)
    total = nothing
    for i in 1:N
        coeff = coeffs[i]
        if i == 1
            mat = product_matrix(ncc_basis, arg_basis, out_basis, i)
            total = spzeros(Float64, size(mat, 1), size(mat, 2))
            if isa(coeff, AbstractArray)
                total = kron(total, coeff)
            end
        end
        if isa(coeff, AbstractArray) || abs(coeff) > cutoff
            mat = product_matrix(ncc_basis, arg_basis, out_basis, i)
            if isa(coeff, AbstractArray)
                total = total + kron(mat, coeff)
            else
                total = total + coeff * mat
            end
        end
    end
    return total
end

"""
    product_matrix(b::Basis, arg_basis, out_basis, i)

Default product matrix for mode `i` (1-based).  When `arg_basis` is `nothing`,
returns a sparse column vector with 1 at position `i`.
"""
function product_matrix(b::Basis, arg_basis, out_basis, i::Integer)
    if arg_basis === nothing
        N = basis_size(b)
        return sparse([i], [1], [1.0], N, 1)
    else
        error("product_matrix not implemented for $(typeof(b)) with arg_basis=$(typeof(arg_basis))")
    end
end

# -- clone_with pattern --

"""
    clone_with(b::Basis; kwargs...)

Create a copy of basis `b` with some fields replaced.  Subtypes should
override this to call their own constructor with updated arguments.
"""
function clone_with end

# ============================================================================
# CardinalBasis
# ============================================================================

"""
    CardinalBasis <: Basis

Cardinal function basis -- already in grid space, no transforms needed.
Used for discrete data without spectral representation.

# Fields
- `coord::CoordinateOrAzimuthal`
- `_size::Int`
- `_cache::Dict{Symbol,Any}`
"""
mutable struct CardinalBasis <: Basis
    coord::CoordinateOrAzimuthal
    _size::Int
    _cache::Dict{Symbol,Any}
end

# -- Constructor caching --
const _cardinal_basis_cache = Dict{Tuple,WeakRef}()

"""
    CardinalBasis(coord, size)

Construct (or retrieve cached) a CardinalBasis.
"""
function CardinalBasis(coord::CoordinateOrAzimuthal, size::Integer)
    key = (coord, Int(size))
    wr = get(_cardinal_basis_cache, key, nothing)
    if wr !== nothing
        inst = wr.value
        if inst !== nothing
            return inst::CardinalBasis
        end
    end
    inst = CardinalBasis(coord, Int(size), Dict{Symbol,Any}())
    _cardinal_basis_cache[key] = WeakRef(inst)
    return inst
end

# -- Interface implementations --

basis_coord(b::CardinalBasis) = b.coord
basis_coordsys(b::CardinalBasis) = b.coord
basis_size(b::CardinalBasis) = b._size
basis_shape(b::CardinalBasis) = (b._size,)
basis_dealias(b::CardinalBasis) = (1,)
basis_dim(::CardinalBasis) = 1
basis_constant(::CardinalBasis) = (false,)
basis_group_shape(::CardinalBasis) = (1,)
basis_subaxis_dependence(::CardinalBasis) = (false,)
volume(b::CardinalBasis) = 0.0  # not well-defined

function basis_add(a::CardinalBasis, b)
    if b === nothing || b === a
        return a
    end
    return nothing
end

function basis_mul(a::CardinalBasis, b)
    if b === nothing || b === a
        return a
    end
    return nothing
end

function basis_rmatmul(a::CardinalBasis, ncc)
    if ncc === nothing || ncc === a
        return a
    end
    return nothing
end

function elements_to_groups(b::CardinalBasis, grid_space, elements)
    return elements
end

function valid_elements(b::CardinalBasis, tensorsig, grid_space, elements)
    vshape = tuple((get_dim(cs) for cs in tensorsig)..., size(elements[1])...)
    return ones(Bool, vshape)
end

function matrix_dependence(b::CardinalBasis, matrix_coupling)
    return matrix_coupling
end

function global_grids(b::CardinalBasis, dist, scales)
    return (global_grid(b, dist, scales[1]),)
end

function global_grid(b::CardinalBasis, dist, scale)
    if scale != 1
        throw(ErrorException("Cardinal basis only supports scale=1."))
    end
    N = b._size
    grid = collect(Float64, 0:N-1)
    return reshape_vector(grid, get_dim(dist), get_basis_axis(dist, b))
end

function local_grids(b::CardinalBasis, dist, scales)
    return (local_grid(b, dist, scales[1]),)
end

function local_grid(b::CardinalBasis, dist, scale)
    if scale != 1
        throw(ErrorException("Cardinal basis only supports scale=1."))
    end
    return collect(Float64, 0:b._size-1)
end

function global_shape(b::CardinalBasis, grid_space, scales)
    return basis_shape(b)
end

function chunk_shape(b::CardinalBasis, grid_space)
    return (1,)
end

function forward_transform(b::CardinalBasis, field, axis, gdata, cdata)
    copyto!(cdata, gdata)
end

function backward_transform(b::CardinalBasis, field, axis, cdata, gdata)
    copyto!(gdata, cdata)
end

function Base.show(io::IO, b::CardinalBasis)
    print(io, "CardinalBasis($(b.coord), $(b._size))")
end

# ============================================================================
# IntervalBasis  (abstract)
# ============================================================================

"""
    IntervalBasis <: Basis

Abstract base type for 1D bases on intervals (Jacobi, Fourier, etc.).

Concrete subtypes must have fields:
- `coord::CoordinateOrAzimuthal`
- `_size::Int`
- `bounds::Tuple{Float64,Float64}`
- `_dealias::Tuple{Float64}`
- `COV::AffineCOV`
- `_cache::Dict{Symbol,Any}`
"""
abstract type IntervalBasis <: Basis end

"""Alias for IntervalBasis, matching the naming convention for abstract types."""
const AbstractIntervalBasis = IntervalBasis

# -- Common interface --

basis_dim(::IntervalBasis) = 1
basis_subaxis_dependence(::IntervalBasis) = (false,)
basis_constant(::IntervalBasis) = (false,)

basis_coord(b::IntervalBasis) = b.coord
basis_coordsys(b::IntervalBasis) = b.coord
basis_size(b::IntervalBasis) = b._size
basis_shape(b::IntervalBasis) = (b._size,)
basis_dealias(b::IntervalBasis) = b._dealias
volume(b::IntervalBasis) = b.bounds[2] - b.bounds[1]

function matrix_dependence(b::IntervalBasis, matrix_coupling)
    return matrix_coupling
end

# -- Grid computation --

"""
    _native_grid(b::IntervalBasis, scale)

Return the native flat global grid for the basis at the given scale.
Must be implemented by concrete subtypes.
"""
function _native_grid end

function global_grids(b::IntervalBasis, dist, scales)
    return (global_grid(b, dist, scales[1]),)
end

function global_grid(b::IntervalBasis, dist, scale)
    ng = _native_grid(b, scale)
    pg = problem_coord(b.COV, ng)
    return reshape_vector(pg, get_dim(dist), get_basis_axis(dist, b))
end

function local_grids(b::IntervalBasis, dist, scales)
    return (local_grid(b, dist, scales[1]),)
end

function local_grid(b::IntervalBasis, dist, scale)
    # For serial usage, local = global (no distribution)
    ng = _native_grid(b, scale)
    pg = problem_coord(b.COV, ng)
    return reshape_vector(pg, get_dim(dist), get_basis_axis(dist, b))
end

function global_grid_spacing(b::IntervalBasis, dist, scale)
    grid = global_grid(b, dist, scale)
    # Numerical gradient along the basis axis
    ax = get_basis_axis(dist, b)  # 1-based
    # Simple central differences
    n = size(grid, ax)
    if n <= 1
        return zeros(size(grid))
    end
    result = similar(grid)
    idx_before(k) = ntuple(d -> d == ax ? max(1, k-1) : Colon(), ndims(grid))
    idx_at(k) = ntuple(d -> d == ax ? k : Colon(), ndims(grid))
    idx_after(k) = ntuple(d -> d == ax ? min(n, k+1) : Colon(), ndims(grid))
    for k in 1:n
        if k == 1
            result[idx_at(k)...] .= grid[idx_after(k)...] .- grid[idx_at(k)...]
        elseif k == n
            result[idx_at(k)...] .= grid[idx_at(k)...] .- grid[idx_before(k)...]
        else
            result[idx_at(k)...] .= (grid[idx_after(k)...] .- grid[idx_before(k)...]) ./ 2
        end
    end
    return result
end

function local_modes(b::IntervalBasis, dist)
    # For serial usage
    elems = collect(0:b._size-1)
    return reshape_vector(elems, get_dim(dist), get_basis_axis(dist, b))
end

function global_shape(b::IntervalBasis, grid_space, scales)
    if grid_space isa Tuple || grid_space isa AbstractVector
        if grid_space[1]
            return grid_shape(b, scales)
        else
            return basis_shape(b)
        end
    else
        if grid_space
            return grid_shape(b, scales)
        else
            return basis_shape(b)
        end
    end
end

function chunk_shape(b::IntervalBasis, grid_space)
    if grid_space isa Tuple || grid_space isa AbstractVector
        gs = grid_space[1]
    else
        gs = grid_space
    end
    if gs
        return (1,)
    else
        return basis_group_shape(b)
    end
end

function forward_transform(b::IntervalBasis, field, axis, gdata, cdata)
    data_axis = length(field.tensorsig) + axis
    grid_size = size(gdata, data_axis)
    plan = transform_plan(b, field.dist, grid_size)
    forward!(plan, gdata, cdata, data_axis)
end

function backward_transform(b::IntervalBasis, field, axis, cdata, gdata)
    data_axis = length(field.tensorsig) + axis
    grid_size = size(gdata, data_axis)
    plan = transform_plan(b, field.dist, grid_size)
    backward!(plan, cdata, gdata, data_axis)
end

"""
    transform_plan(b::IntervalBasis, dist, grid_size)

Return (or build and cache) the transform plan for the given grid size.
Must be implemented by concrete subtypes.
"""
function transform_plan end

# ============================================================================
# Jacobi polynomial basis
# ============================================================================

"""
    JacobiBasis <: IntervalBasis

Jacobi polynomial basis P^{(a,b)} on an interval.

# Fields
- `coord::CoordinateOrAzimuthal` — coordinate
- `_size::Int` — number of modes
- `bounds::Tuple{Float64,Float64}` — problem coordinate bounds
- `a::Float64` — first Jacobi parameter (alpha)
- `b::Float64` — second Jacobi parameter (beta)
- `a0::Float64` — grid alpha parameter
- `b0::Float64` — grid beta parameter
- `_dealias::Tuple{Float64}` — dealias factor
- `library::String` — transform library name
- `COV::AffineCOV` — change of variables
- `grid_params::Tuple` — (coord, bounds, a0, b0, dealias, library)
- `constant_mode_value::Float64` — value of constant mode
- `_cache::Dict{Symbol,Any}`
- `_transform_cache::Dict{Any,Any}`
- `_product_matrix_cache::Dict{Any,Any}`
"""
mutable struct JacobiBasis <: IntervalBasis
    coord::CoordinateOrAzimuthal
    _size::Int
    bounds::Tuple{Float64,Float64}
    a::Float64
    b::Float64
    a0::Float64
    b0::Float64
    _dealias::Tuple{Float64}
    library::String
    COV::AffineCOV
    grid_params::Tuple
    constant_mode_value::Float64
    _cache::Dict{Symbol,Any}
    _transform_cache::Dict{Any,Any}
    _product_matrix_cache::Dict{Any,Any}
end

const JACOBI_NATIVE_BOUNDS = (-1.0, 1.0)
const JACOBI_DEFAULT_DCT = "fftw_dct"
const JACOBI_DEFAULT_LIBRARY = "matrix"

# -- Constructor caching --
const _jacobi_cache = Dict{Any,WeakRef}()

"""
    _preprocess_jacobi_args(coord, size, bounds, a, b, a0, b0, dealias, library)

Preprocess and canonicalize arguments for Jacobi basis construction.
Returns a canonical tuple of arguments.
"""
function _preprocess_jacobi_args(coord, size, bounds, a, b, a0, b0, dealias, library)
    if !(coord isa CoordinateOrAzimuthal)
        throw(ArgumentError("Jacobi coord must be a Coordinate object."))
    end
    size = Int(size)
    if size <= 0
        throw(ArgumentError("Jacobi size must be positive."))
    end
    bounds = (Float64(bounds[1]), Float64(bounds[2]))
    if length(bounds) != 2
        throw(ArgumentError("Jacobi bounds must have length 2."))
    end
    a = Float64(a)
    b = Float64(b)
    if a0 === nothing
        a0 = a
    end
    a0 = Float64(a0)
    if b0 === nothing
        b0 = b
    end
    b0 = Float64(b0)
    if isa(dealias, Number)
        dealias = (Float64(dealias),)
    else
        dealias = (Float64(dealias[1]),)
    end
    if library === nothing
        if a0 == b0 == -0.5
            library = JACOBI_DEFAULT_DCT
        else
            library = JACOBI_DEFAULT_LIBRARY
        end
    end
    return (coord, size, bounds, a, b, a0, b0, dealias, library)
end

"""
    Jacobi(coord, size, bounds, a, b; a0=nothing, b0=nothing, dealias=(1,), library=nothing)

Construct a Jacobi polynomial basis.  Uses constructor caching so that
identical parameter sets return the same object.
"""
function Jacobi(coord::CoordinateOrAzimuthal, size::Integer, bounds, a, b;
                a0=nothing, b0=nothing, dealias=(1,), library=nothing)
    args = _preprocess_jacobi_args(coord, size, bounds, a, b, a0, b0, dealias, library)
    coord_p, size_p, bounds_p, a_p, b_p, a0_p, b0_p, dealias_p, library_p = args

    # Cache lookup
    cache_key = (coord_p, size_p, bounds_p, a_p, b_p, a0_p, b0_p, dealias_p, library_p)
    wr = get(_jacobi_cache, cache_key, nothing)
    if wr !== nothing
        inst = wr.value
        if inst !== nothing
            return inst::JacobiBasis
        end
    end

    # Construct
    check_bounds(coord_p, bounds_p)
    cov = AffineCOV(JACOBI_NATIVE_BOUNDS, bounds_p)
    grid_params = (coord_p, bounds_p, a0_p, b0_p, dealias_p, library_p)
    cmv = 1.0 / sqrt(mass(a_p, b_p))

    inst = JacobiBasis(
        coord_p, size_p, bounds_p, a_p, b_p, a0_p, b0_p,
        dealias_p, library_p, cov, grid_params, cmv,
        Dict{Symbol,Any}(), Dict{Any,Any}(), Dict{Any,Any}()
    )
    _jacobi_cache[cache_key] = WeakRef(inst)
    return inst
end

# -- Interface --

basis_group_shape(::JacobiBasis) = (1,)

function _native_grid(b::JacobiBasis, scale)
    N = grid_shape(b, (scale,))[1]
    return build_grid(N, b.a0, b.b0)
end

function Base.show(io::IO, b::JacobiBasis)
    print(io, "Jacobi($(b.coord), $(b._size), a0=$(b.a0), b0=$(b.b0), a=$(b.a), b=$(b.b), dealias=$(b._dealias[1]))")
end

# -- Basis algebra --

function basis_add(a::JacobiBasis, other)
    if other === nothing || other === a
        return a
    end
    if other isa JacobiBasis
        if a.grid_params == other.grid_params
            sz = max(a._size, other._size)
            a_new = max(a.a, other.a)
            b_new = max(a.b, other.b)
            return clone_with(a; size=sz, a=a_new, b=b_new)
        end
    end
    return nothing
end

function basis_mul(a::JacobiBasis, other)
    if other === nothing || other === a
        return a
    end
    if other isa JacobiBasis
        if a.grid_params == other.grid_params
            sz = max(a._size, other._size)
            # Take grid (a0, b0) for minimal conversions
            return clone_with(a; size=sz, a=a.a0, b=a.b0)
        end
    end
    return nothing
end

function basis_rmatmul(operand::JacobiBasis, ncc)
    # NCC (ncc) * operand (self)
    if ncc === nothing || ncc === operand
        return operand
    end
    if ncc isa JacobiBasis
        if operand.grid_params == ncc.grid_params
            sz = max(operand._size, ncc._size)
            # Take operand (a, b) for minimal conversions
            return clone_with(operand; size=sz, a=operand.a, b=operand.b)
        end
    end
    return nothing
end

function clone_with(basis::JacobiBasis; kwargs...)
    # Extract with explicit field mapping to handle the `b` name conflict
    kw = Dict{Symbol,Any}(kwargs)
    sz = get(kw, :size, basis._size)
    a_val = get(kw, :a, basis.a)
    b_val = get(kw, :b, basis.b)
    a0_val = get(kw, :a0, basis.a0)
    b0_val = get(kw, :b0, basis.b0)
    coord_val = get(kw, :coord, basis.coord)
    bounds_val = get(kw, :bounds, basis.bounds)
    dealias_val = get(kw, :dealias, basis._dealias)
    library_val = get(kw, :library, basis.library)
    return Jacobi(coord_val, sz, bounds_val, a_val, b_val;
                  a0=a0_val, b0=b0_val, dealias=dealias_val, library=library_val)
end

function elements_to_groups(b::JacobiBasis, grid_space, elements)
    return elements
end

function valid_elements(b::JacobiBasis, tensorsig, grid_space, elements)
    vshape = tuple((get_dim(cs) for cs in tensorsig)..., size(elements[1])...)
    return ones(Bool, vshape)
end

# -- Jacobi matrix (tridiagonal recurrence matrix) --

"""
    jacobi_recurrence_matrix(b::JacobiBasis; size=nothing)

Return the Jacobi tridiagonal matrix of the given size (defaults to b._size).
"""
function jacobi_recurrence_matrix(b::JacobiBasis; size=nothing)
    if size === nothing
        size = b._size
    end
    return jacobi_matrix(size, b.a, b.b)
end

# -- Product matrix via Clenshaw --

function product_matrix(b::JacobiBasis, arg_basis, out_basis, i::Integer)
    if arg_basis === nothing
        return invoke(product_matrix, Tuple{Basis, Any, Any, Integer}, b, arg_basis, out_basis, i)
    end
    # Cache lookup
    cache_key = (objectid(arg_basis), objectid(out_basis), i)
    cached = get(b._product_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    # Build via Clenshaw
    coeffs = zeros(i)
    coeffs[i] = 1.0
    mat = _last_axis_component_ncc_matrix(b, arg_basis, out_basis, coeffs)
    b._product_matrix_cache[cache_key] = mat
    return mat
end

"""
    _last_axis_component_ncc_matrix(ncc_basis::JacobiBasis, arg_basis, out_basis, coeffs; cutoff=0.0)

Build NCC component matrix via Clenshaw algorithm on Jacobi polynomials.
"""
function _last_axis_component_ncc_matrix(ncc_basis::JacobiBasis, arg_basis, out_basis, coeffs::AbstractVector;
                                         cutoff::Float64=0.0)
    if arg_basis === nothing
        return ncc_matrix(ncc_basis, arg_basis, out_basis, coeffs; cutoff=cutoff)
    end
    # Jacobi parameters
    a_ncc = ncc_basis.a
    b_ncc = ncc_basis.b
    N = arg_basis._size
    da = Int(round(out_basis.a - arg_basis.a))
    db = Int(round(out_basis.b - arg_basis.b))
    # Pad for dealiasing with conversion
    Nmat = 3 * ((N + 1) ÷ 2) + min((N + 1) ÷ 2, (da + db + 1) ÷ 2)
    J = jacobi_recurrence_matrix(arg_basis; size=Nmat)
    A, B = jacobi_recursion(Nmat, a_ncc, b_ncc, J)
    # f0 = P_0^{(a_ncc, b_ncc)}(1) * I
    # P_0 at x=1 is always 1 for normalized Jacobi polynomials in dedalus_sphere
    p0 = build_polynomials(1, a_ncc, b_ncc, [1.0])[1]
    Imat = sparse(1.0I, Nmat, Nmat)
    f0 = p0 * Imat
    mat = matrix_clenshaw(coeffs, A, B, f0, cutoff)
    # Apply conversion matrix
    conv = conversion_matrix(Nmat, arg_basis.a, arg_basis.b, out_basis.a, out_basis.b)
    mat = conv * mat
    return mat[1:N, 1:N]
end

# -- Derivative basis --

"""
    derivative_basis(b::JacobiBasis; order=1)

Return the Jacobi basis corresponding to the `order`-th derivative.
Raises both parameters by `order`.
"""
function derivative_basis(b::JacobiBasis; order::Int=1)
    return clone_with(b; a=b.a + order, b=b.b + order)
end

# ============================================================================
# Jacobi convenience constructors  (Legendre, Ultraspherical, Chebyshev)
# ============================================================================

"""
    Legendre(coord, size, bounds; dealias=(1,), library=nothing)

Construct a Legendre polynomial basis (Jacobi with a=b=0).
"""
function Legendre(coord::CoordinateOrAzimuthal, size::Integer, bounds;
                  dealias=(1,), library=nothing, kwargs...)
    return Jacobi(coord, size, bounds, 0.0, 0.0; dealias=dealias, library=library, kwargs...)
end

"""
    Ultraspherical(coord, size, bounds; alpha, alpha0=nothing, dealias=(1,), library=nothing)

Construct an Ultraspherical polynomial basis (Jacobi with a=b=alpha-1/2).
"""
function Ultraspherical(coord::CoordinateOrAzimuthal, size::Integer, bounds;
                        alpha, alpha0=nothing, dealias=(1,), library=nothing, kwargs...)
    if alpha0 === nothing
        alpha0 = alpha
    end
    a = alpha - 0.5
    b = alpha - 0.5
    a0 = alpha0 - 0.5
    b0 = alpha0 - 0.5
    return Jacobi(coord, size, bounds, a, b; a0=a0, b0=b0, dealias=dealias, library=library, kwargs...)
end

"""
    ChebyshevT(coord, size, bounds; dealias=(1,), library=nothing)

Construct a Chebyshev-T (first kind) basis (Ultraspherical with alpha=0).
"""
function ChebyshevT(coord::CoordinateOrAzimuthal, size::Integer, bounds;
                    dealias=(1,), library=nothing, kwargs...)
    return Ultraspherical(coord, size, bounds; alpha=0, dealias=dealias, library=library, kwargs...)
end

"""
    ChebyshevU(coord, size, bounds; dealias=(1,), library=nothing)

Construct a Chebyshev-U (second kind) basis (Ultraspherical with alpha=1).
"""
function ChebyshevU(coord::CoordinateOrAzimuthal, size::Integer, bounds;
                    dealias=(1,), library=nothing, kwargs...)
    return Ultraspherical(coord, size, bounds; alpha=1, dealias=dealias, library=library, kwargs...)
end

"""
    ChebyshevV(coord, size, bounds; dealias=(1,), library=nothing)

Construct a ChebyshevV basis (Ultraspherical with alpha=2).
"""
function ChebyshevV(coord::CoordinateOrAzimuthal, size::Integer, bounds;
                    dealias=(1,), library=nothing, kwargs...)
    return Ultraspherical(coord, size, bounds; alpha=2, dealias=dealias, library=library, kwargs...)
end

"""
    Chebyshev

Alias for [`ChebyshevT`](@ref).
"""
const Chebyshev = ChebyshevT

# ============================================================================
# FourierBase  (abstract base for Fourier-type bases)
# ============================================================================

"""
    FourierBase <: IntervalBasis

Abstract base for Fourier-type bases (RealFourier, ComplexFourier).
Provides shared grid computation, wavenumber logic, and coefficient
permutation infrastructure.

Concrete subtypes must have fields:
- `coord::CoordinateOrAzimuthal`
- `_size::Int`
- `bounds::Tuple{Float64,Float64}`
- `_dealias::Tuple{Float64}`
- `library::String`
- `COV::AffineCOV`
- `constant_mode_value::Float64`
- `forward_coeff_permutation::Union{Nothing,Vector{Int}}`
- `backward_coeff_permutation::Union{Nothing,Vector{Int}}`
- `_cache::Dict{Symbol,Any}`
- `_transform_cache::Dict{Any,Any}`
- `_product_matrix_cache::Dict{Any,Any}`
"""
abstract type FourierBase <: IntervalBasis end

const FOURIER_NATIVE_BOUNDS = (0.0, 2 * pi)
const FOURIER_DEFAULT_LIBRARY = "fftw"

# -- Shared Fourier grid --

function _native_grid(b::FourierBase, scale)
    N = grid_shape(b, (scale,))[1]
    return (2 * pi / N) .* collect(Float64, 0:N-1)
end

# -- Wavenumber properties --

"""
    _compute_native_wavenumbers(b::FourierBase)

Return the native wavenumber array (without permutation).
Must be implemented by ComplexFourier and RealFourier.
"""
function _compute_native_wavenumbers end

"""
    native_wavenumbers(b::FourierBase)

Return native wavenumbers, applying forward coefficient permutation if set.
"""
function native_wavenumbers(b::FourierBase)
    nw = _get_cached_attr!(b, :_native_wavenumbers, () -> _compute_native_wavenumbers(b))
    if b.forward_coeff_permutation === nothing
        return nw
    else
        return nw[b.forward_coeff_permutation]
    end
end

"""
    wavenumbers(b::FourierBase)

Return physical wavenumbers (= native_wavenumbers / stretch).
"""
function wavenumbers(b::FourierBase)
    nw = _get_cached_attr!(b, :_native_wavenumbers, () -> _compute_native_wavenumbers(b))
    wn = nw ./ b.COV.stretch
    if b.forward_coeff_permutation === nothing
        return wn
    else
        return wn[b.forward_coeff_permutation]
    end
end

# -- Basis algebra (shared by ComplexFourier and RealFourier) --

function basis_add(a::FourierBase, other)
    if other === nothing || other === a
        return a
    end
    return nothing
end

function basis_mul(a::FourierBase, other)
    if other === nothing || other === a
        return a
    end
    return nothing
end

function basis_rmatmul(a::FourierBase, ncc)
    if ncc === nothing || ncc === a
        return a
    end
    return nothing
end

function basis_pow(a::FourierBase, other)
    return a
end

# -- Elements and groups --

function elements_to_groups(b::FourierBase, grid_space, elements)
    if grid_space isa Tuple || grid_space isa AbstractVector
        gs = grid_space[1]
    else
        gs = grid_space
    end
    if gs
        return elements
    else
        nw = native_wavenumbers(b)
        return nw[elements .+ 1]  # Convert 0-based elements to 1-based indexing
    end
end

# -- Transform with permutation --

function forward_transform(b::FourierBase, field, axis, gdata, cdata)
    # Base transform
    data_axis = length(field.tensorsig) + axis
    grid_size = size(gdata, data_axis)
    plan = transform_plan(b, field.dist, grid_size)
    forward!(plan, gdata, cdata, data_axis)
    # Permute coefficients
    if b.forward_coeff_permutation !== nothing
        permute_axis(cdata, axis + length(field.tensorsig), b.forward_coeff_permutation; out=cdata)
    end
end

function backward_transform(b::FourierBase, field, axis, cdata, gdata)
    # Permute coefficients
    if b.backward_coeff_permutation !== nothing
        permute_axis(cdata, axis + length(field.tensorsig), b.backward_coeff_permutation; out=cdata)
    end
    # Base transform
    data_axis = length(field.tensorsig) + axis
    grid_size = size(gdata, data_axis)
    plan = transform_plan(b, field.dist, grid_size)
    backward!(plan, cdata, gdata, data_axis)
end

# -- Cached attribute helper --

function _get_cached_attr!(b::IntervalBasis, key::Symbol, compute::Function)
    cached = get(b._cache, key, nothing)
    if cached !== nothing
        return cached
    end
    val = compute()
    b._cache[key] = val
    return val
end

# ============================================================================
# ComplexFourier
# ============================================================================

"""
    ComplexFourierBasis <: FourierBase

Fourier complex exponential basis.
Modes: [exp(0j*x), exp(1j*x), exp(2j*x), ..., exp(-kmax*j*x), ..., exp(-1j*x)]
"""
mutable struct ComplexFourierBasis <: FourierBase
    coord::CoordinateOrAzimuthal
    _size::Int
    bounds::Tuple{Float64,Float64}
    _dealias::Tuple{Float64}
    library::String
    COV::AffineCOV
    constant_mode_value::Float64
    forward_coeff_permutation::Union{Nothing,Vector{Int}}
    backward_coeff_permutation::Union{Nothing,Vector{Int}}
    _cache::Dict{Symbol,Any}
    _transform_cache::Dict{Any,Any}
    _product_matrix_cache::Dict{Any,Any}
end

# -- Constructor caching --
const _complex_fourier_cache = Dict{Any,WeakRef}()

"""
    _preprocess_fourier_args(coord, size, bounds, dealias, library)

Preprocess and canonicalize arguments for Fourier basis construction.
"""
function _preprocess_fourier_args(coord, size, bounds, dealias, library)
    if !(coord isa CoordinateOrAzimuthal)
        throw(ArgumentError("Fourier coord must be a Coordinate object."))
    end
    size = Int(size)
    if size <= 0
        throw(ArgumentError("Fourier size must be positive."))
    end
    bounds = (Float64(bounds[1]), Float64(bounds[2]))
    if isa(dealias, Number)
        dealias = (Float64(dealias),)
    else
        dealias = (Float64(dealias[1]),)
    end
    if library === nothing
        library = FOURIER_DEFAULT_LIBRARY
    end
    return (coord, size, bounds, dealias, library)
end

"""
    ComplexFourier(coord, size, bounds; dealias=(1,), library=nothing)

Construct a complex Fourier basis.
"""
function ComplexFourier(coord::CoordinateOrAzimuthal, size::Integer, bounds;
                        dealias=(1,), library=nothing)
    coord_p, size_p, bounds_p, dealias_p, library_p = _preprocess_fourier_args(
        coord, size, bounds, dealias, library)

    cache_key = (coord_p, size_p, bounds_p, dealias_p, library_p)
    wr = get(_complex_fourier_cache, cache_key, nothing)
    if wr !== nothing
        inst = wr.value
        if inst !== nothing
            return inst::ComplexFourierBasis
        end
    end

    check_bounds(coord_p, bounds_p)
    cov = AffineCOV(FOURIER_NATIVE_BOUNDS, bounds_p)
    inst = ComplexFourierBasis(
        coord_p, size_p, bounds_p, dealias_p, library_p,
        cov, 1.0, nothing, nothing,
        Dict{Symbol,Any}(), Dict{Any,Any}(), Dict{Any,Any}()
    )
    _complex_fourier_cache[cache_key] = WeakRef(inst)
    return inst
end

basis_group_shape(::ComplexFourierBasis) = (1,)

function _compute_native_wavenumbers(b::ComplexFourierBasis)
    N = b._size
    kmax = N ÷ 2
    if N % 2 == 1
        # Odd: [0, 1, ..., kmax, -kmax, ..., -1]
        return vcat(collect(0:kmax), collect(-kmax:-1))
    else
        # Even: [0, 1, ..., kmax, 1-kmax, ..., -1]
        return vcat(collect(0:kmax), collect(1-kmax:-1))
    end
end

function valid_elements(b::ComplexFourierBasis, tensorsig, grid_space, elements)
    vshape = tuple((get_dim(cs) for cs in tensorsig)..., size(elements[1])...)
    return ones(Bool, vshape)
end

function product_matrix(b::ComplexFourierBasis, arg_basis, out_basis, i::Integer)
    # Cache lookup
    cache_key = (objectid(arg_basis), objectid(out_basis), i)
    cached = get(b._product_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end

    wn = wavenumbers(b)
    k0 = wn[2]  # wavenumber step (1-based: wn[2] = first nonzero)
    k_ncc = round(Int, wn[i] / k0)
    k_out = round.(Int, wavenumbers(out_basis) ./ k0)

    if arg_basis === nothing
        k_arg = [0]
    else
        k_arg = round.(Int, wavenumbers(arg_basis) ./ k0)
    end

    k_prod = k_arg .+ k_ncc

    # Find matching wavenumbers
    rows = Int[]
    cols = Int[]
    data = Float64[]
    for (ci, kp) in enumerate(k_prod)
        for (ri, ko) in enumerate(k_out)
            if ko == kp
                push!(rows, ri)
                push!(cols, ci)
                push!(data, 1.0)
                break
            end
        end
    end

    mat = sparse(rows, cols, data, length(k_out), length(k_arg))
    b._product_matrix_cache[cache_key] = mat
    return mat
end

function Base.show(io::IO, b::ComplexFourierBasis)
    print(io, "ComplexFourier($(b.coord), $(b._size))")
end

# ============================================================================
# RealFourier
# ============================================================================

"""
    RealFourierBasis <: FourierBase

Fourier real sine/cosine basis.
Modes: [cos(0*x), -sin(0*x), cos(1*x), -sin(1*x), ...]
"""
mutable struct RealFourierBasis <: FourierBase
    coord::CoordinateOrAzimuthal
    _size::Int
    bounds::Tuple{Float64,Float64}
    _dealias::Tuple{Float64}
    library::String
    COV::AffineCOV
    constant_mode_value::Float64
    forward_coeff_permutation::Union{Nothing,Vector{Int}}
    backward_coeff_permutation::Union{Nothing,Vector{Int}}
    _cache::Dict{Symbol,Any}
    _transform_cache::Dict{Any,Any}
    _product_matrix_cache::Dict{Any,Any}
end

# -- Constructor caching --
const _real_fourier_cache = Dict{Any,WeakRef}()

"""
    RealFourier(coord, size, bounds; dealias=(1,), library=nothing)

Construct a real Fourier basis.
"""
function RealFourier(coord::CoordinateOrAzimuthal, size::Integer, bounds;
                     dealias=(1,), library=nothing)
    coord_p, size_p, bounds_p, dealias_p, library_p = _preprocess_fourier_args(
        coord, size, bounds, dealias, library)

    cache_key = (coord_p, size_p, bounds_p, dealias_p, library_p)
    wr = get(_real_fourier_cache, cache_key, nothing)
    if wr !== nothing
        inst = wr.value
        if inst !== nothing
            return inst::RealFourierBasis
        end
    end

    check_bounds(coord_p, bounds_p)
    cov = AffineCOV(FOURIER_NATIVE_BOUNDS, bounds_p)
    inst = RealFourierBasis(
        coord_p, size_p, bounds_p, dealias_p, library_p,
        cov, 1.0, nothing, nothing,
        Dict{Symbol,Any}(), Dict{Any,Any}(), Dict{Any,Any}()
    )
    _real_fourier_cache[cache_key] = WeakRef(inst)
    return inst
end

basis_group_shape(::RealFourierBasis) = (2,)

function _compute_native_wavenumbers(b::RealFourierBasis)
    # Excludes Nyquist mode
    kmax = (b._size - 1) ÷ 2
    # [0, 0, 1, 1, 2, 2, ..., kmax, kmax]  -- repeated for cos/sin pairs
    return repeat(collect(0:kmax); inner=2)
end

function valid_elements(b::RealFourierBasis, tensorsig, grid_space, elements)
    vshape = tuple((get_dim(cs) for cs in tensorsig)..., size(elements[1])...)
    valid = ones(Bool, vshape)
    if grid_space isa Tuple || grid_space isa AbstractVector
        gs = grid_space[1]
    else
        gs = grid_space
    end
    if !gs
        # Drop msin part of k=0 for all Cartesian components and spin scalars
        if !(b.coord isa AzimuthalCoordinate) || isempty(tensorsig)
            groups = elements_to_groups(b, grid_space, elements)
            allcomps = ntuple(_ -> Colon(), length(tensorsig))
            # groups[1] are the wavenumber values; elements are 0-based indices
            elems0 = elements[1]
            grp = groups isa Tuple ? groups[1] : groups
            # Drop: k=0 AND odd element index (the -sin(0*x) mode)
            selection = (grp .== 0) .& (elems0 .% 2 .== 1)
            valid[allcomps..., selection] .= false
        end
    end
    return valid
end

function product_matrix(b::RealFourierBasis, arg_basis, out_basis, i::Integer)
    # Cache lookup
    cache_key = (objectid(arg_basis), objectid(out_basis), i)
    cached = get(b._product_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end

    wn = wavenumbers(b)
    k0 = wn[3]  # 1-based: wn[3] is the first nonzero wavenumber (cos(1*x))
    k_ncc = round(Int, wn[i] / k0)
    k_out = round.(Int, wavenumbers(out_basis) ./ k0)

    if arg_basis === nothing
        k_arg = [0]
    else
        k_arg = round.(Int, wavenumbers(arg_basis) ./ k0)
    end

    Nout = length(k_out)
    Narg = length(k_arg)

    # Skip multiplication by constant
    if k_ncc == 0
        if (i - 1) % 2 == 0  # 0-based i: cos component
            mat = sparse(1.0I, Nout, Narg)
        else  # sin component
            mat = spzeros(Float64, Nout, Narg)
        end
        b._product_matrix_cache[cache_key] = mat
        return mat
    end

    # Find wavenumber couplings for trigonometric product rules
    # k_out indices are 1-based; even indices (1,3,5,...) are cos, odd (2,4,6,...) are sin
    # In 0-based Python: [::2] = cos, [1::2] = sin
    # In 1-based Julia: [1:2:end] = cos, [2:2:end] = sin

    k_out_cos = k_out[1:2:end]   # cosine wavenumbers
    k_arg_cos = k_arg[1:2:end]   # cosine wavenumbers

    # Find intersections for sum and difference frequencies
    # k_out_cos ∩ (k_ncc + k_arg_cos) -> "plus" coupling
    # k_out_cos ∩ (k_ncc - k_arg_cos) -> "minus" coupling
    # k_out_cos[2:end] ∩ (k_arg_cos - k_ncc) -> "minus-neg" coupling (exclude k=0)

    k_sum = k_ncc .+ k_arg_cos
    k_diff = k_ncc .- k_arg_cos
    # For k_out_cos excluding k=0: k_arg_cos - k_ncc
    k_diff_neg = k_arg_cos .- k_ncc

    # Build sparse matrix
    rows_list = Int[]
    cols_list = Int[]
    data_list = Float64[]

    # Helper: find matches between two sorted arrays
    function find_matches(a_arr, b_arr)
        row_idx = Int[]
        col_idx = Int[]
        for (ci, bv) in enumerate(b_arr)
            for (ri, av) in enumerate(a_arr)
                if av == bv
                    push!(row_idx, ri)
                    push!(col_idx, ci)
                    break
                end
            end
        end
        return row_idx, col_idx
    end

    rows_p, cols_p = find_matches(k_out_cos, k_sum)       # plus
    rows_m, cols_m = find_matches(k_out_cos, k_diff)       # minus
    # minus-neg: match k_out_cos[2:end] against k_diff_neg
    k_out_cos_nz = length(k_out_cos) > 1 ? k_out_cos[2:end] : Int[]
    rows_mn, cols_mn = find_matches(k_out_cos_nz, k_diff_neg)
    rows_mn .+= 1  # offset because we skipped k=0

    if (i - 1) % 2 == 0  # cos component (0-based even index)
        # 2 cos(mx) cos(nx) = cos((m+n)x) + cos((m-n)x)
        for (r, c) in zip(rows_p, cols_p)
            push!(rows_list, 2r - 1); push!(cols_list, 2c - 1); push!(data_list, 0.5)
        end
        for (r, c) in zip(rows_m, cols_m)
            push!(rows_list, 2r - 1); push!(cols_list, 2c - 1); push!(data_list, 0.5)
        end
        for (r, c) in zip(rows_mn, cols_mn)
            push!(rows_list, 2r - 1); push!(cols_list, 2c - 1); push!(data_list, 0.5)
        end
        # 2 cos(mx) msin(nx) = msin((m+n)x) - msin((m-n)x)
        for (r, c) in zip(rows_p, cols_p)
            push!(rows_list, 2r); push!(cols_list, 2c); push!(data_list, 0.5)
        end
        for (r, c) in zip(rows_m, cols_m)
            push!(rows_list, 2r); push!(cols_list, 2c); push!(data_list, -0.5)
        end
        for (r, c) in zip(rows_mn, cols_mn)
            push!(rows_list, 2r); push!(cols_list, 2c); push!(data_list, 0.5)
        end
    else  # sin component (0-based odd index)
        # 2 msin(mx) cos(nx) = msin((m+n)x) + msin((m-n)x)
        for (r, c) in zip(rows_p, cols_p)
            push!(rows_list, 2r); push!(cols_list, 2c - 1); push!(data_list, 0.5)
        end
        for (r, c) in zip(rows_m, cols_m)
            push!(rows_list, 2r); push!(cols_list, 2c - 1); push!(data_list, 0.5)
        end
        for (r, c) in zip(rows_mn, cols_mn)
            push!(rows_list, 2r); push!(cols_list, 2c - 1); push!(data_list, -0.5)
        end
        # 2 msin(mx) msin(nx) = -cos((m+n)x) + cos((m-n)x)
        for (r, c) in zip(rows_p, cols_p)
            push!(rows_list, 2r - 1); push!(cols_list, 2c); push!(data_list, -0.5)
        end
        for (r, c) in zip(rows_m, cols_m)
            push!(rows_list, 2r - 1); push!(cols_list, 2c); push!(data_list, 0.5)
        end
        for (r, c) in zip(rows_mn, cols_mn)
            push!(rows_list, 2r - 1); push!(cols_list, 2c); push!(data_list, 0.5)
        end
    end

    mat = sparse(rows_list, cols_list, data_list, Nout, Narg)
    b._product_matrix_cache[cache_key] = mat
    return mat
end

function Base.show(io::IO, b::RealFourierBasis)
    print(io, "RealFourier($(b.coord), $(b._size))")
end

# ============================================================================
# Fourier factory function
# ============================================================================

"""
    Fourier(coord, size, bounds; dtype=nothing, dealias=(1,), library=nothing)

Factory function dispatching to RealFourier or ComplexFourier based on dtype.
"""
function Fourier(coord::CoordinateOrAzimuthal, size::Integer, bounds;
                 dtype=nothing, dealias=(1,), library=nothing)
    if dtype === nothing
        throw(ArgumentError("dtype must be specified"))
    elseif dtype === Float64
        return RealFourier(coord, size, bounds; dealias=dealias, library=library)
    elseif dtype === ComplexF64
        return ComplexFourier(coord, size, bounds; dealias=dealias, library=library)
    else
        throw(ArgumentError("Unrecognized dtype: $dtype"))
    end
end

# ============================================================================
# Operator support types  (Jacobi operators)
# ============================================================================

# These are Julia structs that mirror the Python operator classes.
# They encapsulate the static methods for building operator matrices.

"""
    ConvertJacobiOp

Jacobi polynomial conversion operator.  Provides `full_matrix` for
converting between Jacobi bases with different (a,b) parameters.
"""
struct ConvertJacobiOp end

"""
    convert_jacobi_matrix(input_basis::JacobiBasis, output_basis::JacobiBasis)

Build the full conversion matrix from `input_basis` to `output_basis`.
"""
function convert_jacobi_matrix(input_basis::JacobiBasis, output_basis::JacobiBasis)
    N = input_basis._size
    a0, b0 = input_basis.a, input_basis.b
    a1, b1 = output_basis.a, output_basis.b
    return conversion_matrix(N, a0, b0, a1, b1)
end

"""
    convert_constant_jacobi_matrix(group, input_basis, output_basis::JacobiBasis)

Build the group matrix for converting a constant to a Jacobi basis.
Group `group` is 0-based (Python convention) -- only group 0 produces a nonzero matrix.
"""
function convert_constant_jacobi_matrix(group::Integer, input_basis, output_basis::JacobiBasis)
    if group == 0
        unit_amplitude = 1.0 / output_basis.constant_mode_value
        return reshape([unit_amplitude], 1, 1)
    else
        throw(ArgumentError("ConvertConstantJacobi should only be called for group 0."))
    end
end

"""
    DifferentiateJacobiOp

Jacobi polynomial differentiation operator.
"""
struct DifferentiateJacobiOp end

"""
    differentiate_jacobi_output_basis(input_basis::JacobiBasis)

Return the output basis for Jacobi differentiation.
"""
function differentiate_jacobi_output_basis(input_basis::JacobiBasis)
    return derivative_basis(input_basis; order=1)
end

"""
    differentiate_jacobi_matrix(input_basis::JacobiBasis, output_basis::JacobiBasis)

Build the full differentiation matrix for Jacobi polynomials.
"""
function differentiate_jacobi_matrix(input_basis::JacobiBasis, output_basis::JacobiBasis)
    N = input_basis._size
    a, b = input_basis.a, input_basis.b
    return differentiation_matrix(N, a, b) ./ input_basis.COV.stretch
end

"""
    interpolate_jacobi_matrix(input_basis::JacobiBasis, position)

Build the interpolation matrix (row vector) for Jacobi polynomials at `position`.
"""
function interpolate_jacobi_matrix(input_basis::JacobiBasis, position)
    N = input_basis._size
    a, b = input_basis.a, input_basis.b
    x = native_coord(input_basis.COV, position)
    interp_vec = build_polynomials(N, a, b, x isa Number ? [x] : x)
    # interp_vec should be shape (N,) or (N, 1); return as (1, N)
    if ndims(interp_vec) == 1
        return reshape(interp_vec, 1, :)
    else
        return reshape(interp_vec, 1, :)
    end
end

"""
    integrate_jacobi_matrix(input_basis::JacobiBasis)

Build the integration matrix (row vector) for Jacobi polynomials.
"""
function integrate_jacobi_matrix(input_basis::JacobiBasis)
    N = input_basis._size
    a, b = input_basis.a, input_basis.b
    integ_vec = integration_vector(N, a, b)
    # Rescale by stretch and return as (1, N)
    return reshape(integ_vec .* input_basis.COV.stretch, 1, :)
end

"""
    average_jacobi_matrix(input_basis::JacobiBasis)

Build the averaging matrix (row vector) for Jacobi polynomials.
"""
function average_jacobi_matrix(input_basis::JacobiBasis)
    N = input_basis._size
    a, b = input_basis.a, input_basis.b
    integ_vec = integration_vector(N, a, b)
    ave_vec = integ_vec ./ 2.0
    return reshape(ave_vec, 1, :)
end

# ============================================================================
# Operator support types  (Fourier operators)
# ============================================================================

# -- ComplexFourier operators --

"""
    convert_constant_complex_fourier_matrix(group, input_basis, output_basis::ComplexFourierBasis)

Group matrix for converting a constant to ComplexFourier.
"""
function convert_constant_complex_fourier_matrix(group::Integer, input_basis,
                                                  output_basis::ComplexFourierBasis)
    k = group / output_basis.COV.stretch
    if k == 0
        unit_amplitude = 1.0 / output_basis.constant_mode_value
        return reshape([unit_amplitude], 1, 1)
    else
        return zeros(1, 0)
    end
end

"""
    differentiate_complex_fourier_matrix(group, input_basis::ComplexFourierBasis)

Group matrix for ComplexFourier differentiation.
dx exp(ikx) = ik exp(ikx)
"""
function differentiate_complex_fourier_matrix(group::Integer, input_basis::ComplexFourierBasis)
    k = group / input_basis.COV.stretch
    return reshape([1im * k], 1, 1)
end

"""
    interpolate_complex_fourier_matrix(input_basis::ComplexFourierBasis, position)

Full interpolation matrix for ComplexFourier.
"""
function interpolate_complex_fourier_matrix(input_basis::ComplexFourierBasis, position)
    x = native_coord(input_basis.COV, position)
    k = native_wavenumbers(input_basis)
    interp_vec = exp.(1im .* k .* x)
    return reshape(interp_vec, 1, :)
end

"""
    integrate_complex_fourier_matrix(group, input_basis::ComplexFourierBasis)

Group matrix for ComplexFourier integration.
integral exp(ikx) = L * delta(k, 0)
"""
function integrate_complex_fourier_matrix(group::Integer, input_basis::ComplexFourierBasis)
    k = group / input_basis.COV.stretch
    if k == 0
        L = input_basis.COV.problem_length
        return reshape([L], 1, 1)
    else
        throw(ArgumentError("IntegrateComplexFourier should only be called for group 0."))
    end
end

"""
    average_complex_fourier_matrix(group, input_basis::ComplexFourierBasis)

Group matrix for ComplexFourier averaging.
"""
function average_complex_fourier_matrix(group::Integer, input_basis::ComplexFourierBasis)
    k = group / input_basis.COV.stretch
    if k == 0
        return reshape([1.0], 1, 1)
    else
        throw(ArgumentError("AverageComplexFourier should only be called for group 0."))
    end
end

# -- RealFourier operators --

"""
    convert_constant_real_fourier_matrix(group, input_basis, output_basis::RealFourierBasis)

Group matrix for converting a constant to RealFourier.
1 = cos(0*x)
"""
function convert_constant_real_fourier_matrix(group::Integer, input_basis,
                                              output_basis::RealFourierBasis)
    k = group / output_basis.COV.stretch
    if k == 0
        unit_amplitude = 1.0 / output_basis.constant_mode_value
        return [unit_amplitude 0.0]'  # shape (2, 1)
    else
        return zeros(2, 0)
    end
end

"""
    differentiate_real_fourier_matrix(group, input_basis::RealFourierBasis)

Group matrix for RealFourier differentiation.
dx  cos(kx) = k * (-sin(kx))
dx (-sin(kx)) = -k * cos(kx)
"""
function differentiate_real_fourier_matrix(group::Integer, input_basis::RealFourierBasis)
    k = group / input_basis.COV.stretch
    return [0.0 -k; k 0.0]
end

"""
    interpolate_real_fourier_matrix(input_basis::RealFourierBasis, position)

Full interpolation matrix for RealFourier.
Interleaved: cos(k*x), -sin(k*x)
"""
function interpolate_real_fourier_matrix(input_basis::RealFourierBasis, position)
    x = native_coord(input_basis.COV, position)
    k = native_wavenumbers(input_basis)
    N = length(k)
    interp_vec = zeros(N)
    # Even indices (1-based: 1,3,5,...) = cos; Odd indices (2,4,6,...) = -sin
    interp_vec[1:2:end] .= cos.(k[1:2:end] .* x)
    interp_vec[2:2:end] .= .-sin.(k[2:2:end] .* x)
    return reshape(interp_vec, 1, :)
end

"""
    integrate_real_fourier_matrix(group, input_basis::RealFourierBasis)

Group matrix for RealFourier integration.
integral  cos(kx) = L * delta(k,0)
integral -sin(kx) = 0
"""
function integrate_real_fourier_matrix(group::Integer, input_basis::RealFourierBasis)
    k = group / input_basis.COV.stretch
    if k == 0
        L = input_basis.COV.problem_length
        return reshape([L, 0.0], 1, :)  # shape (1, 2)
    else
        throw(ArgumentError("IntegrateRealFourier should only be called for group 0."))
    end
end

"""
    average_real_fourier_matrix(group, input_basis::RealFourierBasis)

Group matrix for RealFourier averaging.
"""
function average_real_fourier_matrix(group::Integer, input_basis::RealFourierBasis)
    k = group / input_basis.COV.stretch
    if k == 0
        return reshape([1.0, 0.0], 1, :)  # shape (1, 2)
    else
        throw(ArgumentError("AverageRealFourier should only be called for group 0."))
    end
end

# ============================================================================
# Operator support types  (Cardinal operators)
# ============================================================================

"""
    convert_constant_cardinal_matrix(input_basis, output_basis::CardinalBasis)

Full matrix for converting a constant to CardinalBasis.
"""
function convert_constant_cardinal_matrix(input_basis, output_basis::CardinalBasis)
    return ones(output_basis._size, 1)
end

"""
    interpolate_cardinal_matrix(input_basis::CardinalBasis, position::Integer)

Full interpolation matrix for CardinalBasis at integer position.
"""
function interpolate_cardinal_matrix(input_basis::CardinalBasis, position::Integer)
    interp_vec = zeros(input_basis._size)
    interp_vec[position + 1] = 1.0  # Convert 0-based position to 1-based
    return reshape(interp_vec, 1, :)
end

"""
    integrate_cardinal_matrix(input_basis::CardinalBasis)

Full integration matrix for CardinalBasis (sum of all elements).
"""
function integrate_cardinal_matrix(input_basis::CardinalBasis)
    return reshape(ones(input_basis._size), 1, :)
end

"""
    average_cardinal_matrix(input_basis::CardinalBasis)

Full averaging matrix for CardinalBasis.
"""
function average_cardinal_matrix(input_basis::CardinalBasis)
    return reshape(fill(1.0 / input_basis._size, input_basis._size), 1, :)
end

# ============================================================================
# Transform plan stubs
# ============================================================================
# These are placeholders for transform plan objects.  The actual
# implementations will be registered by the transform libraries
# (FFTW, matrix, etc.) when those modules are loaded.

"""
    AbstractTransformPlan

Abstract type for spectral transform plans.
"""
abstract type AbstractTransformPlan end

"""
    forward!(plan::AbstractTransformPlan, gdata, cdata, axis)

Execute the forward (grid -> coeff) transform.
"""
function forward! end

"""
    backward!(plan::AbstractTransformPlan, cdata, gdata, axis)

Execute the backward (coeff -> grid) transform.
"""
function backward! end

"""
    MatrixTransformPlan <: AbstractTransformPlan

Simple dense-matrix-based transform plan.  Forward transform applies
`forward_matrix`; backward applies `backward_matrix`.

# Fields
- `forward_matrix::Matrix{Float64}`
- `backward_matrix::Matrix{Float64}`
"""
struct MatrixTransformPlan <: AbstractTransformPlan
    forward_matrix::Matrix{Float64}
    backward_matrix::Matrix{Float64}
end

function forward!(plan::MatrixTransformPlan, gdata, cdata, axis)
    temp = apply_matrix(plan.forward_matrix, gdata, axis)
    copyto!(cdata, temp)
end

function backward!(plan::MatrixTransformPlan, cdata, gdata, axis)
    temp = apply_matrix(plan.backward_matrix, cdata, axis)
    copyto!(gdata, temp)
end

# ============================================================================
# MultidimensionalBasis  (abstract base for multi-axis bases)
# ============================================================================

"""
    MultidimensionalBasis <: Basis

Abstract base type for bases that span multiple coordinate axes (e.g.
spherical, polar, disk bases).  Concrete subtypes must provide:
- `forward_transforms::Vector` — a vector of callables for each sub-axis
- `backward_transforms::Vector` — a vector of callables for each sub-axis

Transform dispatch delegates to the appropriate sub-axis transform by
computing `subaxis = axis - get_basis_axis(field.dist, basis) + 1`
(1-based in Julia; Python used 0-based).
"""
abstract type MultidimensionalBasis <: Basis end

function forward_transform(b::MultidimensionalBasis, field, axis, gdata, cdata)
    subaxis = axis - get_basis_axis(field.dist, b) + 1  # 1-based
    return b.forward_transforms[subaxis](field, axis, gdata, cdata)
end

function backward_transform(b::MultidimensionalBasis, field, axis, cdata, gdata)
    subaxis = axis - get_basis_axis(field.dist, b) + 1  # 1-based
    return b.backward_transforms[subaxis](field, axis, cdata, gdata)
end

# ============================================================================
# reduced_view_5  (helper for spin recombination)
# ============================================================================

"""
    reduced_view_5(data::AbstractArray, axis1::Int, axis2::Int)

Reshape `data` into a 5D view with dimensions:
  (pre-axis1, axis1, between, axis2, post-axis2)

Both `axis1` and `axis2` are 1-based. `axis1 < axis2` is required.
"""
function reduced_view_5(data::AbstractArray, axis1::Int, axis2::Int)
    shp = size(data)
    N0 = prod(shp[1:axis1-1]; init=1)
    N1 = shp[axis1]
    N2 = prod(shp[axis1+1:axis2-1]; init=1)
    N3 = shp[axis2]
    N4 = prod(shp[axis2+1:end]; init=1)
    return reshape(data, (N0, N1, N2, N3, N4))
end

# ============================================================================
# Spin Recombination functions (from spin_recombination.pyx)
# ============================================================================

const _INVSQRT2 = 1.0 / sqrt(2.0)

"""
    recombine_forward!(s::Int, input::AbstractArray{Float64,5},
                       output::AbstractArray{Float64,5})

Component-to-spin recombination on 5D arrays.

`s` is the 1-based sub-axis index within dimension 2 of the 5D view.
In Python this was 0-based; here we use 1-based indexing.

The recombination operates on the pair of slices `output[:, s, :, :, :]` and
`output[:, s+1, :, :, :]`, mixing even/odd elements of dimension 4 using
a 1/sqrt(2) scaling factor.  Slices outside the pair are copied unchanged.
"""
function recombine_forward!(s::Int, input::AbstractArray{Float64,5},
                            output::AbstractArray{Float64,5})
    size0 = size(input, 1)
    size1 = size(input, 2)
    size2 = size(input, 3)
    size3 = size(input, 4)
    size4 = size(input, 5)
    size3_2 = size3 ÷ 2

    @inbounds for i in 1:size0
        # Copy slices before s
        for j in 1:s-1
            for k in 1:size2, l in 1:size3, m in 1:size4
                output[i, j, k, l, m] = input[i, j, k, l, m]
            end
        end
        # Recombine the s and s+1 pair
        for k in 1:size2
            @simd for l_half in 1:size3_2
                for m in 1:size4
                    l_even = 2 * l_half - 1  # 1-based even index (was 2*l in 0-based)
                    l_odd  = 2 * l_half      # 1-based odd index  (was 2*l+1 in 0-based)
                    # s+1 corresponds to Python s+0 (1-based), s+2 corresponds to Python s+1
                    inp_s0_even = input[i, s,   k, l_even, m]
                    inp_s0_odd  = input[i, s,   k, l_odd,  m]
                    inp_s1_even = input[i, s+1, k, l_even, m]
                    inp_s1_odd  = input[i, s+1, k, l_odd,  m]
                    output[i, s,   k, l_even, m] = (inp_s1_even + inp_s0_odd)  * _INVSQRT2
                    output[i, s+1, k, l_odd,  m] = (inp_s1_odd  + inp_s0_even) * _INVSQRT2
                    output[i, s+1, k, l_even, m] = (inp_s1_even - inp_s0_odd)  * _INVSQRT2
                    output[i, s,   k, l_odd,  m] = (inp_s1_odd  - inp_s0_even) * _INVSQRT2
                end
            end
        end
        # Copy slices after s+1
        for j in s+2:size1
            for k in 1:size2, l in 1:size3, m in 1:size4
                output[i, j, k, l, m] = input[i, j, k, l, m]
            end
        end
    end
    return nothing
end

"""
    recombine_backward!(s::Int, input::AbstractArray{Float64,5},
                        output::AbstractArray{Float64,5})

Spin-to-component recombination on 5D arrays (inverse of `recombine_forward!`).

`s` is 1-based (converted from Python's 0-based `s` parameter).
"""
function recombine_backward!(s::Int, input::AbstractArray{Float64,5},
                             output::AbstractArray{Float64,5})
    size0 = size(input, 1)
    size1 = size(input, 2)
    size2 = size(input, 3)
    size3 = size(input, 4)
    size4 = size(input, 5)
    size3_2 = size3 ÷ 2

    @inbounds for i in 1:size0
        # Copy slices before s
        for j in 1:s-1
            for k in 1:size2, l in 1:size3, m in 1:size4
                output[i, j, k, l, m] = input[i, j, k, l, m]
            end
        end
        # Recombine the s and s+1 pair (inverse of forward)
        for k in 1:size2
            @simd for l_half in 1:size3_2
                for m in 1:size4
                    l_even = 2 * l_half - 1
                    l_odd  = 2 * l_half
                    inp_s0_even = input[i, s,   k, l_even, m]
                    inp_s0_odd  = input[i, s,   k, l_odd,  m]
                    inp_s1_even = input[i, s+1, k, l_even, m]
                    inp_s1_odd  = input[i, s+1, k, l_odd,  m]
                    output[i, s,   k, l_even, m] = (inp_s1_odd  - inp_s0_odd)  * _INVSQRT2
                    output[i, s,   k, l_odd,  m] = (inp_s0_even - inp_s1_even) * _INVSQRT2
                    output[i, s+1, k, l_even, m] = (inp_s0_even + inp_s1_even) * _INVSQRT2
                    output[i, s+1, k, l_odd,  m] = (inp_s0_odd  + inp_s1_odd)  * _INVSQRT2
                end
            end
        end
        # Copy slices after s+1
        for j in s+2:size1
            for k in 1:size2, l in 1:size3, m in 1:size4
                output[i, j, k, l, m] = input[i, j, k, l, m]
            end
        end
    end
    return nothing
end

# ============================================================================
# SpinRecombination trait (Holy trait pattern)
# ============================================================================

"""
    SpinRecombinationTrait

Abstract trait type for the spin recombination dispatch pattern.
"""
abstract type SpinRecombinationTrait end

"""
    HasSpinRecombination <: SpinRecombinationTrait

Marker trait indicating a basis supports spin recombination.
"""
struct HasSpinRecombination <: SpinRecombinationTrait end

"""
    NoSpinRecombination <: SpinRecombinationTrait

Marker trait indicating a basis does NOT support spin recombination.
"""
struct NoSpinRecombination <: SpinRecombinationTrait end

"""
    spin_recombination_trait(basis) -> SpinRecombinationTrait

Return the spin recombination trait for the given basis.  Default is
`NoSpinRecombination()`.  Concrete multidimensional bases that support
spin recombination (SpinBasis subtypes) should override to return
`HasSpinRecombination()`.
"""
spin_recombination_trait(::Basis) = NoSpinRecombination()

"""
    spin_ordering(cs::AbstractCoordinateSystem)

Return the spin ordering tuple for the given coordinate system.
Maps to the `spin_ordering` class attribute in the Python code.
"""
spin_ordering(::S2Coordinates)          = S2_SPIN_ORDERING
spin_ordering(::PolarCoordinates)       = POLAR_SPIN_ORDERING
spin_ordering(::SphericalCoordinates)   = SPHERICAL_SPIN_ORDERING

"""
    spin_recombination_factors(basis, tensorsig)

Build a vector of matrices (or `nothing` entries) for applying spin
recombination to each tensor index.  The recombination factors are
separable over tensor index and depend on which tensor indices share
coordinates with the basis's second coordinate (coords[2], the
colatitude/radial direction).

Uses `_cache` dict for memoization.
"""
function spin_recombination_factors(basis, tensorsig)
    cache_key = (:spin_recombination_factors, objectid(tensorsig))
    cached = get(basis._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    cs = basis_coordsys(basis)
    coord1 = get_coords(cs)[2]  # Python coords[1] → Julia coords[2] (1-based)
    factors = Vector{Union{Nothing, AbstractMatrix}}(undef, length(tensorsig))
    for (i, tcs) in enumerate(tensorsig)
        tcoords = get_coords(tcs)
        idx = findfirst(c -> c == coord1, tcoords)
        if idx !== nothing
            # subaxis is 0-based in Python; forward_intertwiner uses it directly
            subaxis = idx - 1
            factors[i] = forward_intertwiner(tcs, subaxis; order=1, group=nothing)
        else
            factors[i] = nothing
        end
    end
    basis._cache[cache_key] = factors
    return factors
end

"""
    spin_recombination_matrix(basis, tensorsig, n_upstream)

Build the full spin recombination matrix by combining the per-tensor-index
factors.  `nothing` placeholders are replaced by identity matrices.
For real-valued problems (Float64 dtype), the matrix is expanded for
cos/sin (msin) parts.
"""
function spin_recombination_matrix(basis, tensorsig, n_upstream)
    factors = copy(spin_recombination_factors(basis, tensorsig))
    for (i, tcs) in enumerate(tensorsig)
        if factors[i] === nothing
            factors[i] = Matrix{ComplexF64}(I, get_dim(tcs), get_dim(tcs))
        end
    end
    # Add space for upstream groups
    all_factors = copy(factors)
    if n_upstream > 1
        push!(all_factors, Matrix{ComplexF64}(I, n_upstream, n_upstream))
    end
    matrix = kronecker(all_factors...)
    # Expand for cos and msin parts for real-valued problems
    if basis.dtype === Float64
        matrix = kron(real(matrix), [1 0; 0 1]) + kron(imag(matrix), [0 -1; 1 0])
    end
    return matrix
end

"""
    forward_spin_recombination!(basis, tensorsig, colat_axis, gdata, out)

Apply component-to-spin recombination.  `colat_axis` is 1-based.
For complex data, applies the unitary matrices directly via `apply_matrix`.
For real data, uses the optimised `recombine_forward!` on 5D views,
alternating between `gdata` and `out` as input/output buffers.
"""
function forward_spin_recombination!(basis, tensorsig, colat_axis, gdata, out)
    if isempty(tensorsig)
        copyto!(out, gdata)
        return nothing
    end
    azimuth_axis = colat_axis - 1
    factors = spin_recombination_factors(basis, tensorsig)
    if eltype(gdata) <: Complex
        # Complex path: directly apply unitary matrices
        copyto!(out, gdata)
        for (i, Ui) in enumerate(factors)
            if Ui !== nothing
                temp = apply_matrix(Ui, out, i)
                copyto!(out, temp)
            end
        end
    else
        # Real (Float64) path: use recombine_forward! with 5D views
        num_recombinations = 0
        cs = basis_coordsys(basis)
        coord0 = get_coords(cs)[1]  # Python coords[0] → Julia coords[1]
        for (i, Ui) in enumerate(factors)
            if Ui !== nothing
                tcoords = get_coords(tensorsig[i])
                subaxis_idx = findfirst(c -> c == coord0, tcoords)
                # subaxis is 1-based for Julia recombine_forward!
                subaxis = subaxis_idx
                if num_recombinations % 2 == 0
                    input_view  = reduced_view_5(gdata, i, azimuth_axis + length(tensorsig))
                    output_view = reduced_view_5(out,   i, azimuth_axis + length(tensorsig))
                else
                    input_view  = reduced_view_5(out,   i, azimuth_axis + length(tensorsig))
                    output_view = reduced_view_5(gdata, i, azimuth_axis + length(tensorsig))
                end
                recombine_forward!(subaxis, input_view, output_view)
                num_recombinations += 1
            end
        end
        if num_recombinations % 2 == 0
            copyto!(out, gdata)
        end
    end
    return nothing
end

"""
    backward_spin_recombination!(basis, tensorsig, colat_axis, gdata, out)

Apply spin-to-component recombination (inverse of forward).
`colat_axis` is 1-based.
"""
function backward_spin_recombination!(basis, tensorsig, colat_axis, gdata, out)
    if isempty(tensorsig)
        copyto!(out, gdata)
        return nothing
    end
    azimuth_axis = colat_axis - 1
    factors = spin_recombination_factors(basis, tensorsig)
    if eltype(gdata) <: Complex
        # Complex path: apply conjugate transpose of unitary matrices
        copyto!(out, gdata)
        for (i, Ui) in enumerate(factors)
            if Ui !== nothing
                temp = apply_matrix(conj(transpose(Ui)), out, i)
                copyto!(out, temp)
            end
        end
    else
        # Real (Float64) path: use recombine_backward! with 5D views
        num_recombinations = 0
        cs = basis_coordsys(basis)
        coord0 = get_coords(cs)[1]
        for (i, Ui) in enumerate(factors)
            if Ui !== nothing
                tcoords = get_coords(tensorsig[i])
                subaxis_idx = findfirst(c -> c == coord0, tcoords)
                subaxis = subaxis_idx  # 1-based
                if num_recombinations % 2 == 0
                    input_view  = reduced_view_5(gdata, i, azimuth_axis + length(tensorsig))
                    output_view = reduced_view_5(out,   i, azimuth_axis + length(tensorsig))
                else
                    input_view  = reduced_view_5(out,   i, azimuth_axis + length(tensorsig))
                    output_view = reduced_view_5(gdata, i, azimuth_axis + length(tensorsig))
                end
                recombine_backward!(subaxis, input_view, output_view)
                num_recombinations += 1
            end
        end
        if num_recombinations % 2 == 0
            copyto!(out, gdata)
        end
    end
    return nothing
end

# ============================================================================
# SpinBasis  (abstract; common for S2 and D2 / polar bases)
# ============================================================================

"""
    SpinBasis <: MultidimensionalBasis

Abstract base type for multidimensional bases with spin-weighted fields.
Common parent for S2 (sphere) and D2 (disk/annulus/polar) bases.

Concrete subtypes must have fields:
- `coordsys` — the coordinate system (e.g. `S2Coordinates`, `PolarCoordinates`)
- `shape::Tuple{Int,Int}` — (N_azimuth, N_radial_or_colat)
- `dtype::DataType` — `Float64` or `ComplexF64`
- `dealias_tuple::Tuple{Float64,Float64}` — dealias factors per sub-axis
- `mmax::Int` — maximum azimuthal wavenumber = (shape[1] - 1) ÷ 2
- `azimuth_basis` — a `ComplexFourierBasis` or `RealFourierBasis`
- `azimuth_library` — library name for azimuth transforms
- `_cache::Dict{Symbol,Any}` — for cached attributes/methods

## Constructor logic (to be implemented in concrete subtypes)

    SpinBasis constructor:
    - Takes `coordsys`, `shape`, `dtype`, `dealias`, `azimuth_library`
    - Computes `mmax = (shape[1] - 1) ÷ 2`
    - Creates `azimuth_basis`:
        - `ComplexFourier` if `dtype == ComplexF64`
        - `RealFourier` if `dtype == Float64`
"""
abstract type SpinBasis <: MultidimensionalBasis end

# SpinBasis subtypes support spin recombination
spin_recombination_trait(::SpinBasis) = HasSpinRecombination()

# -- Accessor functions for SpinBasis concrete subtypes --

"""
    get_mmax(b::SpinBasis) -> Int

Return the maximum azimuthal wavenumber.
Concrete subtypes must have a `mmax` field.
"""
get_mmax(b::SpinBasis) = b.mmax

"""
    get_azimuth_basis(b::SpinBasis)

Return the azimuth (Fourier) sub-basis.
"""
get_azimuth_basis(b::SpinBasis) = b.azimuth_basis

"""
    basis_constant(b::SpinBasis)

Return tuple of booleans indicating which sub-axes are constant.
The azimuth is constant only when mmax == 0.
"""
function basis_constant(b::SpinBasis)
    cached = get(b._cache, :constant, nothing)
    if cached !== nothing
        return cached
    end
    val = (b.mmax == 0, false)
    b._cache[:constant] = val
    return val
end

"""
    basis_coordsys(b::SpinBasis)

Return the coordinate system for a SpinBasis.
"""
basis_coordsys(b::SpinBasis) = b.coordsys

"""
    basis_shape(b::SpinBasis)

Return the shape tuple for a SpinBasis.
"""
basis_shape(b::SpinBasis) = b.shape

"""
    basis_dealias(b::SpinBasis)

Return the dealias tuple for a SpinBasis.
"""
basis_dealias(b::SpinBasis) = b.dealias_tuple

"""
    spin_weights(basis::SpinBasis, tensorsig)

Compute the spin weight array for the given tensor signature.
Returns an integer array of shape `[dim(cs) for cs in tensorsig]`
containing the total spin weight contribution from each tensor index.

Uses `_cache` for memoization.
"""
function spin_weights(basis::SpinBasis, tensorsig)
    cache_key = (:spin_weights, objectid(tensorsig))
    cached = get(basis._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    cs = basis_coordsys(basis)
    coord1 = get_coords(cs)[2]  # Python coords[1] → Julia coords[2]
    spin_order = collect(spin_ordering(cs))
    dims = [get_dim(tcs) for tcs in tensorsig]
    S = zeros(Int, dims...)
    for (i, tcs) in enumerate(tensorsig)
        tcoords = get_coords(tcs)
        idx = findfirst(c -> c == coord1, tcoords)
        if idx !== nothing
            start = idx - 1  # 0-based start for axslice compatibility
            cs_dim = get_dim(cs)
            # Slice spin_order to match the coordinate system dimension of this tensor index
            # (hack for 2-vectors on S2 with 3D spherical coords)
            tcs_dim = get_dim(tcs)
            spin_sub = spin_order[1:min(tcs_dim, length(spin_order))]
            spin_vec = reshape_vector(spin_sub, length(tensorsig), i)
            # Build index tuple for the slice: all colons except axis i gets start+1:start+cs_dim
            idx_tuple = ntuple(d -> d == i ? (start+1:start+cs_dim) : Colon(), length(tensorsig))
            S[idx_tuple...] .+= spin_vec
        end
    end
    basis._cache[cache_key] = S
    return S
end

"""
    spintotal(basis::SpinBasis, tensorsig, spinindex)

Return the total spin for the given tensor signature at the given spin index.
`spinindex` should be a tuple or CartesianIndex indexing into `spin_weights`.
"""
function spintotal(basis::SpinBasis, tensorsig, spinindex)
    cache_key = (:spintotal, objectid(tensorsig), spinindex)
    cached = get(basis._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    val = Int(spin_weights(basis, tensorsig)[spinindex...])
    basis._cache[cache_key] = val
    return val
end

# ============================================================================
# PolarBasis  (abstract; for disk/annulus bases with dim=2)
# ============================================================================

"""
    PolarBasis <: SpinBasis

Abstract base type for 2D polar-type bases (azimuth + radius) such as
`DiskBasis` and `AnnulusBasis`.

`dim = 2`, `dims = ["azimuth", "radius"]`.

Concrete subtypes must have fields:
- `coordsys` — `PolarCoordinates` or similar 2D curvilinear system
- `shape::Tuple{Int,Int}` — (N_azimuth, N_radial)
- `dtype::DataType` — `Float64` or `ComplexF64`
- `k::Int` — regularity parameter
- `Nmax::Int` — `shape[2] - 1`
- `dealias_tuple::Tuple{Float64,Float64}`
- `mmax::Int` — `(shape[1] - 1) ÷ 2`
- `azimuth_basis` — Fourier sub-basis with permutations applied
- `azimuth_library` — library name
- `forward_m_perm::Union{Nothing,Vector{Int}}` — m permutation for triangular truncation
- `backward_m_perm::Union{Nothing,Vector{Int}}` — inverse m permutation
- `group_shape_val::Tuple{Int,Int}` — group shape
- `forward_transforms::Vector` — transform callables per sub-axis
- `backward_transforms::Vector` — transform callables per sub-axis
- `_cache::Dict{Symbol,Any}`

## m permutation arrays

For complex dtype (`ComplexF64`), when `mmax > 0`:
    az_index = 0:Nphi-1
    az_div, az_mod = divmod(az_index, 2)
    forward_m_perm = az_div + Nphi÷2 * az_mod  (+1 for 1-based)

For real dtype (`Float64`), when `mmax > 0`:
    az_index = 0:Nphi-1
    div2, mod2 = divmod(az_index, 2)
    div22 = div2 % 2
    forward_m_perm = (mod2 + div2) * (1 - div22) + (Nphi - 1 + mod2 - div2) * div22  (+1 for 1-based)
"""
abstract type PolarBasis <: SpinBasis end

"""
    polar_basis_dim(::PolarBasis) -> Int

PolarBasis always has dimension 2.
"""
basis_dim(::PolarBasis) = 2

"""
    polar_basis_dims(::PolarBasis)

Return the sub-axis names: `("azimuth", "radius")`.
"""
polar_basis_dims(::PolarBasis) = ("azimuth", "radius")

"""
    basis_subaxis_dependence(::PolarBasis) -> Tuple{Bool,Bool}

Both sub-axes have dependence.
"""
basis_subaxis_dependence(::PolarBasis) = (false, false)

"""
    basis_group_shape(b::PolarBasis)

Return the group shape for the basis.
"""
basis_group_shape(b::PolarBasis) = b.group_shape_val

# -- m permutation computation helpers --

"""
    compute_forward_m_perm_complex(Nphi::Int) -> Union{Nothing, Vector{Int}}

Compute the forward m permutation for complex dtype.
Returns `nothing` if mmax == 0 (i.e., Nphi leads to mmax = (Nphi-1)÷2 == 0).
"""
function compute_forward_m_perm_complex(Nphi::Int)
    mmax = (Nphi - 1) ÷ 2
    if mmax == 0
        return nothing
    end
    az_index = collect(0:Nphi-1)
    az_div = az_index .÷ 2
    az_mod = az_index .% 2
    perm_0based = az_div .+ (Nphi ÷ 2) .* az_mod
    return perm_0based .+ 1  # Convert to 1-based
end

"""
    compute_forward_m_perm_real(Nphi::Int) -> Union{Nothing, Vector{Int}}

Compute the forward m permutation for real dtype.
Returns `nothing` if mmax == 0.
"""
function compute_forward_m_perm_real(Nphi::Int)
    mmax = (Nphi - 1) ÷ 2
    if mmax == 0
        return nothing
    end
    az_index = collect(0:Nphi-1)
    div2 = az_index .÷ 2
    mod2 = az_index .% 2
    div22 = div2 .% 2
    perm_0based = (mod2 .+ div2) .* (1 .- div22) .+ (Nphi .- 1 .+ mod2 .- div2) .* div22
    return perm_0based .+ 1  # Convert to 1-based
end

"""
    compute_backward_m_perm(forward_perm::Union{Nothing, Vector{Int}}) -> Union{Nothing, Vector{Int}}

Compute the backward (inverse) m permutation from the forward permutation.
"""
function compute_backward_m_perm(forward_perm::Union{Nothing, Vector{Int}})
    if forward_perm === nothing
        return nothing
    end
    backward = similar(forward_perm)
    for (i, p) in enumerate(forward_perm)
        backward[p] = i
    end
    return backward
end

# -- PolarBasis methods --

"""
    matrix_dependence(b::PolarBasis, matrix_coupling)

If radial coupling is present, then azimuthal dependence must also be present.
Returns a copy of `matrix_coupling` with azimuthal dependence set if radial
coupling is active.
"""
function matrix_dependence(b::PolarBasis, matrix_coupling)
    md = copy(matrix_coupling)
    if matrix_coupling[2]
        md[1] = true
    end
    return md
end

"""
    valid_elements(b::PolarBasis, tensorsig, grid_space, elements)

Determine which elements are valid for the given tensor signature and
grid/coeff space configuration.
"""
function valid_elements(b::PolarBasis, tensorsig, grid_space, elements)
    if grid_space[2]
        # grid-grid and coeff-grid: same as Fourier validity
        return valid_elements(b.azimuth_basis, tensorsig, grid_space, elements)
    else
        # coeff-coeff space
        m, n = elements_to_groups(b, grid_space, elements)
        if !isempty(tensorsig)
            rank = length(tensorsig)
            # Expand m, n with leading singleton dimensions for tensor indices
            for _ in 1:rank
                m = reshape(m, 1, size(m)...)
                n = reshape(n, 1, size(n)...)
            end
        end
        valid = n .>= _nmin(b, m)
        if isempty(tensorsig)
            # Drop msin part of m = 0
            elem0 = elements[1]
            valid[(m .== 0) .& (elem0 .% 2 .== 1)] .= false
        end
        return valid
    end
end

"""
    _nmin(b::PolarBasis, m)

Return the minimum n for a given m.  Must be implemented by concrete subtypes
(DiskBasis, AnnulusBasis).  Default returns `0 * m` to preserve array shape.
"""
_nmin(b::PolarBasis, m) = 0 .* m

"""
    S1_basis(b::PolarBasis; radius=1)

Create a 1D azimuth (Fourier) basis with the polar basis's m permutations
applied.  The resulting basis can be used for azimuthal transforms and
validity checks.  Cached via `_cache`.
"""
function S1_basis(b::PolarBasis; radius=1)
    cache_key = (:S1_basis, radius)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    cs = basis_coordsys(b)
    coord0 = get_coords(cs)[1]
    Nphi = b.shape[1]
    bounds = (0.0, 2 * pi)
    dal = b.dealias_tuple[1]
    if b.dtype === ComplexF64
        s1 = ComplexFourier(coord0, Nphi, bounds; dealias=dal, library=b.azimuth_library)
    elseif b.dtype === Float64
        s1 = RealFourier(coord0, Nphi, bounds; dealias=dal, library=b.azimuth_library)
    else
        throw(ErrorException("Unsupported dtype: $(b.dtype)"))
    end
    # Apply m permutations
    s1.forward_coeff_permutation  = b.forward_m_perm
    s1.backward_coeff_permutation = b.backward_m_perm
    b._cache[cache_key] = s1
    return s1
end

"""
    global_shape(b::PolarBasis, grid_space, scales)

Compute the global data shape for the given grid/coeff space and scale factors.
"""
function global_shape(b::PolarBasis, grid_space, scales)
    gs = grid_shape(b, scales)
    if grid_space[1]
        # grid-grid space
        if b.mmax == 0
            return (1, gs[2])
        else
            return gs
        end
    elseif grid_space[2]
        # coeff-grid space
        return (b.shape[1], gs[2])
    else
        # coeff-coeff space
        Nphi = b.shape[1]
        if b.dtype === ComplexF64
            return b.shape
        elseif b.dtype === Float64
            if Nphi > 1
                return b.shape
            else
                return (2, b.shape[2])
            end
        end
    end
end

"""
    chunk_shape(b::PolarBasis, grid_space)

Compute the chunk shape for distribution.
"""
function chunk_shape(b::PolarBasis, grid_space)
    if grid_space[1]
        # grid-grid space
        return (1, 1)
    elseif grid_space[2]
        # coeff-grid space
        if b.dtype === ComplexF64
            return (1, 1)
        elseif b.dtype === Float64
            if b.mmax == 0
                return (1, 1)
            else
                return (2, 1)
            end
        end
    else
        # coeff-coeff space
        if b.dtype === ComplexF64
            return (1, 1)
        elseif b.dtype === Float64
            return (2, 1)
        end
    end
end

"""
    elements_to_groups(b::PolarBasis, grid_space, elements)

Convert element indices to group indices (m, n pairs).
In coeff-coeff space, elements below `_nmin(m)` are masked.
"""
function elements_to_groups(b::PolarBasis, grid_space, elements)
    s1_groups = elements_to_groups(b.azimuth_basis, grid_space, elements[1:1])
    radial_groups = elements[2]
    if isa(s1_groups, Tuple)
        groups = [s1_groups..., radial_groups]
    else
        groups = [s1_groups, radial_groups]
    end
    if !grid_space[2]
        # coeff-coeff space: mask invalid elements
        m = groups[1]
        n = groups[end]
        nmin = _nmin(b, m)
        # Elements below nmin are invalid (would be masked in Python)
        # For Julia, we return the groups as-is; validity is checked by valid_elements
    end
    return groups
end

"""
    n_size(b::PolarBasis, m::Int) -> Int

Return the number of valid radial modes for azimuthal wavenumber `m`.
"""
function n_size(b::PolarBasis, m::Int)
    cache_key = (:n_size, m)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    nmin_val = _nmin(b, m)
    nmax = b.shape[2] - 1  # Nmax
    val = nmax - nmin_val + 1
    b._cache[cache_key] = val
    return val
end

"""
    n_slice(b::PolarBasis, m::Int) -> UnitRange{Int}

Return the range of valid radial mode indices for azimuthal wavenumber `m`.
Returns a 1-based range.
"""
function n_slice(b::PolarBasis, m::Int)
    cache_key = (:n_slice, m)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    nmin_val = _nmin(b, m)
    nmax = b.shape[2] - 1
    val = (nmin_val+1):(nmax+1)  # 1-based range
    b._cache[cache_key] = val
    return val
end

# -- Azimuth transform methods --

"""
    forward_transform_azimuth_Mmax0(b::PolarBasis, field, axis, gdata, cdata)

Forward azimuth transform when mmax == 0 (copy the single m=0 slice).
"""
function forward_transform_azimuth_Mmax0(b::PolarBasis, field, axis, gdata, cdata)
    slice_axis = axis + length(field.tensorsig)
    idx = axslice(slice_axis, 1, 1)  # 1-based: first element only
    copyto!(view(cdata, idx...), gdata)
    return nothing
end

"""
    forward_transform_azimuth(b::PolarBasis, field, axis, gdata, cdata)

Forward azimuth transform: delegate to the azimuth Fourier basis.
Permutation is handled by the azimuth_basis's own permutation arrays.
"""
function forward_transform_azimuth(b::PolarBasis, field, axis, gdata, cdata)
    forward_transform(b.azimuth_basis, field, axis, gdata, cdata)
    return nothing
end

"""
    backward_transform_azimuth_Mmax0(b::PolarBasis, field, axis, cdata, gdata)

Backward azimuth transform when mmax == 0 (copy from the single m=0 slice).
"""
function backward_transform_azimuth_Mmax0(b::PolarBasis, field, axis, cdata, gdata)
    slice_axis = axis + length(field.tensorsig)
    idx = axslice(slice_axis, 1, 1)
    copyto!(gdata, view(cdata, idx...))
    return nothing
end

"""
    backward_transform_azimuth(b::PolarBasis, field, axis, cdata, gdata)

Backward azimuth transform: delegate to the azimuth Fourier basis.
"""
function backward_transform_azimuth(b::PolarBasis, field, axis, cdata, gdata)
    backward_transform(b.azimuth_basis, field, axis, cdata, gdata)
    return nothing
end

# -- Grid methods --

"""
    global_grids(b::PolarBasis, dist, scales)

Return tuple of (azimuth_grid, radius_grid) at the given scales.
Delegates to the azimuth sub-basis and the concrete subtype's
`global_grid_radius` method.
"""
function global_grids(b::PolarBasis, dist, scales)
    return (global_grid(b.azimuth_basis, dist, scales[1]),
            global_grid_radius(b, dist, scales[2]))
end

"""
    local_grids(b::PolarBasis, dist, scales)

Return tuple of (local_azimuth_grid, local_radius_grid).
"""
function local_grids(b::PolarBasis, dist, scales)
    return (local_grid(b.azimuth_basis, dist, scales[1]),
            local_grid_radius(b, dist, scales[2]))
end

"""
    global_grid_radius(b::PolarBasis, dist, scale)

Return the global radial grid.  Must be implemented by concrete subtypes.
"""
function global_grid_radius end

"""
    local_grid_radius(b::PolarBasis, dist, scale)

Return the local radial grid.  Must be implemented by concrete subtypes.
"""
function local_grid_radius end

# -- m_maps for distribution --

"""
    m_maps(b::PolarBasis, dist)

Compute a tuple of `(m, mg_slice, mc_slice, n_slice)` entries for all
local m values.  Used for radial transforms and local element iteration.

Must be implemented by concrete subtypes or overridden once the
distribution infrastructure is in place.
"""
function m_maps(b::PolarBasis, dist)
    cache_key = (:m_maps, objectid(dist))
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    # Placeholder: will be implemented when distributor infrastructure is complete.
    # For serial (non-distributed) usage, iterate over all m values.
    error("m_maps not yet implemented for $(typeof(b)); requires distributor infrastructure")
end

"""
    ell_reversed(b::PolarBasis, dist)

Return a dictionary mapping m -> Bool indicating whether the ell ordering
is reversed for each m.  Default: all false.
"""
function ell_reversed(b::PolarBasis, dist)
    cache_key = (:ell_reversed, objectid(dist))
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    result = Dict{Int, Bool}()
    for (m, mg_slice, mc_slice, ns) in m_maps(b, dist)
        result[m] = false
    end
    b._cache[cache_key] = result
    return result
end

"""
    derivative_basis(b::PolarBasis; order::Int=1)

Return the derivative basis with regularity parameter `k` incremented by `order`.
Delegates to `clone_with(b; k=b.k + order)`.
"""
function derivative_basis(b::PolarBasis; order::Int=1)
    k_new = b.k + order
    return clone_with(b; k=k_new)
end

# ============================================================================
# Exports
# ============================================================================

export AffineCOV,
       problem_coord,
       native_coord,
       Basis,
       AbstractIntervalBasis,
       IntervalBasis,
       basis_coord,
       basis_coordsys,
       basis_size,
       basis_shape,
       basis_dealias,
       basis_dim,
       basis_constant,
       basis_group_shape,
       basis_subaxis_dependence,
       basis_domain,
       basis_add,
       basis_radd,
       basis_mul,
       basis_rmul,
       basis_matmul,
       basis_rmatmul,
       ncc_matrix,
       product_matrix,
       clone_with,
       CardinalBasis,
       IntervalBasis,
       JacobiBasis,
       Jacobi,
       Legendre,
       Ultraspherical,
       ChebyshevT,
       ChebyshevU,
       ChebyshevV,
       Chebyshev,
       FourierBase,
       ComplexFourierBasis,
       ComplexFourier,
       RealFourierBasis,
       RealFourier,
       Fourier,
       native_wavenumbers,
       wavenumbers,
       elements_to_groups,
       valid_elements,
       matrix_dependence,
       global_grids,
       global_grid,
       local_grids,
       local_grid,
       global_grid_spacing,
       local_modes,
       global_shape,
       chunk_shape,
       forward_transform,
       backward_transform,
       transform_plan,
       _native_grid,
       jacobi_recurrence_matrix,
       derivative_basis,
       convert_jacobi_matrix,
       convert_constant_jacobi_matrix,
       differentiate_jacobi_output_basis,
       differentiate_jacobi_matrix,
       interpolate_jacobi_matrix,
       integrate_jacobi_matrix,
       average_jacobi_matrix,
       convert_constant_complex_fourier_matrix,
       differentiate_complex_fourier_matrix,
       interpolate_complex_fourier_matrix,
       integrate_complex_fourier_matrix,
       average_complex_fourier_matrix,
       convert_constant_real_fourier_matrix,
       differentiate_real_fourier_matrix,
       interpolate_real_fourier_matrix,
       integrate_real_fourier_matrix,
       average_real_fourier_matrix,
       convert_constant_cardinal_matrix,
       interpolate_cardinal_matrix,
       integrate_cardinal_matrix,
       average_cardinal_matrix,
       AbstractTransformPlan,
       MatrixTransformPlan,
       forward!,
       backward!,
       MultidimensionalBasis,
       reduced_view_5,
       recombine_forward!,
       recombine_backward!,
       SpinRecombinationTrait,
       HasSpinRecombination,
       NoSpinRecombination,
       spin_recombination_trait,
       spin_ordering,
       spin_recombination_factors,
       spin_recombination_matrix,
       forward_spin_recombination!,
       backward_spin_recombination!,
       SpinBasis,
       get_mmax,
       get_azimuth_basis,
       spin_weights,
       spintotal,
       PolarBasis,
       polar_basis_dims,
       compute_forward_m_perm_complex,
       compute_forward_m_perm_real,
       compute_backward_m_perm,
       S1_basis,
       n_size,
       n_slice,
       forward_transform_azimuth_Mmax0,
       forward_transform_azimuth,
       backward_transform_azimuth_Mmax0,
       backward_transform_azimuth,
       global_grid_radius,
       local_grid_radius,
       m_maps,
       ell_reversed,
       AnnulusBasis,
       DiskBasis,
       SphereBasis,
       ell_size,
       latitude_basis

# ============================================================================
# AnnulusBasis  (concrete polar basis for annular domains)
# ============================================================================

"""
    AnnulusBasis <: PolarBasis

Annular basis on a 2D polar domain with inner and outer radii.
Uses Jacobi polynomials in the radial direction and Fourier in azimuth.

The radial coordinate is mapped to [-1, 1] via:
    z = 2*(r - r_inner) / dR - rho
where dR = r_outer - r_inner and rho = (r_outer + r_inner) / dR.
"""
mutable struct AnnulusBasis <: PolarBasis
    coordsys::PolarCoordinates
    shape::Tuple{Int,Int}
    dtype::DataType
    radii::Tuple{Float64,Float64}
    k::Int
    alpha::Tuple{Float64,Float64}
    dealias_tuple::Tuple{Float64,Float64}
    azimuth_library::String
    radius_library::String
    volume::Float64
    dR::Float64
    rho::Float64
    grid_params::Tuple
    inner_edge
    outer_edge
    Nmax::Int
    mmax::Int
    forward_transforms::Vector
    backward_transforms::Vector
    forward_m_perm::Union{Nothing,Vector{Int}}
    backward_m_perm::Union{Nothing,Vector{Int}}
    group_shape_val::Tuple{Int,Int}
    azimuth_basis
    _cache::Dict{Symbol,Any}
end

# -- Class constants --
basis_subaxis_dependence(::AnnulusBasis) = (false, true)

# -- Constructor caching --
const _annulus_cache = Dict{Any,WeakRef}()

"""
    AnnulusBasis(coordsys, shape, dtype; radii=(1,2), k=0, alpha=(-0.5,-0.5),
                 dealias=(1,1), azimuth_library=nothing, radius_library=nothing)

Construct (or retrieve cached) an AnnulusBasis.
"""
function AnnulusBasis(coordsys::PolarCoordinates, shape, dtype;
                      radii=(1.0, 2.0), k::Int=0,
                      alpha=(-0.5, -0.5),
                      dealias=(1, 1),
                      azimuth_library=nothing,
                      radius_library=nothing)
    # Preprocess arguments
    if !(coordsys isa PolarCoordinates)
        throw(ArgumentError("Annulus coordsys must be PolarCoordinates."))
    end
    shape = (Int(shape[1]), Int(shape[2]))
    if length(shape) != 2
        throw(ArgumentError("Annulus shape must have length 2."))
    end
    radii = (Float64(radii[1]), Float64(radii[2]))
    if min(radii...) <= 0
        throw(ArgumentError("Annulus radii must be positive."))
    end
    if radii[1] >= radii[2]
        throw(ArgumentError("Annulus radii must be in increasing order."))
    end
    k = Int(k)
    if isa(alpha, Number)
        alpha = (Float64(alpha), Float64(alpha))
    else
        alpha = (Float64(alpha[1]), Float64(alpha[2]))
    end
    if isa(dealias, Number)
        dealias = (Float64(dealias), Float64(dealias))
    else
        dealias = (Float64(dealias[1]), Float64(dealias[2]))
    end
    if azimuth_library === nothing
        azimuth_library = FOURIER_DEFAULT_LIBRARY
    end
    if radius_library === nothing
        if alpha[1] == alpha[2] == -0.5
            radius_library = JACOBI_DEFAULT_DCT
        else
            radius_library = JACOBI_DEFAULT_LIBRARY
        end
    end

    # Cache lookup
    cache_key = (objectid(coordsys), shape, dtype, radii, k, alpha, dealias, azimuth_library, radius_library)
    wr = get(_annulus_cache, cache_key, nothing)
    if wr !== nothing
        inst = wr.value
        if inst !== nothing
            return inst::AnnulusBasis
        end
    end

    # Computed attributes
    vol = pi * (radii[2]^2 - radii[1]^2)
    dR = radii[2] - radii[1]
    rho = (radii[2] + radii[1]) / dR
    grid_params = (objectid(coordsys), dtype, radii, alpha, dealias, azimuth_library, radius_library)
    Nmax = shape[2] - 1
    mmax = (shape[1] - 1) ÷ 2

    # Build azimuth basis
    coord0 = get_coords(coordsys)[1]
    Nphi = shape[1]
    az_bounds = (0.0, 2 * pi)
    if dtype === ComplexF64
        az_basis = ComplexFourier(coord0, Nphi, az_bounds; dealias=dealias[1], library=azimuth_library)
    elseif dtype === Float64
        az_basis = RealFourier(coord0, Nphi, az_bounds; dealias=dealias[1], library=azimuth_library)
    else
        throw(ArgumentError("Unsupported dtype: $dtype"))
    end

    # m permutations
    if dtype === ComplexF64
        fwd_perm = compute_forward_m_perm_complex(Nphi)
        group_shp = (1, 1)
    elseif dtype === Float64
        fwd_perm = compute_forward_m_perm_real(Nphi)
        group_shp = (2, 1)
    end
    bwd_perm = compute_backward_m_perm(fwd_perm)

    # Apply permutations to azimuth basis
    if mmax > 0
        az_basis.forward_coeff_permutation = fwd_perm
        az_basis.backward_coeff_permutation = bwd_perm
    end

    _cache = Dict{Symbol,Any}()

    # Placeholder: inner/outer edges will be set after construction
    inst = AnnulusBasis(
        coordsys, shape, dtype, radii, k, alpha, dealias,
        azimuth_library, radius_library,
        vol, dR, rho, grid_params,
        nothing, nothing,  # inner_edge, outer_edge -- set below
        Nmax, mmax,
        Any[], Any[],  # transforms -- set below
        fwd_perm, bwd_perm,
        group_shp, az_basis, _cache
    )

    # Set transform callables
    if mmax == 0
        push!(inst.forward_transforms, (field, axis, gdata, cdata) -> forward_transform_azimuth_Mmax0(inst, field, axis, gdata, cdata))
        push!(inst.backward_transforms, (field, axis, cdata, gdata) -> backward_transform_azimuth_Mmax0(inst, field, axis, cdata, gdata))
    else
        push!(inst.forward_transforms, (field, axis, gdata, cdata) -> forward_transform_azimuth(inst, field, axis, gdata, cdata))
        push!(inst.backward_transforms, (field, axis, cdata, gdata) -> backward_transform_azimuth(inst, field, axis, cdata, gdata))
    end
    push!(inst.forward_transforms, (field, axis, gdata, cdata) -> forward_transform_radius(inst, field, axis, gdata, cdata))
    push!(inst.backward_transforms, (field, axis, cdata, gdata) -> backward_transform_radius(inst, field, axis, cdata, gdata))

    # Set edge bases
    inst.inner_edge = S1_basis(inst; radius=radii[1])
    inst.outer_edge = S1_basis(inst; radius=radii[2])

    _annulus_cache[cache_key] = WeakRef(inst)
    return inst
end

# -- nmin: always 0 for annulus --
_nmin(::AnnulusBasis, m) = 0 .* m

"""
    radial_basis(b::AnnulusBasis)

Return a radial-only version of this annulus basis (shape[1] = 1).
"""
function radial_basis(b::AnnulusBasis)
    cached = get(b._cache, :radial_basis, nothing)
    if cached !== nothing
        return cached
    end
    rb = clone_with(b; shape=(1, b.shape[2]))
    b._cache[:radial_basis] = rb
    return rb
end

# -- Basis algebra --

function basis_add(a::AnnulusBasis, other)
    if other === nothing || other === a
        return a
    end
    if other isa AnnulusBasis
        if a.grid_params == other.grid_params
            shape = (max(a.shape[1], other.shape[1]), max(a.shape[2], other.shape[2]))
            k_new = max(a.k, other.k)
            return clone_with(a; shape=shape, k=k_new)
        end
    end
    return nothing
end

function basis_mul(a::AnnulusBasis, other)
    if other === nothing
        return a
    end
    if other isa AnnulusBasis
        if a.grid_params == other.grid_params
            shape = (max(a.shape[1], other.shape[1]), max(a.shape[2], other.shape[2]))
            k_new = a.k + other.k
            return clone_with(a; shape=shape, k=k_new)
        end
    end
    return nothing
end

function basis_rmatmul(operand::AnnulusBasis, ncc)
    # NCC (ncc) * operand (self) -- same as mul for annulus
    return basis_mul(operand, ncc)
end

# -- Grid methods --

"""
    global_grid_radius(b::AnnulusBasis, dist, scale)

Return the global radial grid for the annulus.
"""
function global_grid_radius(b::AnnulusBasis, dist, scale)
    r = _radius_grid(b, scale)
    return reshape_vector(r, get_dim(dist), get_basis_axis(dist, b) + 1)
end

"""
    local_grid_radius(b::AnnulusBasis, dist, scale)

Return the local radial grid for the annulus.
"""
function local_grid_radius(b::AnnulusBasis, dist, scale)
    r = _radius_grid(b, scale)
    return reshape_vector(r, get_dim(dist), get_basis_axis(dist, b) + 1)
end

"""
    _radius_grid(b::AnnulusBasis, scale)

Compute the radial grid points at the given scale. Uses Jacobi quadrature.
"""
function _radius_grid(b::AnnulusBasis, scale)
    cache_key = (:_radius_grid, scale)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    N = Int(ceil(scale * b.shape[2]))
    z, weights = jacobi_quadrature(N, b.alpha[1], b.alpha[2])
    r = (b.dR / 2) .* (z .+ b.rho)
    result = Float64.(r)
    b._cache[cache_key] = result
    return result
end

"""
    _radius_weights(b::AnnulusBasis, scale)

Compute the radial quadrature weights at the given scale.
"""
function _radius_weights(b::AnnulusBasis, scale)
    cache_key = (:_radius_weights, scale)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    N = Int(ceil(scale * b.shape[2]))
    z_proj, weights_proj = jacobi_quadrature(N, b.alpha[1], b.alpha[2])
    z0, weights0 = jacobi_quadrature(N, 0.0, 0.0)
    Q0 = jacobi_polynomials(N, b.alpha[1], b.alpha[2], z0)
    Q_proj = jacobi_polynomials(N, b.alpha[1], b.alpha[2], z_proj)
    normalization = b.dR / 2
    result = normalization * transpose(Q0 * Diagonal(weights0)) * (Diagonal(weights_proj) * Q_proj)
    b._cache[cache_key] = result
    return result
end

"""
    constant_mode_value(b::AnnulusBasis)

Return the value of the zeroth radial mode (constant mode) for k=0.
"""
function constant_mode_value(b::AnnulusBasis)
    cached = get(b._cache, :constant_mode_value, nothing)
    if cached !== nothing
        return cached
    end
    Q0 = jacobi_polynomials(1, b.alpha[1], b.alpha[2], [0.0])
    val = Q0[1, 1]
    b._cache[:constant_mode_value] = val
    return val
end

"""
    transform_plan(b::AnnulusBasis, dist, grid_size, k)

Build (or retrieve cached) a transform plan for the annulus radial direction.
"""
function transform_plan(b::AnnulusBasis, dist, grid_size, k)
    cache_key = (:transform_plan, objectid(dist), grid_size, k)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    a = b.alpha[1] + k
    bparam = b.alpha[2] + k
    a0 = b.alpha[1]
    b0 = b.alpha[2]
    # Build matrix transform plan using Jacobi polynomials
    N = b.Nmax + 1
    z, w = jacobi_quadrature(grid_size, a0, b0)
    P = jacobi_polynomials(N, a, bparam, z)
    fwd = transpose(P) * Diagonal(w)
    bwd = P
    plan = MatrixTransformPlan(Matrix{Float64}(fwd), Matrix{Float64}(bwd))
    b._cache[cache_key] = plan
    return plan
end

"""
    forward_transform_radius(b::AnnulusBasis, field, axis, gdata, cdata)

Forward radial transform for annulus basis. Applies spin recombination
and radial factor before the Jacobi transform.
"""
function forward_transform_radius(b::AnnulusBasis, field, axis, gdata, cdata)
    data_axis = length(field.tensorsig) + axis
    grid_size = size(gdata, data_axis)
    # Multiply by radial factor for k > 0
    if b.k > 0
        rfactor = _radial_transform_factor(b, field.scales[axis], data_axis, -b.k)
        gdata = gdata .* rfactor
    end
    # Expand gdata if mmax=0 and dtype=float for spin recombination
    if b.mmax == 0 && b.dtype === Float64
        m_axis = length(field.tensorsig) + axis - 1
        gdata = cat(gdata, zeros(eltype(gdata), size(gdata)); dims=m_axis)
    end
    # Apply spin recombination from gdata to temp
    temp = zeros(eltype(gdata), size(gdata))
    forward_spin_recombination!(b, field.tensorsig, axis, gdata, temp)
    fill!(cdata, zero(eltype(cdata)))
    # Transform component-by-component
    S = spin_weights(b, field.tensorsig)
    plan = transform_plan(b, field.dist, grid_size, b.k)
    for i in CartesianIndices(S)
        forward!(plan, view(temp, i), view(cdata, i), axis)
    end
    return nothing
end

"""
    backward_transform_radius(b::AnnulusBasis, field, axis, cdata, gdata)

Backward radial transform for annulus basis.
"""
function backward_transform_radius(b::AnnulusBasis, field, axis, cdata, gdata)
    data_axis = length(field.tensorsig) + axis
    grid_size = size(gdata, data_axis)
    # Handle mmax=0 float expansion
    if b.mmax == 0 && b.dtype === Float64
        m_axis = length(field.tensorsig) + axis - 1
        shp = collect(size(gdata))
        shp[m_axis] = 2
        temp = zeros(eltype(gdata), Tuple(shp))
        gdata_orig = gdata
        gdata = zeros(eltype(gdata), Tuple(shp))
    else
        temp = zeros(eltype(gdata), size(gdata))
    end
    # Transform component-by-component
    S = spin_weights(b, field.tensorsig)
    plan = transform_plan(b, field.dist, grid_size, b.k)
    for i in CartesianIndices(S)
        backward!(plan, view(cdata, i), view(temp, i), axis)
    end
    # Apply backward spin recombination
    fill!(gdata, zero(eltype(gdata)))
    backward_spin_recombination!(b, field.tensorsig, axis, temp, gdata)
    # Multiply by radial factor for k > 0
    if b.k > 0
        rfactor = _radial_transform_factor(b, field.scales[axis], data_axis, b.k)
        gdata .*= rfactor
    end
    # Collapse expanded gdata back
    if b.mmax == 0 && b.dtype === Float64
        m_axis = length(field.tensorsig) + axis - 1
        idx = axslice(m_axis, 1, 1)
        copyto!(gdata_orig, view(gdata, idx...))
    end
    return nothing
end

"""
    _radial_transform_factor(b::AnnulusBasis, scale, data_axis, dk)

Compute the radial prefactor (dR/r)^dk for transforms.
"""
function _radial_transform_factor(b::AnnulusBasis, scale, data_axis, dk)
    cache_key = (:radial_transform_factor, scale, data_axis, dk)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    r = reshape_vector(_radius_grid(b, scale), data_axis, data_axis - 1)
    result = (b.dR ./ r) .^ dk
    b._cache[cache_key] = result
    return result
end

"""
    operator_matrix(b::AnnulusBasis, op, m, spintotal; size=nothing)

Build a radial operator matrix using dedalus_sphere shell operators.
"""
function operator_matrix(b::AnnulusBasis, op, m, spintotal; size=nothing)
    cache_key = (:operator_matrix, op, m, spintotal, size)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    ms = m + spintotal
    if length(op) > 0 && op[end] in ('+', '-')
        o = op[1:end-1]
        p = op[end] == '+' ? 1 : -1
        if ms == 0
            p = 1
        elseif ms < 0
            p = -p
            ms = -ms
        end
        operator = shell_operator(2, b.radii, o; alpha=b.alpha)
        mat = operator(p, ms)
    elseif op == "L"
        D = shell_operator(2, b.radii, "D"; alpha=b.alpha)
        if ms < 0
            mat = compose(D(+1, ms - 1), D(-1, ms))
        else
            mat = compose(D(-1, ms + 1), D(+1, ms))
        end
    else
        operator = shell_operator(2, b.radii, op; alpha=b.alpha)
        mat = operator
    end
    if size === nothing
        size = n_size(b, m)
    end
    result = Float64.(resize_matrix(mat, size, b.k))
    b._cache[cache_key] = result
    return result
end

"""
    conversion_matrix(b::AnnulusBasis, m, spintotal, dk)

Build the Jacobi conversion matrix for the annulus.
"""
function conversion_matrix(b::AnnulusBasis, m, spintotal, dk)
    cache_key = (:conversion_matrix, m, spintotal, dk)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    E = shell_operator(2, b.radii, "E"; alpha=b.alpha)
    # Apply dk-fold conversion
    op = E
    for _ in 2:dk
        op = compose(op, E)
    end
    result = Float64.(resize_matrix(op, n_size(b, m), b.k))
    b._cache[cache_key] = result
    return result
end

"""
    jacobi_conversion(b::AnnulusBasis, m, dk; size=nothing)

Build the Jacobi conversion matrix (AB operator).
"""
function jacobi_conversion(b::AnnulusBasis, m, dk; size=nothing)
    cache_key = (:jacobi_conversion, m, dk, size)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    AB = shell_operator(2, b.radii, "AB"; alpha=b.alpha)
    op = AB
    for _ in 2:dk
        op = compose(op, AB)
    end
    if size === nothing
        size = n_size(b, m)
    end
    result = Float64.(resize_matrix(op, size, b.k))
    b._cache[cache_key] = result
    return result
end

"""
    clone_with(b::AnnulusBasis; kwargs...)

Create a copy of the annulus basis with some fields replaced.
"""
function clone_with(b::AnnulusBasis; kwargs...)
    kw = Dict{Symbol,Any}(kwargs)
    coordsys_val = get(kw, :coordsys, b.coordsys)
    shape_val = get(kw, :shape, b.shape)
    dtype_val = get(kw, :dtype, b.dtype)
    radii_val = get(kw, :radii, b.radii)
    k_val = get(kw, :k, b.k)
    alpha_val = get(kw, :alpha, b.alpha)
    dealias_val = get(kw, :dealias, b.dealias_tuple)
    az_lib = get(kw, :azimuth_library, b.azimuth_library)
    r_lib = get(kw, :radius_library, b.radius_library)
    return AnnulusBasis(coordsys_val, shape_val, dtype_val;
                        radii=radii_val, k=k_val, alpha=alpha_val,
                        dealias=dealias_val, azimuth_library=az_lib,
                        radius_library=r_lib)
end

"""
    _last_axis_component_ncc_matrix(::Type{AnnulusBasis}, subproblem, ncc_basis,
        arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp,
        ncc_tensorsig, arg_tensorsig, out_tensorsig; cutoff=1e-6)

Build NCC component matrix for annulus basis via Clenshaw algorithm.
"""
function _last_axis_component_ncc_matrix(::Type{AnnulusBasis}, subproblem, ncc_basis,
        arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp,
        ncc_tensorsig, arg_tensorsig, out_tensorsig; cutoff::Float64=1e-6)
    error("AnnulusBasis _last_axis_component_ncc_matrix: requires subproblem infrastructure not yet available")
end

function Base.show(io::IO, b::AnnulusBasis)
    print(io, "AnnulusBasis($(b.coordsys), shape=$(b.shape), radii=$(b.radii), k=$(b.k), alpha=$(b.alpha))")
end

# ============================================================================
# DiskBasis  (concrete polar basis for disk domains)
# ============================================================================

"""
    DiskBasis <: PolarBasis

Disk basis on a 2D polar domain with a single radius.
Uses Zernike polynomials in the radial direction and Fourier in azimuth.
Employs triangular truncation: nmin(m) = |m| div 2.
"""
mutable struct DiskBasis <: PolarBasis
    coordsys::PolarCoordinates
    shape::Tuple{Int,Int}
    dtype::DataType
    radius::Float64
    k::Int
    alpha::Float64
    dealias_tuple::Tuple{Float64,Float64}
    azimuth_library::String
    radius_library::String
    volume::Float64
    radial_COV::AffineCOV
    grid_params::Tuple
    edge
    Nmax::Int
    mmax::Int
    forward_transforms::Vector
    backward_transforms::Vector
    forward_m_perm::Union{Nothing,Vector{Int}}
    backward_m_perm::Union{Nothing,Vector{Int}}
    group_shape_val::Tuple{Int,Int}
    azimuth_basis
    _cache::Dict{Symbol,Any}
end

# -- Class constants --
basis_subaxis_dependence(::DiskBasis) = (true, true)

# -- Constructor caching --
const _disk_cache = Dict{Any,WeakRef}()

"""
    DiskBasis(coordsys, shape, dtype; radius=1.0, k=0, alpha=0.0,
              dealias=(1,1), azimuth_library=nothing, radius_library=nothing)

Construct (or retrieve cached) a DiskBasis.
"""
function DiskBasis(coordsys::PolarCoordinates, shape, dtype;
                   radius::Real=1.0, k::Int=0,
                   alpha::Real=0.0,
                   dealias=(1, 1),
                   azimuth_library=nothing,
                   radius_library=nothing)
    # Preprocess arguments
    if !(coordsys isa PolarCoordinates)
        throw(ArgumentError("Disk coordsys must be PolarCoordinates."))
    end
    shape = (Int(shape[1]), Int(shape[2]))
    if length(shape) != 2
        throw(ArgumentError("Disk shape must have length 2."))
    end
    radius = Float64(radius)
    if radius <= 0
        throw(ArgumentError("Disk radius must be positive."))
    end
    k = Int(k)
    alpha = Float64(alpha)
    if isa(dealias, Number)
        dealias = (Float64(dealias), Float64(dealias))
    else
        dealias = (Float64(dealias[1]), Float64(dealias[2]))
    end
    if azimuth_library === nothing
        azimuth_library = FOURIER_DEFAULT_LIBRARY
    end
    if radius_library === nothing
        radius_library = JACOBI_DEFAULT_LIBRARY
    end

    # Cache lookup
    cache_key = (objectid(coordsys), shape, dtype, radius, k, alpha, dealias, azimuth_library, radius_library)
    wr = get(_disk_cache, cache_key, nothing)
    if wr !== nothing
        inst = wr.value
        if inst !== nothing
            return inst::DiskBasis
        end
    end

    # Computed attributes
    vol = pi * radius^2
    radial_cov = AffineCOV((0.0, 1.0), (0.0, radius))
    Nmax = shape[2] - 1
    mmax = (shape[1] - 1) ÷ 2
    grid_params = (objectid(coordsys), dtype, radius, alpha, dealias, azimuth_library, radius_library)

    if mmax > 2 * Nmax
        @warn "You are using more azimuthal modes than can be resolved with your current radial resolution"
    end

    # Build azimuth basis
    coord0 = get_coords(coordsys)[1]
    Nphi = shape[1]
    az_bounds = (0.0, 2 * pi)
    if dtype === ComplexF64
        az_basis = ComplexFourier(coord0, Nphi, az_bounds; dealias=dealias[1], library=azimuth_library)
    elseif dtype === Float64
        az_basis = RealFourier(coord0, Nphi, az_bounds; dealias=dealias[1], library=azimuth_library)
    else
        throw(ArgumentError("Unsupported dtype: $dtype"))
    end

    # m permutations
    if dtype === ComplexF64
        fwd_perm = compute_forward_m_perm_complex(Nphi)
        group_shp = (1, 1)
    elseif dtype === Float64
        fwd_perm = compute_forward_m_perm_real(Nphi)
        group_shp = (2, 1)
    end
    bwd_perm = compute_backward_m_perm(fwd_perm)

    # Apply permutations to azimuth basis
    if mmax > 0
        az_basis.forward_coeff_permutation = fwd_perm
        az_basis.backward_coeff_permutation = bwd_perm
    end

    _cache = Dict{Symbol,Any}()

    inst = DiskBasis(
        coordsys, shape, dtype, radius, k, alpha, dealias,
        azimuth_library, radius_library,
        vol, radial_cov, grid_params,
        nothing,  # edge -- set below
        Nmax, mmax,
        Any[], Any[],  # transforms -- set below
        fwd_perm, bwd_perm,
        group_shp, az_basis, _cache
    )

    # Set transform callables
    if mmax == 0
        push!(inst.forward_transforms, (field, axis, gdata, cdata) -> forward_transform_azimuth_Mmax0(inst, field, axis, gdata, cdata))
        push!(inst.backward_transforms, (field, axis, cdata, gdata) -> backward_transform_azimuth_Mmax0(inst, field, axis, cdata, gdata))
    else
        push!(inst.forward_transforms, (field, axis, gdata, cdata) -> forward_transform_azimuth(inst, field, axis, gdata, cdata))
        push!(inst.backward_transforms, (field, axis, cdata, gdata) -> backward_transform_azimuth(inst, field, axis, cdata, gdata))
    end
    push!(inst.forward_transforms, (field, axis, gdata, cdata) -> forward_transform_radius_disk(inst, field, axis, gdata, cdata))
    push!(inst.backward_transforms, (field, axis, cdata, gdata) -> backward_transform_radius_disk(inst, field, axis, cdata, gdata))

    # Set edge basis
    inst.edge = S1_basis(inst; radius=radius)

    _disk_cache[cache_key] = WeakRef(inst)
    return inst
end

# -- nmin: triangular truncation for disk --
_nmin(::DiskBasis, m) = abs.(m) .÷ 2

"""
    radial_basis(b::DiskBasis)

Return a radial-only version of this disk basis (shape[1] = 1).
"""
function radial_basis(b::DiskBasis)
    cached = get(b._cache, :radial_basis, nothing)
    if cached !== nothing
        return cached
    end
    rb = clone_with(b; shape=(1, b.shape[2]))
    b._cache[:radial_basis] = rb
    return rb
end

# -- Basis algebra --

function basis_add(a::DiskBasis, other)
    if other === nothing || other === a
        return a
    end
    if other isa DiskBasis
        if a.grid_params == other.grid_params
            shape = (max(a.shape[1], other.shape[1]), max(a.shape[2], other.shape[2]))
            k_new = max(a.k, other.k)
            return clone_with(a; shape=shape, k=k_new)
        end
    end
    return nothing
end

function basis_mul(a::DiskBasis, other)
    if other === nothing
        return a
    end
    if other isa DiskBasis
        if a.grid_params == other.grid_params
            shape = (max(a.shape[1], other.shape[1]), max(a.shape[2], other.shape[2]))
            k_new = 0  # minimizes k in RHS expressions
            return clone_with(a; shape=shape, k=k_new)
        end
    end
    return nothing
end

function basis_rmatmul(operand::DiskBasis, ncc)
    if ncc === nothing
        return operand
    end
    if ncc isa DiskBasis
        if operand.grid_params == ncc.grid_params
            shape = (max(operand.shape[1], ncc.shape[1]), max(operand.shape[2], ncc.shape[2]))
            k_new = operand.k  # use operand's k
            return clone_with(operand; shape=shape, k=k_new)
        end
    end
    return nothing
end

# -- Grid methods --

"""
    global_grid_radius(b::DiskBasis, dist, scale)

Return the global radial grid for the disk.
"""
function global_grid_radius(b::DiskBasis, dist, scale)
    r = problem_coord(b.radial_COV, _native_radius_grid(b, scale))
    return reshape_vector(r, get_dim(dist), get_basis_axis(dist, b) + 1)
end

"""
    local_grid_radius(b::DiskBasis, dist, scale)

Return the local radial grid for the disk.
"""
function local_grid_radius(b::DiskBasis, dist, scale)
    r = problem_coord(b.radial_COV, _native_radius_grid(b, scale))
    return reshape_vector(r, get_dim(dist), get_basis_axis(dist, b) + 1)
end

"""
    _native_radius_grid(b::DiskBasis, scale)

Compute the native radial grid (in [0,1]) using Zernike quadrature.
"""
function _native_radius_grid(b::DiskBasis, scale)
    cache_key = (:_native_radius_grid, scale)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    N = Int(ceil(scale * b.shape[2]))
    z, weights = zernike_quadrature(2, N; k=b.alpha)
    r = sqrt.((z .+ 1) ./ 2)
    result = Float64.(r)
    b._cache[cache_key] = result
    return result
end

"""
    constant_mode_value(b::DiskBasis)

Return the constant mode value for the disk basis.
"""
function constant_mode_value(b::DiskBasis)
    cached = get(b._cache, :constant_mode_value, nothing)
    if cached !== nothing
        return cached
    end
    Qk = zernike_polynomials(2, 1, b.alpha + b.k, 0, [0.0])
    val = Qk[1, 1]
    b._cache[:constant_mode_value] = val
    return val
end

"""
    transform_plan(b::DiskBasis, dist, grid_shape_val, axis, s)

Build (or retrieve cached) a transform plan for the disk radial direction.
"""
function transform_plan(b::DiskBasis, dist, grid_shape_val, axis, s)
    cache_key = (:transform_plan, objectid(dist), grid_shape_val, axis, s)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    # Stub: full disk transforms require m_maps distribution infrastructure
    error("DiskBasis transform_plan: requires m_maps distribution infrastructure not yet available")
end

"""
    forward_transform_radius_disk(b::DiskBasis, field, axis, gdata, cdata)

Forward radial transform for disk basis.
"""
function forward_transform_radius_disk(b::DiskBasis, field, axis, gdata, cdata)
    # Expand gdata if mmax=0 and dtype=float for spin recombination
    if b.mmax == 0 && b.dtype === Float64
        m_axis = length(field.tensorsig) + axis - 1
        gdata = cat(gdata, zeros(eltype(gdata), size(gdata)); dims=m_axis)
    end
    # Apply spin recombination from gdata to temp
    temp = zeros(eltype(gdata), size(gdata))
    forward_spin_recombination!(b, field.tensorsig, axis, gdata, temp)
    fill!(cdata, zero(eltype(cdata)))
    # Transform component-by-component
    S = spin_weights(b, field.tensorsig)
    for i in CartesianIndices(S)
        s = S[i]
        grid_shp = size(gdata[i])
        plan = transform_plan(b, field.dist, grid_shp, axis, s)
        forward!(plan, view(temp, i), view(cdata, i), axis)
    end
    return nothing
end

"""
    backward_transform_radius_disk(b::DiskBasis, field, axis, cdata, gdata)

Backward radial transform for disk basis.
"""
function backward_transform_radius_disk(b::DiskBasis, field, axis, cdata, gdata)
    # Handle mmax=0 float expansion
    if b.mmax == 0 && b.dtype === Float64
        m_axis = length(field.tensorsig) + axis - 1
        shp = collect(size(gdata))
        shp[m_axis] = 2
        temp = zeros(eltype(gdata), Tuple(shp))
        gdata_orig = gdata
        gdata = zeros(eltype(gdata), Tuple(shp))
    else
        temp = zeros(eltype(gdata), size(gdata))
    end
    # Transform component-by-component
    S = spin_weights(b, field.tensorsig)
    for i in CartesianIndices(S)
        s = S[i]
        grid_shp = size(gdata[i])
        plan = transform_plan(b, field.dist, grid_shp, axis, s)
        backward!(plan, view(cdata, i), view(temp, i), axis)
    end
    # Apply backward spin recombination
    fill!(gdata, zero(eltype(gdata)))
    backward_spin_recombination!(b, field.tensorsig, axis, temp, gdata)
    # Collapse expanded gdata
    if b.mmax == 0 && b.dtype === Float64
        m_axis = length(field.tensorsig) + axis - 1
        idx = axslice(m_axis, 1, 1)
        copyto!(gdata_orig, view(gdata, idx...))
    end
    return nothing
end

"""
    operator_matrix(b::DiskBasis, op, m, spin; size=nothing)

Build a radial operator matrix using dedalus_sphere Zernike operators.
"""
function operator_matrix(b::DiskBasis, op, m, spin; size=nothing)
    cache_key = (:operator_matrix, op, m, spin, size)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    ms = m + spin
    if length(op) > 0 && op[end] in ('+', '-')
        o = op[1:end-1]
        p = op[end] == '+' ? 1 : -1
        if ms == 0
            p = 1
        elseif ms < 0
            p = -p
        end
        operator = zernike_operator(2, o; radius=b.radius)
        mat = operator(p)
    elseif op == "L"
        D = zernike_operator(2, "D"; radius=b.radius)
        if ms < 0
            mat = compose(D(+1), D(-1))
        else
            mat = compose(D(-1), D(+1))
        end
    else
        operator = zernike_operator(2, op; radius=b.radius)
        mat = operator
    end
    if size === nothing
        size = n_size(b, m)
    end
    result = Float64.(resize_matrix(mat, size, b.alpha + b.k, abs(ms)))
    b._cache[cache_key] = result
    return result
end

"""
    conversion_matrix(b::DiskBasis, m, spintotal, dk)

Build the Zernike conversion matrix for the disk.
"""
function conversion_matrix(b::DiskBasis, m, spintotal, dk)
    cache_key = (:conversion_matrix, m, spintotal, dk)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    E = zernike_operator(2, "E"; radius=b.radius)
    op = E(+1)
    for _ in 2:dk
        op = compose(op, E(+1))
    end
    result = Float64.(resize_matrix(op, n_size(b, m), b.alpha + b.k, abs(m + spintotal)))
    b._cache[cache_key] = result
    return result
end

"""
    interpolation(b::DiskBasis, m, spintotal, position)

Build the interpolation matrix for the disk at the given radial position.
"""
function interpolation(b::DiskBasis, m, spintotal, position)
    cache_key = (:interpolation, m, spintotal, position)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    native_position = native_coord(b.radial_COV, position)
    native_z = 2 * native_position^2 - 1
    result = zernike_polynomials(2, n_size(b, m), b.alpha + b.k, abs(m + spintotal), native_z)
    b._cache[cache_key] = result
    return result
end

"""
    radius_multiplication_matrix(b::DiskBasis, m, spintotal, order, d)

Build the matrix for multiplying by r^order with additional factor.
"""
function radius_multiplication_matrix(b::DiskBasis, m, spintotal, order, d)
    cache_key = (:radius_multiplication_matrix, m, spintotal, order, d)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    if order == 0
        operator = zernike_operator(2, "Id"; radius=b.radius)
    else
        R = zernike_operator(2, "R"; radius=1.0)
        if order < 0
            op = R(-1)
            for _ in 2:abs(order)
                op = compose(op, R(-1))
            end
            operator = op
        else
            op = R(+1)
            for _ in 2:abs(order)
                op = compose(op, R(+1))
            end
            operator = op
        end
    end
    if d > 0
        R = zernike_operator(2, "R"; radius=1.0)
        R2 = compose(R(-1), R(+1))
        for _ in 1:(d ÷ 2)
            operator = compose(R2, operator)
        end
    end
    result = Float64.(resize_matrix(operator, n_size(b, m), b.alpha + b.k, abs(m + spintotal)))
    b._cache[cache_key] = result
    return result
end

"""
    clone_with(b::DiskBasis; kwargs...)

Create a copy of the disk basis with some fields replaced.
"""
function clone_with(b::DiskBasis; kwargs...)
    kw = Dict{Symbol,Any}(kwargs)
    coordsys_val = get(kw, :coordsys, b.coordsys)
    shape_val = get(kw, :shape, b.shape)
    dtype_val = get(kw, :dtype, b.dtype)
    radius_val = get(kw, :radius, b.radius)
    k_val = get(kw, :k, b.k)
    alpha_val = get(kw, :alpha, b.alpha)
    dealias_val = get(kw, :dealias, b.dealias_tuple)
    az_lib = get(kw, :azimuth_library, b.azimuth_library)
    r_lib = get(kw, :radius_library, b.radius_library)
    return DiskBasis(coordsys_val, shape_val, dtype_val;
                     radius=radius_val, k=k_val, alpha=alpha_val,
                     dealias=dealias_val, azimuth_library=az_lib,
                     radius_library=r_lib)
end

"""
    _last_axis_component_ncc_matrix(::Type{DiskBasis}, subproblem, ncc_basis,
        arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp,
        ncc_tensorsig, arg_tensorsig, out_tensorsig; cutoff=1e-6)

Build NCC component matrix for disk basis via Clenshaw algorithm.
"""
function _last_axis_component_ncc_matrix(::Type{DiskBasis}, subproblem, ncc_basis,
        arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp,
        ncc_tensorsig, arg_tensorsig, out_tensorsig; cutoff::Float64=1e-6)
    error("DiskBasis _last_axis_component_ncc_matrix: requires subproblem infrastructure not yet available")
end

function Base.show(io::IO, b::DiskBasis)
    print(io, "DiskBasis($(b.coordsys), shape=$(b.shape), radius=$(b.radius), k=$(b.k), alpha=$(b.alpha))")
end

# ============================================================================
# SphereBasis  (concrete spin basis for S2 / spherical shell surfaces)
# ============================================================================

"""
    SphereBasis <: SpinBasis

Spherical harmonic basis on the 2-sphere (S2).
Uses Fourier in azimuth and spin-weighted spherical harmonics in colatitude.
"""
mutable struct SphereBasis <: SpinBasis
    coordsys::Union{S2Coordinates, SphericalCoordinates}
    shape::Tuple{Int,Int}
    dtype::DataType
    radius::Float64
    dealias_tuple::Tuple{Float64,Float64}
    azimuth_library::String
    colatitude_library::String
    volume::Float64
    Lmax::Int
    grid_params::Tuple
    forward_transforms::Vector
    backward_transforms::Vector
    forward_m_perm::Union{Nothing,Vector{Int}}
    backward_m_perm::Union{Nothing,Vector{Int}}
    group_shape_val::Tuple{Int,Int}
    mmax::Int
    azimuth_basis
    _cache::Dict{Symbol,Any}
end

# -- Class constants --
basis_dim(::SphereBasis) = 2
basis_subaxis_dependence(::SphereBasis) = (true, true)
const SPHERE_CONSTANT_MODE_VALUE = 1.0 / sqrt(2.0)

# -- Constructor caching --
const _sphere_cache = Dict{Any,WeakRef}()

"""
    SphereBasis(coordsys, shape, dtype; radius=1.0, dealias=(1,1),
                azimuth_library=nothing, colatitude_library=nothing)

Construct (or retrieve cached) a SphereBasis.
"""
function SphereBasis(coordsys::Union{S2Coordinates, SphericalCoordinates},
                     shape, dtype;
                     radius::Real=1.0,
                     dealias=(1, 1),
                     azimuth_library=nothing,
                     colatitude_library=nothing)
    # Preprocess arguments
    if !(coordsys isa S2Coordinates || coordsys isa SphericalCoordinates)
        throw(ArgumentError("Sphere coordsys must be S2Coordinates or SphericalCoordinates."))
    end
    shape = (Int(shape[1]), Int(shape[2]))
    if length(shape) != 2
        throw(ArgumentError("Sphere shape must have length 2."))
    end
    radius = Float64(radius)
    if radius < 0
        throw(ArgumentError("Sphere radius must be non-negative."))
    end
    if isa(dealias, Number)
        dealias = (Float64(dealias), Float64(dealias))
    else
        dealias = (Float64(dealias[1]), Float64(dealias[2]))
    end
    if azimuth_library === nothing
        azimuth_library = FOURIER_DEFAULT_LIBRARY
    end
    if colatitude_library === nothing
        colatitude_library = JACOBI_DEFAULT_LIBRARY
    end

    # Cache lookup
    cache_key = (objectid(coordsys), shape, dtype, radius, dealias, azimuth_library, colatitude_library)
    wr = get(_sphere_cache, cache_key, nothing)
    if wr !== nothing
        inst = wr.value
        if inst !== nothing
            return inst::SphereBasis
        end
    end

    # Computed attributes
    vol = 4 * pi * radius^2
    mmax = (shape[1] - 1) ÷ 2

    # Set Lmax for optimal load balancing
    if dtype === Float64
        Lmax = max(0, shape[2] - 2)
    elseif dtype === ComplexF64
        Lmax = shape[2] - 1
    else
        throw(ArgumentError("Unsupported dtype: $dtype"))
    end

    if Lmax > 0 && (mmax > Lmax + 1)
        @warn "You are using more azimuthal modes than can be resolved with your current colatitude resolution"
    end

    grid_params = (objectid(coordsys), dtype, radius, dealias, azimuth_library, colatitude_library)

    # Validate phi resolution
    if shape[1] > 1 && shape[1] % 2 != 0
        throw(ArgumentError("Don't use an odd phi resolution please"))
    end
    if shape[1] > 1 && dtype === Float64 && shape[1] % 4 != 0
        throw(ArgumentError("Don't use a phi resolution that isn't divisible by 4, please"))
    end

    # Build azimuth basis
    if coordsys isa S2Coordinates
        coord0 = coordsys.azimuth
    else
        coord0 = coordsys.azimuth
    end
    Nphi = shape[1]
    az_bounds = (0.0, 2 * pi)
    if dtype === ComplexF64
        az_basis = ComplexFourier(coord0, Nphi, az_bounds; dealias=dealias[1], library=azimuth_library)
    elseif dtype === Float64
        az_basis = RealFourier(coord0, Nphi, az_bounds; dealias=dealias[1], library=azimuth_library)
    end

    # m permutations (triangular truncation repacking)
    if dtype === ComplexF64
        az_index = collect(0:Nphi-1)
        az_div = az_index .÷ 2
        az_mod = az_index .% 2
        fwd_perm_0based = az_div .+ (Nphi ÷ 2) .* az_mod
        fwd_perm = fwd_perm_0based .+ 1  # 1-based
        bwd_perm = similar(fwd_perm)
        for (i, p) in enumerate(fwd_perm)
            bwd_perm[p] = i
        end
        group_shp = (1, 1)
    elseif dtype === Float64
        if Nphi == 1
            az_index = collect(0:1)
        else
            az_index = collect(0:Nphi-1)
        end
        div2 = az_index .÷ 2
        mod2 = az_index .% 2
        div22 = div2 .% 2
        fwd_perm_0based = (mod2 .+ div2) .* (1 .- div22) .+ (Nphi .- 1 .+ mod2 .- div2) .* div22
        fwd_perm = fwd_perm_0based .+ 1  # 1-based
        bwd_perm = similar(fwd_perm)
        for (i, p) in enumerate(fwd_perm)
            bwd_perm[p] = i
        end
        group_shp = (2, 1)
    end

    # Apply permutations to azimuth basis
    if mmax > 0
        az_basis.forward_coeff_permutation = fwd_perm
        az_basis.backward_coeff_permutation = bwd_perm
    end

    _cache = Dict{Symbol,Any}()

    inst = SphereBasis(
        coordsys, shape, dtype, radius, dealias,
        azimuth_library, colatitude_library,
        vol, Lmax, grid_params,
        Any[], Any[],  # transforms -- set below
        fwd_perm, bwd_perm,
        group_shp, mmax, az_basis, _cache
    )

    # Set transform callables
    if mmax == 0
        push!(inst.forward_transforms, (field, axis, gdata, cdata) -> _forward_transform_azimuth_Mmax0_sphere(inst, field, axis, gdata, cdata))
        push!(inst.backward_transforms, (field, axis, cdata, gdata) -> _backward_transform_azimuth_Mmax0_sphere(inst, field, axis, cdata, gdata))
    else
        push!(inst.forward_transforms, (field, axis, gdata, cdata) -> _forward_transform_azimuth_sphere(inst, field, axis, gdata, cdata))
        push!(inst.backward_transforms, (field, axis, cdata, gdata) -> _backward_transform_azimuth_sphere(inst, field, axis, cdata, gdata))
    end
    push!(inst.forward_transforms, (field, axis, gdata, cdata) -> forward_transform_colatitude(inst, field, axis, gdata, cdata))
    push!(inst.backward_transforms, (field, axis, cdata, gdata) -> backward_transform_colatitude(inst, field, axis, cdata, gdata))

    # For SphericalCoordinates, add radial pass-through transforms
    if coordsys isa SphericalCoordinates
        push!(inst.forward_transforms, (field, axis, gdata, cdata) -> nothing)
        push!(inst.backward_transforms, (field, axis, cdata, gdata) -> nothing)
    end

    _sphere_cache[cache_key] = WeakRef(inst)
    return inst
end

# -- SphereBasis interface methods --

"""
    latitude_basis(b::SphereBasis)

Return a latitude-only version of this sphere basis (shape[1] = 1).
"""
function latitude_basis(b::SphereBasis)
    cached = get(b._cache, :latitude_basis, nothing)
    if cached !== nothing
        return cached
    end
    lb = clone_with(b; shape=(1, b.shape[2]))
    b._cache[:latitude_basis] = lb
    return lb
end

"""
    ell_size(b::SphereBasis, m)

Return the number of valid ell modes for azimuthal wavenumber m.
"""
function ell_size(b::SphereBasis, m)
    return b.Lmax + 1 - abs(m)
end

"""
    matrix_dependence(b::SphereBasis, matrix_coupling)

If colatitude coupling is present, azimuthal dependence must also be present.
"""
function matrix_dependence(b::SphereBasis, matrix_coupling)
    md = copy(matrix_coupling)
    if matrix_coupling[2]
        md[1] = true
    end
    return md
end

"""
    global_shape(b::SphereBasis, grid_space, scales)

Compute the global data shape for the given grid/coeff space and scale factors.
"""
function global_shape(b::SphereBasis, grid_space, scales)
    gs = grid_shape(b, scales)
    if grid_space[1]
        # grid-grid space
        if b.mmax == 0
            return (1, gs[2])
        else
            return gs
        end
    elseif grid_space[2]
        # coeff-grid space
        shp = collect(gs)
        if b.mmax > 0
            shp[1] = b.shape[1]
        else
            if b.dtype === ComplexF64
                shp[1] = 1
            elseif b.dtype === Float64
                shp[1] = 2
            end
        end
        return Tuple(shp)
    else
        # coeff-coeff space -- repacked triangular truncation
        Nphi = b.shape[1]
        Lmax = b.Lmax
        if b.mmax > 0
            if b.dtype === ComplexF64
                return (Nphi ÷ 2, Lmax + 1 + max(0, Lmax + 1 - Nphi ÷ 2))
            elseif b.dtype === Float64
                return (Nphi ÷ 2, Lmax + 1 + max(0, Lmax + 2 - Nphi ÷ 2))
            end
        else
            if b.dtype === ComplexF64
                return (1, Lmax + 1)
            elseif b.dtype === Float64
                return (2, Lmax + 1)
            end
        end
    end
end

"""
    chunk_shape(b::SphereBasis, grid_space)

Compute the chunk shape for distribution.
"""
function chunk_shape(b::SphereBasis, grid_space)
    if grid_space[1]
        return (1, 1)
    elseif grid_space[2]
        if b.dtype === ComplexF64
            if b.mmax > 0
                return (2, 1)
            else
                return (1, 1)
            end
        elseif b.dtype === Float64
            if b.mmax > 0
                return (4, 1)
            else
                return (2, 1)
            end
        end
    else
        if b.dtype === ComplexF64
            return (1, 1)
        elseif b.dtype === Float64
            return (2, 1)
        end
    end
end

"""
    elements_to_groups(b::SphereBasis, grid_space, elements)

Convert element indices to group indices (m, ell pairs).
"""
function elements_to_groups(b::SphereBasis, grid_space, elements)
    if grid_space[1]
        return elements
    elseif grid_space[2]
        # coeff-grid space: unpacked m
        permuted_native_wn = native_wavenumbers(b.azimuth_basis)
        groups = copy(elements)
        # elements are 0-based indices; wavenumbers are indexed 1-based
        groups[1] = permuted_native_wn[elements[1] .+ 1]
        return groups
    else
        # coeff-coeff space: repacked triangular truncation
        i_el, j_el = elements[1], elements[2]
        Nphi = b.shape[1]
        Lmax = b.Lmax
        if b.dtype === ComplexF64
            shift = max(0, Lmax + 1 - Nphi ÷ 2)
            m = copy(i_el)
            ell = j_el .- shift
            # Fix for m < 0
            neg_modes = ell .< m
            m[neg_modes] .= i_el[neg_modes] .- (Nphi + 1) ÷ 2
            ell[neg_modes] .= Lmax .- j_el[neg_modes]
            # Fix for m = 0
            m_zero = i_el .== 0
            m[m_zero] .= 0
            ell[m_zero] .= j_el[m_zero]
            # Fix for Nyquist
            nyq_modes = (i_el .== 0) .& (j_el .> Lmax)
            m[nyq_modes] .= Nphi ÷ 2
            ell[nyq_modes] .= j_el[nyq_modes] .- shift
        elseif b.dtype === Float64
            shift = max(0, Lmax + 2 - Nphi ÷ 2)
            m = i_el .÷ 2
            ell = j_el .- shift
            # Fix for Nphi/4 <= m < Nphi/2-1
            neg_modes = ell .< m
            m[neg_modes] .= (Nphi ÷ 2 - 1) .- m[neg_modes]
            ell[neg_modes] .= Lmax .- j_el[neg_modes]
            # Fix for m = 0
            m_zero = i_el .< 2
            m[m_zero] .= 0
            ell[m_zero] .= j_el[m_zero]
            # Fix for m = Nphi/2 - 1
            m_max = (i_el .< 2) .& (j_el .> Lmax)
            m[m_max] .= Nphi ÷ 2 - 1
            ell[m_max] .= j_el[m_max] .- shift
        end
        return [m, ell]
    end
end

"""
    valid_elements(b::SphereBasis, tensorsig, grid_space, elements)

Determine which elements are valid for the given tensor signature.
"""
function valid_elements(b::SphereBasis, tensorsig, grid_space, elements)
    if grid_space[2]
        # grid-grid and coeff-grid: same as Fourier validity
        return valid_elements(b.azimuth_basis, tensorsig, grid_space, elements)
    else
        # coeff-coeff space
        groups = elements_to_groups(b, grid_space, elements)
        m = groups[1]
        ell = groups[2]
        tshape = tuple((get_dim(cs) for cs in tensorsig)...)
        valid = ones(Bool, tshape..., size(m)...)
        if !isempty(tensorsig)
            rank = length(tensorsig)
            dim = ndims(m)
            # Expand m and ell with leading singleton dims for tensor indices
            full_m = reshape(m, ntuple(_ -> 1, rank)..., size(m)...)
            full_ell = reshape(ell, ntuple(_ -> 1, rank)..., size(ell)...)
            s = spin_weights(b, tensorsig)
            full_s = reshape(s, size(s)..., ntuple(_ -> 1, dim)...)
            valid[max.(abs.(full_m), abs.(full_s)) .> full_ell] .= false
        else
            valid[abs.(m) .> ell] .= false
        end
        # Drop msin part of ell == 0 for real scalars and vectors
        if b.dtype === Float64
            if length(tensorsig) == 0
                valid[(ell .== 0) .& (elements[1] .% 2 .== 1)] .= false
            elseif length(tensorsig) == 1
                allcomps = ntuple(_ -> Colon(), 1)
                valid[allcomps..., (ell .== 0) .& (elements[1] .% 2 .== 1)] .= false
            end
        end
        return valid
    end
end

# -- Grid methods --

"""
    global_grids(b::SphereBasis, dist, scales)

Return tuple of (azimuth_grid, colatitude_grid).
"""
function global_grids(b::SphereBasis, dist, scales)
    return (global_grid(b.azimuth_basis, dist, scales[1]),
            global_grid_colatitude(b, dist, scales[2]))
end

"""
    local_grids(b::SphereBasis, dist, scales)

Return tuple of (local_azimuth_grid, local_colatitude_grid).
"""
function local_grids(b::SphereBasis, dist, scales)
    return (local_grid(b.azimuth_basis, dist, scales[1]),
            local_grid_colatitude(b, dist, scales[2]))
end

"""
    global_grid_colatitude(b::SphereBasis, dist, scale)

Return the global colatitude grid.
"""
function global_grid_colatitude(b::SphereBasis, dist, scale)
    theta = _native_colatitude_grid(b, scale)
    return reshape_vector(theta, get_dim(dist), get_basis_axis(dist, b) + 1)
end

"""
    local_grid_colatitude(b::SphereBasis, dist, scale)

Return the local colatitude grid.
"""
function local_grid_colatitude(b::SphereBasis, dist, scale)
    theta = _native_colatitude_grid(b, scale)
    return reshape_vector(theta, get_dim(dist), get_basis_axis(dist, b) + 1)
end

"""
    _native_colatitude_grid(b::SphereBasis, scale)

Compute the colatitude grid at the given scale using sphere quadrature.
"""
function _native_colatitude_grid(b::SphereBasis, scale)
    cache_key = (:_native_colatitude_grid, scale)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    N = Int(ceil(scale * b.shape[2]))
    cos_theta, weights = sphere_quadrature(N - 1)
    theta = Float64.(acos.(cos_theta))
    b._cache[cache_key] = theta
    return theta
end

"""
    global_colatitude_weights(b::SphereBasis, dist; scale=nothing)

Return the global colatitude quadrature weights.
"""
function global_colatitude_weights(b::SphereBasis, dist; scale=nothing)
    if scale === nothing; scale = 1; end
    N = Int(ceil(scale * b.shape[2]))
    cos_theta, weights = sphere_quadrature(N - 1)
    return reshape_vector(Float64.(weights), get_dim(dist), get_basis_axis(dist, b) + 1)
end

"""
    local_colatitude_weights(b::SphereBasis, dist; scale=nothing)

Return the local colatitude quadrature weights.
"""
function local_colatitude_weights(b::SphereBasis, dist; scale=nothing)
    if scale === nothing; scale = 1; end
    N = Int(ceil(scale * b.shape[2]))
    cos_theta, weights = sphere_quadrature(N - 1)
    return reshape_vector(Float64.(weights), get_dim(dist), get_basis_axis(dist, b) + 1)
end

# -- Transform methods --

"""
    _forward_transform_azimuth_Mmax0_sphere(b::SphereBasis, field, axis, gdata, cdata)

Forward azimuth transform for sphere when mmax == 0.
"""
function _forward_transform_azimuth_Mmax0_sphere(b::SphereBasis, field, axis, gdata, cdata)
    slice_axis = axis + length(field.tensorsig)
    temp = zeros(eltype(cdata), size(cdata))
    idx = axslice(slice_axis, 1, 1)
    copyto!(view(temp, idx...), gdata)
    copyto!(cdata, temp)
    return nothing
end

"""
    _forward_transform_azimuth_sphere(b::SphereBasis, field, axis, gdata, cdata)

Forward azimuth transform for sphere: delegate to Fourier basis.
"""
function _forward_transform_azimuth_sphere(b::SphereBasis, field, axis, gdata, cdata)
    forward_transform(b.azimuth_basis, field, axis, gdata, cdata)
    return nothing
end

"""
    _backward_transform_azimuth_Mmax0_sphere(b::SphereBasis, field, axis, cdata, gdata)

Backward azimuth transform for sphere when mmax == 0.
"""
function _backward_transform_azimuth_Mmax0_sphere(b::SphereBasis, field, axis, cdata, gdata)
    slice_axis = axis + length(field.tensorsig)
    idx = axslice(slice_axis, 1, 1)
    copyto!(gdata, view(cdata, idx...))
    return nothing
end

"""
    _backward_transform_azimuth_sphere(b::SphereBasis, field, axis, cdata, gdata)

Backward azimuth transform for sphere: delegate to Fourier basis.
"""
function _backward_transform_azimuth_sphere(b::SphereBasis, field, axis, cdata, gdata)
    backward_transform(b.azimuth_basis, field, axis, cdata, gdata)
    return nothing
end

"""
    transform_plan(b::SphereBasis, dist, Ntheta, s)

Build (or retrieve cached) a colatitude transform plan.
"""
function transform_plan(b::SphereBasis, dist, Ntheta, s)
    cache_key = (:transform_plan, objectid(dist), Ntheta, s)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    # Stub: full sphere transforms require m_maps infrastructure
    error("SphereBasis transform_plan: requires m_maps distribution infrastructure not yet available")
end

"""
    forward_transform_colatitude(b::SphereBasis, field, axis, gdata, cdata)

Forward colatitude transform: spin recombination + SWSH transform.
"""
function forward_transform_colatitude(b::SphereBasis, field, axis, gdata, cdata)
    # Expand gdata if mmax=0 and dtype=float for spin recombination
    if b.mmax == 0 && b.dtype === Float64
        m_axis = length(field.tensorsig) + axis - 1
        gdata = cat(gdata, zeros(eltype(gdata), size(gdata)); dims=m_axis)
    end
    # Apply spin recombination from gdata to temp
    temp = zeros(eltype(gdata), size(gdata))
    forward_spin_recombination!(b, field.tensorsig, axis, gdata, temp)
    fill!(cdata, zero(eltype(cdata)))
    # Transform component-by-component
    S = spin_weights(b, field.tensorsig)
    for i in CartesianIndices(S)
        s = S[i]
        Ntheta = size(gdata[i], axis)
        plan = transform_plan(b, field.dist, Ntheta, s)
        forward!(plan, view(temp, i), view(cdata, i), axis)
    end
    return nothing
end

"""
    backward_transform_colatitude(b::SphereBasis, field, axis, cdata, gdata)

Backward colatitude transform: SWSH transform + backward spin recombination.
"""
function backward_transform_colatitude(b::SphereBasis, field, axis, cdata, gdata)
    # Handle mmax=0 float expansion
    if b.mmax == 0 && b.dtype === Float64
        m_axis = length(field.tensorsig) + axis - 1
        shp = collect(size(gdata))
        shp[m_axis] = 2
        temp = zeros(eltype(gdata), Tuple(shp))
        gdata_orig = gdata
        gdata = zeros(eltype(gdata), Tuple(shp))
    else
        temp = zeros(eltype(gdata), size(gdata))
    end
    # Transform component-by-component
    S = spin_weights(b, field.tensorsig)
    for i in CartesianIndices(S)
        s = S[i]
        Ntheta = size(gdata[i], axis)
        plan = transform_plan(b, field.dist, Ntheta, s)
        backward!(plan, view(cdata, i), view(temp, i), axis)
    end
    # Apply backward spin recombination
    fill!(gdata, zero(eltype(gdata)))
    backward_spin_recombination!(b, field.tensorsig, axis, temp, gdata)
    # Collapse expanded gdata
    if b.mmax == 0 && b.dtype === Float64
        m_axis = length(field.tensorsig) + axis - 1
        idx = axslice(m_axis, 1, 1)
        copyto!(gdata_orig, view(gdata, idx...))
    end
    return nothing
end

# -- Basis algebra --

function basis_add(a::SphereBasis, other)
    if other === nothing || other === a
        return a
    end
    if other isa SphereBasis
        if a.radius != other.radius
            throw(ErrorException("Cannot add two sphere bases with different radii ($(a.radius) and $(other.radius))."))
        end
        if a.grid_params == other.grid_params
            shape = (max(a.shape[1], other.shape[1]), max(a.shape[2], other.shape[2]))
            return clone_with(a; shape=shape)
        end
    end
    return nothing
end

function basis_mul(a::SphereBasis, other)
    if other === nothing || other === a
        return a
    end
    if other isa SphereBasis
        if a.radius != other.radius
            throw(ErrorException("Cannot multiply two sphere bases with different radii ($(a.radius) and $(other.radius))."))
        end
        if a.grid_params == other.grid_params
            shape = (max(a.shape[1], other.shape[1]), max(a.shape[2], other.shape[2]))
            return clone_with(a; shape=shape)
        end
    end
    return nothing
end

function basis_rmatmul(operand::SphereBasis, ncc)
    return basis_mul(operand, ncc)
end

# -- Ell methods --

"""
    k_func(b::SphereBasis, l, s, mu)

Compute the k coupling factor: -mu * sqrt(max(0, (l - mu*s)*(l + mu*s + 1)/2)).
"""
function k_func(b::SphereBasis, l, s, mu)
    return -mu .* sqrt.(max.(0, (l .- mu .* s) .* (l .+ mu .* s .+ 1) ./ 2))
end

"""
    k_vector(b::SphereBasis, mu, m, s, local_l)

Compute the k vector for the given parameters.
"""
function k_vector(b::SphereBasis, mu, m, s, local_l)
    cache_key = (:k_vector, mu, m, s, Tuple(local_l))
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    vec = zeros(length(local_l))
    Lmin = max(abs(m), abs(s), abs(s + mu))
    for (idx, l) in enumerate(local_l)
        if l < Lmin
            vec[idx] = 0
        else
            vec[idx] = sphere_op("k" * (mu > 0 ? "+" : "-"), l, m, s)
        end
    end
    b._cache[cache_key] = vec
    return vec
end

"""
    operator_matrix(b::SphereBasis, op, m, spintotal; size=nothing)

Build operator matrix using dedalus_sphere sphere operators.
"""
function operator_matrix(b::SphereBasis, op, m, spintotal; size=nothing)
    cache_key = (:operator_matrix, op, m, spintotal, size)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    if op == "Cos"
        operator = sphere_operator("Cos")
    else
        error("SphereBasis operator_matrix: operator '$op' not yet supported")
    end
    result = Float64.(resize_matrix(operator, size - 1 + max(abs(m), abs(spintotal)), m, spintotal))
    b._cache[cache_key] = result
    return result
end

"""
    sine_multiplication_matrix(b::SphereBasis, m, spintotal, order; size=nothing)

Build the sine multiplication matrix for the sphere.
"""
function sine_multiplication_matrix(b::SphereBasis, m, spintotal, order; size=nothing)
    cache_key = (:sine_multiplication_matrix, m, spintotal, order, size)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    if size === nothing
        size = ell_size(b, m)
    end
    if order == 0
        result = sparse(1.0I, size, size)
    else
        S_op = sphere_operator("Sin")
        if order < 0
            op = S_op(-1)
            for _ in 2:abs(order)
                op = compose(op, S_op(-1))
            end
        else
            op = S_op(+1)
            for _ in 2:abs(order)
                op = compose(op, S_op(+1))
            end
        end
        sign_factor = (-1)^max(0, -order)
        result = Float64.(sign_factor .* resize_matrix(op, size - 1 + max(abs(m), abs(spintotal)), m, spintotal))
    end
    b._cache[cache_key] = result
    return result
end

"""
    clone_with(b::SphereBasis; kwargs...)

Create a copy of the sphere basis with some fields replaced.
"""
function clone_with(b::SphereBasis; kwargs...)
    kw = Dict{Symbol,Any}(kwargs)
    coordsys_val = get(kw, :coordsys, b.coordsys)
    shape_val = get(kw, :shape, b.shape)
    dtype_val = get(kw, :dtype, b.dtype)
    radius_val = get(kw, :radius, b.radius)
    dealias_val = get(kw, :dealias, b.dealias_tuple)
    az_lib = get(kw, :azimuth_library, b.azimuth_library)
    co_lib = get(kw, :colatitude_library, b.colatitude_library)
    return SphereBasis(coordsys_val, shape_val, dtype_val;
                       radius=radius_val, dealias=dealias_val,
                       azimuth_library=az_lib, colatitude_library=co_lib)
end

"""
    _last_axis_component_ncc_matrix(::Type{SphereBasis}, subproblem, ncc_basis,
        arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp,
        ncc_tensorsig, arg_tensorsig, out_tensorsig; cutoff=1e-6)

Build NCC component matrix for sphere basis via Clenshaw algorithm.
"""
function _last_axis_component_ncc_matrix(::Type{SphereBasis}, subproblem, ncc_basis,
        arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp,
        ncc_tensorsig, arg_tensorsig, out_tensorsig; cutoff::Float64=1e-6)
    error("SphereBasis _last_axis_component_ncc_matrix: requires subproblem infrastructure not yet available")
end

# -- m_maps and ell_maps --

"""
    m_maps(b::SphereBasis, dist)

Compute a tuple of (m, mg_slice, mc_slice, ell_slice) for all local m values.
"""
function m_maps(b::SphereBasis, dist)
    cache_key = (:m_maps, objectid(dist))
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    error("SphereBasis m_maps: requires distributor infrastructure not yet available")
end

"""
    ell_reversed(b::SphereBasis, dist)

Return a dictionary mapping m -> Bool indicating whether ell ordering is reversed.
"""
function ell_reversed(b::SphereBasis, dist)
    cache_key = (:ell_reversed, objectid(dist))
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    result = Dict{Int, Bool}()
    for (m, mg_slice, mc_slice, ell_sl) in m_maps(b, dist)
        result[m] = false
        if ell_sl isa StepRange && step(ell_sl) < 0
            result[m] = true
        end
    end
    b._cache[cache_key] = result
    return result
end

"""
    ell_maps(b::SphereBasis, dist)

Compute a tuple of (ell, m_slice, ell_slice) for all local ells in coeff space.
"""
function ell_maps(b::SphereBasis, dist)
    cache_key = (:ell_maps, objectid(dist))
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    error("SphereBasis ell_maps: requires distributor infrastructure not yet available")
end

function Base.show(io::IO, b::SphereBasis)
    print(io, "SphereBasis($(b.coordsys), shape=$(b.shape), radius=$(b.radius), Lmax=$(b.Lmax))")
end
