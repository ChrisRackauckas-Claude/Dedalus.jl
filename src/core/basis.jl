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
       backward!
