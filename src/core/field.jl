"""
    Field types for Dedalus.jl

Julia translation of `dedalus/core/field.py`. Provides the operand type
hierarchy and the `Field` type that stores scalar, vector, and tensor field
data across distributed spectral layouts.

## Type hierarchy

    AbstractOperand
    +-- AbstractCurrent  <: AbstractOperand
    |   +-- Field        <: AbstractCurrent
    |   +-- LockedField  <: AbstractCurrent (wraps Field with layout restrictions)
    +-- AbstractFuture   <: AbstractOperand  (defined in future.jl)

## Key translation choices

- Python's `Operand` base class -> `AbstractOperand` abstract type.
- Python's `Current` base class -> `AbstractCurrent <: AbstractOperand`.
- Python's `Future` base class  -> `AbstractFuture <: AbstractOperand` (forward ref).
- Python's `__add__`, `__mul__`, etc. -> Julia `Base.:+`, `Base.:*`, etc.
- Python's `isinstance(arg, Operand)` -> `isa(arg, AbstractOperand)`.
- Python's `weakref` -> `WeakRef` in Julia.
- Python's `CachedAttribute` -> `_cache::Dict{Symbol,Any}` + accessor helpers.
- Python's `np.result_type` -> `promote_type`.
- Python's `Number` -> `Number` (Julia base).
- Python `fftw.create_buffer` -> plain `zeros` (FFTW SIMD alignment handled
  separately or not needed for correctness).
"""

using SparseArrays
using LinearAlgebra

# ============================================================================
# Abstract type hierarchy
# ============================================================================

"""
    AbstractOperand

Root abstract type for all operand classes in the Dedalus expression tree.
Provides arithmetic operator overloads that build lazy expression trees.
"""
abstract type AbstractOperand end

"""
    AbstractCurrent <: AbstractOperand

Abstract base for "current" (evaluated, data-holding) operands -- primarily
`Field` and `LockedField`.
"""
abstract type AbstractCurrent <: AbstractOperand end

"""
    AbstractFuture <: AbstractOperand

Abstract base for "future" (deferred) operands -- expression-tree nodes that
produce a result upon evaluation.  Defined fully in `future.jl`.
"""
abstract type AbstractFuture <: AbstractOperand end

# ============================================================================
# Forward-reference functions for arithmetic operators
# ============================================================================
#
# These are populated by arithmetic.jl once it is loaded.

"""
    dedalus_add(a, b)

Lazy addition of operands. Implemented in arithmetic.jl.
"""
function dedalus_add end

"""
    dedalus_multiply(a, b)

Lazy multiplication of operands. Implemented in arithmetic.jl.
"""
function dedalus_multiply end

"""
    dedalus_power(a, b)

Lazy exponentiation of operands. Implemented in operators.jl.
"""
function dedalus_power end

# ============================================================================
# AbstractOperand -- arithmetic operator overloads
# ============================================================================

# Addition: Operand + anything, anything + Operand
Base.:+(a::AbstractOperand, b) = dedalus_add(a, b)
Base.:+(a, b::AbstractOperand) = dedalus_add(a, b)
Base.:+(a::AbstractOperand, b::AbstractOperand) = dedalus_add(a, b)

# Negation
Base.:-(a::AbstractOperand) = dedalus_multiply(-1, a)

# Subtraction
Base.:-(a::AbstractOperand, b) = a + (-b)
Base.:-(a, b::AbstractOperand) = a + (-b)
Base.:-(a::AbstractOperand, b::AbstractOperand) = a + (-b)

# Multiplication: Operand * anything, anything * Operand
Base.:*(a::AbstractOperand, b) = dedalus_multiply(a, b)
Base.:*(a, b::AbstractOperand) = dedalus_multiply(a, b)
Base.:*(a::AbstractOperand, b::AbstractOperand) = dedalus_multiply(a, b)

# Division
Base.:/(a::AbstractOperand, b) = a * b^(-1)
Base.:/(a, b::AbstractOperand) = a * b^(-1)
Base.:/(a::AbstractOperand, b::AbstractOperand) = a * b^(-1)

# Power
function Base.:^(a::AbstractOperand, b)
    b == 0 && return 1
    b == 1 && return a
    return dedalus_power(a, b)
end

function Base.:^(a, b::AbstractOperand)
    return dedalus_power(a, b)
end

function Base.:^(a::AbstractOperand, b::AbstractOperand)
    return dedalus_power(a, b)
end

# ============================================================================
# AbstractOperand -- interface methods (defaults / stubs)
# ============================================================================

"""
    operand_cast(arg, dist, tensorsig, dtype)

Cast `arg` to an operand compatible with `dist`, `tensorsig`, and `dtype`.
Numbers are cast to constant Fields.
"""
function operand_cast(arg, dist, tensorsig, dtype)
    if isa(arg, AbstractOperand)
        if arg.dist !== dist
            throw(ArgumentError("Mismatching distributor."))
        elseif arg.tensorsig != tensorsig
            throw(ArgumentError("Mismatching tensorsig."))
        elseif arg.dtype != dtype
            throw(ArgumentError("Mismatching dtype."))
        else
            return arg
        end
    elseif isa(arg, Number)
        out = Field(dist; tensorsig=tensorsig, dtype=dtype)
        out[Symbol("g")] = arg  # set in grid space
        out.name = string(arg)
        return out
    else
        throw(ArgumentError("Cannot cast type $(typeof(arg))"))
    end
end

"""
    atoms(op::AbstractOperand, types...)

Gather all leaf-operands of specified types in the expression tree.
Must be implemented by subtypes.
"""
function atoms end

"""
    has_operand(op::AbstractOperand, vars...)

Determine if the expression tree contains any specified operands/operators.
Must be implemented by subtypes.
"""
function has_operand end

"""
    split_operand(op::AbstractOperand, vars...)

Split expression into parts containing and not containing specified operands.
Must be implemented by subtypes.
"""
function split_operand end

"""
    replace_operand(op::AbstractOperand, old, new_val)

Replace specified operand/operator in the expression tree.
Must be implemented by subtypes.
"""
function replace_operand end

"""
    replace_dict(op::AbstractOperand, subs::Dict)

Replace operands/operators according to a dictionary of substitutions.
Must be implemented by subtypes.
"""
function replace_dict end

"""
    sym_diff(op::AbstractOperand, var)

Symbolically differentiate with respect to specified operand.
Must be implemented by subtypes.
"""
function sym_diff end

"""
    expand_operand(op::AbstractOperand, vars...)

Expand expression over specified variables.
Must be implemented by subtypes.
"""
function expand_operand end

"""
    require_linearity(op::AbstractOperand, vars...; allow_affine=false,
                      self_name=nothing, vars_name=nothing, error_type=AssertionError)

Require expression to be linear in specified operands/operators.
"""
function require_linearity end

"""
    require_first_order(op::AbstractOperand, ops...; self_name=nothing,
                        ops_name=nothing, error_type=AssertionError)

Require expression to be maximally first order in specified operators.
"""
function require_first_order end

"""
    require_independent(op::AbstractOperand, vars...; self_name=nothing,
                        vars_name=nothing, error_type=AssertionError)

Require expression to be independent of specified operands/operators.
"""
function require_independent(op::AbstractOperand, vars...;
                             self_name=nothing, vars_name=nothing,
                             error_type=AssertionError)
    if has_operand(op, vars...)
        if self_name === nothing
            self_name = string(op)
        end
        if vars_name === nothing
            vars_name = [string(v) for v in vars]
        end
        throw(error_type("$(self_name) must be independent of $(vars_name)."))
    end
end

"""
    build_ncc_matrices(op::AbstractOperand, separability, vars; kw...)

Precompute non-constant coefficients and build multiplication matrices.
"""
function build_ncc_matrices end

"""
    expression_matrices(op::AbstractOperand, subproblem, vars; kw...)

Build expression matrices for a specific subproblem and variables.
"""
function expression_matrices end

"""
    frechet_differential(op::AbstractOperand, variables, perturbations;
                         backgrounds=nothing)

Compute Frechet differential with respect to specified variables/perturbations.

Parameters
----------
variables : list of Field objects
    Variables to differentiate around.
perturbations : list of Field objects
    Perturbation directions for each variable.
backgrounds : list of Field objects, optional
    Backgrounds for each variable. Default: variables.
"""
function frechet_differential(op::AbstractOperand, variables, perturbations;
                              backgrounds=nothing)
    dist = op.dist
    tensorsig = op.tensorsig
    dtype = op.dtype
    # Compute differential
    epsilon = Field(dist; dtype=dtype)
    # d/de F(X0 + e*X1)
    subs = Dict(var => var + epsilon * pert
                for (var, pert) in zip(variables, perturbations))
    diff = replace_dict(op, subs)
    diff = sym_diff(diff, epsilon)
    # e -> 0
    if diff !== nothing && diff != 0
        diff = operand_cast(diff, dist; tensorsig=tensorsig, dtype=dtype)
        diff = replace_operand(diff, epsilon, 0)
    end
    # Replace variables with backgrounds, if specified
    if diff !== nothing && diff != 0
        if backgrounds !== nothing
            bg_subs = Dict(var => bg for (var, bg) in zip(variables, backgrounds))
            diff = replace_dict(diff, bg_subs)
        end
    end
    return diff
end

"""
    is_complex_operand(op::AbstractOperand) -> Bool

Check whether the operand's dtype is complex.
"""
function is_complex_operand(op::AbstractOperand)
    return is_complex_dtype(op.dtype)
end

"""
    is_real_operand(op::AbstractOperand) -> Bool

Check whether the operand's dtype is real.
"""
function is_real_operand(op::AbstractOperand)
    return is_real_dtype(op.dtype)
end

"""
    valid_modes(op::AbstractOperand)

Get general coefficient-space valid modes, returning a copy.
"""
function valid_modes(op::AbstractOperand)
    vm = op.dist.coeff_layout.valid_elements(op.tensorsig, op.domain, 1)
    return copy(vm)
end

# ============================================================================
# AbstractCurrent -- concrete interface implementations
# ============================================================================

function Base.show(io::IO, c::AbstractCurrent)
    print(io, "<", typeof(c), " ", objectid(c), ">")
end

function Base.string(c::AbstractCurrent)
    if c.name !== nothing
        return c.name
    else
        return repr(c)
    end
end

# --- Expression tree methods for Current (leaf nodes) ---

function atoms(c::AbstractCurrent, types...)
    result = OrderedSet{Any}()
    if isempty(types) || any(t -> isa(c, t), types)
        push!(result, c)
    end
    return result
end

function has_operand(c::AbstractCurrent, vars...)
    return isempty(vars) || (c in vars)
end

function split_operand(c::AbstractCurrent, vars...)
    if c in vars
        return (c, 0)
    else
        return (0, c)
    end
end

function replace_operand(c::AbstractCurrent, old, new_val)
    if c == old
        return new_val
    else
        return c
    end
end

function replace_dict(c::AbstractCurrent, subs::AbstractDict)
    if c in keys(subs)
        return subs[c]
    else
        return c
    end
end

function sym_diff(c::AbstractCurrent, var)
    if c == var
        return 1
    else
        return 0
    end
end

function expand_operand(c::AbstractCurrent, vars...)
    return c
end

function require_linearity(c::AbstractCurrent, vars...;
                           allow_affine::Bool=false,
                           self_name=nothing, vars_name=nothing,
                           error_type=AssertionError)
    if !allow_affine && !(c in vars)
        if self_name === nothing
            self_name = string(c)
        end
        if vars_name === nothing
            vars_name = [string(v) for v in vars]
        end
        throw(error_type("$(self_name) must be strictly linear in $(vars_name)."))
    end
end

function require_first_order(c::AbstractCurrent, args...; kw...)
    # No-op for leaf nodes
    return nothing
end

function matrix_dependence(c::AbstractCurrent, vars...)
    require_linearity(c, vars...)
    return falses(get_dim(c.domain.dist))
end

function matrix_coupling(c::AbstractCurrent, vars...)
    require_linearity(c, vars...)
    return falses(get_dim(c.domain.dist))
end

function build_ncc_matrices(c::AbstractCurrent, separability, vars; kw...)
    require_linearity(c, vars...)
    return nothing
end

function expression_matrices(c::AbstractCurrent, subproblem, vars; kw...)
    require_linearity(c, vars...)
    size_val = subproblem.field_size(c)
    matrix = sparse(1.0I, size_val, size_val)
    return Dict(c => matrix)
end

"""
    attempt(c::AbstractCurrent; id=nothing)

Leaf operands are already evaluated; return self.
"""
function attempt(c::AbstractCurrent; id=nothing)
    return c
end

"""
    evaluate_operand(c::AbstractCurrent)

Leaf operands are already evaluated; return self.
"""
function evaluate_operand(c::AbstractCurrent)
    return c
end

"""
    reinitialize(c::AbstractCurrent; kw...)

Reinitialize a current operand. Default is identity.
"""
function reinitialize(c::AbstractCurrent; kw...)
    return c
end

# ============================================================================
# Buffer management helpers
# ============================================================================

"""
    create_buffer(buffer_size::Int) -> Vector{Float64}

Create a buffer for Field data. Uses plain `zeros` allocation.
In the Python Dedalus, this uses FFTW SIMD-aligned allocation; here
we use standard Julia allocation (Julia's allocator typically provides
adequate alignment for SIMD).
"""
function create_buffer(buffer_size::Int)
    if buffer_size == 0
        return zeros(Float64, 0)
    else
        alloc_doubles = buffer_size ÷ 8
        return zeros(Float64, alloc_doubles)
    end
end

# ============================================================================
# Field <: AbstractCurrent
# ============================================================================

"""
    Field <: AbstractCurrent

Scalar field over a domain.  Can also represent vector / tensor fields
via the `tensorsig` tuple.

# Fields
- `dist` -- distributor managing data distribution across processes.
- `name::Union{Nothing,String}` -- human-readable name.
- `tensorsig::Tuple` -- tuple of coordinate systems for tensor indices.
- `dtype::DataType` -- element type (`Float64`, `ComplexF64`, etc.).
- `domain::Domain` -- the spectral domain this field lives on.
- `scales::Union{Nothing,Tuple}` -- current scale factors per axis.
- `buffer::Vector{Float64}` -- raw memory buffer backing the data.
- `layout` -- current layout object.
- `data::Array` -- view of the buffer in the current layout.
- `_cache::Dict{Symbol,Any}` -- lazy-attribute cache.
- `buffer_size::Int` -- current buffer size in bytes (for reallocation logic).

# Examples
```julia
f = Field(dist; name="u", dtype=Float64)
f[Symbol("g")] = 1.0       # set in grid space
data = f[Symbol("c")]       # get coefficient-space data
```
"""
mutable struct Field <: AbstractCurrent
    dist::Any
    name::Union{Nothing,String}
    tensorsig::Tuple
    dtype::DataType
    domain::Domain
    scales::Union{Nothing,Tuple}
    buffer::Vector{Float64}
    layout::Any
    data::Array
    _cache::Dict{Symbol,Any}
    buffer_size::Int

    function Field(dist; bases=nothing, name=nothing, tensorsig=nothing, dtype=nothing)
        if bases === nothing
            bases = ()
        end
        # Accept single basis in place of tuple/list
        if !isa(bases, Tuple)
            if isa(bases, AbstractVector)
                bases = Tuple(bases)
            else
                bases = (bases,)
            end
        end
        if tensorsig === nothing
            tensorsig = ()
        end
        if dtype === nothing
            if dist.dtype === nothing
                throw(ArgumentError("dtype must be specified for Distributor or Field."))
            end
            dtype = dist.dtype
        end
        domain = Domain(dist, bases)
        # Create a minimal instance; scales/buffer/layout/data will be set by preset_scales
        f = new(dist, name, tensorsig, dtype, domain,
                nothing,              # scales
                zeros(Float64, 0),    # buffer placeholder
                nothing,              # layout placeholder
                Array{dtype}(undef, ntuple(_->0, get_dim(dist))...), # data placeholder
                Dict{Symbol,Any}(),
                -1)
        # Set initial layout to coefficient space
        f.layout = dist.get_layout_object(Symbol("c"))
        # Set initial scales and build buffer/data
        preset_scales!(f, ntuple(_ -> 1, get_dim(dist)))
        # Add weak reference to distributor's field set
        if hasproperty(dist, :fields) && dist.fields !== nothing
            push!(dist.fields, f)
        end
        return f
    end
end

# ============================================================================
# Field -- indexing (getindex / setindex!)
# ============================================================================

"""
    Base.getindex(f::Field, key)

Return data viewed in the specified layout. If `key` is a tuple
`(layout, scales)`, change both scales and layout first.
"""
function Base.getindex(f::Field, key)
    if isa(key, Tuple)
        layout_key, sc = key
        change_scales!(f, sc)
        change_layout!(f, layout_key)
    else
        change_layout!(f, key)
    end
    return f.data
end

"""
    Base.setitem!(f::Field, key, data)

Set data viewed in the specified layout. If `key` is a tuple
`(layout, scales)`, preset both scales and layout first.
"""
function Base.setindex!(f::Field, data, key)
    if isa(key, Tuple)
        layout_key, sc = key
        preset_scales!(f, sc)
        layout = f.dist.get_layout_object(layout_key)
    else
        layout = f.dist.get_layout_object(key)
    end
    preset_layout!(f, layout)
    dedalus_copyto!(f.data, data)
    return data
end

# ============================================================================
# Field -- accessors
# ============================================================================

"""
    get_basis(f::Field, coord)

Return the basis covering the given coordinate for this field's domain.
"""
function get_basis(f::Field, coord)
    return get_basis(f.domain, coord)
end

"""
    global_shape(f::Field)

Return the global data shape in the current layout and scales.
"""
function global_shape(f::Field)
    return f.layout.global_shape(f.domain, f.scales)
end

"""
    copy_field(f::Field)

Return a deep copy of the field with the same layout, scales, and data.
"""
function copy_field(f::Field)
    c = Field(f.dist; bases=f.domain.bases, tensorsig=f.tensorsig, dtype=f.dtype)
    preset_scales!(c, f.scales)
    c[f.layout] = f.data
    return c
end

"""
    is_scalar_field(f::Field) -> Bool

Check whether the field is a scalar (all bases are nothing).
"""
function is_scalar_field(f::Field)
    return all(b === nothing for b in f.domain.bases)
end

"""
    local_elements(f::Field)

Return the local elements in the current layout.
"""
function local_elements(f::Field)
    return f.layout.local_elements(f.domain, f.scales)
end

# ============================================================================
# Field -- scale and layout management
# ============================================================================

"""
    _dealias_buffer_size(f::Field) -> Int

Compute and cache the buffer size needed for dealias-scale data.
"""
function _dealias_buffer_size(f::Field)
    return get!(f._cache, :dealias_buffer_size) do
        f.dist.buffer_size(f.domain, domain_dealias(f.domain); dtype=f.dtype)
    end
end

"""
    _dealias_buffer(f::Field) -> Vector{Float64}

Build and cache a buffer large enough for dealias-scale data.
"""
function _dealias_buffer(f::Field)
    return get!(f._cache, :dealias_buffer) do
        bs = _dealias_buffer_size(f)
        ncomp = prod(get_dim(vs) for vs in f.tensorsig; init=1)
        create_buffer(ncomp * bs)
    end
end

"""
    preset_scales!(f::Field, scales)

Set new transform scales, reallocating the buffer if necessary.
Does not transform data -- use `change_scales!` for that.
"""
function preset_scales!(f::Field, scales)
    new_scales = f.dist.remedy_scales(scales)
    old_scales = f.scales
    # Return if scales are unchanged
    if new_scales == old_scales
        return nothing
    end
    # Get required buffer size
    buffer_size = f.dist.buffer_size(f.domain, new_scales; dtype=f.dtype)
    # Use dealias buffer if possible
    dbs = _dealias_buffer_size(f)
    if buffer_size <= dbs
        f.buffer = _dealias_buffer(f)
    else
        ncomp = prod(get_dim(vs) for vs in f.tensorsig; init=1)
        f.buffer = create_buffer(ncomp * buffer_size)
    end
    # Reset layout to build new data view
    f.scales = new_scales
    if f.layout !== nothing
        preset_layout!(f, f.layout)
    end
    return nothing
end

"""
    preset_layout!(f::Field, layout)

Interpret the buffer as data in the specified layout. Rebuilds the data view.
"""
function preset_layout!(f::Field, layout)
    layout = f.dist.get_layout_object(layout)
    f.layout = layout
    tens_shape = [get_dim(vs) for vs in f.tensorsig]
    local_shape = layout.local_shape(f.domain, f.scales)
    total_shape = Tuple(vcat(tens_shape, collect(local_shape)))
    total_len = prod(total_shape; init=1)
    # Build a view into the buffer
    if total_len > 0 && length(f.buffer) >= total_len
        data_flat = reinterpret(f.dtype, view(f.buffer, 1:total_len * sizeof(f.dtype) ÷ sizeof(Float64)))
        f.data = reshape(data_flat, total_shape)
    else
        f.data = Array{f.dtype}(undef, total_shape...)
    end
    return nothing
end

"""
    change_scales!(f::Field, scales)

Change data to specified scales, preserving data via transforms if needed.
"""
function change_scales!(f::Field, scales)
    new_scales = f.dist.remedy_scales(scales)
    old_scales = f.scales
    # Quit if new scales aren't new
    if new_scales == old_scales
        return nothing
    end
    # Forward transform until remaining scales match
    dim = get_dim(f.dist)
    for ax in dim:-1:1
        if !f.layout.grid_space[ax]
            break
        end
        if old_scales[ax] != new_scales[ax]
            require_coeff_space!(f, ax)
            break
        end
    end
    # Copy over scale change
    old_data = copy(f.data)
    preset_scales!(f, scales)
    dedalus_copyto!(f.data, old_data)
    return nothing
end

"""
    change_layout!(f::Field, layout)

Change data to the specified layout via sequential transforms.
"""
function change_layout!(f::Field, layout)
    layout = f.dist.get_layout_object(layout)
    if f.layout.index < layout.index
        while f.layout.index < layout.index
            towards_grid_space!(f)
        end
    elseif f.layout.index > layout.index
        while f.layout.index > layout.index
            towards_coeff_space!(f)
        end
    end
    return nothing
end

"""
    towards_grid_space!(f::Field)

Change to the next layout towards grid space.
"""
function towards_grid_space!(f::Field)
    index = f.layout.index
    f.dist.paths[index].increment([f])
    return nothing
end

"""
    towards_coeff_space!(f::Field)

Change to the next layout towards coefficient space.
"""
function towards_coeff_space!(f::Field)
    index = f.layout.index
    f.dist.paths[index - 1].decrement([f])
    return nothing
end

"""
    require_grid_space!(f::Field, axis=nothing)

Require one axis (default: all axes) to be in grid space.
"""
function require_grid_space!(f::Field, axis=nothing)
    if axis === nothing
        while !all(f.layout.grid_space)
            towards_grid_space!(f)
        end
    else
        while !f.layout.grid_space[axis]
            towards_grid_space!(f)
        end
    end
    return nothing
end

"""
    require_coeff_space!(f::Field, axis=nothing)

Require one axis (default: all axes) to be in coefficient space.
"""
function require_coeff_space!(f::Field, axis=nothing)
    if axis === nothing
        while any(f.layout.grid_space)
            towards_coeff_space!(f)
        end
    else
        while f.layout.grid_space[axis]
            towards_coeff_space!(f)
        end
    end
    return nothing
end

"""
    require_local!(f::Field, axis::Int)

Require an axis to be local.
"""
function require_local!(f::Field, axis::Int)
    if f.layout.grid_space[axis]
        while !f.layout.local[axis]
            towards_coeff_space!(f)
        end
    else
        while !f.layout.local[axis]
            towards_grid_space!(f)
        end
    end
    return nothing
end

# ============================================================================
# Field -- distributed data operations
# ============================================================================

"""
    allgather_data(f::Field; layout=nothing)

Build global data on all processes using Allreduce.
"""
function allgather_data(f::Field; layout=nothing)
    if layout !== nothing
        change_layout!(f, layout)
    end
    # Shortcut for serial execution
    if f.dist.comm.size == 1
        return copy(f.data)
    end
    # Build global buffers
    tensor_shape = Tuple(get_dim(cs) for cs in f.tensorsig)
    gs = f.layout.global_shape(f.domain, f.scales)
    global_shape_full = (tensor_shape..., gs...)
    component_slices = ntuple(_ -> Colon(), length(f.tensorsig))
    spatial_slices = f.layout.slices(f.domain, f.scales)
    local_slices = (component_slices..., spatial_slices...)
    send_buff = zeros(f.dtype, global_shape_full)
    recv_buff = similar(send_buff)
    send_buff[local_slices...] .= f.data
    # MPI Allreduce -- not communication-optimal but simple
    f.dist.comm.Allreduce(send_buff, recv_buff, MPI.SUM)
    return recv_buff
end

"""
    gather_data(f::Field; root=0, layout=nothing)

Gather global data on a single root process.
"""
function gather_data(f::Field; root::Int=0, layout=nothing)
    if layout !== nothing
        change_layout!(f, layout)
    end
    if f.dist.comm.size == 1
        return copy(f.data)
    end
    pieces = f.dist.comm.gather(f.data; root=root)
    if f.dist.comm.rank == root
        ext_mesh = f.layout.ext_mesh
        n = prod(ext_mesh)
        combined = Vector{Any}(undef, n)
        combined .= pieces
        # Assemble blocks using numpy-like np.block via reshaping
        return _block_assemble(combined, ext_mesh)
    end
    return nothing
end

"""
    _block_assemble(pieces, mesh)

Assemble array pieces into a single block array (helper for gather_data).
"""
function _block_assemble(pieces, mesh)
    # Simple case: just concatenate
    if length(mesh) == 1
        return cat(pieces...; dims=1)
    end
    # General case: recursive block assembly
    return cat(pieces...; dims=1)
end

"""
    allreduce_data_norm(f::Field; layout=nothing, order=2)

Compute a global data norm via MPI Allreduce.
"""
function allreduce_data_norm(f::Field; layout=nothing, order=2)
    if layout !== nothing
        change_layout!(f, layout)
    end
    # Compute local norm
    if length(f.data) == 0
        norm_val = 0.0
    elseif order == Inf
        norm_val = maximum(abs.(f.data))
    else
        norm_val = sum(abs.(f.data) .^ order)
    end
    # Reduce
    if order == Inf
        if f.dist.comm.size > 1
            norm_val = f.dist.comm.allreduce(norm_val, MPI.MAX)
        end
    else
        if f.dist.comm.size > 1
            norm_val = f.dist.comm.allreduce(norm_val, MPI.SUM)
        end
        norm_val = norm_val ^ (1 / order)
    end
    return norm_val
end

"""
    allreduce_data_max(f::Field; layout=nothing)

Compute the global maximum absolute value via MPI Allreduce.
"""
function allreduce_data_max(f::Field; layout=nothing)
    return allreduce_data_norm(f; layout=layout, order=Inf)
end

"""
    fill_random!(f::Field; layout=nothing, scales=nothing, seed=nothing,
                 chunk_size=2^20, distribution="standard_normal", kw...)

Fill field with random data. If a seed is specified, the global data is
reproducibly generated for any process mesh.

Parameters
----------
layout : optional
    Layout for setting field data. Default: current layout.
scales : optional
    Scales for setting field data. Default: current scales.
seed : optional Int
    RNG seed.
chunk_size : Int
    Chunk size for drawing from distribution.
distribution : String
    Distribution name. Default: "standard_normal".
"""
function fill_random!(f::Field; layout=nothing, scales=nothing, seed=nothing,
                      chunk_size::Int=2^20, distribution::String="standard_normal",
                      kw...)
    init_layout = f.layout
    # Set scales if requested
    if scales !== nothing
        preset_scales!(f, scales)
        if layout === nothing
            preset_layout!(f, init_layout)
        end
    end
    # Set layout if requested
    if layout !== nothing
        preset_layout!(f, layout)
    end
    # Build global chunked random array
    gs = global_shape(f)
    shape = (Tuple(get_dim(cs) for cs in f.tensorsig)..., gs...)
    if is_complex_operand(f)
        shape = (shape..., 2)
    end
    global_data = ChunkedRandomArray(shape, seed, chunk_size, distribution)
    # Extract local data
    component_slices = ntuple(_ -> Colon(), length(f.tensorsig))
    spatial_slices = f.layout.slices(f.domain, f.scales)
    local_slices = (component_slices..., spatial_slices...)
    local_data = global_data[local_slices...]
    if is_real_operand(f)
        f.data .= local_data
    else
        f.data .= complex.(local_data[.., 1], local_data[.., 2])
    end
    return nothing
end

"""
    low_pass_filter!(f::Field; shape=nothing, scales=nothing)

Apply a spectral low-pass filter by zeroing modes above specified relative scales.
"""
function low_pass_filter!(f::Field; shape=nothing, scales=nothing)
    original_scales = f.scales
    if shape !== nothing
        if scales !== nothing
            throw(ArgumentError("Specify either shape or scales."))
        end
        gs = f.dist.grid_layout.global_shape(f.domain, 1)
        scales = Tuple(shape[i] / gs[i] for i in eachindex(shape))
    end
    change_scales!(f, scales)
    require_grid_space!(f)
    change_scales!(f, original_scales)
    return nothing
end

"""
    high_pass_filter!(f::Field; shape=nothing, scales=nothing)

Apply a spectral high-pass filter by zeroing modes below specified relative scales.
"""
function high_pass_filter!(f::Field; shape=nothing, scales=nothing)
    data_orig = copy(f[Symbol("c")])
    low_pass_filter!(f; shape=shape, scales=scales)
    data_filt = copy(f[Symbol("c")])
    f[Symbol("c")] = data_orig .- data_filt
    return nothing
end

"""
    load_from_hdf5(f::Field, file, index; task=nothing, func=nothing)

Load grid data from an HDF5 file. Task corresponds to field name by default.
"""
function load_from_hdf5(f::Field, file, index; task=nothing, func=nothing)
    if task === nothing
        task = f.name
    end
    dset = file["tasks"][task]
    grid_space_flags = dset.attrs["grid_space"]
    if all(grid_space_flags)
        load_from_global_grid_data!(f, dset; pre_slices=(index,), func=func)
    elseif all(.!grid_space_flags)
        load_from_global_coeff_data!(f, dset; pre_slices=(index,), func=func)
    else
        throw(ArgumentError("Can only load global data from pure grid or coeff space"))
    end
    return nothing
end

"""
    load_from_global_coeff_data!(f::Field, global_data; pre_slices=(), func=nothing)

Load local coeff data from array-like global coeff data.
"""
function load_from_global_coeff_data!(f::Field, global_data;
                                      pre_slices::Tuple=(), func=nothing)
    dim = get_dim(f.dist)
    layout = f.dist.coeff_layout
    # Check shapes
    data_shape = size(global_data)[(end-dim+1):end]
    self_shape = layout.global_shape(f.domain; scales=1)
    if data_shape != self_shape
        throw(ArgumentError("Cannot change global shape when loading coeff data."))
    end
    # Extract local data from global data
    component_slices = ntuple(_ -> Colon(), length(f.tensorsig))
    spatial_slices = layout.slices(f.domain, 1)
    local_slices = (pre_slices..., component_slices..., spatial_slices...)
    if func === nothing
        f[layout] = global_data[local_slices...]
    else
        f[layout] = func(global_data[local_slices...])
    end
    return nothing
end

"""
    load_from_global_grid_data!(f::Field, global_data; pre_slices=(), func=nothing)

Load local grid data from array-like global grid data.
"""
function load_from_global_grid_data!(f::Field, global_data;
                                     pre_slices::Tuple=(), func=nothing)
    dim = get_dim(f.dist)
    layout = f.dist.grid_layout
    # Set scales to match saved data
    saved_shape = size(global_data)[(end-dim+1):end]
    base_shape = layout.global_shape(f.domain; scales=1)
    sc = Tuple(saved_shape[i] / base_shape[i] for i in 1:dim)
    preset_scales!(f, sc)
    # Extract local data
    component_slices = ntuple(_ -> Colon(), length(f.tensorsig))
    spatial_slices = layout.slices(f.domain, sc)
    local_slices = (pre_slices..., component_slices..., spatial_slices...)
    if func === nothing
        f[layout] = global_data[local_slices...]
    else
        f[layout] = func(global_data[local_slices...])
    end
    # Change scales back to dealias scales
    change_scales!(f, domain_dealias(f.domain))
    return nothing
end

"""
    set_global_data!(f::Field, global_data)

Set local data from global data using layout elements.
"""
function set_global_data!(f::Field, global_data)
    elements = f.layout.local_elements(f.domain, f.scales)
    local_data = global_data[elements...]
    dedalus_copyto!(f.data, local_data)
    return nothing
end

"""
    set_local_data!(f::Field, local_data)

Set data directly from local-shaped array.
"""
function set_local_data!(f::Field, local_data)
    dedalus_copyto!(f.data, local_data)
    return nothing
end

"""
    broadcast_ghosts(f::Field, output_nonconst_dims)

Copy data over constant distributed dimensions for arithmetic broadcasting.
"""
function broadcast_ghosts(f::Field, output_nonconst_dims)
    self_const_dims = collect(domain_constant(f.domain))
    distributed = .!f.layout.local
    broadcast_dims = output_nonconst_dims .& self_const_dims
    deploy_dims_ext = broadcast_dims .& distributed
    deploy_dims = deploy_dims_ext[distributed]
    if !any(deploy_dims)
        return f.data
    end
    # Broadcast on subgrid communicator
    comm_sub = f.domain.dist.comm_cart.Sub(remain_dims=Int.(deploy_dims))
    if comm_sub.rank == 0
        data = f.data
    else
        shape = collect(size(f.data))
        shape[shape .== 0] .= 1
        data = similar(f.data, Tuple(shape))
    end
    comm_sub.Bcast(data; root=0)
    return data
end

# ============================================================================
# ScalarField alias
# ============================================================================

"""
    ScalarField

Alias for `Field`. A scalar field has an empty `tensorsig`.
"""
const ScalarField = Field

# ============================================================================
# VectorField convenience constructor
# ============================================================================

"""
    VectorField(dist, coordsys; kw...)

Convenience function to create a vector `Field` with `tensorsig = (coordsys,)`.
"""
function VectorField(dist, coordsys, args...; kw...)
    return Field(dist, args...; tensorsig=(coordsys,), kw...)
end

# ============================================================================
# TensorField convenience constructor
# ============================================================================

"""
    TensorField(dist, coordsys; order=2, kw...)

Convenience function to create a tensor `Field`.

If `coordsys` is a tuple or vector, it is used directly as the `tensorsig`.
Otherwise, it is repeated `order` times to form the `tensorsig`.
"""
function TensorField(dist, coordsys, args...; order::Int=2, kw...)
    if isa(coordsys, Tuple) || isa(coordsys, AbstractVector)
        tensorsig = Tuple(coordsys)
    else
        tensorsig = ntuple(_ -> coordsys, order)
    end
    return Field(dist, args...; tensorsig=tensorsig, kw...)
end

# ============================================================================
# LockedField <: AbstractCurrent
# ============================================================================

"""
    LockedField <: AbstractCurrent

A Field that is locked to particular layouts, disallowing changes to
layouts not in `allowed_layouts`.

Wraps a `Field` and overrides `change_scales!`, `towards_grid_space!`,
and `towards_coeff_space!` to enforce layout restrictions.
"""
mutable struct LockedField <: AbstractCurrent
    # Delegate all field storage to the inner Field
    dist::Any
    name::Union{Nothing,String}
    tensorsig::Tuple
    dtype::DataType
    domain::Domain
    scales::Union{Nothing,Tuple}
    buffer::Vector{Float64}
    layout::Any
    data::Array
    _cache::Dict{Symbol,Any}
    buffer_size::Int
    allowed_layouts::Tuple

    function LockedField(dist; bases=nothing, name=nothing, tensorsig=nothing,
                         dtype=nothing)
        f = Field(dist; bases=bases, name=name, tensorsig=tensorsig, dtype=dtype)
        return new(f.dist, f.name, f.tensorsig, f.dtype, f.domain,
                   f.scales, f.buffer, f.layout, f.data, f._cache, f.buffer_size,
                   ())  # no allowed layouts initially
    end
end

"""
    change_scales!(f::LockedField, scales)

Override: locked fields cannot change scales.
"""
function change_scales!(f::LockedField, scales)
    sc = f.dist.remedy_scales(scales)
    if sc != f.scales
        throw(ArgumentError("Cannot change locked scales."))
    end
    return nothing
end

"""
    towards_grid_space!(f::LockedField)

Override: only allowed if the target layout is in `allowed_layouts`.
"""
function towards_grid_space!(f::LockedField)
    index = f.layout.index
    new_index = index + 1
    new_layout = f.dist.layouts[new_index]
    if new_layout in f.allowed_layouts
        # Delegate to the standard Field logic via the dist paths
        f.dist.paths[index].increment([f])
    else
        throw(ArgumentError("Cannot change locked layout."))
    end
    return nothing
end

"""
    towards_coeff_space!(f::LockedField)

Override: only allowed if the target layout is in `allowed_layouts`.
"""
function towards_coeff_space!(f::LockedField)
    index = f.layout.index
    new_index = index - 1
    new_layout = f.dist.layouts[new_index]
    if new_layout in f.allowed_layouts
        f.dist.paths[index - 1].decrement([f])
    else
        throw(ArgumentError("Cannot change locked layout."))
    end
    return nothing
end

"""
    lock_to_layouts!(f::LockedField, layouts...)

Set the allowed layouts for this locked field.
"""
function lock_to_layouts!(f::LockedField, layouts...)
    f.allowed_layouts = Tuple(layouts)
    return nothing
end

"""
    lock_axis_to_grid!(f::LockedField, axis::Int)

Lock to all layouts where the given axis is in grid space.
"""
function lock_axis_to_grid!(f::LockedField, axis::Int)
    f.allowed_layouts = Tuple(l for l in f.dist.layouts if l.grid_space[axis])
    return nothing
end

"""
    unlock(f::LockedField) -> Field

Return a regular Field object with the same data and no layout locking.
"""
function unlock(f::LockedField)
    field = Field(f.dist; bases=f.domain.bases, name=f.name,
                  tensorsig=f.tensorsig, dtype=f.dtype)
    preset_scales!(field, f.scales)
    field[f.layout] = f.data
    return field
end

# Delegate common methods from LockedField to its inner field-like storage
# (LockedField has the same struct fields as Field, so these work directly)

Base.getindex(f::LockedField, key) = Base.getindex(f::Field, key)
Base.setindex!(f::LockedField, data, key) = Base.setindex!(f::Field, data, key)

# LockedField shares all AbstractCurrent methods via the abstract type dispatch.

# ============================================================================
# Shared Field/LockedField operations
# ============================================================================

# Make getindex/setindex! work for LockedField via reinterpretation as Field-like
# Since LockedField has all the same fields, the Field methods above work on it too
# via the AbstractCurrent dispatch. We need to provide explicit methods though
# because Julia dispatches on concrete types.

function Base.getindex(f::LockedField, key)
    if isa(key, Tuple)
        layout_key, sc = key
        change_scales!(f, sc)  # Will throw if locked
        change_layout!(f, layout_key)
    else
        change_layout!(f, key)
    end
    return f.data
end

function Base.setindex!(f::LockedField, data, key)
    if isa(key, Tuple)
        layout_key, sc = key
        preset_scales!(f, sc)
        layout = f.dist.get_layout_object(layout_key)
    else
        layout = f.dist.get_layout_object(key)
    end
    preset_layout!(f, layout)
    dedalus_copyto!(f.data, data)
    return data
end

# preset_scales!/preset_layout! for LockedField (delegate to the Field versions)
function preset_scales!(f::LockedField, scales)
    new_scales = f.dist.remedy_scales(scales)
    old_scales = f.scales
    if new_scales == old_scales
        return nothing
    end
    buffer_size = f.dist.buffer_size(f.domain, new_scales; dtype=f.dtype)
    dbs = _dealias_buffer_size(f)
    if buffer_size <= dbs
        f.buffer = _dealias_buffer(f)
    else
        ncomp = prod(get_dim(vs) for vs in f.tensorsig; init=1)
        f.buffer = create_buffer(ncomp * buffer_size)
    end
    f.scales = new_scales
    if f.layout !== nothing
        preset_layout!(f, f.layout)
    end
    return nothing
end

function preset_layout!(f::LockedField, layout)
    layout = f.dist.get_layout_object(layout)
    f.layout = layout
    tens_shape = [get_dim(vs) for vs in f.tensorsig]
    local_shape = layout.local_shape(f.domain, f.scales)
    total_shape = Tuple(vcat(tens_shape, collect(local_shape)))
    total_len = prod(total_shape; init=1)
    if total_len > 0 && length(f.buffer) >= total_len
        data_flat = reinterpret(f.dtype, view(f.buffer, 1:total_len * sizeof(f.dtype) ÷ sizeof(Float64)))
        f.data = reshape(data_flat, total_shape)
    else
        f.data = Array{f.dtype}(undef, total_shape...)
    end
    return nothing
end

function change_layout!(f::LockedField, layout)
    layout = f.dist.get_layout_object(layout)
    if f.layout.index < layout.index
        while f.layout.index < layout.index
            towards_grid_space!(f)
        end
    elseif f.layout.index > layout.index
        while f.layout.index > layout.index
            towards_coeff_space!(f)
        end
    end
    return nothing
end

function require_grid_space!(f::LockedField, axis=nothing)
    if axis === nothing
        while !all(f.layout.grid_space)
            towards_grid_space!(f)
        end
    else
        while !f.layout.grid_space[axis]
            towards_grid_space!(f)
        end
    end
    return nothing
end

function require_coeff_space!(f::LockedField, axis=nothing)
    if axis === nothing
        while any(f.layout.grid_space)
            towards_coeff_space!(f)
        end
    else
        while f.layout.grid_space[axis]
            towards_coeff_space!(f)
        end
    end
    return nothing
end

# ============================================================================
# AssertionError compatibility
# ============================================================================
# The Python Dedalus codebase uses "AssertionError" (their consistent spelling
# throughout the project).  We define it here as a Julia exception type.

"""
    AssertionError <: Exception

Exception type matching the Python Dedalus `AssertionError` name. Used for
linearity / independence / first-order checks in the expression tree.
"""
struct AssertionError <: Exception
    msg::String
end

Base.showerror(io::IO, e::AssertionError) =
    print(io, "AssertionError: ", e.msg)

# ============================================================================
# Exports
# ============================================================================

export AbstractOperand,
       AbstractCurrent,
       AbstractFuture,
       Field,
       ScalarField,
       VectorField,
       TensorField,
       LockedField,
       dedalus_add,
       dedalus_multiply,
       dedalus_power,
       operand_cast,
       atoms,
       has_operand,
       split_operand,
       replace_operand,
       replace_dict,
       sym_diff,
       expand_operand,
       require_linearity,
       require_first_order,
       require_independent,
       build_ncc_matrices,
       expression_matrices,
       frechet_differential,
       is_complex_operand,
       is_real_operand,
       valid_modes,
       get_basis,
       global_shape,
       copy_field,
       is_scalar_field,
       local_elements,
       change_scales!,
       change_layout!,
       towards_grid_space!,
       towards_coeff_space!,
       require_grid_space!,
       require_coeff_space!,
       require_local!,
       preset_scales!,
       preset_layout!,
       allgather_data,
       gather_data,
       allreduce_data_norm,
       allreduce_data_max,
       fill_random!,
       low_pass_filter!,
       high_pass_filter!,
       load_from_hdf5,
       load_from_global_coeff_data!,
       load_from_global_grid_data!,
       set_global_data!,
       set_local_data!,
       broadcast_ghosts,
       create_buffer,
       attempt,
       evaluate_operand,
       reinitialize,
       matrix_dependence,
       matrix_coupling,
       lock_to_layouts!,
       lock_axis_to_grid!,
       unlock
