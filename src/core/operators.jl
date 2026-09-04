# Operators module for Dedalus.jl
# Translates the 1D, Cartesian, and DirectProduct portions of operators.py
#
# Types available from earlier includes:
#   AbstractOperand, AbstractCurrent, AbstractFuture, FutureField,
#   FutureLockedField, Field, LockedField, Domain
#
# Functions available from earlier includes:
#   dedalus_add, dedalus_multiply, evaluate_future, check_conditions,
#   enforce_conditions, operate, get_out, build_out, preset_layout!,
#   require_grid_space!, require_coeff_space!, change_layout!,
#   unify_attributes, get_basis, get_axis, get_basis_axis, substitute_basis

using SparseArrays
using LinearAlgebra

# ============================================================================
# Abstract operator types
# ============================================================================

"""Abstract base for all operators in the expression tree."""
abstract type AbstractOperator <: AbstractFuture end

"""Abstract base for linear operators (matrix representable)."""
abstract type AbstractLinearOperator <: AbstractOperator end

"""Abstract base for spectral operators acting on specific bases."""
abstract type SpectralOperator <: AbstractLinearOperator end

"""Abstract base for spectral operators acting on a single coordinate."""
abstract type SpectralOperator1D <: SpectralOperator end

"""Abstract base for nonlinear operators."""
abstract type NonlinearOperator <: AbstractOperator end

# ============================================================================
# AbstractField (requested by code review)
# ============================================================================

"""
    AbstractField

Abstract type encompassing all field-like objects: both current (evaluated)
and future (deferred) fields.
"""
const AbstractField = Union{AbstractCurrent, FutureField}

# ============================================================================
# Operator alias registry
# ============================================================================

"""Global alias dictionary for operator dispatch from string names."""
const OPERATOR_ALIASES = Dict{String, Any}()

function register_operator_alias!(name::String, op)
    OPERATOR_ALIASES[name] = op
    return op
end

# ============================================================================
# Common interface methods for LinearOperator types
# ============================================================================

"""
    operator_operand(op)

Return the primary operand of a linear operator. By convention this is args[1].
"""
operator_operand(op::AbstractLinearOperator) = op.args[1]

"""
    new_operand(op::AbstractLinearOperator, operand; kw...)

Create a new operator of the same kind with a different operand.
Subtypes should override this.
"""
function new_operand(op::AbstractLinearOperator, operand; kw...)
    error("$(typeof(op)) has not implemented new_operand")
end

"""
    matrix_dependence(op::AbstractLinearOperator, vars...)

Determine dimension-by-dimension matrix dependence.
Default delegates to operand.
"""
function matrix_dependence(op::AbstractLinearOperator, vars...)
    return matrix_dependence(operator_operand(op), vars...)
end

"""
    matrix_coupling(op::AbstractLinearOperator, vars...)

Determine dimension-by-dimension matrix coupling.
Default delegates to operand.
"""
function matrix_coupling(op::AbstractLinearOperator, vars...)
    return matrix_coupling(operator_operand(op), vars...)
end

"""
    build_ncc_matrices(op::AbstractLinearOperator, separability, vars; kw...)

Precompute NCC matrices. Default delegates to operand.
"""
function build_ncc_matrices(op::AbstractLinearOperator, separability, vars; kw...)
    build_ncc_matrices(operator_operand(op), separability, vars; kw...)
end

"""
    expression_matrices(op::AbstractLinearOperator, subproblem, vars; kw...)

Build expression matrices. Default: gets operand matrices and left-multiplies
by the operator's subproblem_matrix.
"""
function expression_matrices(op::AbstractLinearOperator, subproblem, vars; kw...)
    # Intercept calls when self is a variable
    if op in vars
        size_val = field_size(subproblem, op)
        matrix = sparse(1.0I, size_val, size_val)
        return Dict(op => matrix)
    end
    # Build operand matrices
    operand = operator_operand(op)
    operand_mats = expression_matrices(operand, subproblem, vars; kw...)
    # Apply operator matrix if subproblem_matrix is defined
    if hasmethod(subproblem_matrix, Tuple{typeof(op), typeof(subproblem)})
        op_mat = subproblem_matrix(op, subproblem)
        return Dict(var => op_mat * operand_mats[var] for var in keys(operand_mats))
    end
    return operand_mats
end

"""
    subproblem_matrix(op, subproblem)

Build the sparse matrix representation of this operator for a subproblem.
Subtypes should override.
"""
function subproblem_matrix(op::AbstractLinearOperator, subproblem)
    error("$(typeof(op)) has not implemented subproblem_matrix")
end

"""
    reinitialize(op::AbstractLinearOperator; kw...)

Reinitialize with reinitialized operand.
"""
function reinitialize(op::AbstractLinearOperator; kw...)
    operand = reinitialize(operator_operand(op); kw...)
    return new_operand(op, operand; kw...)
end

"""
    split_op(op::AbstractLinearOperator, vars...)

Split into expressions containing and not containing specified operands.
"""
function split_op(op::AbstractLinearOperator, vars...)
    for var in vars
        if isa(var, DataType) && isa(op, var)
            return (op, 0)
        end
    end
    operand = operator_operand(op)
    s = split_operand(operand, vars...)
    return (new_operand(op, s[1]), new_operand(op, s[2]))
end

"""
    sym_diff(op::AbstractLinearOperator, var)

Symbolically differentiate: distributes over the operand.
"""
function sym_diff(op::AbstractLinearOperator, var)
    return new_operand(op, sym_diff(operator_operand(op), var))
end

# ============================================================================
# Common interface methods for NonlinearOperator types
# ============================================================================

function split_op(op::NonlinearOperator, vars...)
    if has_operand(op, vars...)
        return (op, 0)
    else
        return (0, op)
    end
end

function expand_operand(op::NonlinearOperator, vars...)
    return op
end

split(op::AbstractOperator, vars...) = split_op(op, vars...)
expand(op::NonlinearOperator, vars...) = expand_operand(op, vars...)

function expand(op::AbstractLinearOperator, vars...)
    operand = operator_operand(op)
    e = expand(operand, vars...)
    if isa(e, Add)
        return sum(new_operand(op, a) for a in e.args)
    else
        return new_operand(op, e)
    end
end

# ============================================================================
# Power operator
# ============================================================================

"""
    Power

Lazy exponentiation operator: base^exponent where base is a field operand
and exponent is typically a Number.
"""
mutable struct Power <: NonlinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
    _grid_layout::Any
    _coeff_layout::Any
end

function Power(base_operand, exponent; out=nothing)
    if exponent == 0
        return 1
    elseif exponent == 1
        return base_operand
    end
    args = Any[base_operand, exponent]
    dist = base_operand.dist
    domain = base_operand.domain
    tensorsig = base_operand.tensorsig
    dtype = base_operand.dtype
    Power(args, (base_operand, exponent), out, dist, domain, tensorsig, dtype,
          "Pow", nothing, nothing, false, 1,
          dist.grid_layout, dist.coeff_layout)
end

"""Replace the stub from field.jl with a working Power operator."""
function dedalus_power(a, b)
    if isa(a, AbstractOperand)
        return Power(a, b)
    else
        return a^b
    end
end

function check_conditions(op::Power)
    arg = op.args[1]
    return isa(arg, AbstractCurrent) && all(arg.layout.grid_space)
end

function enforce_conditions(op::Power)
    arg = op.args[1]
    if isa(arg, Field)
        require_grid_space!(arg)
    end
end

function operate(op::Power, out)::Nothing
    arg0 = op.args[1]
    arg1 = op.args[2]
    preset_layout!(out, arg0.layout)
    if length(out.data) > 0
        out.data .= arg0.data .^ arg1
    end
    return nothing
end

function new_operands(op::Power, arg0, arg1; kw...)
    return Power(arg0, arg1)
end

function reinitialize(op::Power; kw...)
    arg0 = reinitialize(op.args[1]; kw...)
    arg1 = op.args[2]
    return new_operands(op, arg0, arg1; kw...)
end

function sym_diff(op::Power, var)
    base, power = op.args[1], op.args[2]
    return power * base^(power - 1) * sym_diff(base, var)
end

# Register alias
register_operator_alias!("pow", Power)

# ============================================================================
# FieldCopy (identity operator, wraps a Field as a FutureField)
# ============================================================================

"""
    FieldCopy

Identity operator that wraps a Field to produce a FutureField. Used by
cast_future to ensure all operands are FutureField.
"""
mutable struct FieldCopy <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function FieldCopy(arg; out=nothing)
    FieldCopy(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
              arg.dtype, "Copy", arg, nothing, nothing, false, 1)
end

# Wire up the forward reference from future.jl
_field_copy(arg) = FieldCopy(arg)

check_conditions(::FieldCopy) = true
enforce_conditions(::FieldCopy) = nothing

function operate(op::FieldCopy, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
    return nothing
end

function new_operand(op::FieldCopy, operand; kw...)
    return FieldCopy(operand; kw...)
end

function subproblem_matrix(op::FieldCopy, subproblem)
    sz = field_size(subproblem, op.operand)
    return sparse(1.0I, sz, sz)
end

# ============================================================================
# UnaryGridFunction (wraps numpy/scipy ufuncs)
# ============================================================================

"""
    UnaryGridFunction

Wraps a unary function for lazy application to field data in grid space.
The function must be vectorized (element-wise).
"""
mutable struct UnaryGridFunction <: NonlinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    func::Any
    deriv::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
    _grid_layout::Any
    _coeff_layout::Any
end

"""Built-in symbolic derivatives for common math functions."""
const UFUNC_DERIVATIVES = Dict{Any, Any}(
    abs   => x -> sign.(x),
    sign  => x -> 0,
    exp   => x -> exp.(x),
    log   => x -> x .^ (-1),
    sqrt  => x -> 0.5 .* x .^ (-0.5),
    sin   => x -> cos.(x),
    cos   => x -> -(sin.(x)),
    tan   => x -> cos.(x) .^ (-2),
    asin  => x -> (1 .- x .^ 2) .^ (-0.5),
    acos  => x -> -((1 .- x .^ 2) .^ (-0.5)),
    atan  => x -> (1 .+ x .^ 2) .^ (-1),
    sinh  => x -> cosh.(x),
    cosh  => x -> sinh.(x),
    tanh  => x -> 1 .- tanh.(x) .^ 2,
    asinh => x -> (x .^ 2 .+ 1) .^ (-0.5),
    acosh => x -> (x .^ 2 .- 1) .^ (-0.5),
    atanh => x -> (1 .- x .^ 2) .^ (-1),
)

function UnaryGridFunction(func, arg; deriv=nothing, out=nothing)
    if deriv === nothing
        deriv = get(UFUNC_DERIVATIVES, func, nothing)
    end
    dist = arg.dist
    UnaryGridFunction(Any[arg], (arg,), out, dist, arg.domain, arg.tensorsig,
                      arg.dtype, string(func), func, deriv,
                      nothing, nothing, false, 1,
                      dist.grid_layout, dist.coeff_layout)
end

function check_conditions(op::UnaryGridFunction)
    arg = op.args[1]
    return isa(arg, AbstractCurrent) && all(arg.layout.grid_space)
end

function enforce_conditions(op::UnaryGridFunction)
    arg = op.args[1]
    if isa(arg, Field)
        require_grid_space!(arg)
    end
end

function operate(op::UnaryGridFunction, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= op.func.(arg.data)
    end
    return nothing
end

function sym_diff(op::UnaryGridFunction, var)
    if op.deriv === nothing
        error("Symbolic derivative not implemented for $(op.name)")
    end
    arg = op.args[1]
    return op.deriv(arg) * sym_diff(arg, var)
end

function new_operands(op::UnaryGridFunction, arg; kw...)
    return UnaryGridFunction(op.func, arg; deriv=op.deriv)
end

# Register common function aliases
for (name, func) in [("exp", exp), ("log", log), ("sin", sin), ("cos", cos),
                      ("tan", tan), ("abs", abs), ("sqrt", sqrt),
                      ("sinh", sinh), ("cosh", cosh), ("tanh", tanh)]
    register_operator_alias!(name, func)
end

# ============================================================================
# Grid / Coeff operators (Lock to specific layout)
# ============================================================================

"""
    GridOperator

Locks the operand to grid layout. Equivalent to Python's Grid(operand).
"""
mutable struct GridOperator <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function GridOperator(arg; out=nothing)
    GridOperator(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
                 arg.dtype, "Grid", arg, nothing, nothing, false, 1)
end

function check_conditions(op::GridOperator)
    arg = op.args[1]
    return isa(arg, AbstractCurrent) && all(arg.layout.grid_space)
end

function enforce_conditions(op::GridOperator)
    arg = op.args[1]
    if isa(arg, Field)
        require_grid_space!(arg)
    end
end

function operate(op::GridOperator, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
    return nothing
end

function new_operand(op::GridOperator, operand; kw...)
    return GridOperator(operand; kw...)
end

"""
    CoeffOperator

Locks the operand to coefficient layout. Equivalent to Python's Coeff(operand).
"""
mutable struct CoeffOperator <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CoeffOperator(arg; out=nothing)
    CoeffOperator(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
                  arg.dtype, "Coeff", arg, nothing, nothing, false, 1)
end

function check_conditions(op::CoeffOperator)
    arg = op.args[1]
    return isa(arg, AbstractCurrent) && all(.!arg.layout.grid_space)
end

function enforce_conditions(op::CoeffOperator)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg)
    end
end

function operate(op::CoeffOperator, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
    return nothing
end

function new_operand(op::CoeffOperator, operand; kw...)
    return CoeffOperator(operand; kw...)
end

# ============================================================================
# TimeDerivative
# ============================================================================

"""
    TimeDerivative

Symbolic time derivative operator. Used during equation parsing; not
evaluated directly but rather used by timestepper infrastructure.
"""
mutable struct TimeDerivative <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function TimeDerivative(arg; out=nothing)
    TimeDerivative(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
                   arg.dtype, "dt", arg, nothing, nothing, false, 1)
end

check_conditions(::TimeDerivative) = true
enforce_conditions(::TimeDerivative) = nothing

function operate(::TimeDerivative, out)
    error("TimeDerivative should not be evaluated directly -- used symbolically by solvers")
end

function new_operand(op::TimeDerivative, operand; kw...)
    return TimeDerivative(operand; kw...)
end

function matrix_dependence(op::TimeDerivative, vars...)
    return matrix_dependence(op.operand, vars...)
end

function matrix_coupling(op::TimeDerivative, vars...)
    return matrix_coupling(op.operand, vars...)
end

register_operator_alias!("dt", TimeDerivative)

# ============================================================================
# Convert
# ============================================================================

"""
    Convert

Convert an operand between two spectral bases. In grid space, acts as
identity (copy). In coefficient space, applies the conversion matrix.
"""
mutable struct Convert <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    input_basis::Any
    output_basis::Any
    coord::Any
    first_axis::Int
    last_axis::Int
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Convert(arg, output_basis; out=nothing)
    dist = arg.dist
    coords = basis_coordsys(output_basis)
    input_basis = get_basis(arg.domain, coords)
    first_axis = get_basis_axis(dist, output_basis)
    last_axis = first_axis + get_dim(output_basis) - 1
    new_domain = substitute_basis(arg.domain, input_basis, output_basis)
    ndim = get_dim(output_basis)
    subaxis_dep = fill(true, ndim)
    subaxis_coup = fill(true, ndim)
    Convert(Any[arg], (arg,), out, dist, new_domain, arg.tensorsig, arg.dtype,
            "Convert", arg, input_basis, output_basis, coords,
            first_axis, last_axis, subaxis_dep, subaxis_coup,
            nothing, nothing, false, 1)
end

function check_conditions(op::Convert)
    arg = op.args[1]
    last_axis = op.last_axis
    last_is_coeff = !arg.layout.grid_space[last_axis]
    last_is_local = arg.layout.local[last_axis]
    if last_is_coeff && op.subaxis_coupling[end]
        return last_is_local
    end
    return true
end

function enforce_conditions(op::Convert)
    arg = op.args[1]
    last_axis = op.last_axis
    last_is_coeff = !arg.layout.grid_space[last_axis]
    if last_is_coeff && op.subaxis_coupling[end]
        require_local!(arg, last_axis)
    end
end

function operate(op::Convert, out)::Nothing
    arg = op.args[1]
    layout = arg.layout
    # Grid space: identity (copy data)
    if layout.grid_space[op.last_axis]
        preset_layout!(out, layout)
        if length(out.data) > 0
            out.data .= arg.data
        end
    else
        # Coefficient space: apply conversion matrix
        preset_layout!(out, layout)
        if length(arg.data) > 0 && length(out.data) > 0
            data_axis = op.last_axis + length(arg.tensorsig)
            apply_matrix(subspace_matrix(op, layout), arg.data, data_axis; out=out.data)
        else
            out.data .= 0
        end
    end
    return nothing
end

function new_operand(op::Convert, operand; kw...)
    # Pass through without conversion (matches Python behavior)
    return operand
end

"""
    convert_operand(arg, bases)

Convert an operand to the specified output bases. Applies Convert iteratively
for each non-nothing basis.
"""
function convert_operand(arg, bases)
    if isa(arg, Number)
        return arg
    end
    # Filter out Nothings
    real_bases = [b for b in bases if b !== nothing]
    for basis in real_bases
        arg = Convert(arg, basis)
    end
    return arg
end

# ============================================================================
# Differentiate
# ============================================================================

"""
    Differentiate

Spectral differentiation along a single coordinate. Applies the derivative
matrix in coefficient space.
"""
mutable struct Differentiate <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coord::Any
    input_basis::Any
    output_basis::Any
    first_axis::Int
    last_axis::Int
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Differentiate(arg, coord; out=nothing)
    if isa(arg, Number)
        return 0
    end
    dist = arg.dist
    input_basis = get_basis(arg.domain, coord)
    # Get output basis (derivative basis)
    output_basis = if input_basis !== nothing && hasmethod(derivative_basis, Tuple{typeof(input_basis)})
        derivative_basis(input_basis)
    else
        input_basis
    end
    axis = get_axis(dist, coord)
    new_domain = if input_basis !== nothing && output_basis !== nothing
        substitute_basis(arg.domain, input_basis, output_basis)
    else
        arg.domain
    end
    Differentiate(Any[arg], (arg,), out, dist, new_domain, arg.tensorsig,
                  arg.dtype, "d$(coord.name)", arg, coord, input_basis, output_basis,
                  axis, axis, [true], [false],
                  nothing, nothing, false, 1)
end

function check_conditions(op::Differentiate)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.last_axis]
end

function enforce_conditions(op::Differentiate)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.last_axis)
    end
end

function operate(op::Differentiate, out)::Nothing
    arg = op.args[1]
    layout = arg.layout
    preset_layout!(out, layout)
    if length(arg.data) > 0 && length(out.data) > 0
        data_axis = op.last_axis + length(arg.tensorsig)
        apply_matrix(subspace_matrix(op, layout), arg.data, data_axis; out=out.data)
    else
        out.data .= 0
    end
    return nothing
end

function new_operand(op::Differentiate, operand; kw...)
    return Differentiate(operand, op.coord; kw...)
end

function Base.show(io::IO, op::Differentiate)
    print(io, "d", op.coord.name, "(", string(operator_operand(op)), ")")
end

# ============================================================================
# Interpolate
# ============================================================================

"""
    Interpolate

Interpolation along one coordinate at a specified position.
"""
mutable struct Interpolate <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coord::Any
    position::Any
    input_basis::Any
    output_basis::Any
    first_axis::Int
    last_axis::Int
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Interpolate(arg, coord, position; out=nothing)
    if isa(arg, Number)
        return arg
    end
    dist = arg.dist
    input_basis = get_basis(arg.domain, coord)
    output_basis = if input_basis !== nothing && hasmethod(interpolate_basis, Tuple{typeof(input_basis), typeof(position)})
        interpolate_basis(input_basis, position)
    else
        nothing
    end
    first_axis = get_axis(dist, coord)
    last_axis = first_axis
    new_domain = substitute_basis(arg.domain, input_basis, output_basis)
    Interpolate(Any[arg], (arg,), out, dist, new_domain, arg.tensorsig,
                arg.dtype, "interp", arg, coord, position,
                input_basis, output_basis, first_axis, last_axis,
                [true], [true],
                nothing, nothing, false, 1)
end

function check_conditions(op::Interpolate)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.last_axis]
end

function enforce_conditions(op::Interpolate)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.last_axis)
    end
end

function operate(op::Interpolate, out)::Nothing
    arg = op.args[1]
    layout = arg.layout
    preset_layout!(out, layout)
    if length(arg.data) > 0 && length(out.data) > 0
        data_axis = op.last_axis + length(arg.tensorsig)
        apply_matrix(subspace_matrix(op, layout), arg.data, data_axis; out=out.data)
    else
        out.data .= 0
    end
    return nothing
end

function new_operand(op::Interpolate, operand; kw...)
    return Interpolate(operand, op.coord, op.position; kw...)
end

"""
    interpolate(arg; positions...)

Apply Interpolate iteratively for each coord=position pair.
"""
function interpolate(arg; positions...)
    for (coord, position) in positions
        arg = Interpolate(arg, coord, position)
    end
    return arg
end

function interpolate(arg, coord, position)
    return Interpolate(arg, coord, position)
end

# ============================================================================
# Integrate
# ============================================================================

"""
    Integrate

Integration over one or more coordinates.
"""
mutable struct Integrate <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coord::Any
    input_basis::Any
    output_basis::Any
    first_axis::Int
    last_axis::Int
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Integrate(arg, coord; out=nothing)
    if arg == 0
        return 0
    end
    dist = arg.dist
    input_basis = get_basis(arg.domain, coord)
    output_basis = nothing  # Integration removes the basis
    first_axis = get_axis(dist, coord)
    last_axis = first_axis
    new_domain = substitute_basis(arg.domain, input_basis, output_basis)
    Integrate(Any[arg], (arg,), out, dist, new_domain, arg.tensorsig,
              arg.dtype, "Integrate", arg, coord, input_basis, output_basis,
              first_axis, last_axis, [true], [false],
              nothing, nothing, false, 1)
end

function check_conditions(op::Integrate)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.last_axis]
end

function enforce_conditions(op::Integrate)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.last_axis)
    end
end

function operate(op::Integrate, out)::Nothing
    arg = op.args[1]
    layout = arg.layout
    preset_layout!(out, layout)
    if length(arg.data) > 0 && length(out.data) > 0
        data_axis = op.last_axis + length(arg.tensorsig)
        apply_matrix(subspace_matrix(op, layout), arg.data, data_axis; out=out.data)
    else
        out.data .= 0
    end
    return nothing
end

function new_operand(op::Integrate, operand; kw...)
    return Integrate(operand, op.coord; kw...)
end

function integrate(arg, coord)
    return Integrate(arg, coord)
end

function integrate(arg)
    # Integrate over all bases
    for basis in arg.domain.bases
        if basis !== nothing
            arg = Integrate(arg, basis.coordsys)
        end
    end
    return arg
end

register_operator_alias!("integ", Integrate)

# ============================================================================
# Average
# ============================================================================

"""
    Average

Average over one or more coordinates.
"""
mutable struct Average <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coord::Any
    input_basis::Any
    output_basis::Any
    first_axis::Int
    last_axis::Int
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Average(arg, coord; out=nothing)
    if isa(arg, Number)
        return arg
    end
    dist = arg.dist
    input_basis = get_basis(arg.domain, coord)
    output_basis = nothing  # Averaging removes the basis
    first_axis = get_axis(dist, coord)
    last_axis = first_axis
    new_domain = substitute_basis(arg.domain, input_basis, output_basis)
    Average(Any[arg], (arg,), out, dist, new_domain, arg.tensorsig,
            arg.dtype, "Average", arg, coord, input_basis, output_basis,
            first_axis, last_axis, [true], [false],
            nothing, nothing, false, 1)
end

function check_conditions(op::Average)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.last_axis]
end

function enforce_conditions(op::Average)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.last_axis)
    end
end

function operate(op::Average, out)::Nothing
    arg = op.args[1]
    layout = arg.layout
    preset_layout!(out, layout)
    if length(arg.data) > 0 && length(out.data) > 0
        data_axis = op.last_axis + length(arg.tensorsig)
        apply_matrix(subspace_matrix(op, layout), arg.data, data_axis; out=out.data)
    else
        out.data .= 0
    end
    return nothing
end

function new_operand(op::Average, operand; kw...)
    return Average(operand, op.coord; kw...)
end

function average(arg, coord)
    return Average(arg, coord)
end

function average(arg)
    for basis in arg.domain.bases
        if basis !== nothing
            arg = Average(arg, basis.coordsys)
        end
    end
    return arg
end

register_operator_alias!("ave", Average)

# ============================================================================
# Lift
# ============================================================================

"""
    Lift

Lift (tau) operator. Adds boundary condition enforcement by placing data
into specific polynomial modes of the output basis.
"""
mutable struct Lift <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    output_basis::Any
    n::Int
    input_basis::Any
    first_axis::Int
    last_axis::Int
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Lift(arg, output_basis, n; out=nothing)
    if arg == 0
        return 0
    end
    if n >= 0
        throw(ArgumentError("Only negative mode specifiers allowed for Lift"))
    end
    dist = arg.dist
    input_basis = get_basis(arg.domain, basis_coordsys(output_basis))
    first_axis = get_basis_axis(dist, output_basis)
    last_axis = first_axis + get_dim(output_basis) - 1
    new_domain = substitute_basis(arg.domain, input_basis, output_basis)
    ndim = get_dim(output_basis)
    Lift(Any[arg], (arg,), out, dist, new_domain, arg.tensorsig, arg.dtype,
         "Lift", arg, output_basis, n, input_basis, first_axis, last_axis,
         fill(true, ndim), fill(true, ndim),
         nothing, nothing, false, 1)
end

function check_conditions(op::Lift)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.last_axis]
end

function enforce_conditions(op::Lift)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.last_axis)
    end
end

function operate(op::Lift, out)::Nothing
    arg = op.args[1]
    layout = arg.layout
    preset_layout!(out, layout)
    if length(arg.data) > 0 && length(out.data) > 0
        data_axis = op.last_axis + length(arg.tensorsig)
        apply_matrix(subspace_matrix(op, layout), arg.data, data_axis; out=out.data)
    else
        out.data .= 0
    end
    return nothing
end

function new_operand(op::Lift, operand; kw...)
    return Lift(operand, op.output_basis, op.n; kw...)
end

function lift(arg, basis, n)
    return Lift(arg, basis, n)
end

# ============================================================================
# Cartesian vector calculus operators
# ============================================================================

# --------------------------------------------------------------------------
# CartesianGradient
# --------------------------------------------------------------------------

"""
    CartesianGradient

Gradient in Cartesian coordinates. Assembles partial derivatives along each
coordinate into a vector field. Output tensorsig is (coordsys,) + input tensorsig.
"""
mutable struct CartesianGradient <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianGradient(operand, coordsys; out=nothing)
    if isa(operand, Number)
        return 0
    end
    # Build partial derivatives along each coordinate
    diff_args = Any[Differentiate(operand, coord) for coord in coordsys.coords]
    # Output bases: union of all differentiate bases
    new_tensorsig = (coordsys, operand.tensorsig...)
    # Build domain from the sum of derivative domains
    all_bases = []
    for darg in diff_args
        if isa(darg, AbstractOperand)
            for b in darg.domain.bases
                if b !== nothing
                    push!(all_bases, b)
                end
            end
        end
    end
    new_domain = isempty(all_bases) ? operand.domain : Domain(operand.dist, Tuple(unique(all_bases)))
    CartesianGradient(diff_args, Tuple(diff_args), out, operand.dist, new_domain,
                      new_tensorsig, operand.dtype, "Grad", operand, coordsys,
                      nothing, nothing, false, 1)
end

function check_conditions(op::CartesianGradient)
    # All component derivatives must be in the same layout
    layouts = Set()
    for arg in op.args
        if isa(arg, AbstractCurrent)
            push!(layouts, arg.layout)
        end
    end
    return length(layouts) <= 1
end

function enforce_conditions(op::CartesianGradient)
    # Require all in coefficient layout
    layout = op.dist.coeff_layout
    for arg in op.args
        if isa(arg, AbstractOperand)
            change_layout!(arg, layout)
        end
    end
end

function operate(op::CartesianGradient, out)::Nothing
    operands = op.args
    # Find a layout from the args
    layout = nothing
    for arg in operands
        if isa(arg, AbstractCurrent)
            layout = arg.layout
            break
        end
    end
    if layout === nothing
        layout = op.dist.coeff_layout
    end
    preset_layout!(out, layout)
    # Copy each component's data into the output vector slots
    for (i, comp) in enumerate(operands)
        if isa(comp, AbstractCurrent) && length(comp.data) > 0
            @inbounds out.data[i, ntuple(_ -> Colon(), ndims(out.data) - 1)...] .= comp.data
        else
            @inbounds out.data[i, ntuple(_ -> Colon(), ndims(out.data) - 1)...] .= 0
        end
    end
    return nothing
end

function new_operand(op::CartesianGradient, operand; kw...)
    return CartesianGradient(operand, op.coordsys; kw...)
end

function subproblem_matrix(op::CartesianGradient, subproblem)
    # Vertical stack of component expression matrices
    mats = [expression_matrices(arg, subproblem, [op.operand])[op.operand] for arg in op.args]
    return vcat(mats...)
end

# --------------------------------------------------------------------------
# CartesianDivergence
# --------------------------------------------------------------------------

"""
    CartesianDivergence

Divergence in Cartesian coordinates. Sums d(u_i)/dx_i over all coordinates.
Input must be a vector field (first tensorsig index is the coordsys).
"""
mutable struct CartesianDivergence <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    index::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianDivergence(operand; index=1, out=nothing)
    if isa(operand, Number)
        return 0
    end
    coordsys = operand.tensorsig[index]
    # Build sum of d(component_i)/dx_i
    comps = [CartesianComponent(operand, index, coord) for coord in coordsys.coords]
    diffs = [Differentiate(comp, coord) for (comp, coord) in zip(comps, coordsys.coords)]
    inner_arg = sum(diffs)
    new_tensorsig = (operand.tensorsig[1:index-1]..., operand.tensorsig[index+1:end]...)
    new_domain = isa(inner_arg, AbstractOperand) ? inner_arg.domain : operand.domain
    CartesianDivergence(Any[inner_arg], (inner_arg,), out, operand.dist, new_domain,
                        new_tensorsig, operand.dtype, "Div", operand, coordsys, index,
                        nothing, nothing, false, 1)
end

check_conditions(::CartesianDivergence) = true
enforce_conditions(::CartesianDivergence) = nothing

function operate(op::CartesianDivergence, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
    return nothing
end

function new_operand(op::CartesianDivergence, operand; kw...)
    return CartesianDivergence(operand; index=op.index, kw...)
end

function subproblem_matrix(op::CartesianDivergence, subproblem)
    return expression_matrices(op.args[1], subproblem, [op.operand])[op.operand]
end

# --------------------------------------------------------------------------
# CartesianCurl
# --------------------------------------------------------------------------

"""
    CartesianCurl

Curl in Cartesian coordinates. Only implemented for 3D vector fields.
Computes curl as the cross product of nabla and the vector field.
"""
mutable struct CartesianCurl <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    index::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianCurl(operand; index=1, out=nothing)
    if isa(operand, Number)
        return 0
    end
    coordsys = operand.tensorsig[index]
    if get_dim(coordsys) != 3
        throw(ArgumentError("CartesianCurl only implemented for 3D vector fields"))
    end
    coords = coordsys.coords
    # Get components
    comps = [CartesianComponent(operand, index, c) for c in coords]
    # curl = (dy(uz)-dz(uy), dz(ux)-dx(uz), dx(uy)-dy(ux))
    x_comp = Differentiate(comps[3], coords[2]) - Differentiate(comps[2], coords[3])
    y_comp = Differentiate(comps[1], coords[3]) - Differentiate(comps[3], coords[1])
    z_comp = Differentiate(comps[2], coords[1]) - Differentiate(comps[1], coords[2])
    # Assemble result via unit vectors
    inner_arg = x_comp + y_comp + z_comp  # Simplified; full assembly needs vector fields
    if !coordsys.right_handed
        inner_arg = -inner_arg
    end
    new_domain = isa(inner_arg, AbstractOperand) ? inner_arg.domain : operand.domain
    CartesianCurl(Any[inner_arg], (inner_arg,), out, operand.dist, new_domain,
                  operand.tensorsig, operand.dtype, "Curl", operand, coordsys, index,
                  nothing, nothing, false, 1)
end

check_conditions(::CartesianCurl) = true
enforce_conditions(::CartesianCurl) = nothing

function operate(op::CartesianCurl, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
    return nothing
end

function new_operand(op::CartesianCurl, operand; kw...)
    return CartesianCurl(operand; index=op.index, kw...)
end

# --------------------------------------------------------------------------
# CartesianLaplacian
# --------------------------------------------------------------------------

"""
    CartesianLaplacian

Laplacian in Cartesian coordinates. Sums d^2/dx_i^2 over all coordinates.
"""
mutable struct CartesianLaplacian <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianLaplacian(operand, coordsys; out=nothing)
    if isa(operand, Number)
        return 0
    end
    # Build sum of second derivatives
    parts = [Differentiate(Differentiate(operand, c), c) for c in coordsys.coords]
    inner_arg = sum(parts)
    new_domain = isa(inner_arg, AbstractOperand) ? inner_arg.domain : operand.domain
    CartesianLaplacian(Any[inner_arg], (inner_arg,), out, operand.dist, new_domain,
                       operand.tensorsig, operand.dtype, "Lap", operand, coordsys,
                       nothing, nothing, false, 1)
end

check_conditions(::CartesianLaplacian) = true
enforce_conditions(::CartesianLaplacian) = nothing

function operate(op::CartesianLaplacian, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
    return nothing
end

function new_operand(op::CartesianLaplacian, operand; kw...)
    return CartesianLaplacian(operand, op.coordsys; kw...)
end

function subproblem_matrix(op::CartesianLaplacian, subproblem)
    return expression_matrices(op.args[1], subproblem, [op.operand])[op.operand]
end

function matrix_dependence(op::CartesianLaplacian, vars...)
    return matrix_dependence(op.args[1], vars...)
end

function matrix_coupling(op::CartesianLaplacian, vars...)
    return matrix_coupling(op.args[1], vars...)
end

# --------------------------------------------------------------------------
# CartesianTrace
# --------------------------------------------------------------------------

"""
    CartesianTrace

Trace of a rank-2+ tensor in Cartesian coordinates. Contracts the first
two tensor indices. Output tensorsig = input tensorsig[3:end].
"""
mutable struct CartesianTrace <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianTrace(operand; out=nothing)
    if isa(operand, Number)
        return 0
    end
    coordsys = operand.tensorsig[1]
    new_tensorsig = operand.tensorsig[3:end]
    CartesianTrace(Any[operand], (operand,), out, operand.dist, operand.domain,
                   new_tensorsig, operand.dtype, "Trace", operand, coordsys,
                   nothing, nothing, false, 1)
end

function check_conditions(op::CartesianTrace)
    arg = op.args[1]
    return isa(arg, AbstractCurrent) && all(arg.layout.grid_space)
end

function enforce_conditions(op::CartesianTrace)
    arg = op.args[1]
    if isa(arg, Field)
        require_grid_space!(arg)
    end
end

function operate(op::CartesianTrace, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    # Perform trace: sum over diagonal of first two tensor indices
    cs = op.coordsys
    d = cs_dim(cs)
    out.data .= 0
    for i in 1:d
        spatial_idx = ntuple(_ -> Colon(), ndims(arg.data) - 2)
        @inbounds out.data .+= arg.data[i, i, spatial_idx...]
    end
    return nothing
end

function new_operand(op::CartesianTrace, operand; kw...)
    return CartesianTrace(operand; kw...)
end

function subproblem_matrix(op::CartesianTrace, subproblem)
    d = cs_dim(op.coordsys)
    trace_vec = vec(Matrix{Float64}(I, d, d))
    # Kronecker with remaining tensor components and coefficient size
    n_eye = prod(cs_dim(cs) for cs in op.tensorsig; init=1)
    n_eye *= coeff_size(subproblem, op.domain)
    eye_mat = sparse(1.0I, n_eye, n_eye)
    return kron(sparse(trace_vec'), eye_mat)
end

register_operator_alias!("trace", CartesianTrace)

# --------------------------------------------------------------------------
# CartesianTransposeComponents
# --------------------------------------------------------------------------

"""
    CartesianTransposeComponents

Transpose the first two tensor indices of a tensor field.
"""
mutable struct CartesianTransposeComponents <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    indices::Tuple{Int,Int}
    new_axis_order::Tuple
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianTransposeComponents(operand; indices=(1,2), out=nothing)
    if isa(operand, Number)
        return 0
    end
    i0, i1 = indices
    ts = operand.tensorsig
    # Build swapped tensorsig
    new_ts = collect(ts)
    new_ts[i0], new_ts[i1] = new_ts[i1], new_ts[i0]
    # Build new axis permutation (tensor dims + spatial dims)
    total_dims = length(ts) + get_dim(operand.dist)
    new_order = collect(1:total_dims)
    new_order[i0], new_order[i1] = new_order[i1], new_order[i0]
    CartesianTransposeComponents(Any[operand], (operand,), out, operand.dist,
                                  operand.domain, Tuple(new_ts), operand.dtype,
                                  "Trans", operand, indices, Tuple(new_order),
                                  nothing, nothing, false, 1)
end

check_conditions(::CartesianTransposeComponents) = true
enforce_conditions(::CartesianTransposeComponents) = nothing

function operate(op::CartesianTransposeComponents, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= permutedims(arg.data, collect(op.new_axis_order))
    end
    return nothing
end

function new_operand(op::CartesianTransposeComponents, operand; kw...)
    return CartesianTransposeComponents(operand; indices=op.indices, kw...)
end

register_operator_alias!("transpose", CartesianTransposeComponents)
register_operator_alias!("trans", CartesianTransposeComponents)

# --------------------------------------------------------------------------
# CartesianComponent
# --------------------------------------------------------------------------

"""
    CartesianComponent

Extract a single component from a vector/tensor field along a specified
tensor index and coordinate.
"""
mutable struct CartesianComponent <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    index::Int
    comp::Any
    coord_subaxis::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianComponent(operand, index, comp; out=nothing)
    coordsys = operand.tensorsig[index]
    coord_subaxis = get_axis(operand.dist, comp) - get_axis(operand.dist, coordsys) + 1
    new_tensorsig = (operand.tensorsig[1:index-1]..., operand.tensorsig[index+1:end]...)
    CartesianComponent(Any[operand], (operand,), out, operand.dist, operand.domain,
                       new_tensorsig, operand.dtype, "Comp", operand, index, comp,
                       coord_subaxis, nothing, nothing, false, 1)
end

check_conditions(::CartesianComponent) = true
enforce_conditions(::CartesianComponent) = nothing

function operate(op::CartesianComponent, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        # Select the component along the tensor index
        idx = ntuple(i -> i == op.index ? op.coord_subaxis : Colon(), ndims(arg.data))
        out.data .= arg.data[idx...]
    end
    return nothing
end

function new_operand(op::CartesianComponent, operand; kw...)
    return CartesianComponent(operand, op.index, op.comp; kw...)
end

function subproblem_matrix(op::CartesianComponent, subproblem)
    # Build selection matrix via Kronecker product
    factors = [sparse(1.0I, cs_dim(cs), cs_dim(cs)) for cs in op.operand.tensorsig]
    push!(factors, sparse(1.0I, coeff_size(subproblem, op.domain), coeff_size(subproblem, op.domain)))
    # Build selection row for the indexed coordinate
    sel = zeros(1, get_dim(op.operand.tensorsig[op.index]))
    sel[1, op.coord_subaxis] = 1.0
    factors[op.index] = sparse(sel)
    return reduce(kron, factors)
end

# ============================================================================
# DirectProduct operators (delegate to per-sub-coordsys operations)
# ============================================================================

"""
    DirectProductGradient

Gradient for DirectProduct coordinate systems. Assembles sub-gradients
for each constituent coordinate system.
"""
mutable struct DirectProductGradient <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function DirectProductGradient(operand, coordsys; out=nothing)
    if isa(operand, Number)
        return 0
    end
    sub_grads = Any[gradient(operand, cs) for cs in coordsys.coordsystems]
    new_tensorsig = (coordsys, operand.tensorsig...)
    new_domain = operand.domain
    DirectProductGradient(sub_grads, Tuple(sub_grads), out, operand.dist, new_domain,
                          new_tensorsig, operand.dtype, "Grad", operand, coordsys,
                          nothing, nothing, false, 1)
end

function check_conditions(op::DirectProductGradient)
    layouts = Set()
    for arg in op.args
        if isa(arg, AbstractCurrent)
            push!(layouts, arg.layout)
        end
    end
    return length(layouts) <= 1
end

function enforce_conditions(op::DirectProductGradient)
    layout = op.dist.coeff_layout
    for arg in op.args
        if isa(arg, AbstractOperand)
            change_layout!(arg, layout)
        end
    end
end

function operate(op::DirectProductGradient, out)::Nothing
    layout = nothing
    for arg in op.args
        if isa(arg, AbstractCurrent)
            layout = arg.layout
            break
        end
    end
    if layout === nothing
        layout = op.dist.coeff_layout
    end
    preset_layout!(out, layout)
    i0 = 1
    for (cs_grad, cs) in zip(op.args, op.coordsys.coordsystems)
        dim = get_dim(cs)
        if isa(cs_grad, AbstractCurrent) && length(cs_grad.data) > 0
            @inbounds out.data[i0:i0+dim-1, ntuple(_ -> Colon(), ndims(out.data) - 1)...] .= cs_grad.data
        else
            @inbounds out.data[i0:i0+dim-1, ntuple(_ -> Colon(), ndims(out.data) - 1)...] .= 0
        end
        i0 += dim
    end
    return nothing
end

function new_operand(op::DirectProductGradient, operand; kw...)
    return DirectProductGradient(operand, op.coordsys; kw...)
end

function subproblem_matrix(op::DirectProductGradient, subproblem)
    mats = [expression_matrices(arg, subproblem, [op.operand])[op.operand] for arg in op.args]
    return vcat(mats...)
end

"""
    DirectProductDivergence

Divergence for DirectProduct coordinate systems.
"""
mutable struct DirectProductDivergence <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    index::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function DirectProductDivergence(operand; index=1, out=nothing)
    if isa(operand, Number)
        return 0
    end
    coordsys = operand.tensorsig[index]
    # Get sub-components and diverge each
    comps = [DirectProductComponent(operand, index, cs) for cs in coordsys.coordsystems]
    divs = [divergence(comp) for comp in comps]
    inner_arg = sum(divs)
    new_tensorsig = (operand.tensorsig[1:index-1]..., operand.tensorsig[index+1:end]...)
    new_domain = isa(inner_arg, AbstractOperand) ? inner_arg.domain : operand.domain
    DirectProductDivergence(Any[inner_arg], (inner_arg,), out, operand.dist, new_domain,
                            new_tensorsig, operand.dtype, "Div", operand, coordsys, index,
                            nothing, nothing, false, 1)
end

check_conditions(::DirectProductDivergence) = true
enforce_conditions(::DirectProductDivergence) = nothing

function operate(op::DirectProductDivergence, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
    return nothing
end

function new_operand(op::DirectProductDivergence, operand; kw...)
    return DirectProductDivergence(operand; index=op.index, kw...)
end

function subproblem_matrix(op::DirectProductDivergence, subproblem)
    return expression_matrices(op.args[1], subproblem, [op.operand])[op.operand]
end

"""
    DirectProductCurl

Curl for DirectProduct coordinate systems. Only for 3D total dimension.
"""
mutable struct DirectProductCurl <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    index::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function DirectProductCurl(operand; index=1, out=nothing)
    if isa(operand, Number)
        return 0
    end
    coordsys = operand.tensorsig[index]
    if get_dim(coordsys) != 3
        throw(ArgumentError("DirectProductCurl only implemented for 3D vector fields"))
    end
    throw(ArgumentError("DirectProductCurl requires Skew and IdentityTensor operators " *
                         "from Milestone 2 (Polar/Disk geometry). Use CartesianCurl for " *
                         "Cartesian coordinate systems."))
end

check_conditions(::DirectProductCurl) = true
enforce_conditions(::DirectProductCurl) = nothing

function operate(op::DirectProductCurl, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
    return nothing
end

function new_operand(op::DirectProductCurl, operand; kw...)
    return DirectProductCurl(operand; index=op.index, kw...)
end

"""
    DirectProductLaplacian

Laplacian for DirectProduct coordinate systems. Sums sub-Laplacians.
"""
mutable struct DirectProductLaplacian <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function DirectProductLaplacian(operand, coordsys; out=nothing)
    if isa(operand, Number)
        return 0
    end
    parts = [laplacian(operand, cs) for cs in coordsys.coordsystems]
    inner_arg = sum(parts)
    new_domain = isa(inner_arg, AbstractOperand) ? inner_arg.domain : operand.domain
    DirectProductLaplacian(Any[inner_arg], (inner_arg,), out, operand.dist, new_domain,
                           operand.tensorsig, operand.dtype, "Lap", operand, coordsys,
                           nothing, nothing, false, 1)
end

check_conditions(::DirectProductLaplacian) = true
enforce_conditions(::DirectProductLaplacian) = nothing

function operate(op::DirectProductLaplacian, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
    return nothing
end

function new_operand(op::DirectProductLaplacian, operand; kw...)
    return DirectProductLaplacian(operand, op.coordsys; kw...)
end

function subproblem_matrix(op::DirectProductLaplacian, subproblem)
    return expression_matrices(op.args[1], subproblem, [op.operand])[op.operand]
end

"""
    DirectProductComponent

Extract a sub-coordsystem slice from a DirectProduct vector field.
"""
mutable struct DirectProductComponent <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    index::Int
    comp::Any
    comp_subaxis::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function DirectProductComponent(operand, index, comp; out=nothing)
    coordsys = operand.tensorsig[index]
    comp_subaxis = get_axis(operand.dist, comp) - get_axis(operand.dist, coordsys) + 1
    new_tensorsig = collect(operand.tensorsig)
    new_tensorsig[index] = comp
    DirectProductComponent(Any[operand], (operand,), out, operand.dist, operand.domain,
                           Tuple(new_tensorsig), operand.dtype, "Comp", operand, index,
                           comp, comp_subaxis, nothing, nothing, false, 1)
end

check_conditions(::DirectProductComponent) = true
enforce_conditions(::DirectProductComponent) = nothing

function operate(op::DirectProductComponent, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        comp_dim = get_dim(op.comp)
        comp_slice = op.comp_subaxis:(op.comp_subaxis + comp_dim - 1)
        idx = ntuple(i -> i == op.index ? comp_slice : Colon(), ndims(arg.data))
        out.data .= arg.data[idx...]
    end
    return nothing
end

function new_operand(op::DirectProductComponent, operand; kw...)
    return DirectProductComponent(operand, op.index, op.comp; kw...)
end

# ============================================================================
# AdvectiveCFL (needed by extras/flow_tools.jl)
# ============================================================================

"""
    AdvectiveCFL

Computes the scalar advective grid-crossing frequency for CFL condition.
"""
mutable struct AdvectiveCFL <: NonlinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    velocity::Any
    coordsys::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
    _grid_layout::Any
    _coeff_layout::Any
end

function AdvectiveCFL(velocity, coordsys; out=nothing)
    dist = velocity.dist
    AdvectiveCFL(Any[velocity], (velocity,), out, dist, velocity.domain,
                 (), velocity.dtype, "AdvCFL", velocity, coordsys,
                 nothing, nothing, false, 1,
                 dist.grid_layout, dist.coeff_layout)
end

function check_conditions(op::AdvectiveCFL)
    arg = op.args[1]
    return isa(arg, AbstractCurrent) && all(arg.layout.grid_space)
end

function enforce_conditions(op::AdvectiveCFL)
    arg = op.args[1]
    if isa(arg, Field)
        require_grid_space!(arg)
    end
end

# ============================================================================
# Constructor dispatch functions (PUBLIC API)
# ============================================================================

"""Gradient dispatch: selects CartesianGradient or DirectProductGradient."""
gradient(field, cs::CartesianCoordinates) = CartesianGradient(field, cs)
gradient(field, cs::DirectProduct) = DirectProductGradient(field, cs)
gradient(field, cs) = CartesianGradient(field, cs)  # fallback

# Divergence dispatch moved to end of file (after all operator types defined)

"""Curl dispatch: selects CartesianCurl or DirectProductCurl."""
function curl(field; index=1)
    if isa(field, Number) || field == 0
        return 0
    end
    cs = field.tensorsig[index]
    if isa(cs, SphericalCoordinates)
        return SphericalCurl(field; index=index)
    elseif isa(cs, CartesianCoordinates)
        return CartesianCurl(field; index=index)
    elseif isa(cs, DirectProduct)
        return DirectProductCurl(field; index=index)
    else
        return CartesianCurl(field; index=index)  # fallback
    end
end

# Laplacian dispatch moved to end of file (after all operator types defined)

"""Trace of tensor field (contracts first two indices)."""
trace_op(field) = CartesianTrace(field)

"""Transpose first two tensor components."""
transpose_components(field) = CartesianTransposeComponents(field)

"""Lock field to grid space."""
grid_op(field) = GridOperator(field)

"""Lock field to coefficient space."""
coeff_op(field) = CoeffOperator(field)

"""Symbolic time derivative."""
time_derivative(field) = TimeDerivative(field)

"""Extract component from vector/tensor field."""
function component(field, index, comp)
    CartesianComponent(field, index, comp)
end

# Aliases for equation namespace
const dt = time_derivative

# ============================================================================
# Display methods for all operator types
# ============================================================================

for T in [Power, FieldCopy, UnaryGridFunction, TimeDerivative,
          GridOperator, CoeffOperator, Convert,
          Interpolate, Integrate, Average, Lift,
          CartesianGradient, CartesianDivergence, CartesianCurl,
          CartesianLaplacian, CartesianTrace, CartesianTransposeComponents,
          CartesianComponent,
          DirectProductGradient, DirectProductDivergence, DirectProductCurl,
          DirectProductLaplacian, DirectProductComponent,
          AdvectiveCFL]
    @eval begin
        function Base.show(io::IO, op::$T)
            print(io, op.name, "(", join(map(string, op.args), ", "), ")")
        end
    end
end

# Differentiate has its own show method defined above

# ============================================================================
# subspace_matrix placeholder
# ============================================================================

"""
    subspace_matrix(op, layout)

Return the subspace (basis conversion/differentiation/interpolation) matrix
for operator `op` at the given layout.  Generic fallback returns an identity
matrix.  Concrete dispatch methods for each operator+basis combination are
defined in transforms.jl (loaded after both operators.jl and basis.jl).
"""
function subspace_matrix(op, layout)
    error("subspace_matrix not implemented for operator type $(typeof(op)). " *
          "Add a dispatch method in transforms.jl for this operator+basis combination.")
end

# ============================================================================
# AdvectiveCFL operate method
# ============================================================================

function operate(op::AdvectiveCFL, out)::Nothing
    vel = op.args[1]
    preset_layout!(out, vel.layout)
    # Compute |velocity| / grid_spacing for each component
    # Simplified: just use absolute value of velocity components
    out.data .= 0
    for i in 1:length(vel.tensorsig)
        @inbounds out.data .+= abs.(vel.data[i, fill(:, ndims(vel.data)-1)...])
    end
    return nothing
end

# ============================================================================
# Exports
# ============================================================================

export AbstractOperator, AbstractLinearOperator, SpectralOperator, SpectralOperator1D,
       NonlinearOperator,
       AbstractField,
       Power, dedalus_power,
       FieldCopy,
       UnaryGridFunction, UFUNC_DERIVATIVES,
       GridOperator, CoeffOperator,
       TimeDerivative,
       Convert, convert_operand,
       Differentiate, differentiate,
       Interpolate, interpolate,
       Integrate, integrate,
       Average, average,
       Lift, lift,
       CartesianGradient, CartesianDivergence, CartesianCurl, CartesianLaplacian,
       CartesianTrace, CartesianTransposeComponents, CartesianComponent,
       DirectProductGradient, DirectProductDivergence, DirectProductCurl,
       DirectProductLaplacian, DirectProductComponent,
       AdvectiveCFL,
       SpectralOperatorS2, SeparableSphereOperator, PolarMOperator,
       spinindex_out, l_matrix, m_matrix,
       symbol, local_symbols,
       radial_matrix, _output_basis,
       polar_m_operator_subaxis_dependence, polar_m_operator_subaxis_coupling,
       init_polar_m_operator!,
       # Curvilinear geometry operators (Group 1)
       SphericalTrace, PolarTrace, DirectProductTrace,
       StandardTransposeComponents, SphericalTransposeComponents,
       CartesianSkew, SpinSkew, skew,
       # Component operators (Group 2)
       RadialComponent, AngularComponent, AzimuthalComponent,
       radial_component, angular_component, azimuthal_component,
       # Polar differential operators (Group 3)
       MulCosine, PolarGradient, PolarDivergence, PolarLaplacian,
       # S2 operators (Group 4)
       SphereEllProduct,
       # Sphere differential operators (Group 5)
       SphereGradient, SphereDivergence, SphereLaplacian,
       sphere_basis_k,
       # 3D Spherical operators (Group 6)
       SphericalEllOperator,
       SphericalGradient, SphericalDivergence, SphericalCurl,
       SphericalLaplacian, SphericalEllProduct,
       regindex_out,
       gradient, divergence, curl, laplacian,
       trace_op, transpose_components, grid_op, coeff_op,
       time_derivative, component,
       OPERATOR_ALIASES, register_operator_alias!,
       operator_operand, new_operand,
       subproblem_matrix, expression_matrices,
       subspace_matrix,
       dt, lap

# ============================================================================
# Convenience functions (re-added after duplicate removal)
# ============================================================================

differentiate(arg, coord) = Differentiate(arg, coord)
interpolate(arg, coord, position) = Interpolate(arg, coord, position)
integrate(arg, coord) = Integrate(arg, coord)
average(arg, coord) = Average(arg, coord)
lift(arg, basis, n) = Lift(arg, basis, n)

# ============================================================================
# Stubs for subsystem/solver wiring
# ============================================================================

function local_groupset_slices(layout, group, domain; scales=1)
    return (Colon(),)
end

function basis_matrix_dependence(basis, axis)
    return false
end

# ============================================================================
# SpectralOperatorS2 — abstract base for linear operators on the 2-sphere
# ============================================================================

"""
    SpectralOperatorS2 <: SpectralOperator

Abstract base type for linear operators acting on the 2-sphere (S2).
Subtypes operate on SpinBasis bases and build their matrix representation
from per-ell matrix blocks (`l_matrix`) or per-m matrix blocks (`m_matrix`).

## Subtypes must implement
- `spinindex_out(op, spinindex_in)` — return tuple of valid output spin indices
- `l_matrix(op, input_basis, output_basis, spinindex_in, spinindex_out, ell)` — per-ell block
- `m_matrix(op, spinindex_in, spinindex_out, m)` — per-m block (for ell-coupled operators)

## Subtypes must have fields
- `operand`, `input_basis`, `output_basis`, `first_axis`, `last_axis`
- `tensorsig`, `dtype`, `domain`, `dist`
- `subaxis_dependence::Vector{Bool}`, `subaxis_coupling::Vector{Bool}`
- `args::Vector{Any}`
"""
abstract type SpectralOperatorS2 <: SpectralOperator end

"""
    spinindex_out(op, spinindex_in)

Return a tuple of valid output spin indices for the given input spin index.
Must be implemented by concrete SpectralOperatorS2 subtypes.
"""
function spinindex_out(op::SpectralOperatorS2, spinindex_in)
    error("spinindex_out not implemented for type $(typeof(op))")
end

"""
    l_matrix(op, input_basis, output_basis, spinindex_in, spinindex_out, ell)

Return the per-ell matrix block for a SpectralOperatorS2.
Must be implemented by concrete subtypes.
"""
function l_matrix(op::SpectralOperatorS2, input_basis, output_basis, spinindex_in, spinindex_out, ell)
    error("l_matrix not implemented for type $(typeof(op))")
end

"""
    m_matrix(op, spinindex_in, spinindex_out, m)

Return the per-m matrix block for ell-coupled SpectralOperatorS2 operators.
Must be implemented by concrete subtypes that couple in ell.
"""
function m_matrix(op::SpectralOperatorS2, spinindex_in, spinindex_out, m)
    error("m_matrix not implemented for type $(typeof(op))")
end

"""
    subproblem_matrix(op::SpectralOperatorS2, subproblem)

Build operator matrix for a specific subproblem.

Loops over spin components, constructing per-ell or per-m blocks and
assembling them into a block-diagonal structure via Kronecker products.
"""
function subproblem_matrix(op::SpectralOperatorS2, subproblem)
    operand = op.args[1]
    if op.input_basis === nothing
        basis = op.output_basis
        domain = op.domain
    else
        basis = op.input_basis
        domain = operand.domain
    end
    S_in = spin_weights(basis, operand.tensorsig)
    S_out = spin_weights(basis, op.tensorsig)
    # subproblem.group is 1-based; last_axis-1 and last_axis give m and ell
    m = subproblem.group[op.last_axis - 1]
    l = subproblem.group[op.last_axis]
    m_coupled = (m === nothing)
    l_coupled = (l === nothing)
    if op.subaxis_coupling[1] && !m_coupled
        error("SpectralOperatorS2: m must be coupled (group[m_axis] === nothing) " *
              "when subaxis_coupling[1] is true.")
    end
    if op.subaxis_coupling[2] && !l_coupled
        error("SpectralOperatorS2: ell must be coupled (group[ell_axis] === nothing) " *
              "when subaxis_coupling[2] is true.")
    end
    m_dep = op.subaxis_dependence[1]
    l_dep = op.subaxis_dependence[2]
    m_axis = first_axis(op.dist, basis)
    # Loop over spin components
    submatrices = []
    for si_out in CartesianIndices(size(S_out))
        spintotal_out = S_out[si_out]
        submatrix_row = []
        for si_in in CartesianIndices(size(S_in))
            spintotal_in = S_in[si_in]
            subshape_in = coeff_shape(subproblem, operand.domain)
            subshape_out = coeff_shape(subproblem, op.domain)
            if Tuple(si_out) in spinindex_out(op, Tuple(si_in))
                # Build identity matrices for each axis
                factors = [sparse(1.0I, subshape_out[i], subshape_in[i]) for i in eachindex(subshape_out)]
                # Substitute factor for the operator's last axis
                if l_coupled && op.subaxis_coupling[2]
                    matrix = m_matrix(op, Tuple(si_in), Tuple(si_out), m)
                elseif l_coupled || (!m_dep)
                    if l_coupled
                        local_groups = local_group_arrays(op.dist.coeff_layout, domain; scales=1)
                        local_m = local_groups[m_axis]
                        local_ell = local_groups[m_axis + 1]
                        ell_list = local_ell[local_m .== m][:]
                    elseif !m_dep
                        ell_list = [l]
                    end
                    blocks = []
                    for ell in ell_list
                        if abs(spintotal_in) <= ell && abs(spintotal_out) <= ell
                            block = l_matrix(op, op.input_basis, op.output_basis,
                                             Tuple(si_in), Tuple(si_out), ell)
                        else
                            block = spzeros(1, 1)  # HACK: placeholder for invalid ell
                        end
                        push!(blocks, block)
                    end
                    matrix = sparse_block_diag(blocks)
                else
                    error("SpectralOperatorS2: unsupported subaxis configuration")
                end
                factors[op.last_axis] = matrix
                comp_matrix = reduce(kron, factors; init=sparse(ones(1, 1)))
            else
                # Build zero matrix
                comp_matrix = spzeros(prod(subshape_out), prod(subshape_in))
            end
            push!(submatrix_row, comp_matrix)
        end
        push!(submatrices, submatrix_row)
    end
    # Assemble block matrix [submatrices[i][j] for i in rows, j in cols]
    nrows = length(submatrices)
    ncols = length(submatrices[1])
    block_mat = hvcat(
        ntuple(_ -> ncols, nrows),
        [submatrices[i][j] for i in 1:nrows for j in 1:ncols]...
    )
    return sparse(block_mat)
end

"""
    operate(op::SpectralOperatorS2, out)

Explicit evaluation of an S2 spectral operator.

For operators without ell-coupling, loops over spin components and applies
the subspace matrix via `apply_matrix`.
"""
function operate(op::SpectralOperatorS2, out)::Nothing
    operand = op.args[1]
    input_basis = op.input_basis
    layout = operand.layout
    axis = op.first_axis
    dim_val = 2  # S2 has 2 sub-axes
    if op.subaxis_coupling[2]
        error("Explicit evaluation not implemented for ell-coupled S2 operators.")
    end
    # Set output layout
    preset_layout!(out, layout)
    out.data .= 0
    # Return for size-zero data
    if length(operand.data) == 0 || length(out.data) == 0
        return
    end
    # Apply operator over spin components
    S_in = spin_weights(input_basis, operand.tensorsig)
    for si_in in CartesianIndices(size(S_in))
        spintotal_in = S_in[si_in]
        @inbounds comp_in = operand.data[Tuple(si_in)...]
        reduced_in = reduced_view_3(comp_in, axis)
        for si_out_tuple in spinindex_out(op, Tuple(si_in))
            @inbounds comp_out = out.data[si_out_tuple...]
            reduced_out = reduced_view_3(comp_out, axis)
            matrix = subspace_matrix(op, layout, Tuple(si_in), si_out_tuple)
            reduced_out .+= apply_matrix(matrix, reduced_in, 1)
        end
    end
    return nothing
end

# ============================================================================
# SeparableSphereOperator — abstract base for separable (diagonal) S2 ops
# ============================================================================

"""
    SeparableSphereOperator <: SpectralOperator

Abstract base type for sphere operators that are separable (diagonal) in
the (m, ell) spectral space. These operators are defined by scalar symbols
that multiply each (m, ell) mode independently.

`subaxis_coupling = [false, false]` — no coupling in either m or ell.

## Subtypes must implement
- `symbol(op, spinindex_in, spinindex_out, spintotal_in, spintotal_out, ...)` — per-mode symbol
- `spinindex_out(op, spinindex_in)` — valid output spin indices

## Subtypes must have fields
- `operand`, `input_basis`, `output_basis`, `first_axis`, `last_axis`
- `tensorsig`, `dtype`, `domain`, `dist`
- `subaxis_dependence::Vector{Bool}`, `subaxis_coupling::Vector{Bool}`
- `complex_operator::Bool`
- `args::Vector{Any}`
"""
abstract type SeparableSphereOperator <: SpectralOperator end

"""
    spinindex_out(op::SeparableSphereOperator, spinindex_in)

Return a tuple of valid output spin indices.
Must be implemented by concrete subtypes.
"""
function spinindex_out(op::SeparableSphereOperator, spinindex_in)
    error("spinindex_out not implemented for type $(typeof(op))")
end

"""
    symbol(op::SeparableSphereOperator, spinindex_in, spinindex_out, spintotal_in, spintotal_out, args...)

Return the scalar symbol for a given spin component at the given mode.
Must be implemented by concrete subtypes.
"""
function symbol(op::SeparableSphereOperator, spinindex_in, spinindex_out,
                spintotal_in, spintotal_out, args...)
    error("symbol not implemented for type $(typeof(op))")
end

"""
    local_symbols(op::SeparableSphereOperator, layout, spinindex_in, spinindex_out,
                  spintotal_in, spintotal_out)

Return the array of symbols for all local (m, ell) groups in the given layout.
Dispatches based on `subaxis_dependence`:
- `[false, true]`: symbols depend on ell only (most common, e.g. SphereEllProduct)
- `[false, false]`: symbols are constant (depend only on spin indices and radius)
- `[true, ...]`: not yet implemented
"""
function local_symbols(op::SeparableSphereOperator, layout, spinindex_in, spinindex_out,
                       spintotal_in, spintotal_out)
    operand = op.args[1]
    if op.input_basis === nothing
        domain = op.domain
        radius = op.output_basis.radius
    else
        domain = operand.domain
        radius = op.input_basis.radius
    end
    if op.subaxis_dependence[1]
        error("local_symbols not implemented for m-dependent SeparableSphereOperator")
    elseif op.subaxis_dependence[2]
        colat_axis = op.first_axis + 1
        local_ell = local_group_arrays(layout, domain; scales=basis_dealias(domain))[colat_axis]
        return symbol(op, spinindex_in, spinindex_out, spintotal_in, spintotal_out,
                      local_ell, radius)
    else
        return symbol(op, spinindex_in, spinindex_out, spintotal_in, spintotal_out, radius)
    end
end

"""
    subproblem_matrix(op::SeparableSphereOperator, subproblem)

Build the operator matrix for a specific subproblem.

Since the operator is separable, the matrix is diagonal with entries given
by the symbol values at each (m, ell) mode.
"""
function subproblem_matrix(op::SeparableSphereOperator, subproblem)
    operand = op.args[1]
    if op.input_basis === nothing
        basis = op.output_basis
        domain = op.domain
    else
        basis = op.input_basis
        domain = operand.domain
    end
    layout = op.dist.coeff_layout
    S_in = spin_weights(basis, operand.tensorsig)
    S_out = spin_weights(basis, op.tensorsig)
    groupset_slices_val = local_groupset_slices(op.dist.coeff_layout, subproblem.group, domain; scales=1)
    # Select overlapping data
    subshape_in = coeff_shape(subproblem, operand.domain)
    subshape_out = coeff_shape(subproblem, op.domain)
    subshape = min.(subshape_in, subshape_out)
    slices = Tuple(1:n for n in subshape)
    size_in = prod(subshape_in)
    size_out = prod(subshape_out)
    # Build block matrix over spin components
    submatrices = []
    for si_out in CartesianIndices(size(S_out))
        spintotal_out = S_out[si_out]
        block_row = []
        for si_in in CartesianIndices(size(S_in))
            spintotal_in = S_in[si_in]
            if prod(subshape) > 0 && (Tuple(si_out) in spinindex_out(op, Tuple(si_in)))
                # Get symbols for overlapping data
                symbols_val = local_symbols(op, layout, Tuple(si_in), Tuple(si_out),
                                            spintotal_in, spintotal_out)
                if isa(symbols_val, Number)
                    symbols_vec = fill(symbols_val, prod(subshape))
                else
                    # Concatenate symbol slices from groupset_slices
                    symbols_vec = vcat([vec(symbols_val[sl...]) for sl in groupset_slices_val]...)
                end
                # Build diagonal component matrix
                block = spdiagm(0 => symbols_vec)
                # Ensure correct shape (size_out x size_in)
                if size(block) != (size_out, size_in)
                    block_full = spzeros(eltype(symbols_vec), size_out, size_in)
                    n = min(size(block, 1), size_out, size_in)
                    for k in 1:n
                        block_full[k, k] = block[k, k]
                    end
                    block = block_full
                end
            else
                # Zeros
                block = spzeros(size_out, size_in)
            end
            push!(block_row, block)
        end
        push!(submatrices, block_row)
    end
    # Assemble block matrix
    nrows = length(submatrices)
    ncols = length(submatrices[1])
    block_mat = hvcat(
        ntuple(_ -> ncols, nrows),
        [submatrices[i][j] for i in 1:nrows for j in 1:ncols]...
    )
    return sparse(block_mat)
end

"""
    operate(op::SeparableSphereOperator, out)

Explicit evaluation of a separable sphere operator.

Multiplies each spin component of the operand data by the corresponding
symbol array, accumulating into the output.
"""
function operate(op::SeparableSphereOperator, out)::Nothing
    operand = op.args[1]
    layout = operand.layout
    basis = op.input_basis
    if basis === nothing
        basis = op.output_basis
    end
    # Set output layout
    preset_layout!(out, layout)
    out.data .= 0
    # Return for size-zero data
    if length(operand.data) == 0 || length(out.data) == 0
        return
    end
    # Select overlapping data if shapes differ
    rank_in = length(operand.tensorsig)
    rank_out = length(out.tensorsig)
    local_shape_in = size(operand.data)[(rank_in + 1):end]
    local_shape_out = size(out.data)[(rank_out + 1):end]
    if local_shape_in == local_shape_out
        slices = nothing
        data_in = operand.data
        data_out = out.data
    else
        overlap = min.(local_shape_in, local_shape_out)
        slices = Tuple(1:n for n in overlap)
        # Build full index tuples including tensor dimensions
        in_idx = ntuple(i -> i <= rank_in ? Colon() : slices[i - rank_in], rank_in + length(overlap))
        out_idx = ntuple(i -> i <= rank_out ? Colon() : slices[i - rank_out], rank_out + length(overlap))
        data_in = view(operand.data, in_idx...)
        data_out = view(out.data, out_idx...)
    end
    # Apply operator over spin components
    S_in = spin_weights(basis, operand.tensorsig)
    for si_in in CartesianIndices(size(S_in))
        spintotal_in = S_in[si_in]
        @inbounds comp_in = data_in[Tuple(si_in)...]
        for si_out_tuple in spinindex_out(op, Tuple(si_in))
            # Get symbols
            spintotal_out = spintotal(basis, out.tensorsig, si_out_tuple)
            symbols_val = local_symbols(op, layout, Tuple(si_in), si_out_tuple,
                                        spintotal_in, spintotal_out)
            if slices !== nothing && !isa(symbols_val, Number)
                symbols_val = symbols_val[slices...]
            end
            # Multiply by symbols and accumulate
            @inbounds comp_out = data_out[si_out_tuple...]
            comp_out .+= symbols_val .* comp_in
        end
    end
    return nothing
end

# ============================================================================
# PolarMOperator — abstract base for polar/disk/annulus m-dependent operators
# ============================================================================

"""
    PolarMOperator <: SpectralOperator

Abstract base type for operators on polar/disk/annulus bases that act in the
azimuthal-m subspace. These operators couple only in the radial direction,
with a separate radial matrix for each azimuthal wavenumber m.

`subaxis_dependence = [true, true]` — depends on both m and radial n.
`subaxis_coupling = [false, true]` — couples only in the radial direction.

## Subtypes must implement
- `spinindex_out(op, spinindex_in)` — return tuple of valid output spin indices
- `radial_matrix(op, spinindex_in, spinindex_out, m)` — per-m radial matrix
- `_output_basis(op, input_basis)` — determine output basis from input

## Subtypes must have fields
- `operand`, `input_basis`, `output_basis`, `first_axis`, `last_axis`
- `coordsys`, `radius_axis`
- `tensorsig`, `dtype`, `domain`, `dist`
- `args::Vector{Any}`
"""
abstract type PolarMOperator <: SpectralOperator end

"""
    spinindex_out(op::PolarMOperator, spinindex_in)

Return a tuple of valid output spin indices for the given input spin index.
Must be implemented by concrete PolarMOperator subtypes.
"""
function spinindex_out(op::PolarMOperator, spinindex_in)
    error("spinindex_out not implemented for type $(typeof(op))")
end

"""
    radial_matrix(op::PolarMOperator, spinindex_in, spinindex_out, m)

Return the radial matrix for the given spin indices and azimuthal wavenumber m.
Must be implemented by concrete PolarMOperator subtypes.
"""
function radial_matrix(op::PolarMOperator, spinindex_in, spinindex_out, m)
    error("radial_matrix not implemented for type $(typeof(op))")
end

"""
    _output_basis(op::PolarMOperator, input_basis)

Determine the output basis given the input basis.
Must be implemented by concrete PolarMOperator subtypes.
"""
function _output_basis(op::PolarMOperator, input_basis)
    error("_output_basis not implemented for type $(typeof(op))")
end

"""
    polar_m_operator_subaxis_dependence(::PolarMOperator) -> Vector{Bool}

Default subaxis_dependence for PolarMOperator: depends on both m and radial n.
"""
polar_m_operator_subaxis_dependence(::PolarMOperator) = [true, true]

"""
    polar_m_operator_subaxis_coupling(::PolarMOperator) -> Vector{Bool}

Default subaxis_coupling for PolarMOperator: couples only in radial direction.
"""
polar_m_operator_subaxis_coupling(::PolarMOperator) = [false, true]

"""
    init_polar_m_operator!(op::PolarMOperator, operand, coordsys)

Common initialization logic for PolarMOperator subtypes.
Sets `coordsys`, `radius_axis`, `input_basis`, `output_basis`,
`first_axis`, `last_axis`, and `operand` on the operator.

This should be called from the concrete subtype's constructor.
Subtypes must have the fields:
    coordsys, radius_axis, input_basis, output_basis,
    first_axis, last_axis, operand, dist,
    subaxis_dependence, subaxis_coupling
"""
function init_polar_m_operator!(op, operand, coordsys)
    op.coordsys = coordsys
    op.radius_axis = get_axis(op.dist, coordsys.coords[2])
    input_basis = get_basis(operand.domain, coordsys)
    if input_basis === nothing
        input_basis = get_basis(operand.domain, coordsys.radius)
    end
    op.input_basis = input_basis
    op.output_basis = _output_basis(op, input_basis)
    op.first_axis = first_axis(op.dist, input_basis)
    op.last_axis = last_axis(op.dist, input_basis)
    op.operand = operand
    op.subaxis_dependence = [true, true]
    op.subaxis_coupling = [false, true]
    return nothing
end

"""
    operate(op::PolarMOperator, out)

Explicit evaluation of a polar m-dependent operator.

Loops over spin components and m-maps, applying the per-m radial matrix
to each azimuthal slice of the operand data.
"""
function operate(op::PolarMOperator, out)::Nothing
    operand = op.args[1]
    if hasfield(typeof(op.output_basis), :m_maps) || hasmethod(m_maps, Tuple{typeof(op.output_basis), Any})
        basis = op.output_basis
    else
        basis = op.input_basis
    end
    axis = op.last_axis
    # Set output layout
    preset_layout!(out, operand.layout)
    out.data .= 0
    # Return for size-zero data
    if length(operand.data) == 0 || length(out.data) == 0
        return
    end
    # Apply operator
    S_in = spin_weights(basis, operand.tensorsig)
    ndim = length(size(operand.data)) - length(operand.tensorsig)
    for si_in in CartesianIndices(size(S_in))
        spintotal_in = S_in[si_in]
        for si_out_tuple in spinindex_out(op, Tuple(si_in))
            comp_in = operand.data[Tuple(si_in)...]
            comp_out = out.data[si_out_tuple...]
            for (m, mg_slice, mc_slice, n_slice_val) in m_maps(basis, op.dist)
                # Build slice tuple: all colons except axis-1 gets mc_slice,
                # axis gets n_slice_val (1-based indexing)
                slices_in = ntuple(i -> i == (axis - 1) ? mc_slice :
                                        i == axis ? n_slice_val : Colon(), ndim)
                slices_out = ntuple(i -> i == (axis - 1) ? mc_slice :
                                         i == axis ? n_slice_val : Colon(), ndim)
                vec_in = view(comp_in, slices_in...)
                vec_out = view(comp_out, slices_out...)
                if length(vec_in) > 0 && length(vec_out) > 0
                    A = radial_matrix(op, Tuple(si_in), si_out_tuple, m)
                    vec_out .+= apply_matrix(A, collect(vec_in), axis)
                end
            end
        end
    end
    return nothing
end

"""
    subproblem_matrix(op::PolarMOperator, subproblem)

Build the operator matrix for a specific subproblem.

Constructs a block matrix over spin components, where each nonzero block
is built from Kronecker products of identity matrices with the per-m
radial matrix substituted at the radial axis.
"""
function subproblem_matrix(op::PolarMOperator, subproblem)
    operand = op.args[1]
    if op.input_basis === nothing
        radial_basis = op.output_basis
    else
        radial_basis = op.input_basis
    end
    S_in = spin_weights(radial_basis, operand.tensorsig)
    S_out = spin_weights(radial_basis, op.tensorsig)
    m = subproblem.group[op.last_axis - 1]
    # Loop over spin components
    submatrices = []
    for si_out in CartesianIndices(size(S_out))
        spintotal_out = S_out[si_out]
        submatrix_row = []
        for si_in in CartesianIndices(size(S_in))
            spintotal_in = S_in[si_in]
            # Build identity matrices for each axis
            subshape_in = coeff_shape(subproblem, operand.domain)
            subshape_out = coeff_shape(subproblem, op.domain)
            if (Tuple(si_out) in spinindex_out(op, Tuple(si_in))) &&
               prod(subshape_out) > 0 && prod(subshape_in) > 0
                # Build per-axis identity factors
                factors = [sparse(1.0I, subshape_out[i], subshape_in[i]) for i in eachindex(subshape_out)]
                # Get the radial matrix for this m
                rad_matrix = radial_matrix(op, Tuple(si_in), Tuple(si_out), m)
                # Reverse matrices to match memory order for flipped groups
                if hasmethod(ell_reversed, Tuple{typeof(radial_basis), typeof(op.dist)})
                    ell_rev = ell_reversed(radial_basis, op.dist)
                    if haskey(ell_rev, m) && ell_rev[m]
                        rad_matrix = rad_matrix[end:-1:1, end:-1:1]
                    end
                end
                factors[op.last_axis] = sparse(rad_matrix)
                comp_matrix = reduce(kron, factors; init=sparse(ones(1, 1)))
            else
                # Build zero matrix
                comp_matrix = spzeros(prod(subshape_out), prod(subshape_in))
            end
            push!(submatrix_row, comp_matrix)
        end
        push!(submatrices, submatrix_row)
    end
    # Assemble block matrix
    nrows = length(submatrices)
    ncols = length(submatrices[1])
    block_mat = hvcat(
        ntuple(_ -> ncols, nrows),
        [submatrices[i][j] for i in 1:nrows for j in 1:ncols]...
    )
    return sparse(block_mat)
end

# ============================================================================
# Group 1: Geometry-specific Trace, TransposeComponents, Skew
# ============================================================================

# --------------------------------------------------------------------------
# SphericalTrace
# --------------------------------------------------------------------------

"""
    SphericalTrace

Trace of a rank-2+ tensor in spherical coordinates. Contracts the first
two tensor indices using regularity recombination matrices.
"""
mutable struct SphericalTrace <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    radius_axis::Int
    radial_basis::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function SphericalTrace(operand; out=nothing)
    if isa(operand, Number)
        return 0
    end
    coordsys = operand.tensorsig[1]
    new_tensorsig = operand.tensorsig[3:end]
    input_basis = get_basis(operand.domain, coordsys)
    radius_axis = get_axis(operand.dist, coordsys.coords[3])
    radial_basis = get_radial_basis(input_basis)
    SphericalTrace(Any[operand], (operand,), out, operand.dist, operand.domain,
                   new_tensorsig, operand.dtype, "Trace", operand, coordsys,
                   input_basis, radius_axis, radial_basis,
                   nothing, nothing, false, 1)
end

function check_conditions(op::SphericalTrace)
    arg = op.args[1]
    return isa(arg, AbstractCurrent) && all(arg.layout.grid_space)
end

function enforce_conditions(op::SphericalTrace)
    arg = op.args[1]
    if isa(arg, Field)
        require_grid_space!(arg)
    end
end

function operate(op::SphericalTrace, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    # Perform trace: einsum 'ii...' equivalent
    d = cs_dim(op.coordsys)
    out.data .= 0
    for i in 1:d
        spatial_idx = ntuple(_ -> Colon(), ndims(arg.data) - 2)
        out.data .+= arg.data[i, i, spatial_idx...]
    end
    return nothing
end

function new_operand(op::SphericalTrace, operand; kw...)
    return SphericalTrace(operand; kw...)
end

function subproblem_matrix(op::SphericalTrace, subproblem)
    input_basis = op.input_basis
    radial_basis = op.radial_basis
    # subproblem.group is 1-based; radius_axis-1 and radius_axis give m and ell
    m = subproblem.group[op.radius_axis - 1]
    ell = subproblem.group[op.radius_axis]
    # Spin trace: [-+, +-, 00] are 1, other components are 0
    # In 0-based flat indexing [1, 3, 8] -> 1-based [2, 4, 9]
    trace_spin = zeros(9)
    trace_spin[[2, 4, 9]] .= 1
    trace_mat = kron(sparse(trace_spin'), sparse(1.0I, 3^length(op.tensorsig), 3^length(op.tensorsig)))
    # Stack ells
    if ell === nothing
        ell_list = collect(abs(m):input_basis.Lmax)
        if hasmethod(ell_reversed, Tuple{typeof(input_basis), typeof(op.dist)})
            ell_rev = ell_reversed(input_basis, op.dist)
            if haskey(ell_rev, m) && ell_rev[m]
                reverse!(ell_list)
            end
        end
    else
        ell_list = [ell]
    end
    Q_in = radial_recombinations(radial_basis, op.operand.tensorsig; ell_list=Tuple(ell_list))
    Q_out = radial_recombinations(radial_basis, op.tensorsig; ell_list=Tuple(ell_list))
    # Apply Q's and interleave
    trace_list = [sparse(Q_out[ell_val]') * trace_mat * sparse(Q_in[ell_val]) for ell_val in ell_list]
    # Block-diag for sin/cos parts for real dtype
    if op.dtype == Float64
        I2 = sparse(1.0I, 2, 2)
        trace_list = [kron(trace_ell, I2) for trace_ell in trace_list]
    end
    trace_final = interleave_matrices(trace_list)
    # Apply to all n
    if ell === nothing
        eye = sparse(1.0I, n_size(radial_basis, 0), n_size(radial_basis, 0))
    else
        eye = sparse(1.0I, n_size(radial_basis, ell), n_size(radial_basis, ell))
    end
    return kron(trace_final, eye)
end

# --------------------------------------------------------------------------
# PolarTrace
# --------------------------------------------------------------------------

"""
    PolarTrace

Trace of a rank-2+ tensor in polar coordinates. Contracts the first
two tensor indices using spin recombination.
"""
mutable struct PolarTrace <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    radius_axis::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function PolarTrace(operand; out=nothing)
    if isa(operand, Number)
        return 0
    end
    coordsys = operand.tensorsig[1]
    new_tensorsig = operand.tensorsig[3:end]
    input_basis = get_basis(operand.domain, coordsys)
    radius_axis = get_axis(operand.dist, coordsys.coords[2])
    PolarTrace(Any[operand], (operand,), out, operand.dist, operand.domain,
               new_tensorsig, operand.dtype, "Trace", operand, coordsys,
               input_basis, radius_axis,
               nothing, nothing, false, 1)
end

function check_conditions(op::PolarTrace)
    arg = op.args[1]
    return isa(arg, AbstractCurrent) && all(arg.layout.grid_space)
end

function enforce_conditions(op::PolarTrace)
    arg = op.args[1]
    if isa(arg, Field)
        require_grid_space!(arg)
    end
end

function operate(op::PolarTrace, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    d = cs_dim(op.coordsys)
    out.data .= 0
    for i in 1:d
        spatial_idx = ntuple(_ -> Colon(), ndims(arg.data) - 2)
        out.data .+= arg.data[i, i, spatial_idx...]
    end
    return nothing
end

function new_operand(op::PolarTrace, operand; kw...)
    return PolarTrace(operand; kw...)
end

function subproblem_matrix(op::PolarTrace, subproblem)
    m = subproblem.group[op.radius_axis]
    # [-+, +-] are 1, other components are 0
    # In 0-based flat indexing [1, 2] -> 1-based [2, 3]
    trace_spin = zeros(4)
    trace_spin[[2, 3]] .= 1
    # Kronecker up identity for remaining tensor components
    n_eye = prod(cs_dim(cs) for cs in op.tensorsig; init=1)
    # Kronecker up identity for coeff size
    n_eye *= coeff_size(subproblem, op.domain)
    eye = sparse(1.0I, n_eye, n_eye)
    return kron(sparse(trace_spin'), eye)
end

# --------------------------------------------------------------------------
# DirectProductTrace
# --------------------------------------------------------------------------

"""
    DirectProductTrace

Trace of a rank-2+ tensor in direct product coordinates.
Delegates to sub-system traces.
"""
mutable struct DirectProductTrace <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function DirectProductTrace(operand; out=nothing)
    if isa(operand, Number)
        return 0
    end
    coordsys = operand.tensorsig[1]
    new_tensorsig = operand.tensorsig[3:end]
    DirectProductTrace(Any[operand], (operand,), out, operand.dist, operand.domain,
                       new_tensorsig, operand.dtype, "Trace", operand, coordsys,
                       nothing, nothing, false, 1)
end

function check_conditions(op::DirectProductTrace)
    arg = op.args[1]
    return isa(arg, AbstractCurrent) && all(arg.layout.grid_space)
end

function enforce_conditions(op::DirectProductTrace)
    arg = op.args[1]
    if isa(arg, Field)
        require_grid_space!(arg)
    end
end

function operate(op::DirectProductTrace, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    d = cs_dim(op.coordsys)
    out.data .= 0
    for i in 1:d
        spatial_idx = ntuple(_ -> Colon(), ndims(arg.data) - 2)
        out.data .+= arg.data[i, i, spatial_idx...]
    end
    return nothing
end

function new_operand(op::DirectProductTrace, operand; kw...)
    return DirectProductTrace(operand; kw...)
end

function subproblem_matrix(op::DirectProductTrace, subproblem)
    # Delegate to sub-system traces: extract diagonal block for each sub-coordsys
    comps = [DirectProductComponent(DirectProductComponent(op.operand, 1, cs), 2, cs)
             for cs in op.coordsys.coordsystems]
    fulltrace = sum(trace_op(comp) for comp in comps)
    return expression_matrices(fulltrace, subproblem, [op.operand])[op.operand]
end

# --------------------------------------------------------------------------
# StandardTransposeComponents (for Cartesian, Polar, S2, DirectProduct)
# --------------------------------------------------------------------------

"""
    StandardTransposeComponents

General transpose of the first two tensor indices. Works for
CartesianCoordinates, PolarCoordinates, S2Coordinates, and DirectProduct.
Uses a permutation matrix approach.
"""
mutable struct StandardTransposeComponents <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    indices::Tuple{Int,Int}
    new_axis_order::Tuple
    _transpose_matrix_cache::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function StandardTransposeComponents(operand; indices=(1,2), out=nothing)
    if isa(operand, Number)
        return 0
    end
    i0, i1 = indices
    ts = operand.tensorsig
    coordsys = ts[i0]
    input_basis = get_basis(operand.domain, coordsys)
    # Build new axis permutation (tensor dims + spatial dims)
    total_dims = length(ts) + get_dim(operand.dist)
    new_order = collect(1:total_dims)
    new_order[i0], new_order[i1] = new_order[i1], new_order[i0]
    StandardTransposeComponents(Any[operand], (operand,), out, operand.dist,
                                operand.domain, ts, operand.dtype,
                                "Trans", operand, coordsys, input_basis,
                                indices, Tuple(new_order), nothing,
                                nothing, nothing, false, 1)
end

check_conditions(::StandardTransposeComponents) = true
enforce_conditions(::StandardTransposeComponents) = nothing

function _get_transpose_matrix(op::StandardTransposeComponents)
    if op._transpose_matrix_cache !== nothing
        return op._transpose_matrix_cache
    end
    rank = length(op.tensorsig)
    tensor_shape = [cs_dim(cs) for cs in op.tensorsig]
    total = prod(tensor_shape)
    # Build index permutation
    I1 = reshape(collect(1:total), Tuple(tensor_shape))
    perm_order = collect(op.new_axis_order[1:rank])
    I2 = permutedims(I1, perm_order)
    i2 = vec(I2)
    # Build permutation matrix
    P = perm_matrix(i2; source_index=true, use_sparse=true)
    op._transpose_matrix_cache = P
    return P
end

function operate(op::StandardTransposeComponents, out)::Nothing
    operand = op.args[1]
    preset_layout!(out, operand.layout)
    if length(out.data) > 0
        out.data .= permutedims(operand.data, collect(op.new_axis_order))
    end
    return nothing
end

function new_operand(op::StandardTransposeComponents, operand; kw...)
    return StandardTransposeComponents(operand; indices=op.indices, kw...)
end

function subproblem_matrix(op::StandardTransposeComponents, subproblem)
    transpose_mat = _get_transpose_matrix(op)
    eye = sparse(1.0I, coeff_size(subproblem, op.domain), coeff_size(subproblem, op.domain))
    return kron(transpose_mat, eye)
end

# --------------------------------------------------------------------------
# SphericalTransposeComponents
# --------------------------------------------------------------------------

"""
    SphericalTransposeComponents

Transpose the first two tensor indices for spherical coordinates.
Requires special handling for regularity recombination.
"""
mutable struct SphericalTransposeComponents <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    indices::Tuple{Int,Int}
    new_axis_order::Tuple
    radius_axis::Int
    radial_basis::Any
    _transpose_matrix_cache::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function SphericalTransposeComponents(operand; indices=(1,2), out=nothing)
    if isa(operand, Number)
        return 0
    end
    i0, i1 = indices
    ts = operand.tensorsig
    coordsys = ts[i0]
    input_basis = get_basis(operand.domain, coordsys)
    radius_axis = get_axis(operand.dist, coordsys.coords[3])
    radial_basis = get_radial_basis(input_basis)
    total_dims = length(ts) + get_dim(operand.dist)
    new_order = collect(1:total_dims)
    new_order[i0], new_order[i1] = new_order[i1], new_order[i0]
    SphericalTransposeComponents(Any[operand], (operand,), out, operand.dist,
                                  operand.domain, ts, operand.dtype,
                                  "Trans", operand, coordsys, input_basis,
                                  indices, Tuple(new_order),
                                  radius_axis, radial_basis, nothing,
                                  nothing, nothing, false, 1)
end

check_conditions(::SphericalTransposeComponents) = true
enforce_conditions(::SphericalTransposeComponents) = nothing

function _get_transpose_matrix(op::SphericalTransposeComponents)
    if op._transpose_matrix_cache !== nothing
        return op._transpose_matrix_cache
    end
    rank = length(op.tensorsig)
    tensor_shape = [cs_dim(cs) for cs in op.tensorsig]
    total = prod(tensor_shape)
    I1 = reshape(collect(1:total), Tuple(tensor_shape))
    perm_order = collect(op.new_axis_order[1:rank])
    I2 = permutedims(I1, perm_order)
    i2 = vec(I2)
    P = perm_matrix(i2; source_index=true, use_sparse=true)
    op._transpose_matrix_cache = P
    return P
end

function operate(op::SphericalTransposeComponents, out)::Nothing
    operand = op.args[1]
    radius_axis = op.radius_axis
    layout = operand.layout
    preset_layout!(out, layout)
    if layout.grid_space[radius_axis]
        # Not in regularity components: can directly transpose
        if length(out.data) > 0
            out.data .= permutedims(operand.data, collect(op.new_axis_order))
        end
    else
        # Coefficient space: commute transposition with regularity recombination
        if length(out.data) > 0
            out.data .= operand.data
            radial_basis = op.radial_basis
            ell_maps_val = ell_maps(op.input_basis, op.dist)
            backward_regularity_recombination!(radial_basis, operand.tensorsig, radius_axis, out.data; ell_maps=ell_maps_val)
            out.data .= permutedims(out.data, collect(op.new_axis_order))
            forward_regularity_recombination!(radial_basis, operand.tensorsig, radius_axis, out.data; ell_maps=ell_maps_val)
        end
    end
    return nothing
end

function new_operand(op::SphericalTransposeComponents, operand; kw...)
    return SphericalTransposeComponents(operand; indices=op.indices, kw...)
end

function subproblem_matrix(op::SphericalTransposeComponents, subproblem)
    input_basis = op.input_basis
    radial_basis = op.radial_basis
    m = subproblem.group[op.radius_axis - 1]
    ell = subproblem.group[op.radius_axis]
    # Get transpose permutation matrix
    transpose_mat = _get_transpose_matrix(op)
    # Stack ells
    if ell === nothing
        ell_list = collect(abs(m):input_basis.Lmax)
        if hasmethod(ell_reversed, Tuple{typeof(input_basis), typeof(op.dist)})
            ell_rev = ell_reversed(input_basis, op.dist)
            if haskey(ell_rev, m) && ell_rev[m]
                reverse!(ell_list)
            end
        end
    else
        ell_list = [ell]
    end
    Q_in = radial_recombinations(radial_basis, op.operand.tensorsig; ell_list=Tuple(ell_list))
    Q_out = radial_recombinations(radial_basis, op.tensorsig; ell_list=Tuple(ell_list))
    # Apply Q's and interleave
    transpose_list = [sparse(Q_out[ell_val]') * transpose_mat * sparse(Q_in[ell_val]) for ell_val in ell_list]
    # Block-diag for sin/cos parts for real dtype
    if op.dtype == Float64
        I2 = sparse(1.0I, 2, 2)
        transpose_list = [kron(tr_ell, I2) for tr_ell in transpose_list]
    end
    transpose_final = interleave_matrices(transpose_list)
    # Apply to all n
    if ell === nothing
        eye = sparse(1.0I, n_size(radial_basis, 0), n_size(radial_basis, 0))
    else
        eye = sparse(1.0I, n_size(radial_basis, ell), n_size(radial_basis, ell))
    end
    return kron(transpose_final, eye)
end

# --------------------------------------------------------------------------
# CartesianSkew
# --------------------------------------------------------------------------

"""
    CartesianSkew

Skew operator for 2D Cartesian coordinates.
Rotates a 2D vector by 90 degrees: (x,y) -> (-y,x).
"""
mutable struct CartesianSkew <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    index::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianSkew(operand; index=1, out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    coordsys = operand.tensorsig[index]
    if cs_dim(coordsys) != 2
        throw(ArgumentError("Skew only valid on 2D coordsystems."))
    end
    CartesianSkew(Any[operand], (operand,), out, operand.dist, operand.domain,
                  operand.tensorsig, operand.dtype, "Skew", operand, coordsys, index,
                  nothing, nothing, false, 1)
end

check_conditions(::CartesianSkew) = true
enforce_conditions(::CartesianSkew) = nothing

function operate(op::CartesianSkew, out)::Nothing
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(arg.data) > 0
        sx = axslice(op.index, 1, 1)
        sy = axslice(op.index, 2, 2)
        out.data[sx...] .= -(arg.data[sy...])
        out.data[sy...] .= arg.data[sx...]
    end
    return nothing
end

function new_operand(op::CartesianSkew, operand; kw...)
    return CartesianSkew(operand; index=op.index, kw...)
end

function subproblem_matrix(op::CartesianSkew, subproblem)
    # Build identity factors for each tangent space and coeffs
    factors = [sparse(1.0I, cs_dim(cs), cs_dim(cs)) for cs in op.operand.tensorsig]
    push!(factors, sparse(1.0I, coeff_size(subproblem, op.domain), coeff_size(subproblem, op.domain)))
    # Substitute skew matrix at the indexed position
    skew_mat = sparse([0.0 -1.0; 1.0 0.0])
    factors[op.index] = skew_mat
    return reduce(kron, factors; init=sparse(ones(1, 1)))
end

# --------------------------------------------------------------------------
# SpinSkew (for PolarCoordinates and S2Coordinates)
# --------------------------------------------------------------------------

"""
    SpinSkew

Skew operator for polar and S2 coordinates.
Multiplies spin-minus component by -1j and spin-plus by +1j.
"""
mutable struct SpinSkew <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    index::Int
    azimuth_axis::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function SpinSkew(operand; index=1, out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    coordsys = operand.tensorsig[index]
    if cs_dim(coordsys) != 2
        throw(ArgumentError("Skew only valid on 2D coordsystems."))
    end
    azimuth_axis = get_axis(operand.dist, coordsys.coords[1])
    SpinSkew(Any[operand], (operand,), out, operand.dist, operand.domain,
             operand.tensorsig, operand.dtype, "Skew", operand, coordsys, index,
             azimuth_axis,
             nothing, nothing, false, 1)
end

check_conditions(::SpinSkew) = true
enforce_conditions(::SpinSkew) = nothing

function operate(op::SpinSkew, out)::Nothing
    arg = op.args[1]
    index = op.index
    azimuth_axis = op.azimuth_axis
    rank = length(op.tensorsig)
    preset_layout!(out, arg.layout)
    if length(arg.data) > 0
        if arg.layout.grid_space[azimuth_axis + 1]
            # Grid space: left-handed rotation
            sx = axslice(index, 1, 1)
            sy = axslice(index, 2, 2)
            out.data[sx...] .= arg.data[sy...]
            out.data[sy...] .= -(arg.data[sx...])
        else
            # Coefficient space: spinorder -, +
            minus = axslice(index, 1, 1)
            plus = axslice(index, 2, 2)
            arg_plus = arg.data[plus...]
            arg_minus = arg.data[minus...]
            out_plus = view(out.data, plus...)
            out_minus = view(out.data, minus...)
            if is_complex_dtype(op.dtype)
                # out = 1j * s * arg; s=-1 for minus, s=+1 for plus
                out_plus .= 1im .* arg_plus
                out_minus .= (-1im) .* arg_minus
            else
                # Real: (1j * s) * (arg_cos + 1j * arg_msin)
                # = -s * arg_msin + 1j * s * arg_cos
                cos_sl = axslice(rank + azimuth_axis, 1, nothing, 2)
                msin_sl = axslice(rank + azimuth_axis, 2, nothing, 2)
                out_plus[cos_sl...] .= -(arg_plus[msin_sl...])
                out_plus[msin_sl...] .= arg_plus[cos_sl...]
                out_minus[cos_sl...] .= arg_minus[msin_sl...]
                out_minus[msin_sl...] .= -(arg_minus[cos_sl...])
            end
        end
    end
    return nothing
end

function new_operand(op::SpinSkew, operand; kw...)
    return SpinSkew(operand; index=op.index, kw...)
end

function subproblem_matrix(op::SpinSkew, subproblem)
    # Build identity factors for each dimension in field_shape
    shape = field_shape(subproblem, op)
    factors = [sparse(1.0I, sz, sz) for sz in shape]
    # Weight by spin (spinorder: -, +)
    factors[op.index] = sparse([-1.0 0.0; 0.0 1.0])
    # Multiply by 1j
    if is_complex_dtype(op.dtype)
        factors[op.index] = 1im .* factors[op.index]
    else
        azimuth_index = length(op.tensorsig) + op.azimuth_axis
        id_m = sparse(1.0I, shape[op.azimuth_axis] ÷ 2, shape[op.azimuth_axis] ÷ 2)
        mul_1j = sparse([0.0 -1.0; 1.0 0.0])
        factors[azimuth_index] = kron(id_m, mul_1j)
    end
    return reduce(kron, factors; init=sparse(ones(1, 1)))
end

# --------------------------------------------------------------------------
# Updated dispatch functions for trace, transpose, skew
# --------------------------------------------------------------------------

function trace_op(field)
    if isa(field, Number) || field == 0
        return 0
    end
    cs = field.tensorsig[1]
    if isa(cs, SphericalCoordinates)
        return SphericalTrace(field)
    elseif isa(cs, PolarCoordinates)
        return PolarTrace(field)
    elseif isa(cs, DirectProduct)
        return DirectProductTrace(field)
    else
        return CartesianTrace(field)
    end
end

function transpose_components(field; indices=(1,2))
    if isa(field, Number) || field == 0
        return 0
    end
    i0, i1 = indices
    cs = field.tensorsig[i0]
    if isa(cs, SphericalCoordinates)
        return SphericalTransposeComponents(field; indices=indices)
    else
        # Standard transpose works for Cartesian, Polar, S2, DirectProduct
        return StandardTransposeComponents(field; indices=indices)
    end
end

"""Skew dispatch: selects CartesianSkew or SpinSkew based on coordinate system."""
function skew(field; index=1)
    if isa(field, Number) || field == 0
        return 0
    end
    cs = field.tensorsig[index]
    if isa(cs, CartesianCoordinates) || isa(cs, Coordinate)
        return CartesianSkew(field; index=index)
    elseif isa(cs, PolarCoordinates) || isa(cs, S2Coordinates)
        return SpinSkew(field; index=index)
    else
        return CartesianSkew(field; index=index)  # fallback
    end
end

register_operator_alias!("skew", skew)

# ============================================================================
# Group 2: Component operators (RadialComponent, AngularComponent, AzimuthalComponent)
# ============================================================================

# --------------------------------------------------------------------------
# RadialComponent
# --------------------------------------------------------------------------

"""
    RadialComponent

Extract the radial component from a vector/tensor field.
Dispatches by basis type for subproblem_matrix.
"""
mutable struct RadialComponent <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    index::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function RadialComponent(operand; index=1, out=nothing)
    if isa(operand, Number)
        return 0
    end
    if index < 0
        index += length(operand.tensorsig) + 1
    end
    coordsys = operand.tensorsig[index]
    input_basis = get_basis(operand.domain, coordsys)
    new_tensorsig = (operand.tensorsig[1:index-1]..., operand.tensorsig[index+1:end]...)
    RadialComponent(Any[operand], (operand,), out, operand.dist, operand.domain,
                    new_tensorsig, operand.dtype, "Radial", operand, coordsys,
                    input_basis, index,
                    nothing, nothing, false, 1)
end

check_conditions(::RadialComponent) = true
enforce_conditions(::RadialComponent) = nothing

function operate(op::RadialComponent, out)::Nothing
    operand = op.args[1]
    preset_layout!(out, operand.layout)
    if length(out.data) > 0
        # For polar coords, radial is spin index 2 (1-based);
        # For spherical coords, radial is spin index 3 (the 0-component, 1-based index 3)
        if isa(op.coordsys, SphericalCoordinates)
            comp_idx = 3
        else
            # Polar: radius is second coord -> spin index 2
            comp_idx = 2
        end
        idx = axindex(op.index, comp_idx)
        out.data .= operand.data[idx...]
    end
    return nothing
end

function new_operand(op::RadialComponent, operand; kw...)
    return RadialComponent(operand; index=op.index, kw...)
end

function subproblem_matrix(op::RadialComponent, subproblem)
    operand = op.args[1]
    basis = get_basis(op.domain, op.coordsys)
    if basis === nothing
        basis = op.input_basis
    end
    # Check if this is a spherical basis (SphereBasis) or polar (IntervalBasis)
    if isa(op.coordsys, SphericalCoordinates)
        # S2RadialComponent: select spin index 2 (0-based) = 3 (1-based)
        S_in = spin_weights(basis, operand.tensorsig)
        S_out = spin_weights(basis, op.tensorsig)
        matrix = zeros(Int, length(S_out), length(S_in))
        for (j, si_in) in enumerate(CartesianIndices(size(S_in)))
            si_in_tuple = Tuple(si_in)
            if si_in_tuple[op.index] == 3  # 0-spin (radial) is index 3 in 1-based
                si_check = (si_in_tuple[1:op.index-1]..., si_in_tuple[op.index+1:end]...)
                for (i, si_out) in enumerate(CartesianIndices(size(S_out)))
                    if Tuple(si_out) == si_check
                        matrix[i, j] = 1
                    end
                end
            end
        end
        matrix = sparse(Float64.(matrix))
        # Block-diag for sin/cos parts for real dtype
        if op.dtype == Float64
            matrix = kron(matrix, sparse(1.0I, 2, 2))
        end
        # Block over ell
        la = last_axis(op.dist, op.input_basis)
        m = subproblem.group[la - 1]
        ell = subproblem.group[la]
        if ell === nothing
            n_ell = op.input_basis.Lmax + 1 - abs(m)
            matrix = kron(matrix, sparse(1.0I, n_ell, n_ell))
        end
        return matrix
    else
        # PolarRadialComponent: spin index 1 (1-based) = + component
        # In polar spin ordering (-, +), radial is index 2 (1-based)
        input_dim = length(operand.tensorsig)
        output_dim = length(op.tensorsig)
        n_in = 2^input_dim
        n_out = 2^output_dim
        matrix = zeros(Int, n_out, n_in)
        for input_idx in 0:(n_in - 1)
            index_in = reverse(digits(input_idx, base=2, pad=input_dim))
            for output_idx in 0:(n_out - 1)
                index_out = reverse(digits(output_idx, base=2, pad=output_dim))
                # Check: removing op.index from index_in gives index_out, and the removed value is 1 (radial)
                idx_removed = (index_in[1:op.index-1]..., index_in[op.index+1:end]...)
                if collect(idx_removed) == index_out && index_in[op.index] == 1
                    matrix[output_idx + 1, input_idx + 1] = 1
                end
            end
        end
        matrix = sparse(Float64.(matrix))
        if op.dtype == Float64
            matrix = kron(matrix, sparse(1.0I, 2, 2))
        end
        return matrix
    end
end

# --------------------------------------------------------------------------
# AngularComponent
# --------------------------------------------------------------------------

"""
    AngularComponent

Extract the angular (meridional/colatitude) component from a vector/tensor field.
"""
mutable struct AngularComponent <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    index::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function AngularComponent(operand; index=1, out=nothing)
    if isa(operand, Number)
        return 0
    end
    if index < 0
        index += length(operand.tensorsig) + 1
    end
    coordsys = operand.tensorsig[index]
    input_basis = get_basis(operand.domain, coordsys)
    # Determine output tensorsig
    if isa(coordsys, PolarCoordinates)
        new_tensorsig = (operand.tensorsig[1:index-1]..., operand.tensorsig[index+1:end]...)
    elseif isa(coordsys, SphericalCoordinates)
        S2coordsys = coordsys.S2coordsys
        new_tensorsig = (operand.tensorsig[1:index-1]..., S2coordsys, operand.tensorsig[index+1:end]...)
    else
        error("AngularComponent not supported for coordinate system type $(typeof(coordsys))")
    end
    AngularComponent(Any[operand], (operand,), out, operand.dist, operand.domain,
                     new_tensorsig, operand.dtype, "Angular", operand, coordsys,
                     input_basis, index,
                     nothing, nothing, false, 1)
end

check_conditions(::AngularComponent) = true
enforce_conditions(::AngularComponent) = nothing

function operate(op::AngularComponent, out)::Nothing
    operand = op.args[1]
    preset_layout!(out, operand.layout)
    if length(out.data) > 0
        if isa(op.coordsys, SphericalCoordinates)
            # Angular: first 2 spin components (-, +)
            idx = axslice(op.index, 1, 2)
            out.data .= operand.data[idx...]
        else
            # Polar: azimuthal is spin index 1 (the - component)
            idx = axindex(op.index, 1)
            out.data .= operand.data[idx...]
        end
    end
    return nothing
end

function new_operand(op::AngularComponent, operand; kw...)
    return AngularComponent(operand; index=op.index, kw...)
end

function subproblem_matrix(op::AngularComponent, subproblem)
    operand = op.args[1]
    basis = get_basis(op.domain, op.coordsys)
    if basis === nothing
        basis = op.input_basis
    end
    if isa(op.coordsys, SphericalCoordinates)
        # S2AngularComponent: identity mapping (keeps -, + components)
        S_in = spin_weights(basis, operand.tensorsig)
        S_out = spin_weights(basis, op.tensorsig)
        matrix = zeros(Int, length(S_out), length(S_in))
        for (j, si_in) in enumerate(CartesianIndices(size(S_in)))
            for (i, si_out) in enumerate(CartesianIndices(size(S_out)))
                if Tuple(si_in) == Tuple(si_out)
                    matrix[i, j] = 1
                end
            end
        end
        matrix = sparse(Float64.(matrix))
        if op.dtype == Float64
            matrix = kron(matrix, sparse(1.0I, 2, 2))
        end
        la = last_axis(op.dist, op.input_basis)
        m = subproblem.group[la - 1]
        ell = subproblem.group[la]
        if ell === nothing
            n_ell = op.input_basis.Lmax + 1 - abs(m)
            matrix = kron(matrix, sparse(1.0I, n_ell, n_ell))
        end
        return matrix
    else
        # PolarAngularComponent: select first component (azimuthal = -)
        matrix = sparse(Float64.([1 0]))
        if op.dtype == Float64
            matrix = kron(matrix, sparse(1.0I, 2, 2))
        end
        return matrix
    end
end

# --------------------------------------------------------------------------
# AzimuthalComponent
# --------------------------------------------------------------------------

"""
    AzimuthalComponent

Extract the azimuthal component from a polar vector/tensor field.
Only valid for PolarCoordinates.
"""
mutable struct AzimuthalComponent <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    index::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function AzimuthalComponent(operand; index=1, out=nothing)
    if isa(operand, Number)
        return 0
    end
    if index < 0
        index += length(operand.tensorsig) + 1
    end
    coordsys = operand.tensorsig[index]
    if !isa(coordsys, PolarCoordinates)
        error("Can only take the AzimuthalComponent of a PolarCoordinate vector")
    end
    input_basis = get_basis(operand.domain, coordsys)
    new_tensorsig = (operand.tensorsig[1:index-1]..., operand.tensorsig[index+1:end]...)
    AzimuthalComponent(Any[operand], (operand,), out, operand.dist, operand.domain,
                       new_tensorsig, operand.dtype, "Azimuthal", operand, coordsys,
                       input_basis, index,
                       nothing, nothing, false, 1)
end

check_conditions(::AzimuthalComponent) = true
enforce_conditions(::AzimuthalComponent) = nothing

function operate(op::AzimuthalComponent, out)::Nothing
    operand = op.args[1]
    preset_layout!(out, operand.layout)
    if length(out.data) > 0
        # Azimuthal = spin index 1 (the - component, 1-based)
        idx = axindex(op.index, 1)
        out.data .= operand.data[idx...]
    end
    return nothing
end

function new_operand(op::AzimuthalComponent, operand; kw...)
    return AzimuthalComponent(operand; index=op.index, kw...)
end

function subproblem_matrix(op::AzimuthalComponent, subproblem)
    operand = op.args[1]
    input_dim = length(operand.tensorsig)
    output_dim = length(op.tensorsig)
    n_in = 2^input_dim
    n_out = 2^output_dim
    matrix = zeros(Int, n_out, n_in)
    for input_idx in 0:(n_in - 1)
        index_in = reverse(digits(input_idx, base=2, pad=input_dim))
        for output_idx in 0:(n_out - 1)
            index_out = reverse(digits(output_idx, base=2, pad=output_dim))
            # Check: removing op.index from index_in gives index_out, and the removed value is 0 (azimuthal)
            idx_removed = (index_in[1:op.index-1]..., index_in[op.index+1:end]...)
            if collect(idx_removed) == index_out && index_in[op.index] == 0
                matrix[output_idx + 1, input_idx + 1] = 1
            end
        end
    end
    matrix = sparse(Float64.(matrix))
    if op.dtype == Float64
        matrix = kron(matrix, sparse(1.0I, 2, 2))
    end
    return matrix
end

# --------------------------------------------------------------------------
# Component dispatch functions
# --------------------------------------------------------------------------

"""Extract the radial component from a vector/tensor field."""
radial_component(field; index=1) = RadialComponent(field; index=index)

"""Extract the angular (meridional) component from a vector/tensor field."""
angular_component(field; index=1) = AngularComponent(field; index=index)

"""Extract the azimuthal component from a polar vector/tensor field."""
azimuthal_component(field; index=1) = AzimuthalComponent(field; index=index)

register_operator_alias!("radial", radial_component)
register_operator_alias!("angular", angular_component)
register_operator_alias!("azimuthal", azimuthal_component)

# ============================================================================
# Group 3: Polar differential operators
# ============================================================================

# --------------------------------------------------------------------------
# MulCosine
# --------------------------------------------------------------------------

"""
    MulCosine <: PolarMOperator

Multiplication by cosine of latitude for S2 / polar coordinates.
Uses the sphere operator "Cos" from dedalus_sphere.
"""
mutable struct MulCosine <: PolarMOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Any
    last_axis::Any
    radius_axis::Any
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    _radial_matrix_cache::Dict{Any,Any}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function MulCosine(operand, coordsys=nothing; out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    if coordsys === nothing
        coordsys = operand.dist.single_coordsys
        if coordsys === false
            error("coordsys must be specified.")
        end
    end
    dist = operand.dist
    op = MulCosine(Any[operand], (operand,), out, dist, operand.domain,
                   operand.tensorsig, operand.dtype, "MulCos", operand,
                   nothing, nothing, nothing, nothing, nothing, nothing,
                   [true, true], [false, true], Dict{Any,Any}(),
                   nothing, nothing, false, 1)
    init_polar_m_operator!(op, operand, coordsys)
    op.domain = operand.domain
    op.tensorsig = operand.tensorsig
    return op
end

function _output_basis(op::MulCosine, input_basis)
    return input_basis
end

function spinindex_out(op::MulCosine, spinindex_in)
    # Spinorder: -, +, 0 -- cosine is diagonal
    return (spinindex_in,)
end

function new_operand(op::MulCosine, operand; kw...)
    return MulCosine(operand, op.coordsys; kw...)
end

function check_conditions(op::MulCosine)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.radius_axis] && arg.layout.local[op.radius_axis]
end

function enforce_conditions(op::MulCosine)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.radius_axis)
        require_local!(arg, op.radius_axis)
    end
end

function radial_matrix(op::MulCosine, spinindex_in, spinindex_out_val, m)
    cache_key = (spinindex_in, spinindex_out_val, m)
    cached = get(op._radial_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    radial_basis = op.input_basis
    spintotal_in = spintotal(radial_basis, op.operand.tensorsig, spinindex_in)
    if spinindex_out_val in spinindex_out(op, spinindex_in)
        matrix = _mulcosine_radial_matrix(radial_basis.Lmax, spintotal_in, m, op.dtype)
        op._radial_matrix_cache[cache_key] = matrix
        return matrix
    else
        error("Invalid spinindex_out for MulCosine")
    end
end

"""Static helper: compute the radial matrix for MulCosine."""
function _mulcosine_radial_matrix(Lmax, spintotal_val, m, dtype)
    matrix = sphere_operator("Cos", dtype, Lmax, m, spintotal_val)
    # Pad to include invalid ells
    trunc = abs(spintotal_val) - abs(m)
    if trunc > 0
        pad = spzeros(trunc, trunc)
        matrix = sparse_block_diag([pad, sparse(matrix)])
    end
    return matrix
end

# --------------------------------------------------------------------------
# PolarGradient
# --------------------------------------------------------------------------

"""
    PolarGradient <: PolarMOperator

Gradient in polar coordinates. Produces a vector field from a scalar/tensor
field. Uses D- and D+ operator matrices from the radial basis.
"""
mutable struct PolarGradient <: PolarMOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Any
    last_axis::Any
    radius_axis::Any
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    _radial_matrix_cache::Dict{Any,Any}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function PolarGradient(operand, coordsys; out=nothing)
    if isa(operand, Number)
        return 0
    end
    dist = operand.dist
    op = PolarGradient(Any[operand], (operand,), out, dist, operand.domain,
                       (coordsys, operand.tensorsig...), operand.dtype, "Grad", operand,
                       nothing, nothing, nothing, nothing, nothing, nothing,
                       [true, true], [false, true], Dict{Any,Any}(),
                       nothing, nothing, false, 1)
    init_polar_m_operator!(op, operand, coordsys)
    # Update domain and tensorsig after init
    op.domain = substitute_basis(operand.domain, op.input_basis, op.output_basis)
    op.tensorsig = (coordsys, operand.tensorsig...)
    return op
end

function _output_basis(op::PolarGradient, input_basis)
    return derivative_basis(input_basis, 1)
end

function check_conditions(op::PolarGradient)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.radius_axis] && arg.layout.local[op.radius_axis]
end

function enforce_conditions(op::PolarGradient)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.radius_axis)
        require_local!(arg, op.radius_axis)
    end
end

function spinindex_out(op::PolarGradient, spinindex_in)
    # Spinorder: -, +  -- Gradient hits both - and +
    # Prepend 1 (for -) and 2 (for +) as first index (1-based)
    return ((1, spinindex_in...), (2, spinindex_in...))
end

function new_operand(op::PolarGradient, operand; kw...)
    return PolarGradient(operand, op.coordsys; kw...)
end

function radial_matrix(op::PolarGradient, spinindex_in, spinindex_out_val, m)
    cache_key = (spinindex_in, spinindex_out_val, m)
    cached = get(op._radial_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    radial_basis = op.input_basis
    spintotal_in = spintotal(radial_basis, op.operand.tensorsig, spinindex_in)
    if spinindex_out_val in spinindex_out(op, spinindex_in)
        matrix = _polar_gradient_radial_matrix(radial_basis, spinindex_out_val[1], spintotal_in, m)
        op._radial_matrix_cache[cache_key] = matrix
        return matrix
    else
        error("Invalid spinindex_out for PolarGradient")
    end
end

"""Static helper: compute the radial matrix for PolarGradient."""
function _polar_gradient_radial_matrix(radial_basis, spinindex_out0, spintotal_val, m)
    if spinindex_out0 == 1
        # D- operator (spin-minus), 1-based index 1 = Python index 0
        return (1 / sqrt(2)) * operator_matrix(radial_basis, "D-", m, spintotal_val)
    elseif spinindex_out0 == 2
        # D+ operator (spin-plus), 1-based index 2 = Python index 1
        return (1 / sqrt(2)) * operator_matrix(radial_basis, "D+", m, spintotal_val)
    else
        error("Invalid spinindex_out0 for PolarGradient: $spinindex_out0")
    end
end

# --------------------------------------------------------------------------
# PolarDivergence
# --------------------------------------------------------------------------

"""
    PolarDivergence <: PolarMOperator

Divergence in polar coordinates. Contracts the first tensor index of a
vector/tensor field. Uses D+ and D- operator matrices from the radial basis.
"""
mutable struct PolarDivergence <: PolarMOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Any
    last_axis::Any
    radius_axis::Any
    index::Int
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    _radial_matrix_cache::Dict{Any,Any}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function PolarDivergence(operand; index=1, out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    if index != 1
        error("Divergence only implemented along index 1.")
    end
    coordsys = operand.tensorsig[index]
    dist = operand.dist
    new_tensorsig = (operand.tensorsig[1:index-1]..., operand.tensorsig[index+1:end]...)
    op = PolarDivergence(Any[operand], (operand,), out, dist, operand.domain,
                         new_tensorsig, operand.dtype, "Div", operand,
                         nothing, nothing, nothing, nothing, nothing, nothing,
                         index,
                         [true, true], [false, true], Dict{Any,Any}(),
                         nothing, nothing, false, 1)
    init_polar_m_operator!(op, operand, coordsys)
    op.domain = substitute_basis(operand.domain, op.input_basis, op.output_basis)
    op.tensorsig = new_tensorsig
    return op
end

function _output_basis(op::PolarDivergence, input_basis)
    return derivative_basis(input_basis, 1)
end

function check_conditions(op::PolarDivergence)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.radius_axis] && arg.layout.local[op.radius_axis]
end

function enforce_conditions(op::PolarDivergence)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.radius_axis)
        require_local!(arg, op.radius_axis)
    end
end

function spinindex_out(op::PolarDivergence, spinindex_in)
    # Spinorder: -, +  -- Divergence feels - and +
    # In 1-based: 1 = -, 2 = +
    if spinindex_in[1] in (1, 2)
        return (spinindex_in[2:end],)
    else
        return ()
    end
end

function new_operand(op::PolarDivergence, operand; kw...)
    return PolarDivergence(operand; index=op.index, kw...)
end

function radial_matrix(op::PolarDivergence, spinindex_in, spinindex_out_val, m)
    cache_key = (spinindex_in, spinindex_out_val, m)
    cached = get(op._radial_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    radial_basis = op.input_basis
    spintotal_in = spintotal(radial_basis, op.operand.tensorsig, spinindex_in)
    if spinindex_in[1] != 3 && spinindex_in[2:end] == spinindex_out_val
        matrix = _polar_divergence_radial_matrix(radial_basis, spinindex_in[1], spintotal_in, m)
        op._radial_matrix_cache[cache_key] = matrix
        return matrix
    else
        error("Invalid spinindex for PolarDivergence")
    end
end

"""Static helper: compute the radial matrix for PolarDivergence."""
function _polar_divergence_radial_matrix(radial_basis, spinindex_in0, spintotal_val, m)
    if spinindex_in0 == 1
        # D+ operator for spin-minus input (1-based index 1 = Python index 0)
        return (1 / sqrt(2)) * operator_matrix(radial_basis, "D+", m, spintotal_val)
    elseif spinindex_in0 == 2
        # D- operator for spin-plus input (1-based index 2 = Python index 1)
        return (1 / sqrt(2)) * operator_matrix(radial_basis, "D-", m, spintotal_val)
    else
        error("Invalid spinindex_in0 for PolarDivergence: $spinindex_in0")
    end
end

# --------------------------------------------------------------------------
# PolarLaplacian
# --------------------------------------------------------------------------

"""
    PolarLaplacian <: PolarMOperator

Laplacian in polar coordinates. Uses the 'L' operator matrix from the
radial basis.
"""
mutable struct PolarLaplacian <: PolarMOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Any
    last_axis::Any
    radius_axis::Any
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    _radial_matrix_cache::Dict{Any,Any}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function PolarLaplacian(operand, coordsys; out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    dist = operand.dist
    op = PolarLaplacian(Any[operand], (operand,), out, dist, operand.domain,
                        operand.tensorsig, operand.dtype, "Lap", operand,
                        nothing, nothing, nothing, nothing, nothing, nothing,
                        [true, true], [false, true], Dict{Any,Any}(),
                        nothing, nothing, false, 1)
    init_polar_m_operator!(op, operand, coordsys)
    op.domain = substitute_basis(operand.domain, op.input_basis, op.output_basis)
    op.tensorsig = operand.tensorsig
    return op
end

function _output_basis(op::PolarLaplacian, input_basis)
    return derivative_basis(input_basis, 2)
end

function check_conditions(op::PolarLaplacian)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.radius_axis] && arg.layout.local[op.radius_axis]
end

function enforce_conditions(op::PolarLaplacian)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.radius_axis)
        require_local!(arg, op.radius_axis)
    end
end

function spinindex_out(op::PolarLaplacian, spinindex_in)
    return (spinindex_in,)
end

function new_operand(op::PolarLaplacian, operand; kw...)
    return PolarLaplacian(operand, op.coordsys; kw...)
end

function radial_matrix(op::PolarLaplacian, spinindex_in, spinindex_out_val, m)
    cache_key = (spinindex_in, spinindex_out_val, m)
    cached = get(op._radial_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    radial_basis = op.input_basis
    spintotal_in = spintotal(radial_basis, op.operand.tensorsig, spinindex_in)
    if spinindex_in == spinindex_out_val
        matrix = operator_matrix(radial_basis, "L", m, spintotal_in)
        op._radial_matrix_cache[cache_key] = matrix
        return matrix
    else
        error("Invalid spinindex for PolarLaplacian")
    end
end

# --------------------------------------------------------------------------
# Updated dispatch: gradient, divergence, laplacian for PolarCoordinates
# --------------------------------------------------------------------------

gradient(field, cs::PolarCoordinates) = PolarGradient(field, cs)
gradient(field, cs::S2Coordinates) = SphereGradient(field, cs)
gradient(field, cs::SphericalCoordinates) = SphericalGradient(field, cs)

function divergence(field; index=1)
    if isa(field, Number) || field == 0
        return 0
    end
    cs = field.tensorsig[index]
    if isa(cs, SphericalCoordinates)
        return SphericalDivergence(field; index=index)
    elseif isa(cs, S2Coordinates)
        return SphereDivergence(field; index=index)
    elseif isa(cs, PolarCoordinates)
        return PolarDivergence(field; index=index)
    elseif isa(cs, CartesianCoordinates) || isa(cs, Coordinate)
        return CartesianDivergence(field; index=index)
    elseif isa(cs, DirectProduct)
        return DirectProductDivergence(field; index=index)
    else
        return CartesianDivergence(field; index=index)  # fallback
    end
end

function laplacian(field, cs)
    if isa(field, Number) || field == 0
        return 0
    end
    if isa(cs, SphericalCoordinates)
        return SphericalLaplacian(field, cs)
    elseif isa(cs, S2Coordinates)
        return SphereLaplacian(field, cs)
    elseif isa(cs, PolarCoordinates)
        return PolarLaplacian(field, cs)
    elseif isa(cs, CartesianCoordinates) || isa(cs, Coordinate)
        return CartesianLaplacian(field, cs)
    elseif isa(cs, DirectProduct)
        return DirectProductLaplacian(field, cs)
    else
        return CartesianLaplacian(field, cs)  # fallback
    end
end

# ============================================================================
# Group 4: SphereEllProduct
# ============================================================================

"""
    SphereEllProduct <: SeparableSphereOperator

Product with an ell-dependent (and optionally radius-dependent) scalar.
Used for angular Laplacian eigenvalue multiplication: ell*(ell+1) etc.

The `ell_r_func(ell, radius)` callable defines the per-mode symbol.
"""
mutable struct SphereEllProduct <: SeparableSphereOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Any
    last_axis::Any
    ell_r_func::Any
    complex_operator::Bool
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function SphereEllProduct(operand, coordsys, ell_r_func; out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    dist = operand.dist
    input_basis = get_basis(operand.domain, coordsys)
    output_basis = input_basis
    fa = first_axis(dist, input_basis)
    la = last_axis(dist, input_basis)
    SphereEllProduct(Any[operand], (operand,), out, dist, operand.domain,
                     operand.tensorsig, operand.dtype, "SphereEllProduct", operand,
                     coordsys, input_basis, output_basis, fa, la,
                     ell_r_func, false,
                     [false, true], [false, false],
                     nothing, nothing, false, 1)
end

function symbol(op::SphereEllProduct, spinindex_in, spinindex_out_val,
                spintotal_in, spintotal_out, local_ell, radius)
    return op.ell_r_func(local_ell, radius)
end

function spinindex_out(op::SphereEllProduct, spinindex_in)
    return (spinindex_in,)
end

function new_operand(op::SphereEllProduct, operand; kw...)
    return SphereEllProduct(operand, op.coordsys, op.ell_r_func; kw...)
end

function check_conditions(op::SphereEllProduct)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    # Require colatitude axis to be in coefficient space
    colat_axis = op.first_axis + 1
    return !arg.layout.grid_space[colat_axis]
end

function enforce_conditions(op::SphereEllProduct)
    arg = op.args[1]
    if isa(arg, Field)
        colat_axis = op.first_axis + 1
        require_coeff_space!(arg, colat_axis)
    end
end

# ============================================================================
# Group 5: Sphere differential operators (SphereGradient, SphereDivergence, SphereLaplacian)
# ============================================================================

# --------------------------------------------------------------------------
# SphereBasis spin coupling coefficient k(ell, s, mu)
# --------------------------------------------------------------------------

"""
    sphere_basis_k(ell, s, mu)

Spin coupling coefficient for sphere operators.
Translated from Python `SphereBasis.k(l, s, mu)`:
    k(l, s, mu) = -mu * sqrt(max(0, (l - mu*s) * (l + mu*s + 1) / 2))
"""
function sphere_basis_k(ell, s, mu)
    return -mu * sqrt(max(0, (ell - mu * s) * (ell + mu * s + 1) / 2))
end

# --------------------------------------------------------------------------
# SphereGradient
# --------------------------------------------------------------------------

"""
    SphereGradient <: SeparableSphereOperator

Gradient on S2 (the 2-sphere). A separable sphere operator whose symbol
is the spin-raising/lowering coefficient `k(ell, s, mu) / radius`.

Prepends a vector index (the S2 coordinate system) to the tensor signature.
"""
mutable struct SphereGradient <: SeparableSphereOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Any
    last_axis::Any
    complex_operator::Bool
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function SphereGradient(operand, coordsys; out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    dist = operand.dist
    input_basis = get_basis(operand.domain, coordsys)
    output_basis = input_basis  # Gradient stays on the same basis
    fa = first_axis(dist, input_basis)
    la = last_axis(dist, input_basis)
    new_tensorsig = (coordsys, operand.tensorsig...)
    SphereGradient(Any[operand], (operand,), out, dist, operand.domain,
                   new_tensorsig, operand.dtype, "Grad", operand,
                   coordsys, input_basis, output_basis, fa, la,
                   false,
                   [false, true], [false, false],
                   nothing, nothing, false, 1)
end

function spinindex_out(op::SphereGradient, spinindex_in)
    # Gradient prepends spin index 1 (minus) and 2 (plus) to each existing spin index
    return ((1, spinindex_in...), (2, spinindex_in...))
end

function symbol(op::SphereGradient, spinindex_in, spinindex_out_val,
                spintotal_in, spintotal_out, local_ell, radius)
    mu = spintotal_out - spintotal_in
    k_val = sphere_basis_k.(local_ell, spintotal_in, mu)
    # Zero out entries where |spintotal_in| > ell or |spintotal_out| > ell
    k_val = @. ifelse(abs(spintotal_in) > local_ell, zero(k_val), k_val)
    k_val = @. ifelse(abs(spintotal_out) > local_ell, zero(k_val), k_val)
    return k_val ./ radius
end

function new_operand(op::SphereGradient, operand; kw...)
    return SphereGradient(operand, op.coordsys; kw...)
end

function check_conditions(op::SphereGradient)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    colat_axis = op.first_axis + 1
    return !arg.layout.grid_space[colat_axis]
end

function enforce_conditions(op::SphereGradient)
    arg = op.args[1]
    if isa(arg, Field)
        colat_axis = op.first_axis + 1
        require_coeff_space!(arg, colat_axis)
    end
end

# --------------------------------------------------------------------------
# SphereDivergence
# --------------------------------------------------------------------------

"""
    SphereDivergence <: SeparableSphereOperator

Divergence on S2 (the 2-sphere). A separable sphere operator whose symbol
reuses `SphereGradient.symbol` (the gradient and divergence are dual).

Contracts the first tensor index.
"""
mutable struct SphereDivergence <: SeparableSphereOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Any
    last_axis::Any
    index::Int
    complex_operator::Bool
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function SphereDivergence(operand; index=1, out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    if index != 1
        error("SphereDivergence only implemented along index 1.")
    end
    coordsys = operand.tensorsig[index]
    dist = operand.dist
    input_basis = get_basis(operand.domain, coordsys)
    output_basis = input_basis  # Divergence stays on the same basis
    fa = first_axis(dist, input_basis)
    la = last_axis(dist, input_basis)
    new_tensorsig = (operand.tensorsig[1:index-1]..., operand.tensorsig[index+1:end]...)
    SphereDivergence(Any[operand], (operand,), out, dist, operand.domain,
                     new_tensorsig, operand.dtype, "Div", operand,
                     coordsys, input_basis, output_basis, fa, la,
                     index, false,
                     [false, true], [false, false],
                     nothing, nothing, false, 1)
end

function spinindex_out(op::SphereDivergence, spinindex_in)
    # Divergence contracts: spin index 1 (minus) or 2 (plus) in first position
    if spinindex_in[1] in (1, 2)
        return (spinindex_in[2:end],)
    else
        return ()
    end
end

function symbol(op::SphereDivergence, spinindex_in, spinindex_out_val,
                spintotal_in, spintotal_out, local_ell, radius)
    # Divergence symbol is the same as gradient symbol (duality)
    mu = spintotal_out - spintotal_in
    k_val = sphere_basis_k.(local_ell, spintotal_in, mu)
    k_val = @. ifelse(abs(spintotal_in) > local_ell, zero(k_val), k_val)
    k_val = @. ifelse(abs(spintotal_out) > local_ell, zero(k_val), k_val)
    return k_val ./ radius
end

function new_operand(op::SphereDivergence, operand; kw...)
    return SphereDivergence(operand; index=op.index, kw...)
end

function check_conditions(op::SphereDivergence)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    colat_axis = op.first_axis + 1
    return !arg.layout.grid_space[colat_axis]
end

function enforce_conditions(op::SphereDivergence)
    arg = op.args[1]
    if isa(arg, Field)
        colat_axis = op.first_axis + 1
        require_coeff_space!(arg, colat_axis)
    end
end

# --------------------------------------------------------------------------
# SphereLaplacian
# --------------------------------------------------------------------------

"""
    SphereLaplacian <: SeparableSphereOperator

Laplacian on S2 (the 2-sphere). A separable sphere operator whose symbol
is the composition of eth and eth-bar: k_lap / radius^2.

Preserves the tensor signature.
"""
mutable struct SphereLaplacian <: SeparableSphereOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Any
    last_axis::Any
    complex_operator::Bool
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function SphereLaplacian(operand, coordsys; out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    dist = operand.dist
    input_basis = get_basis(operand.domain, coordsys)
    output_basis = input_basis  # Laplacian stays on the same basis
    fa = first_axis(dist, input_basis)
    la = last_axis(dist, input_basis)
    SphereLaplacian(Any[operand], (operand,), out, dist, operand.domain,
                    operand.tensorsig, operand.dtype, "Lap", operand,
                    coordsys, input_basis, output_basis, fa, la,
                    false,
                    [false, true], [false, false],
                    nothing, nothing, false, 1)
end

function spinindex_out(op::SphereLaplacian, spinindex_in)
    # Laplacian preserves spin indices
    return (spinindex_in,)
end

function symbol(op::SphereLaplacian, spinindex_in, spinindex_out_val,
                spintotal_in, spintotal_out, local_ell, radius)
    # Laplacian = composition of eth-bar and eth:
    # k_lap = k(ell, s-1, +1) * k(ell, s, -1) + k(ell, s+1, -1) * k(ell, s, +1)
    kp = sphere_basis_k.(local_ell, spintotal_in, +1)   # spin-raise
    km = sphere_basis_k.(local_ell, spintotal_in, -1)   # spin-lower
    kp_1 = sphere_basis_k.(local_ell, spintotal_in - 1, +1)  # spin-raise from one step below
    km_1 = sphere_basis_k.(local_ell, spintotal_in + 1, -1)  # spin-lower from one step above
    k_lap = km_1 .* kp .+ kp_1 .* km
    # Zero out entries where |spintotal| > ell
    k_lap = @. ifelse(abs(spintotal_in) > local_ell, zero(k_lap), k_lap)
    k_lap = @. ifelse(abs(spintotal_out) > local_ell, zero(k_lap), k_lap)
    return k_lap ./ (radius^2)
end

function new_operand(op::SphereLaplacian, operand; kw...)
    return SphereLaplacian(operand, op.coordsys; kw...)
end

function check_conditions(op::SphereLaplacian)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    colat_axis = op.first_axis + 1
    return !arg.layout.grid_space[colat_axis]
end

function enforce_conditions(op::SphereLaplacian)
    arg = op.args[1]
    if isa(arg, Field)
        colat_axis = op.first_axis + 1
        require_coeff_space!(arg, colat_axis)
    end
end

# ============================================================================
# Group 6: 3D Spherical operators (SphericalEllOperator family)
# ============================================================================

# --------------------------------------------------------------------------
# SphericalEllOperator — abstract base for 3D spherical operators
# --------------------------------------------------------------------------

"""
    SphericalEllOperator <: SpectralOperator

Abstract base type for operators on 3D spherical bases (ShellBasis, BallBasis)
that depend on the spherical harmonic degree ell and couple only in the radial
direction. These are the 3D analogues of PolarMOperator.

`subaxis_dependence = [false, true, true]` — depends on ell and radial n.
`subaxis_coupling = [false, false, true]` — couples only in radial direction.

## Subtypes must implement
- `regindex_out(op, regindex_in)` — return tuple of valid output regularity indices
- `radial_matrix(op, regindex_in, regindex_out, ell)` — per-ell radial matrix
- `_output_basis(input_basis)` — determine output basis from input

## Subtypes must have fields
- `operand`, `input_basis`, `output_basis`, `first_axis`, `last_axis`
- `coordsys`, `radius_axis`
- `tensorsig`, `dtype`, `domain`, `dist`
- `args::Vector{Any}`
"""
abstract type SphericalEllOperator <: SpectralOperator end

"""
    regindex_out(op::SphericalEllOperator, regindex_in)

Return a tuple of valid output regularity indices for the given input regularity index.
Must be implemented by concrete SphericalEllOperator subtypes.
"""
function regindex_out(op::SphericalEllOperator, regindex_in)
    error("regindex_out not implemented for type $(typeof(op))")
end

"""
    radial_matrix(op::SphericalEllOperator, regindex_in, regindex_out, ell)

Return the radial matrix for the given regularity indices and spherical harmonic degree ell.
Must be implemented by concrete SphericalEllOperator subtypes.
"""
function radial_matrix(op::SphericalEllOperator, regindex_in, regindex_out, ell)
    error("radial_matrix not implemented for type $(typeof(op))")
end

"""
    _spherical_ell_get_radial_basis(op::SphericalEllOperator)

Get the radial basis for a SphericalEllOperator.
"""
function _spherical_ell_get_radial_basis(op::SphericalEllOperator)
    return get_radial_basis(op.input_basis)
end

"""
    _spherical_ell_get_S2_basis(op::SphericalEllOperator)

Get the S2 (sphere) basis for a SphericalEllOperator.
"""
function _spherical_ell_get_S2_basis(op::SphericalEllOperator)
    return S2_basis(op.input_basis)
end

"""
    operate(op::SphericalEllOperator, out)

Explicit evaluation of a 3D spherical operator.

Loops over regularity components and ell-maps, applying the per-ell radial matrix
to each (m, ell) slice of the operand data.
"""
function operate(op::SphericalEllOperator, out)::Nothing
    operand = op.args[1]
    if op.input_basis === nothing
        basis = op.output_basis
    else
        basis = op.input_basis
    end
    radial_basis = _spherical_ell_get_radial_basis(op)
    axis = last_axis(op.dist, radial_basis)
    # Set output layout
    preset_layout!(out, operand.layout)
    out.data .= 0
    # Apply operator
    R_in = regularity_classes(radial_basis, operand.tensorsig)
    ndim = length(size(operand.data)) - length(operand.tensorsig)
    for idx_in in CartesianIndices(size(R_in))
        regindex_in = Tuple(idx_in)
        for regindex_out_val in regindex_out(op, regindex_in)
            comp_in = operand.data[regindex_in...]
            comp_out = out.data[regindex_out_val...]
            for (ell, m_ind, ell_ind) in ell_maps(basis, op.dist)
                allowed_in = regularity_allowed(radial_basis, ell, regindex_in)
                allowed_out = regularity_allowed(radial_basis, ell, regindex_out_val)
                if allowed_in && allowed_out
                    n_sl = n_slice(radial_basis, ell)
                    # Build slice tuple: axis-2 gets m_ind, axis-1 gets ell_ind, axis gets n_slice
                    slices = ntuple(i -> i == (axis - 2) ? m_ind :
                                        i == (axis - 1) ? ell_ind :
                                        i == axis ? n_sl : Colon(), ndim)
                    vec_in = view(comp_in, slices...)
                    vec_out = view(comp_out, slices...)
                    if length(vec_in) > 0 && length(vec_out) > 0
                        A = radial_matrix(op, regindex_in, regindex_out_val, ell)
                        vec_out .+= apply_matrix(A, collect(vec_in), axis)
                    end
                end
            end
        end
    end
    return nothing
end

"""
    subproblem_matrix(op::SphericalEllOperator, subproblem)

Build the operator matrix for a specific subproblem.

Constructs a block matrix over regularity components, where each nonzero block
is built from Kronecker products of identity matrices with the per-ell
radial matrix substituted at the radial axis.
"""
function subproblem_matrix(op::SphericalEllOperator, subproblem)
    operand = op.args[1]
    radial_basis = _spherical_ell_get_radial_basis(op)
    R_in = regularity_classes(radial_basis, operand.tensorsig)
    R_out = regularity_classes(radial_basis, op.tensorsig)
    # subproblem group indices: last_axis-2 is m, last_axis-1 is ell (1-based)
    m_val = subproblem.group[op.last_axis - 2]
    ell_val = subproblem.group[op.last_axis - 1]
    # Shortcut if empty
    size_in = field_size(subproblem, op.operand)
    size_out = field_size(subproblem, op)
    if size_in == 0 || size_out == 0
        return spzeros(size_out, size_in)
    end
    # Build identity matrices for each axis
    subshape_in = coeff_shape(subproblem, operand.domain)
    subshape_out = coeff_shape(subproblem, op.domain)
    factors_template = [sparse(1.0I, subshape_out[i], subshape_in[i]) for i in eachindex(subshape_out)]
    if ell_val === nothing
        factors_template[op.last_axis - 1] = sparse(1.0I, 1, 1)
    end
    # Assemble block matrix over components
    zero_block = spzeros(prod(subshape_out), prod(subshape_in))
    submatrices = []
    for idx_out in CartesianIndices(size(R_out))
        regindex_out_val = Tuple(idx_out)
        block_row = []
        for idx_in in CartesianIndices(size(R_in))
            regindex_in = Tuple(idx_in)
            if ell_val === nothing
                matrix = _coupled_ell_matrices(op, regindex_in, regindex_out_val, m_val)
            else
                matrix = _wrap_radial_matrix(op, regindex_in, regindex_out_val, ell_val; return_zeros=false)
            end
            if matrix === nothing
                block = zero_block
            else
                factors = copy(factors_template)
                factors[op.last_axis] = matrix
                block = reduce(kron, factors; init=sparse(ones(1, 1)))
            end
            push!(block_row, block)
        end
        push!(submatrices, block_row)
    end
    # Assemble block matrix
    nrows = length(submatrices)
    ncols = length(submatrices[1])
    block_mat = hvcat(
        ntuple(_ -> ncols, nrows),
        [submatrices[i][j] for i in 1:nrows for j in 1:ncols]...
    )
    return sparse(block_mat)
end

"""
    _coupled_ell_matrices(op::SphericalEllOperator, regindex_in, regindex_out, m_val)

Build a block-diagonal matrix over ells for a given m value.
Used when ell is not specified (ell === nothing) in the subproblem group.
"""
function _coupled_ell_matrices(op::SphericalEllOperator, regindex_in, regindex_out_val, m_val)
    basis = _spherical_ell_get_S2_basis(op)
    ell_list = collect(abs(m_val):basis.Lmax)
    ell_rev = ell_reversed(basis, op.dist)
    if haskey(ell_rev, m_val) && ell_rev[m_val]
        reverse!(ell_list)
    end
    ell_matrices = [_wrap_radial_matrix(op, regindex_in, regindex_out_val, ell; return_zeros=true) for ell in ell_list]
    return sparse_block_diag(ell_matrices)
end

"""
    _wrap_radial_matrix(op::SphericalEllOperator, regindex_in, regindex_out, ell; return_zeros=false)

Get the radial matrix for a given (regindex_in, regindex_out, ell) triple,
or return a zero matrix if the regularity is not allowed.
"""
function _wrap_radial_matrix(op::SphericalEllOperator, regindex_in, regindex_out_val, ell; return_zeros=false)
    radial_basis = _spherical_ell_get_radial_basis(op)
    if (regindex_out_val in regindex_out(op, regindex_in)) &&
       regularity_allowed(radial_basis, ell, regindex_in) &&
       regularity_allowed(radial_basis, ell, regindex_out_val)
        return radial_matrix(op, regindex_in, regindex_out_val, ell)
    elseif return_zeros
        if basis_dim(op.input_basis) == 2
            n_in = 1
        else
            n_in = n_size(radial_basis, ell)
        end
        if basis_dim(op.output_basis) == 2
            n_out = 1
        else
            n_out = n_size(radial_basis, ell)
        end
        return spzeros(n_out, n_in)
    else
        return nothing
    end
end

# --------------------------------------------------------------------------
# SphericalGradient — gradient on 3D spherical domains
# --------------------------------------------------------------------------

"""
    SphericalGradient <: SphericalEllOperator

Gradient on 3D spherical domains (Ball, Shell). Prepends a vector index
(the SphericalCoordinates) to the tensor signature.

Uses regularity-based radial matrices with D+/D- operators and xi coupling.
"""
mutable struct SphericalGradient <: SphericalEllOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Int
    last_axis::Int
    radius_axis::Int
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
    _radial_matrix_cache::Dict{Any,Any}
end

function SphericalGradient(operand, coordsys; out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    dist = operand.dist
    input_basis = get_basis(operand.domain, coordsys)
    if input_basis === nothing
        input_basis = get_basis(operand.domain, coordsys.radius)
    end
    output_basis = derivative_basis(input_basis; order=1)
    fa = first_axis(dist, input_basis)
    la = last_axis(dist, input_basis)
    ra = get_axis(dist, coordsys) + 2  # radius is 3rd coordinate (1-based + 2)
    new_domain = substitute_basis(operand.domain, input_basis, output_basis)
    new_tensorsig = (coordsys, operand.tensorsig...)
    SphericalGradient(Any[operand], (operand,), out, dist, new_domain,
                      new_tensorsig, operand.dtype, "Grad", operand,
                      coordsys, input_basis, output_basis, fa, la, ra,
                      [false, true, true], [false, false, true],
                      nothing, nothing, false, 1, Dict{Any,Any}())
end

function regindex_out(op::SphericalGradient, regindex_in)
    # Regorder: -, +, 0 -> indices 1, 2, 3
    # Gradient hits - (1) and + (2), not 0 (3)
    return ((1, regindex_in...), (2, regindex_in...))
end

function radial_matrix(op::SphericalGradient, regindex_in, regindex_out_val, ell)
    cache_key = (regindex_in, regindex_out_val, ell)
    cached = get(op._radial_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    radial_basis = _spherical_ell_get_radial_basis(op)
    regtotal_val = regtotal(regindex_in)
    # Python: regindex_out[0] != 2 (0-based) -> Julia: regindex_out_val[1] != 3 (1-based)
    if regindex_out_val[1] != 3 && regindex_in == regindex_out_val[2:end]
        result = _spherical_gradient_radial_matrix(radial_basis, regindex_out_val[1], regtotal_val, ell)
    else
        error("SphericalGradient: invalid regindex_in/regindex_out_val combination")
    end
    op._radial_matrix_cache[cache_key] = result
    return result
end

function _spherical_gradient_radial_matrix(radial_basis, regindex_out0, regtotal_val, ell)
    # Python: regindex_out0 == 0 -> Julia: regindex_out0 == 1 (minus component)
    if regindex_out0 == 1
        return xi(radial_basis, -1, ell + regtotal_val) * operator_matrix(radial_basis, "D-", ell, regtotal_val)
    # Python: regindex_out0 == 1 -> Julia: regindex_out0 == 2 (plus component)
    elseif regindex_out0 == 2
        return xi(radial_basis, +1, ell + regtotal_val) * operator_matrix(radial_basis, "D+", ell, regtotal_val)
    else
        error("SphericalGradient: invalid regindex_out0 = $regindex_out0")
    end
end

function check_conditions(op::SphericalGradient)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.radius_axis] && arg.layout.local[op.radius_axis]
end

function enforce_conditions(op::SphericalGradient)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.radius_axis)
        require_local!(arg, op.radius_axis)
    end
end

function new_operand(op::SphericalGradient, operand; kw...)
    return SphericalGradient(operand, op.coordsys; kw...)
end

# --------------------------------------------------------------------------
# SphericalDivergence — divergence on 3D spherical domains
# --------------------------------------------------------------------------

"""
    SphericalDivergence <: SphericalEllOperator

Divergence on 3D spherical domains (Ball, Shell). Contracts the first
tensor index (SphericalCoordinates) from the tensor signature.

Uses regularity-based radial matrices with D+/D- operators and xi coupling.
"""
mutable struct SphericalDivergence <: SphericalEllOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Int
    last_axis::Int
    radius_axis::Int
    index::Int
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
    _radial_matrix_cache::Dict{Any,Any}
end

function SphericalDivergence(operand; index=1, out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    if index != 1
        error("SphericalDivergence only implemented along index 1.")
    end
    coordsys = operand.tensorsig[index]
    dist = operand.dist
    input_basis = get_basis(operand.domain, coordsys)
    if input_basis === nothing
        input_basis = get_basis(operand.domain, coordsys.radius)
    end
    output_basis = derivative_basis(input_basis; order=1)
    fa = first_axis(dist, input_basis)
    la = last_axis(dist, input_basis)
    ra = get_axis(dist, coordsys) + 2
    new_domain = substitute_basis(operand.domain, input_basis, output_basis)
    new_tensorsig = (operand.tensorsig[1:index-1]..., operand.tensorsig[index+1:end]...)
    SphericalDivergence(Any[operand], (operand,), out, dist, new_domain,
                        new_tensorsig, operand.dtype, "Div", operand,
                        coordsys, input_basis, output_basis, fa, la, ra,
                        index,
                        [false, true, true], [false, false, true],
                        nothing, nothing, false, 1, Dict{Any,Any}())
end

function regindex_out(op::SphericalDivergence, regindex_in)
    # Regorder: -, +, 0 -> indices 1, 2, 3
    # Divergence feels - (1) and + (2), not 0 (3)
    if regindex_in[1] in (1, 2)
        return (regindex_in[2:end],)
    else
        return ()
    end
end

function radial_matrix(op::SphericalDivergence, regindex_in, regindex_out_val, ell)
    cache_key = (regindex_in, regindex_out_val, ell)
    cached = get(op._radial_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    radial_basis = _spherical_ell_get_radial_basis(op)
    regtotal_val = regtotal(regindex_in)
    # Python: regindex_in[0] != 2 (0-based) -> Julia: regindex_in[1] != 3 (1-based)
    if regindex_in[1] != 3 && regindex_in[2:end] == regindex_out_val
        result = _spherical_divergence_radial_matrix(radial_basis, regindex_in[1], regtotal_val, ell)
    else
        error("SphericalDivergence: invalid regindex_in/regindex_out_val combination")
    end
    op._radial_matrix_cache[cache_key] = result
    return result
end

function _spherical_divergence_radial_matrix(radial_basis, regindex_in0, regtotal_val, ell)
    # Python: regindex_in0 == 0 -> Julia: regindex_in0 == 1 (minus component)
    if regindex_in0 == 1
        return xi(radial_basis, -1, ell + regtotal_val + 1) * operator_matrix(radial_basis, "D+", ell, regtotal_val)
    # Python: regindex_in0 == 1 -> Julia: regindex_in0 == 2 (plus component)
    elseif regindex_in0 == 2
        return xi(radial_basis, +1, ell + regtotal_val - 1) * operator_matrix(radial_basis, "D-", ell, regtotal_val)
    else
        error("SphericalDivergence: invalid regindex_in0 = $regindex_in0")
    end
end

function check_conditions(op::SphericalDivergence)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.radius_axis] && arg.layout.local[op.radius_axis]
end

function enforce_conditions(op::SphericalDivergence)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.radius_axis)
        require_local!(arg, op.radius_axis)
    end
end

function new_operand(op::SphericalDivergence, operand; kw...)
    return SphericalDivergence(operand; index=op.index, kw...)
end

# --------------------------------------------------------------------------
# SphericalCurl — curl on 3D spherical domains
# --------------------------------------------------------------------------

"""
    SphericalCurl <: SphericalEllOperator

Curl on 3D spherical domains (Ball, Shell). For vector fields, produces a
vector output. The tensor index is contracted and a new one is prepended.

Uses regularity-based radial matrices with D+/D- operators, xi coupling,
and imaginary factors for cross-product structure.

Has special handling for real (Float64) data types where the imaginary
coupling must be encoded in the matrix structure.
"""
mutable struct SphericalCurl <: SphericalEllOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Int
    last_axis::Int
    radius_axis::Int
    index::Int
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
    _radial_matrix_cache::Dict{Any,Any}
end

function SphericalCurl(operand; index=1, out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    if index != 1
        error("SphericalCurl only implemented along index 1.")
    end
    coordsys = operand.tensorsig[index]
    dist = operand.dist
    input_basis = get_basis(operand.domain, coordsys)
    if input_basis === nothing
        input_basis = get_basis(operand.domain, coordsys.radius)
    end
    output_basis = derivative_basis(input_basis; order=1)
    fa = first_axis(dist, input_basis)
    la = last_axis(dist, input_basis)
    ra = get_axis(dist, coordsys) + 2
    new_domain = substitute_basis(operand.domain, input_basis, output_basis)
    # Curl: contracts index `index` and prepends coordsys
    new_tensorsig = (coordsys, operand.tensorsig[1:index-1]..., operand.tensorsig[index+1:end]...)
    SphericalCurl(Any[operand], (operand,), out, dist, new_domain,
                  new_tensorsig, operand.dtype, "Curl", operand,
                  coordsys, input_basis, output_basis, fa, la, ra,
                  index,
                  [false, true, true], [false, false, true],
                  nothing, nothing, false, 1, Dict{Any,Any}())
end

function regindex_out(op::SphericalCurl, regindex_in)
    # Regorder: -, +, 0 -> indices 1, 2, 3
    # - (1) and + (2) map to 0 (3)
    if regindex_in[1] in (1, 2)
        return ((3, regindex_in[2:end]...),)
    # 0 (3) maps to - (1) and + (2)
    else
        return ((1, regindex_in[2:end]...), (2, regindex_in[2:end]...))
    end
end

function radial_matrix(op::SphericalCurl, regindex_in, regindex_out_val, ell)
    cache_key = (regindex_in, regindex_out_val, ell)
    cached = get(op._radial_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    radial_basis = _spherical_ell_get_radial_basis(op)
    regtotal_in = regtotal(regindex_in)
    regtotal_out = regtotal(regindex_out_val)
    if regindex_in[2:end] == regindex_out_val[2:end]
        result = _spherical_curl_radial_matrix(radial_basis, regindex_in[1], regindex_out_val[1],
                                                regtotal_in, regtotal_out, ell)
    else
        error("SphericalCurl: invalid regindex_in/regindex_out_val combination")
    end
    op._radial_matrix_cache[cache_key] = result
    return result
end

function _spherical_curl_radial_matrix(radial_basis, regindex_in0, regindex_out0,
                                        regtotal_in, regtotal_out, ell)
    # Python uses 0-based indices (0=-, 1=+, 2=0)
    # Julia uses 1-based indices (1=-, 2=+, 3=0)
    if regindex_in0 == 1 && regindex_out0 == 3     # - -> 0
        return -1im * xi(radial_basis, +1, ell + regtotal_in + 1) *
               operator_matrix(radial_basis, "D+", ell, regtotal_in)
    elseif regindex_in0 == 2 && regindex_out0 == 3  # + -> 0
        return 1im * xi(radial_basis, -1, ell + regtotal_in - 1) *
               operator_matrix(radial_basis, "D-", ell, regtotal_in)
    elseif regindex_in0 == 3 && regindex_out0 == 1  # 0 -> -
        return -1im * xi(radial_basis, +1, ell + regtotal_in) *
               operator_matrix(radial_basis, "D-", ell, regtotal_in)
    elseif regindex_in0 == 3 && regindex_out0 == 2  # 0 -> +
        return 1im * xi(radial_basis, -1, ell + regtotal_in) *
               operator_matrix(radial_basis, "D+", ell, regtotal_in)
    else
        error("SphericalCurl: invalid regindex_in0=$regindex_in0, regindex_out0=$regindex_out0")
    end
end

function check_conditions(op::SphericalCurl)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.radius_axis] && arg.layout.local[op.radius_axis]
end

function enforce_conditions(op::SphericalCurl)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.radius_axis)
        require_local!(arg, op.radius_axis)
    end
end

function new_operand(op::SphericalCurl, operand; kw...)
    return SphericalCurl(operand; index=op.index, kw...)
end

"""
    subproblem_matrix(op::SphericalCurl, subproblem)

Specialized subproblem matrix for SphericalCurl. For complex data types,
delegates to the generic SphericalEllOperator method. For real data types
(Float64), encodes the imaginary coupling using the 2x2 block structure
of the real-valued m Fourier modes.
"""
function subproblem_matrix(op::SphericalCurl, subproblem)
    if op.dtype == ComplexF64
        # Use generic SphericalEllOperator subproblem_matrix
        return invoke(subproblem_matrix, Tuple{SphericalEllOperator, typeof(subproblem)}, op, subproblem)
    end
    # Real dtype (Float64) — special handling for imaginary factors
    operand = op.args[1]
    radial_basis = _spherical_ell_get_radial_basis(op)
    R_in = regularity_classes(radial_basis, operand.tensorsig)
    R_out = regularity_classes(radial_basis, op.tensorsig)
    ell_val = subproblem.group[op.last_axis - 1]
    # Loop over components
    submatrices = []
    for idx_out in CartesianIndices(size(R_out))
        regindex_out_val = Tuple(idx_out)
        submatrix_row = []
        for idx_in in CartesianIndices(size(R_in))
            regindex_in = Tuple(idx_in)
            # Build identity matrices for each axis
            subshape_in = coeff_shape(subproblem, operand.domain)
            subshape_out = coeff_shape(subproblem, op.domain)
            if (regindex_out_val in regindex_out(op, regindex_in)) &&
               regularity_allowed(radial_basis, ell_val, regindex_in) &&
               regularity_allowed(radial_basis, ell_val, regindex_out_val)
                factors = [sparse(1.0I, subshape_out[i], subshape_in[i]) for i in eachindex(subshape_out)]
                rad_matrix = radial_matrix(op, regindex_in, regindex_out_val, ell_val)
                # Real part
                factors[op.last_axis] = sparse(real(rad_matrix))
                comp_matrix_real = reduce(kron, factors; init=sparse(ones(1, 1)))
                # Imaginary part — encode via 2x2 block structure [[0,-1],[1,0]]
                m_size = subshape_in[op.first_axis]
                mult_1j = [0.0 -1.0; 1.0 0.0]
                m_blocks = sparse(1.0I, fld(m_size, 2), fld(m_size, 2))
                factors[op.first_axis] = kron(sparse(mult_1j), m_blocks)
                factors[op.last_axis] = sparse(imag(rad_matrix))
                comp_matrix_imag = reduce(kron, factors; init=sparse(ones(1, 1)))
                comp_matrix = comp_matrix_real + comp_matrix_imag
            else
                comp_matrix = spzeros(prod(subshape_out), prod(subshape_in))
            end
            push!(submatrix_row, comp_matrix)
        end
        push!(submatrices, submatrix_row)
    end
    # Assemble block matrix
    nrows = length(submatrices)
    ncols = length(submatrices[1])
    matrix = hvcat(
        ntuple(_ -> ncols, nrows),
        [submatrices[i][j] for i in 1:nrows for j in 1:ncols]...
    )
    return sparse(matrix)
end

"""
    operate(op::SphericalCurl, out)

Explicit evaluation of SphericalCurl. For complex data types, uses the
generic SphericalEllOperator evaluation. For real data types (Float64),
encodes the imaginary coupling by splitting into cos/sin components.
"""
function operate(op::SphericalCurl, out)::Nothing
    if op.dtype == ComplexF64
        invoke(operate, Tuple{SphericalEllOperator, typeof(out)}, op, out)
        return nothing
    end
    # Real dtype (Float64) — special handling
    operand = op.args[1]
    input_basis = op.input_basis
    radial_basis = _spherical_ell_get_radial_basis(op)
    axis = last_axis(op.dist, radial_basis)
    # Set output layout
    preset_layout!(out, operand.layout)
    out.data .= 0
    # Apply operator
    R_in = regularity_classes(radial_basis, operand.tensorsig)
    ndim = length(size(operand.data)) - length(operand.tensorsig)
    for idx_in in CartesianIndices(size(R_in))
        regindex_in = Tuple(idx_in)
        for regindex_out_val in regindex_out(op, regindex_in)
            comp_in = operand.data[regindex_in...]
            comp_out = out.data[regindex_out_val...]
            for (ell, m_ind, ell_ind) in ell_maps(input_basis, op.dist)
                allowed_in = regularity_allowed(radial_basis, ell, regindex_in)
                allowed_out = regularity_allowed(radial_basis, ell, regindex_out_val)
                if allowed_in && allowed_out
                    n_sl = n_slice(radial_basis, ell)
                    slices = ntuple(i -> i == (axis - 2) ? m_ind :
                                        i == (axis - 1) ? ell_ind :
                                        i == axis ? n_sl : Colon(), ndim)
                    # Extract cos and -sin components (even/odd m indices)
                    cos_sl = axslice(axis - 2, 1, size(comp_in, axis - 2), 2)
                    msin_sl = axslice(axis - 2, 2, size(comp_in, axis - 2), 2)
                    vec_in_data = comp_in[slices...]
                    vec_in_cos = vec_in_data[cos_sl...]
                    vec_in_msin = vec_in_data[msin_sl...]
                    vec_in_complex = vec_in_cos .+ 1im .* vec_in_msin
                    A = radial_matrix(op, regindex_in, regindex_out_val, ell)
                    vec_out_complex = apply_matrix(A, vec_in_complex, axis)
                    comp_out_view = view(comp_out, slices...)
                    cos_view = view(comp_out_view, cos_sl...)
                    msin_view = view(comp_out_view, msin_sl...)
                    cos_view .+= real.(vec_out_complex)
                    msin_view .+= imag.(vec_out_complex)
                end
            end
        end
    end
    return nothing
end

# --------------------------------------------------------------------------
# SphericalLaplacian — Laplacian on 3D spherical domains
# --------------------------------------------------------------------------

"""
    SphericalLaplacian <: SphericalEllOperator

Laplacian on 3D spherical domains (Ball, Shell). Preserves the tensor
signature. Uses derivative_basis(2) for the output basis.

Uses regularity-based radial matrices with the 'L' (Laplacian) operator.
"""
mutable struct SphericalLaplacian <: SphericalEllOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Int
    last_axis::Int
    radius_axis::Int
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
    _radial_matrix_cache::Dict{Any,Any}
end

function SphericalLaplacian(operand, coordsys; out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    dist = operand.dist
    input_basis = get_basis(operand.domain, coordsys)
    if input_basis === nothing
        input_basis = get_basis(operand.domain, coordsys.radius)
    end
    output_basis = derivative_basis(input_basis; order=2)
    fa = first_axis(dist, input_basis)
    la = last_axis(dist, input_basis)
    ra = get_axis(dist, coordsys) + 2
    new_domain = substitute_basis(operand.domain, input_basis, output_basis)
    SphericalLaplacian(Any[operand], (operand,), out, dist, new_domain,
                       operand.tensorsig, operand.dtype, "Lap", operand,
                       coordsys, input_basis, output_basis, fa, la, ra,
                       [false, true, true], [false, false, true],
                       nothing, nothing, false, 1, Dict{Any,Any}())
end

function regindex_out(op::SphericalLaplacian, regindex_in)
    # Laplacian preserves regularity indices
    return (regindex_in,)
end

function radial_matrix(op::SphericalLaplacian, regindex_in, regindex_out_val, ell)
    cache_key = (regindex_in, regindex_out_val, ell)
    cached = get(op._radial_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    radial_basis = _spherical_ell_get_radial_basis(op)
    regtotal_val = regtotal(regindex_in)
    if regindex_in == regindex_out_val
        result = operator_matrix(radial_basis, "L", ell, regtotal_val)
    else
        error("SphericalLaplacian: regindex_in must equal regindex_out_val")
    end
    op._radial_matrix_cache[cache_key] = result
    return result
end

function check_conditions(op::SphericalLaplacian)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.radius_axis] && arg.layout.local[op.radius_axis]
end

function enforce_conditions(op::SphericalLaplacian)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.radius_axis)
        require_local!(arg, op.radius_axis)
    end
end

function new_operand(op::SphericalLaplacian, operand; kw...)
    return SphericalLaplacian(operand, op.coordsys; kw...)
end

# --------------------------------------------------------------------------
# SphericalEllProduct — ell-dependent product on 3D spherical domains
# --------------------------------------------------------------------------

"""
    SphericalEllProduct <: SphericalEllOperator

Product with an ell-dependent scalar on 3D spherical domains (Ball, Shell).
The `ell_func(ell + regtotal)` callable defines the per-mode scalar multiplier.

Preserves the tensor signature and the basis. Uses the identity radial matrix
scaled by the ell-dependent function.
"""
mutable struct SphericalEllProduct <: SphericalEllOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    operand::Any
    coordsys::Any
    input_basis::Any
    output_basis::Any
    first_axis::Int
    last_axis::Int
    radius_axis::Int
    ell_func::Any
    subaxis_dependence::Vector{Bool}
    subaxis_coupling::Vector{Bool}
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
    _radial_matrix_cache::Dict{Any,Any}
end

function SphericalEllProduct(operand, coordsys, ell_func; out=nothing)
    if isa(operand, Number) || operand == 0
        return 0
    end
    dist = operand.dist
    input_basis = get_basis(operand.domain, coordsys)
    if input_basis === nothing
        input_basis = get_basis(operand.domain, coordsys.radius)
    end
    output_basis = input_basis  # EllProduct preserves the basis
    fa = first_axis(dist, input_basis)
    la = last_axis(dist, input_basis)
    ra = get_axis(dist, coordsys) + 2
    SphericalEllProduct(Any[operand], (operand,), out, dist, operand.domain,
                        operand.tensorsig, operand.dtype, "SphericalEllProduct", operand,
                        coordsys, input_basis, output_basis, fa, la, ra,
                        ell_func,
                        [false, true, true], [false, false, true],
                        nothing, nothing, false, 1, Dict{Any,Any}())
end

function regindex_out(op::SphericalEllProduct, regindex_in)
    # EllProduct preserves regularity indices
    return (regindex_in,)
end

function radial_matrix(op::SphericalEllProduct, regindex_in, regindex_out_val, ell)
    cache_key = (regindex_in, regindex_out_val, ell)
    cached = get(op._radial_matrix_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    radial_basis = _spherical_ell_get_radial_basis(op)
    regtotal_val = regtotal(regindex_in)
    if regindex_in == regindex_out_val
        result = op.ell_func(ell + regtotal_val) * operator_matrix(radial_basis, "Id", ell, regtotal_val)
    else
        error("SphericalEllProduct: regindex_in must equal regindex_out_val")
    end
    op._radial_matrix_cache[cache_key] = result
    return result
end

function check_conditions(op::SphericalEllProduct)
    arg = op.args[1]
    if !isa(arg, AbstractCurrent)
        return false
    end
    return !arg.layout.grid_space[op.radius_axis] && arg.layout.local[op.radius_axis]
end

function enforce_conditions(op::SphericalEllProduct)
    arg = op.args[1]
    if isa(arg, Field)
        require_coeff_space!(arg, op.radius_axis)
        require_local!(arg, op.radius_axis)
    end
end

function new_operand(op::SphericalEllProduct, operand; kw...)
    return SphericalEllProduct(operand, op.coordsys, op.ell_func; kw...)
end

# ============================================================================
# Display methods for new operator types
# ============================================================================

for T in [SphericalTrace, PolarTrace, DirectProductTrace,
          StandardTransposeComponents, SphericalTransposeComponents,
          CartesianSkew, SpinSkew,
          RadialComponent, AngularComponent, AzimuthalComponent,
          MulCosine, PolarGradient, PolarDivergence, PolarLaplacian,
          SphereEllProduct,
          SphereGradient, SphereDivergence, SphereLaplacian,
          SphericalGradient, SphericalDivergence, SphericalCurl,
          SphericalLaplacian, SphericalEllProduct]
    @eval begin
        function Base.show(io::IO, op::$T)
            print(io, op.name, "(", join(map(string, op.args), ", "), ")")
        end
    end
end

# ============================================================================
# Helper stubs for functions that may not yet exist
# ============================================================================

# Stub for sphere_operator (delegates to dedalus_sphere)
function sphere_operator(name, dtype, Lmax, m, spintotal_val)
    error("sphere_operator not available — dedalus_sphere library not loaded")
end

# Stub for operator_matrix on radial basis
function operator_matrix(basis, name, m, spintotal_val)
    if hasproperty(basis, :operator_matrix)
        return basis.operator_matrix(name, m, spintotal_val)
    end
    # Fallback
    error("operator_matrix not yet implemented for basis type $(typeof(basis))")
end

# Stub for spintotal on radial basis
function spintotal(basis, tensorsig, spinindex)
    if hasproperty(basis, :spintotal)
        return basis.spintotal(tensorsig, spinindex)
    end
    error("spintotal not implemented for basis type $(typeof(basis))")
end

# Stub for derivative_basis with order argument
function derivative_basis(basis, order)
    if order == 1 && hasmethod(derivative_basis, Tuple{typeof(basis)})
        return derivative_basis(basis)
    end
    error("derivative_basis(order=$order) not implemented for basis type $(typeof(basis))")
end

# Stubs for spherical-specific functions
function get_radial_basis(basis)
    if hasproperty(basis, :radial_basis)
        return basis.radial_basis
    end
    error("get_radial_basis not implemented for basis type $(typeof(basis))")
end

function radial_recombinations(basis, tensorsig; ell_list=())
    if hasproperty(basis, :radial_recombinations)
        return basis.radial_recombinations(tensorsig; ell_list=ell_list)
    end
    error("radial_recombinations not implemented for basis type $(typeof(basis))")
end

function n_size(basis, ell)
    if hasproperty(basis, :n_size)
        return basis.n_size(ell)
    end
    error("n_size not implemented for basis type $(typeof(basis))")
end

function ell_reversed(basis, dist)
    if hasproperty(basis, :ell_reversed)
        return basis.ell_reversed(dist)
    end
    error("ell_reversed not implemented for basis type $(typeof(basis))")
end

function ell_maps(basis, dist)
    if hasproperty(basis, :ell_maps)
        return basis.ell_maps(dist)
    end
    error("ell_maps not implemented for basis type $(typeof(basis))")
end

function backward_regularity_recombination!(basis, tensorsig, axis, data; ell_maps=nothing)
    backward_regularity_recombination(basis, tensorsig, axis, data, ell_maps)
end

function forward_regularity_recombination!(basis, tensorsig, axis, data; ell_maps=nothing)
    forward_regularity_recombination(basis, tensorsig, axis, data; ell_maps=ell_maps)
end

function field_shape(subproblem, op)
    if hasproperty(subproblem, :field_shape)
        return subproblem.field_shape(op)
    end
    # Fallback: product of tensor dims and coeff shape
    tshape = Tuple(cs_dim(cs) for cs in op.tensorsig)
    cshape = coeff_shape(subproblem, op.domain)
    return (tshape..., cshape...)
end

function basis_dealias(domain)
    if hasproperty(domain, :dealias)
        return domain.dealias
    end
    return 1
end

function m_maps(basis, dist)
    if hasproperty(basis, :m_maps)
        return basis.m_maps(dist)
    end
    error("m_maps not implemented for basis type $(typeof(basis))")
end

function require_local!(field, axis)
    error("require_local! not yet implemented for $(typeof(field))")
end

function coeff_size(subproblem, domain)
    # Try as method on subproblem
    if hasproperty(subproblem, :coeff_size)
        return subproblem.coeff_size(domain)
    end
    shape = coeff_shape(subproblem, domain)
    return prod(shape; init=1)
end


const lap = laplacian
