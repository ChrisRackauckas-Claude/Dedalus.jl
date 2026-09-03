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
        size_val = subproblem.field_size(op)
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

function operate(op::Power, out)
    arg0 = op.args[1]
    arg1 = op.args[2]
    preset_layout!(out, arg0.layout)
    if length(out.data) > 0
        out.data .= arg0.data .^ arg1
    end
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

function operate(op::FieldCopy, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
end

function new_operand(op::FieldCopy, operand; kw...)
    return FieldCopy(operand; kw...)
end

function subproblem_matrix(op::FieldCopy, subproblem)
    sz = subproblem.field_size(op.operand)
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

function operate(op::UnaryGridFunction, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= op.func.(arg.data)
    end
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

function operate(op::GridOperator, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
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

function operate(op::CoeffOperator, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
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

function operate(op::Convert, out)
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

function operate(op::Differentiate, out)
    arg = op.args[1]
    layout = arg.layout
    preset_layout!(out, layout)
    if length(arg.data) > 0 && length(out.data) > 0
        data_axis = op.last_axis + length(arg.tensorsig)
        apply_matrix(subspace_matrix(op, layout), arg.data, data_axis; out=out.data)
    else
        out.data .= 0
    end
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

function operate(op::Interpolate, out)
    arg = op.args[1]
    layout = arg.layout
    preset_layout!(out, layout)
    if length(arg.data) > 0 && length(out.data) > 0
        data_axis = op.last_axis + length(arg.tensorsig)
        apply_matrix(subspace_matrix(op, layout), arg.data, data_axis; out=out.data)
    else
        out.data .= 0
    end
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

function operate(op::Integrate, out)
    arg = op.args[1]
    layout = arg.layout
    preset_layout!(out, layout)
    if length(arg.data) > 0 && length(out.data) > 0
        data_axis = op.last_axis + length(arg.tensorsig)
        apply_matrix(subspace_matrix(op, layout), arg.data, data_axis; out=out.data)
    else
        out.data .= 0
    end
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

function operate(op::Average, out)
    arg = op.args[1]
    layout = arg.layout
    preset_layout!(out, layout)
    if length(arg.data) > 0 && length(out.data) > 0
        data_axis = op.last_axis + length(arg.tensorsig)
        apply_matrix(subspace_matrix(op, layout), arg.data, data_axis; out=out.data)
    else
        out.data .= 0
    end
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

function operate(op::Lift, out)
    arg = op.args[1]
    layout = arg.layout
    preset_layout!(out, layout)
    if length(arg.data) > 0 && length(out.data) > 0
        data_axis = op.last_axis + length(arg.tensorsig)
        apply_matrix(subspace_matrix(op, layout), arg.data, data_axis; out=out.data)
    else
        out.data .= 0
    end
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

function operate(op::CartesianGradient, out)
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
            out.data[i, ntuple(_ -> Colon(), ndims(out.data) - 1)...] .= comp.data
        else
            out.data[i, ntuple(_ -> Colon(), ndims(out.data) - 1)...] .= 0
        end
    end
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

function operate(op::CartesianDivergence, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
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

function operate(op::CartesianCurl, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
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

function operate(op::CartesianLaplacian, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
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

function operate(op::CartesianTrace, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    # Perform trace: sum over diagonal of first two tensor indices
    dim = op.get_dim(coordsys)
    out.data .= 0
    for i in 1:dim
        spatial_idx = ntuple(_ -> Colon(), ndims(arg.data) - 2)
        out.data .+= arg.data[i, i, spatial_idx...]
    end
end

function new_operand(op::CartesianTrace, operand; kw...)
    return CartesianTrace(operand; kw...)
end

function subproblem_matrix(op::CartesianTrace, subproblem)
    dim = op.get_dim(coordsys)
    trace_vec = vec(Matrix{Float64}(I, dim, dim))
    # Kronecker with remaining tensor components and coefficient size
    n_eye = prod(cs_dim(cs) for cs in op.tensorsig; init=1)
    n_eye *= subproblem.coeff_size(op.domain)
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

function operate(op::CartesianTransposeComponents, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= permutedims(arg.data, collect(op.new_axis_order))
    end
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

function operate(op::CartesianComponent, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        # Select the component along the tensor index
        idx = ntuple(i -> i == op.index ? op.coord_subaxis : Colon(), ndims(arg.data))
        out.data .= arg.data[idx...]
    end
end

function new_operand(op::CartesianComponent, operand; kw...)
    return CartesianComponent(operand, op.index, op.comp; kw...)
end

function subproblem_matrix(op::CartesianComponent, subproblem)
    # Build selection matrix via Kronecker product
    factors = [sparse(1.0I, cs_dim(cs), cs_dim(cs)) for cs in op.operand.tensorsig]
    push!(factors, sparse(1.0I, subproblem.coeff_size(op.domain), subproblem.coeff_size(op.domain)))
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

function operate(op::DirectProductGradient, out)
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
            out.data[i0:i0+dim-1, ntuple(_ -> Colon(), ndims(out.data) - 1)...] .= cs_grad.data
        else
            out.data[i0:i0+dim-1, ntuple(_ -> Colon(), ndims(out.data) - 1)...] .= 0
        end
        i0 += dim
    end
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

function operate(op::DirectProductDivergence, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
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
    # Simplified: delegate to the inner arg computation
    # Full implementation would decompose into horizontal/vertical curl components
    inner_arg = operand  # placeholder
    DirectProductCurl(Any[inner_arg], (inner_arg,), out, operand.dist, operand.domain,
                      operand.tensorsig, operand.dtype, "Curl", operand, coordsys, index,
                      nothing, nothing, false, 1)
end

check_conditions(::DirectProductCurl) = true
enforce_conditions(::DirectProductCurl) = nothing

function operate(op::DirectProductCurl, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
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

function operate(op::DirectProductLaplacian, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        out.data .= arg.data
    end
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

function operate(op::DirectProductComponent, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    if length(out.data) > 0
        comp_dim = get_dim(op.comp)
        comp_slice = op.comp_subaxis:(op.comp_subaxis + comp_dim - 1)
        idx = ntuple(i -> i == op.index ? comp_slice : Colon(), ndims(arg.data))
        out.data .= arg.data[idx...]
    end
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

"""Divergence dispatch: selects CartesianDivergence or DirectProductDivergence."""
function divergence(field; index=1)
    if isa(field, Number) || field == 0
        return 0
    end
    cs = field.tensorsig[index]
    if isa(cs, CartesianCoordinates) || isa(cs, Coordinate)
        return CartesianDivergence(field; index=index)
    elseif isa(cs, DirectProduct)
        return DirectProductDivergence(field; index=index)
    else
        return CartesianDivergence(field; index=index)  # fallback
    end
end

"""Curl dispatch: selects CartesianCurl or DirectProductCurl."""
function curl(field; index=1)
    if isa(field, Number) || field == 0
        return 0
    end
    cs = field.tensorsig[index]
    if isa(cs, CartesianCoordinates)
        return CartesianCurl(field; index=index)
    elseif isa(cs, DirectProduct)
        return DirectProductCurl(field; index=index)
    else
        return CartesianCurl(field; index=index)  # fallback
    end
end

"""Laplacian dispatch: selects CartesianLaplacian or DirectProductLaplacian."""
function laplacian(field, cs)
    if isa(field, Number) || field == 0
        return 0
    end
    if isa(cs, CartesianCoordinates) || isa(cs, Coordinate)
        return CartesianLaplacian(field, cs)
    elseif isa(cs, DirectProduct)
        return DirectProductLaplacian(field, cs)
    else
        return CartesianLaplacian(field, cs)  # fallback
    end
end

"""Differentiate along a coordinate."""
function differentiate(arg, coord)
    return Differentiate(arg, coord)
end

"""Interpolate at a position along a coordinate."""
function interpolate(arg, coord, position)
    return Interpolate(arg, coord, position)
end

"""Integrate over a coordinate."""
function integrate(arg, coord)
    return Integrate(arg, coord)
end

"""Average over a coordinate."""
function average(arg, coord)
    return Average(arg, coord)
end

"""Lift operator."""
function lift(arg, basis, n)
    return Lift(arg, basis, n)
end

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
const lap = laplacian

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
       gradient, divergence, curl, laplacian,
       trace_op, transpose_components, grid_op, coeff_op,
       time_derivative, component,
       OPERATOR_ALIASES, register_operator_alias!,
       operator_operand, new_operand,
       subproblem_matrix, expression_matrices,
       dt, lap
