# Operators module for Dedalus.jl
# Translates the 1D, Cartesian, and DirectProduct portions of operators.py

using SparseArrays
using LinearAlgebra

# ============================================================================
# Abstract operator types
# ============================================================================

abstract type AbstractOperator <: AbstractFuture end
abstract type AbstractLinearOperator <: AbstractOperator end
abstract type SpectralOperator <: AbstractLinearOperator end
abstract type SpectralOperator1D <: SpectralOperator end
abstract type NonlinearOperator <: AbstractOperator end

# ============================================================================
# Power operator
# ============================================================================

mutable struct Power <: AbstractOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    exponent::Any
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
          "Power", exponent, nothing, nothing, false, 1, nothing, nothing)
end

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
    arg = op.args[1]
    exp = op.args[2]
    preset_layout!(out, arg.layout)
    out.data .= arg.data .^ exp
end

# ============================================================================
# FieldCopy
# ============================================================================

mutable struct FieldCopy <: AbstractLinearOperator
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
end

function FieldCopy(arg; out=nothing)
    FieldCopy(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
              arg.dtype, "FieldCopy", nothing, nothing, false, 1)
end

function check_conditions(op::FieldCopy)
    return true
end

function enforce_conditions(op::FieldCopy)
end

function operate(op::FieldCopy, out)
    arg = op.args[1]
    preset_layout!(out, arg.layout)
    out.data .= arg.data
end

# ============================================================================
# UnaryGridFunction
# ============================================================================

mutable struct UnaryGridFunction <: AbstractOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    func::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function UnaryGridFunction(func, arg; out=nothing)
    UnaryGridFunction(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
                      arg.dtype, string(func), func, nothing, nothing, false, 1)
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
    out.data .= op.func.(arg.data)
end

# ============================================================================
# TimeDerivative
# ============================================================================

mutable struct TimeDerivative <: AbstractLinearOperator
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
end

function TimeDerivative(arg; out=nothing)
    TimeDerivative(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
                   arg.dtype, "dt", nothing, nothing, false, 1)
end

function check_conditions(::TimeDerivative)
    return true
end

function enforce_conditions(::TimeDerivative)
end

function operate(op::TimeDerivative, out)
    error("TimeDerivative should not be evaluated directly — used symbolically by solvers")
end

# ============================================================================
# GridOperator / CoeffOperator
# ============================================================================

mutable struct GridOperator <: AbstractOperator
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
end

function GridOperator(arg; out=nothing)
    GridOperator(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
                 arg.dtype, "Grid", nothing, nothing, false, 1)
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
    out.data .= arg.data
end

mutable struct CoeffOperator <: AbstractOperator
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
end

function CoeffOperator(arg; out=nothing)
    CoeffOperator(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
                  arg.dtype, "Coeff", nothing, nothing, false, 1)
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
    out.data .= arg.data
end

# ============================================================================
# Convert
# ============================================================================

mutable struct Convert <: SpectralOperator1D
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    input_basis::Any
    output_basis::Any
    coord::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Convert(arg, output_basis; out=nothing)
    Convert(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
            arg.dtype, "Convert", nothing, output_basis, nothing,
            nothing, nothing, false, 1)
end

function convert_operand(arg, bases)
    return arg
end

# ============================================================================
# Differentiate
# ============================================================================

mutable struct Differentiate <: SpectralOperator1D
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    coord::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Differentiate(arg, coord; out=nothing)
    Differentiate(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
                  arg.dtype, "d$(coord.name)", coord, nothing, nothing, false, 1)
end

function differentiate(arg, coord)
    Differentiate(arg, coord)
end

# ============================================================================
# Interpolate
# ============================================================================

mutable struct Interpolate <: SpectralOperator1D
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    coord::Any
    position::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Interpolate(arg, coord, position; out=nothing)
    Interpolate(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
                arg.dtype, "Interpolate", coord, position, nothing, nothing, false, 1)
end

function interpolate(arg, args...; kw...)
    if length(args) == 2
        return Interpolate(arg, args[1], args[2])
    else
        return Interpolate(arg, args...)
    end
end

# ============================================================================
# Integrate
# ============================================================================

mutable struct Integrate <: SpectralOperator1D
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    coord::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Integrate(arg, coord; out=nothing)
    Integrate(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
              arg.dtype, "Integrate", coord, nothing, nothing, false, 1)
end

function Integrate(arg; out=nothing)
    Integrate(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
              arg.dtype, "Integrate", nothing, nothing, nothing, false, 1)
end

function integrate(arg, coord)
    Integrate(arg, coord)
end

# ============================================================================
# Average
# ============================================================================

mutable struct Average <: SpectralOperator1D
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    coord::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Average(arg, coord; out=nothing)
    Average(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
            arg.dtype, "Average", coord, nothing, nothing, false, 1)
end

function Average(arg; out=nothing)
    Average(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
            arg.dtype, "Average", nothing, nothing, nothing, false, 1)
end

function average(arg, coord)
    Average(arg, coord)
end

# ============================================================================
# Lift
# ============================================================================

mutable struct Lift <: SpectralOperator1D
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    basis::Any
    n::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function Lift(arg, basis, n; out=nothing)
    Lift(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
         arg.dtype, "Lift", basis, n, nothing, nothing, false, 1)
end

function lift(arg, basis, n)
    Lift(arg, basis, n)
end

# ============================================================================
# Cartesian operators
# ============================================================================

mutable struct CartesianGradient <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    cs::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianGradient(arg, cs; out=nothing)
    new_tensorsig = (cs, arg.tensorsig...)
    CartesianGradient(Any[arg], (arg,), out, arg.dist, arg.domain, new_tensorsig,
                      arg.dtype, "Grad", cs, nothing, nothing, false, 1)
end

mutable struct CartesianDivergence <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    cs::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianDivergence(arg, cs; out=nothing)
    new_tensorsig = arg.tensorsig[2:end]
    CartesianDivergence(Any[arg], (arg,), out, arg.dist, arg.domain, new_tensorsig,
                        arg.dtype, "Div", cs, nothing, nothing, false, 1)
end

mutable struct CartesianCurl <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    cs::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianCurl(arg, cs; out=nothing)
    CartesianCurl(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
                  arg.dtype, "Curl", cs, nothing, nothing, false, 1)
end

mutable struct CartesianLaplacian <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    cs::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianLaplacian(arg, cs; out=nothing)
    CartesianLaplacian(Any[arg], (arg,), out, arg.dist, arg.domain, arg.tensorsig,
                       arg.dtype, "Lap", cs, nothing, nothing, false, 1)
end

mutable struct CartesianTrace <: AbstractLinearOperator
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
end

function CartesianTrace(arg; out=nothing)
    new_tensorsig = arg.tensorsig[3:end]
    CartesianTrace(Any[arg], (arg,), out, arg.dist, arg.domain, new_tensorsig,
                   arg.dtype, "Trace", nothing, nothing, false, 1)
end

mutable struct CartesianTransposeComponents <: AbstractLinearOperator
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
end

function CartesianTransposeComponents(arg; out=nothing)
    ts = arg.tensorsig
    new_ts = length(ts) >= 2 ? (ts[2], ts[1], ts[3:end]...) : ts
    CartesianTransposeComponents(Any[arg], (arg,), out, arg.dist, arg.domain, new_ts,
                                 arg.dtype, "Trans", nothing, nothing, false, 1)
end

mutable struct CartesianComponent <: AbstractLinearOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    index::Int
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function CartesianComponent(arg, index; out=nothing)
    new_tensorsig = arg.tensorsig[2:end]
    CartesianComponent(Any[arg], (arg,), out, arg.dist, arg.domain, new_tensorsig,
                       arg.dtype, "Comp", index, nothing, nothing, false, 1)
end

# ============================================================================
# DirectProduct operators (delegate to per-coordinate Cartesian ops)
# ============================================================================

const DirectProductGradient = CartesianGradient
const DirectProductDivergence = CartesianDivergence
const DirectProductCurl = CartesianCurl
const DirectProductLaplacian = CartesianLaplacian

# ============================================================================
# Constructor dispatch functions (PUBLIC API)
# ============================================================================

gradient(field, cs::CartesianCoordinates) = CartesianGradient(field, cs)
gradient(field, cs::DirectProduct) = CartesianGradient(field, cs)
gradient(field, cs) = CartesianGradient(field, cs)

divergence(field, cs::CartesianCoordinates) = CartesianDivergence(field, cs)
divergence(field, cs::DirectProduct) = CartesianDivergence(field, cs)
divergence(field, cs) = CartesianDivergence(field, cs)

curl(field, cs::CartesianCoordinates) = CartesianCurl(field, cs)
curl(field, cs::DirectProduct) = CartesianCurl(field, cs)
curl(field, cs) = CartesianCurl(field, cs)

laplacian(field, cs::CartesianCoordinates) = CartesianLaplacian(field, cs)
laplacian(field, cs::DirectProduct) = CartesianLaplacian(field, cs)
laplacian(field, cs) = CartesianLaplacian(field, cs)

trace_op(field) = CartesianTrace(field)
transpose_components(field) = CartesianTransposeComponents(field)
grid_op(field) = GridOperator(field)
coeff_op(field) = CoeffOperator(field)
time_derivative(field) = TimeDerivative(field)

# Aliases for equation namespace
const dt = time_derivative
const lap = laplacian
const div = divergence
const grad = gradient

function component(field, index)
    CartesianComponent(field, index)
end

# ============================================================================
# AdvectiveCFL (needed by extras/flow_tools.jl)
# ============================================================================

mutable struct AdvectiveCFL <: AbstractOperator
    args::Vector{Any}
    original_args::Tuple
    out::Any
    dist::Any
    domain::Any
    tensorsig::Tuple
    dtype::DataType
    name::String
    velocity::Any
    cs::Any
    last_id::Any
    last_out::Any
    store_last::Bool
    scales::Any
end

function AdvectiveCFL(velocity, cs; out=nothing)
    AdvectiveCFL(Any[velocity], (velocity,), out, velocity.dist, velocity.domain,
                 (), velocity.dtype, "AdvCFL", velocity, cs, nothing, nothing, false, 1)
end

# ============================================================================
# Default methods for all operators
# ============================================================================

for T in [Power, FieldCopy, UnaryGridFunction, TimeDerivative, GridOperator,
          CoeffOperator, Convert, Differentiate, Interpolate, Integrate, Average,
          Lift, CartesianGradient, CartesianDivergence, CartesianCurl,
          CartesianLaplacian, CartesianTrace, CartesianTransposeComponents,
          CartesianComponent, AdvectiveCFL]
    @eval begin
        function Base.show(io::IO, op::$T)
            print(io, op.name, "(", join(map(string, op.args), ", "), ")")
        end
        Base.string(op::$T) = op.name
    end
end
