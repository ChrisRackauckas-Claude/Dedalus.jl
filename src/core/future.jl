"""
    Future types for Dedalus.jl

Julia translation of `dedalus/core/future.py`. Provides abstract and concrete
types for deferred (lazy) operations on fields. Futures form the internal nodes
of the Dedalus expression tree; when evaluated, they recursively evaluate their
argument sub-trees and then perform their own operation.

## Type hierarchy

    AbstractOperand (from field.jl)
    +-- AbstractFuture <: AbstractOperand
        +-- FutureField       (deferred ops producing a Field)
        +-- FutureLockedField (deferred ops producing a LockedField)

## Key translation choices

- Python `Future.__init__` -> `init_future!` helper called by concrete constructors.
- Python `store_last` class variable -> module-level constant + per-instance field.
- Python `CachedAttribute` for `bases` -> lazy evaluation via `_cache` dict.
- Python `isinstance(a, Field/Future)` -> `isa(a, Field)` / `isa(a, AbstractFuture)`.
- Python `self.reset()` -> `reset_future!`.
- Python `self.evaluate(id, force)` -> `evaluate_future(future; id, force)`.
"""

using SparseArrays

# ============================================================================
# Configuration
# ============================================================================

const STORE_OUTPUTS = get_config_bool("memory", "STORE_OUTPUTS")
const STORE_LAST_DEFAULT = get_config_bool("memory", "STORE_LAST_DEFAULT")

# ============================================================================
# AbstractFuture interface
# ============================================================================
#
# AbstractFuture is declared in field.jl. Here we define the concrete
# interface that all futures must implement.

# --- Common fields pattern ---
# Every concrete future subtype should have these fields:
#   args::Vector{Any}
#   original_args::Tuple
#   out::Union{Nothing, <concrete output type>}
#   dist::Any
#   _grid_layout::Any
#   _coeff_layout::Any
#   last_id::Any
#   last_out::Any
#   scales::Any
#   store_last::Bool
#   name::Union{Nothing,String}
#   tensorsig::Tuple
#   dtype::DataType
#   domain::Domain

# ============================================================================
# Initialization helper
# ============================================================================

"""
    init_future!(future, args...; out=nothing)

Common initialization for all AbstractFuture subtypes. Populates the
standard fields (`args`, `original_args`, `out`, `dist`, layout caches,
etc.).

The caller must have already set `future.name`, `future.tensorsig`,
`future.dtype`, and `future.domain` before calling this, as those are
determined by the specific operator logic.
"""
function init_future!(future::AbstractFuture, args...; out=nothing)
    future.args = collect(Any, args)
    future.original_args = Tuple(args)
    future.out = out
    future.dist = unify_attributes(
        [a for a in args if isa(a, AbstractOperand)], :dist; require=false)
    future._grid_layout = future.dist.grid_layout
    future._coeff_layout = future.dist.coeff_layout
    future.last_id = nothing
    future.last_out = nothing
    future.scales = 1
    future.store_last = STORE_LAST_DEFAULT
    return nothing
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, f::AbstractFuture)
    repr_args = join([repr(a) for a in f.args], ", ")
    print(io, f.name, "(", repr_args, ")")
end

function Base.string(f::AbstractFuture)
    str_args = join([string(a) for a in f.args], ", ")
    return "$(f.name)($(str_args))"
end

# ============================================================================
# Expression tree methods
# ============================================================================

"""
    reset_future!(future::AbstractFuture)

Restore original arguments (undo any in-place argument substitution from
evaluation).
"""
function reset_future!(future::AbstractFuture)
    future.args = collect(Any, future.original_args)
    return nothing
end

"""
    atoms(f::AbstractFuture, types...)

Recursively gather all leaf-operands of specified types.
"""
function atoms(f::AbstractFuture, types...)
    result = OrderedSet{Any}()
    for arg in f.args
        if isa(arg, AbstractOperand)
            union!(result, atoms(arg, types...))
        end
    end
    return result
end

"""
    has_operand(f::AbstractFuture, vars...)

Determine if the expression tree contains any specified operands/operators.
Checks whether the future itself is an instance of a type in `vars`, then
recursively checks arguments.
"""
function has_operand(f::AbstractFuture, vars...)
    # Check for matching operator type
    for v in vars
        if isa(v, DataType) && isa(f, v)
            return true
        end
    end
    # Check arguments recursively
    for arg in f.args
        if isa(arg, AbstractOperand) && has_operand(arg, vars...)
            return true
        end
    end
    return false
end

"""
    replace_operand(f::AbstractFuture, old, new_val)

Replace specified operand/operator in the expression tree. If the
entire expression matches `old`, return `new_val`. If `old` is a type
and the future is an instance of it, call `new_val` with replaced args.
Otherwise, rebuild with replaced arguments.
"""
function replace_operand(f::AbstractFuture, old, new_val)
    # Check for entire expression match
    if f == old
        return new_val
    end
    # Check for type-based replacement
    if isa(old, DataType) && isa(f, old)
        new_args = [isa(a, AbstractOperand) ? replace_operand(a, old, new_val) : a
                    for a in f.args]
        return new_val(new_args...)
    end
    # Rebuild with replaced arguments
    new_args = [isa(a, AbstractOperand) ? replace_operand(a, old, new_val) : a
                for a in f.args]
    return new_operands(f, new_args...)
end

"""
    replace_dict(f::AbstractFuture, subs::AbstractDict)

Replace specified operands/operators according to a dictionary.
"""
function replace_dict(f::AbstractFuture, subs::AbstractDict)
    # Check for entire expression match
    if f in keys(subs)
        return subs[f]
    end
    # Check for type-based replacement
    ft = typeof(f)
    if ft in keys(subs)
        new_args = [isa(a, AbstractOperand) ? replace_dict(a, subs) : a
                    for a in f.args]
        return subs[ft](new_args...)
    end
    # Rebuild with replaced arguments
    new_args = [isa(a, AbstractOperand) ? replace_dict(a, subs) : a
                for a in f.args]
    return new_operands(f, new_args...)
end

"""
    new_operands(f::AbstractFuture, args...)

Rebuild the future operator with new arguments. This should be overridden
by concrete subtypes to call the appropriate constructor. Default
implementation calls the type constructor.
"""
function new_operands(f::AbstractFuture, args...)
    return typeof(f)(args...)
end

"""
    prep_nccs(f::AbstractFuture, vars)

Recursively prepare non-constant coefficients.
"""
function prep_nccs(f::AbstractFuture, vars)
    for arg in f.args
        if isa(arg, AbstractOperand)
            prep_nccs(arg, vars)
        end
    end
    return nothing
end

"""
    gather_ncc_coeffs(f::AbstractFuture)

Recursively gather NCC coefficients.
"""
function gather_ncc_coeffs(f::AbstractFuture)
    for arg in f.args
        if isa(arg, AbstractOperand)
            gather_ncc_coeffs(arg)
        end
    end
    return nothing
end

# ============================================================================
# Evaluation
# ============================================================================

"""
    evaluate_future(f::AbstractFuture; id=nothing, force::Bool=true)

Recursively evaluate the operation.

1. Check storage for previously cached results (if `store_last` and `id`
   are set).
2. Recursively evaluate all operand arguments.
3. Check/enforce operator conditions.
4. Allocate output field.
5. Perform the operation.
6. Reset arguments.
7. Cache result if requested.
"""
function evaluate_future(f::AbstractFuture; id=nothing, force::Bool=true)::Union{AbstractCurrent, Nothing}
    # Check storage
    if f.store_last && id !== nothing
        if id == f.last_id
            return f.last_out
        else
            # Clear cache to free output field
            f.last_id = nothing
            f.last_out = nothing
        end
    end

    # Recursively attempt evaluation of all operator arguments
    all_eval = true
    for i in eachindex(f.args)
        a = f.args[i]
        if isa(a, Field)
            change_scales!(a, domain_dealias(a.domain))
        end
        if isa(a, AbstractFuture)
            a_eval = evaluate_future(a; id=id, force=force)
            if a_eval !== nothing
                f.args[i] = a_eval
            else
                all_eval = false
            end
        end
    end

    # Return nothing if any arguments are not evaluable
    if !all_eval
        return nothing
    end

    # Check conditions unless forcing evaluation
    if force
        enforce_conditions(f)
    else
        if !check_conditions(f)
            return nothing
        end
    end

    # Allocate output field if necessary
    out = get_out(f)

    # Copy metadata
    preset_scales!(out, domain_dealias(f.domain))

    # Perform operation
    operate(f, out)

    # Reset to free temporary field arguments
    reset_future!(f)

    # Update storage
    if f.store_last && id !== nothing
        f.last_id = id
        f.last_out = out
    end

    return out
end

"""
    get_out(f::AbstractFuture)

Get or create the output field for this future.
"""
function get_out(f::AbstractFuture)::AbstractCurrent
    if f.out !== nothing
        return f.out
    end
    out = build_out(f)
    if STORE_OUTPUTS
        f.out = out
    end
    return out
end

"""
    build_out(f::AbstractFuture)

Build a new output field for this future. Uses `future_type(f)` to
determine what kind of output to create.
"""
function build_out(f::AbstractFuture)::AbstractCurrent
    bases = f.domain.bases
    ft = future_type(f)
    if any(b !== nothing for b in bases)
        return ft(f.dist; bases=bases, tensorsig=f.tensorsig, dtype=f.dtype,
                  name=string(f))
    else
        return ft(f.dist; tensorsig=f.tensorsig, dtype=f.dtype, name=string(f))
    end
end

"""
    attempt(f::AbstractFuture; id=nothing)

Recursively attempt to evaluate operation (non-forcing).
"""
function attempt(f::AbstractFuture; id=nothing)::Union{AbstractCurrent, Nothing}
    return evaluate_future(f; id=id, force=false)
end

# --- Abstract methods that concrete subtypes must implement ---

"""
    check_conditions(f::AbstractFuture) -> Bool

Check that arguments are in a proper layout for the operation.
Must be implemented by concrete subtypes.
"""
function check_conditions end

"""
    enforce_conditions(f::AbstractFuture)

Require arguments to be in a proper layout for the operation.
Must be implemented by concrete subtypes.
"""
function enforce_conditions end

"""
    operate(f::AbstractFuture, out)

Perform the operation, writing results into `out`.
Must be implemented by concrete subtypes.
"""
function operate end

"""
    future_type(f::AbstractFuture) -> DataType

Return the output field type for this future (Field or LockedField).
Must be implemented by concrete subtypes.
"""
function future_type end

# ============================================================================
# FutureField
# ============================================================================

"""
    FutureField

Abstract type for deferred operations producing a `Field`.
Concrete operator types (Add, Multiply, Power, etc.) should subtype this.
"""
abstract type FutureField <: AbstractFuture end

"""
    future_type(::FutureField) -> Type{Field}

FutureField operations produce Field output.
"""
future_type(::FutureField) = Field

"""
    Base.getindex(f::FutureField, layout)

Evaluate the future and return data viewed in the specified layout.
"""
function Base.getindex(f::FutureField, layout)
    field = evaluate_future(f)
    return field[layout]
end

"""
    parse_future(str::AbstractString, namespace, dist)

Build a FutureField from a string expression.
"""
function parse_future(str::AbstractString, namespace, dist)
    expression = eval(Meta.parse(str))
    return cast_future(expression, dist)
end

"""
    cast_future(arg, dist)

Cast an object to a FutureField. Fields are wrapped in a FieldCopy operator;
existing FutureFields are returned as-is.
"""
function cast_future(arg, dist)
    # Cast to operand (checks dist)
    arg = operand_cast(arg, dist, (), dist.dtype)
    if isa(arg, FutureField)
        return arg
    else
        # Forward reference to operators.FieldCopy
        return _field_copy(arg)
    end
end

"""
    _field_copy(arg)

Placeholder for operators.FieldCopy. Will be overwritten when operators.jl
is loaded.
"""
function _field_copy end

# ============================================================================
# FutureLockedField
# ============================================================================

"""
    FutureLockedField

Abstract type for deferred operations producing a `LockedField`.
"""
abstract type FutureLockedField <: AbstractFuture end

"""
    future_type(::FutureLockedField) -> Type{LockedField}

FutureLockedField operations produce LockedField output.
"""
future_type(::FutureLockedField) = LockedField

# ============================================================================
# Exports
# ============================================================================

"""
    evaluate

Alias for `evaluate_future` — provided for convenience so that call sites
(tests, arithmetic helpers) can write `evaluate(expr)` instead of the
longer `evaluate_future(expr)`.
"""
const evaluate = evaluate_future

export FutureField,
       FutureLockedField,
       init_future!,
       reset_future!,
       evaluate_future,
       evaluate,
       get_out,
       build_out,
       check_conditions,
       enforce_conditions,
       operate,
       future_type,
       new_operands,
       prep_nccs,
       gather_ncc_coeffs,
       parse_future,
       cast_future,
       STORE_OUTPUTS,
       STORE_LAST_DEFAULT
