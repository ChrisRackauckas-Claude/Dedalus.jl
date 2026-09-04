"""
Dispatch utilities for constructing concrete subtypes from abstract type hierarchies.

Translated from dedalus/tools/dispatch.py. Python's `MultiClass` metaclass
dispatches instantiation to the correct subclass by iterating over
`__subclasses__()` and calling `_check_args`. Julia has native multiple dispatch
but lacks runtime subclass enumeration, so we provide an explicit registry and
a `dispatch_construct` function that replicates the pattern.

# Overview

- **`register_subtype!(AbstractT, ConcreteT)`** -- register a concrete subtype
  for dispatch.
- **`dispatch_construct(T, args...; kwargs...)`** -- find the unique registered
  subtype whose `check_args` returns `true`, then construct it.
- **`preprocess_args`**, **`check_args`**, **`postprocess_args`** -- overridable
  hooks (mirror Python's `_preprocess_args`, `_check_args`, `_postprocess_args`).
- **`@register_dispatch`** -- convenience macro for registering a subtype at
  definition time.
- **`CachedDispatch`** -- wrapper combining dispatch with LRU-style caching.
"""

# ---- registry ---------------------------------------------------------------

"""
Global dispatch registry.  Maps an abstract supertype to a vector of its
registered concrete subtypes (in registration order).
"""
const _DISPATCH_REGISTRY = Dict{DataType, Vector{DataType}}()

"""
    register_subtype!(AbstractT::Type, ConcreteT::Type)

Register `ConcreteT` as a dispatchable subtype of `AbstractT`.
`ConcreteT` must be a subtype of `AbstractT`.  Duplicate registrations
are silently ignored.

# Examples
```julia
abstract type MyBase end
struct MyImpl <: MyBase end
register_subtype!(MyBase, MyImpl)
```
"""
function register_subtype!(AbstractT::Type, ConcreteT::Type)
    ConcreteT <: AbstractT ||
        throw(ArgumentError("$ConcreteT is not a subtype of $AbstractT"))
    subtypes = get!(() -> DataType[], _DISPATCH_REGISTRY, AbstractT)
    if ConcreteT in subtypes
        return nothing  # already registered
    end
    push!(subtypes, ConcreteT)
    return nothing
end

"""
    registered_subtypes(T::Type) -> Vector{DataType}

Return the registered dispatchable subtypes for `T`.
"""
registered_subtypes(T::Type) = get(() -> DataType[], _DISPATCH_REGISTRY, T)

"""
    clear_registry!(T::Type)

Remove all registered subtypes for `T`.  Mostly useful for testing.
"""
function clear_registry!(T::Type)
    delete!(_DISPATCH_REGISTRY, T)
    return nothing
end

# ---- @register_dispatch macro -----------------------------------------------

"""
    @register_dispatch AbstractT ConcreteT

Convenience macro that calls `register_subtype!(AbstractT, ConcreteT)` at
top-level (module-load time).

# Examples
```julia
abstract type Operator end
struct Add <: Operator end
@register_dispatch Operator Add
```
"""
macro register_dispatch(AbstractT, ConcreteT)
    esc(:(register_subtype!($AbstractT, $ConcreteT)))
end

# ---- overridable hooks ------------------------------------------------------

"""
    preprocess_args(::Type{T}, args...; kwargs...) -> (args, kwargs)

Called before dispatch iteration to canonicalise arguments.
Default implementation returns `(args, kwargs)` unchanged.
Override for a specific abstract type to normalise inputs.
"""
preprocess_args(::Type{T}, args...; kwargs...) where {T} = (args, kwargs)

"""
    check_args(::Type{T}, args...; kwargs...) -> Bool

Return `true` if `T` is the correct concrete type for the given arguments.
Must be overridden for each registered concrete subtype; the default returns
`false`.
"""
check_args(::Type{T}, args...; kwargs...) where {T} = false

"""
    postprocess_args(::Type{T}, args...; kwargs...) -> (args, kwargs)

Called after a unique subtype is selected but before construction.
Default implementation returns `(args, kwargs)` unchanged.
"""
postprocess_args(::Type{T}, args...; kwargs...) where {T} = (args, kwargs)

"""
    stop_dispatch(::Type{T}) -> Bool

If `true`, this subtype is excluded from the dispatch search.
Mirrors Python's `stop_dispatch` class attribute.  Default is `false`.
"""
stop_dispatch(::Type{T}) where {T} = false

# ---- dispatch_construct -----------------------------------------------------

"""
    dispatch_construct(::Type{T}, args...; kwargs...)

Find the unique registered subtype of `T` whose `check_args` returns `true`
for the given arguments, then construct and return an instance of that subtype.

The algorithm mirrors Python's `MultiClass.__call__`:

1. `preprocess_args(T, args...; kwargs...)` -- canonicalise inputs.
2. Collect registered subtypes, skipping those with `stop_dispatch(S) == true`.
3. If no subtypes are registered and `check_args(T, ...)` passes, construct `T`
   directly (leaf-type behaviour).
4. Otherwise iterate subtypes; each that passes `check_args` is appended to a
   pass-list.
5. If exactly one subtype passes, call `postprocess_args` and recurse via
   `dispatch_construct` on that subtype (allowing multi-level dispatch chains).
6. Zero or multiple matches raise an error.

`SkipDispatchException` may be thrown from any hook; if caught, its `.output`
field is returned immediately.

# Examples
```julia
abstract type Op end
struct AddOp <: Op; a; b; end
register_subtype!(Op, AddOp)
check_args(::Type{AddOp}, a, b; kw...) = a isa Number && b isa Number
obj = dispatch_construct(Op, 1, 2)  # returns AddOp(1, 2)
```
"""
function dispatch_construct(::Type{T}, args...; kwargs...) where {T}
    # --- preprocess -----------------------------------------------------------
    local processed_args, processed_kwargs
    try
        processed_args, processed_kwargs = preprocess_args(T, args...; kwargs...)
    catch e
        e isa SkipDispatchException && return e.output
        rethrow()
    end

    # --- gather candidate subtypes --------------------------------------------
    subtypes_list = [S for S in registered_subtypes(T) if !stop_dispatch(S)]

    # --- leaf type (no subtypes registered) -----------------------------------
    if isempty(subtypes_list)
        local passes::Bool
        try
            passes = check_args(T, processed_args...; processed_kwargs...)
        catch e
            e isa SkipDispatchException && return e.output
            rethrow()
        end
        if passes
            return T(processed_args...; processed_kwargs...)
        else
            throw(TypeError("No registered subtypes for $T and provided arguments do not pass dispatch check."))
        end
    end

    # --- find matching subtype(s) ---------------------------------------------
    passlist = DataType[]
    for S in subtypes_list
        local ok::Bool
        try
            ok = check_args(S, processed_args...; processed_kwargs...)
        catch e
            e isa SkipDispatchException && return e.output
            rethrow()
        end
        ok && push!(passlist, S)
    end

    # --- exactly one match required -------------------------------------------
    if isempty(passlist)
        throw(TypeError(
            "None of the registered subtypes of $T passed dispatch check for " *
            "args=$(processed_args), kwargs=$(processed_kwargs). " *
            "Registered subtypes: $(subtypes_list)."
        ))
    elseif length(passlist) > 1
        throw(TypeError(
            "Multiple registered subtypes of $T passed dispatch check: " *
            "$(passlist). Dispatch requires exactly one match."
        ))
    end

    subtype = only(passlist)

    # --- postprocess & recurse ------------------------------------------------
    local final_args, final_kwargs
    try
        final_args, final_kwargs = postprocess_args(subtype, processed_args...; processed_kwargs...)
    catch e
        e isa SkipDispatchException && return e.output
        rethrow()
    end

    return dispatch_construct(subtype, final_args...; final_kwargs...)
end

# ---- TypeError helper (Julia has no built-in TypeError) ----------------------

"""
    TypeError(msg)

Lightweight exception mirroring Python's `TypeError`, used when dispatch
cannot find a unique matching subtype.
"""
struct TypeError <: Exception
    msg::String
end

Base.showerror(io::IO, e::TypeError) = print(io, "TypeError: ", e.msg)

# ---- CachedDispatch ---------------------------------------------------------

"""
    CachedDispatch{T}

Wrapper that combines `dispatch_construct` with a result cache keyed by
the call arguments.  Useful when repeated constructions with identical
arguments should return the same object (mirrors Python's `CachedMultiClass`).

# Usage
```julia
cached = CachedDispatch{MyAbstractType}()
obj1 = cached(arg1, arg2)
obj2 = cached(arg1, arg2)
obj1 === obj2  # true
```
"""
mutable struct CachedDispatch{T}
    cache::Dict{UInt, Any}
end

CachedDispatch{T}() where {T} = CachedDispatch{T}(Dict{UInt, Any}())

"""
    (cd::CachedDispatch{T})(args...; kwargs...) where {T}

Dispatch-construct with caching.  The cache key is derived from a hash of
`(args, kwargs)`.
"""
function (cd::CachedDispatch{T})(args...; kwargs...) where {T}
    key = hash((args, kwargs))
    cached = get(cd.cache, key, nothing)
    cached !== nothing && return cached
    result = dispatch_construct(T, args...; kwargs...)
    cd.cache[key] = result
    return result
end

"""
    clear_cache!(cd::CachedDispatch)

Remove all cached entries.
"""
function clear_cache!(cd::CachedDispatch)
    empty!(cd.cache)
    return nothing
end

# ---- exports ----------------------------------------------------------------

export dispatch_construct,
       register_subtype!,
       registered_subtypes,
       clear_registry!,
       @register_dispatch,
       preprocess_args,
       check_args,
       postprocess_args,
       stop_dispatch,
       CachedDispatch,
       clear_cache!,
       TypeError
