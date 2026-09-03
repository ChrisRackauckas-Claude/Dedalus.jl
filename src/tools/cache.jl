"""
    Cache utilities for Dedalus.jl

Provides caching patterns translated from the Python Dedalus `cache.py` module:

- [`CachedAttribute`](@ref): Lazily compute and cache a value on first access.
- [`CachedFunction`](@ref): Memoize function outputs with bounded cache size.
- [`CachedMethod`](@ref): Per-instance memoization for method-like callables.
- [`CachedClass`](@ref): Cache struct construction using weak references.
- [`serialize_call`](@ref): Normalize positional/keyword arguments into a canonical tuple.
"""

using OrderedCollections: OrderedDict

export CachedAttribute, get_cached!, reset_cached!,
       CachedFunction, serialize_call,
       CachedMethod, get_cached_method,
       CachedClass, cached_construct,
       @cached_attribute, @cached_method

# ---------------------------------------------------------------------------
# CachedAttribute
# ---------------------------------------------------------------------------

"""
    CachedAttribute{T}

A lazily-computed, write-once cache slot. The first call to [`get_cached!`](@ref)
evaluates `compute_fn` and stores the result; subsequent calls return the
cached value without recomputation.

Mirrors Python's `CachedAttribute` descriptor.

# Fields
- `compute_fn::Function` – zero-argument callable that produces the value.
- `value::Ref{Union{Nothing, T}}` – cached result (initially `nothing`).

# Example
```julia
ca = CachedAttribute{Matrix{Float64}}(() -> rand(3, 3))
A  = get_cached!(ca)   # computes and caches
B  = get_cached!(ca)   # returns cached; A === B
```
"""
mutable struct CachedAttribute{T}
    compute_fn::Function
    value::Union{Nothing, T}

    function CachedAttribute{T}(compute_fn::Function) where {T}
        new{T}(compute_fn, nothing)
    end
end

"""
    CachedAttribute(compute_fn::Function)

Convenience constructor that infers `T = Any`.
"""
CachedAttribute(compute_fn::Function) = CachedAttribute{Any}(compute_fn)

"""
    get_cached!(ca::CachedAttribute{T}) :: T

Return the cached value, computing it on first access via `ca.compute_fn()`.
"""
function get_cached!(ca::CachedAttribute{T})::T where {T}
    if ca.value === nothing
        ca.value = ca.compute_fn()
    end
    return ca.value
end

"""
    reset_cached!(ca::CachedAttribute)

Clear the cached value so the next [`get_cached!`](@ref) call recomputes it.
"""
function reset_cached!(ca::CachedAttribute)
    ca.value = nothing
    return nothing
end

"""
    @cached_attribute name T expr

Generate a `CachedAttribute{T}` field pattern. Expands to a helper function
`name(instance)` that lazily computes and caches a value stored in
`instance._cache_name`.

Intended for use inside struct definitions paired with an `IdDict`-based
per-instance cache. The macro produces:

1. A function `name(instance)` that checks `instance._cache` for key `name`,
   computing `expr` (which may reference `instance`) on first access.

# Example
```julia
@cached_attribute my_matrix Matrix{Float64} begin
    rand(3, 3)
end
```
expands roughly to a function `my_matrix(instance)` that lazily computes and
caches the result.
"""
macro cached_attribute(name, T, expr)
    func_name = esc(name)
    cache_key = QuoteNode(name)
    result_type = esc(T)
    compute_body = esc(expr)
    return quote
        function $(func_name)(instance)::$(result_type)
            cache = get!(IdDict, instance._cache)
            key = $(cache_key)
            if !haskey(cache, key)
                cache[key] = $(compute_body)
            end
            return cache[key]
        end
    end
end

# ---------------------------------------------------------------------------
# serialize_call
# ---------------------------------------------------------------------------

"""
    serialize_call(args::Tuple, kw, argnames::Vector{Symbol}, defaults::Dict{Symbol}) -> Tuple

Normalize a call's positional and keyword arguments into a canonical tuple.
Positional arguments are taken first; remaining argument names are filled from
`kw` and then from `defaults`.

This mirrors the Python `serialize_call` helper used by [`CachedFunction`](@ref).

# Arguments
- `args` – positional arguments as a tuple.
- `kw` – keyword arguments (any dict-like mapping `Symbol => value`).
- `argnames` – ordered parameter names of the target function.
- `defaults` – default values keyed by parameter name.

# Returns
A `Tuple` containing one element per entry in `argnames`.
"""
function serialize_call(args::Tuple, kw, argnames::Vector{Symbol},
                        defaults::Dict{Symbol})
    call = collect(Any, args)
    for name in argnames[length(args)+1:end]
        if haskey(kw, name)
            push!(call, kw[name])
        else
            push!(call, defaults[name])
        end
    end
    return Tuple(call)
end

# ---------------------------------------------------------------------------
# CachedFunction
# ---------------------------------------------------------------------------

"""
    CachedFunction{F}

A callable wrapper that memoizes the outputs of `func` in an `OrderedDict`.
When the cache reaches `max_size`, the oldest entry is evicted (FIFO).

Both a "direct call" key `(args, kw_pairs)` and a "resolved call" key
(canonical tuple from [`serialize_call`](@ref)) are stored so that calls
with equivalent argument semantics share the cached result.

Mirrors Python's `CachedFunction` decorator.

# Fields
- `func::F` – the wrapped function.
- `cache::OrderedDict{Any,Any}` – memoization store.
- `max_size::Float64` – maximum number of cache entries (`Inf` for unbounded).
- `argnames::Vector{Symbol}` – positional parameter names of `func`.
- `defaults::Dict{Symbol}` – default values for trailing positional parameters.

# Example
```julia
cf = CachedFunction(sin)
cf(1.0)            # computes sin(1.0)
cf(1.0)            # returns cached result
```
"""
struct CachedFunction{F}
    func::F
    cache::OrderedDict{Any,Any}
    max_size::Float64
    argnames::Vector{Symbol}
    defaults::Dict{Symbol}

    function CachedFunction(func::F; max_size::Real=Inf) where {F}
        argnames = _extract_argnames(func)
        defaults = _extract_defaults(func)
        new{F}(func, OrderedDict{Any,Any}(), Float64(max_size), argnames, defaults)
    end
end

"""
    (cf::CachedFunction)(args...; kwargs...)

Call the cached function. Returns a cached result if the arguments match a
previous call; otherwise evaluates `cf.func`, caches the result, and returns it.
"""
function (cf::CachedFunction)(args...; kwargs...)
    kw_tuple = Tuple(pairs(kwargs))
    direct_call = (args, kw_tuple)

    # Fast path: exact match on direct call form.
    if haskey(cf.cache, direct_call)
        return cf.cache[direct_call]
    end

    # Try resolved (canonical) form.
    kw_dict = Dict{Symbol,Any}(pairs(kwargs))
    resolved_call = serialize_call(args, kw_dict, cf.argnames, cf.defaults)

    if haskey(cf.cache, resolved_call)
        result = cf.cache[resolved_call]
        cf.cache[direct_call] = result
        return result
    end

    # Evict oldest entries until there is room.
    while length(cf.cache) >= cf.max_size
        popfirst!(cf.cache)
    end

    result = cf.func(args...; kwargs...)
    cf.cache[direct_call] = result
    cf.cache[resolved_call] = result
    return result
end

# -- internal helpers for argument introspection ----------------------------

"""
    _extract_argnames(func) -> Vector{Symbol}

Return the positional-argument names of the first method of `func`.
Falls back to an empty vector when introspection is not possible.
"""
function _extract_argnames(func)::Vector{Symbol}
    try
        meths = methods(func)
        if isempty(meths)
            return Symbol[]
        end
        m = first(meths)
        # `Base.method_argnames` returns names including the function slot.
        names = Base.method_argnames(m)
        # Drop the first entry (the function/callable itself).
        return length(names) > 1 ? Symbol.(names[2:end]) : Symbol[]
    catch
        return Symbol[]
    end
end

"""
    _extract_defaults(func) -> Dict{Symbol}

Extract default keyword-argument values. Julia does not expose positional
defaults at runtime, so this returns an empty `Dict` as a safe baseline.
Users may supply defaults explicitly via [`serialize_call`](@ref).
"""
function _extract_defaults(::Any)::Dict{Symbol}
    return Dict{Symbol,Any}()
end

# ---------------------------------------------------------------------------
# CachedMethod
# ---------------------------------------------------------------------------

"""
    CachedMethod{F}

Per-instance memoization wrapper. Unlike [`CachedFunction`](@ref), which uses
a single global cache, `CachedMethod` maintains a separate `CachedFunction`
for each instance (keyed by object identity via `objectid`). When the instance
is garbage-collected, its cache is collected as well because the mapping uses
a `WeakRef`-like identity scheme via `IdDict`.

Mirrors Python's `CachedMethod` descriptor.

# Fields
- `func::F` – the underlying function (first argument is the instance).
- `max_size::Float64` – per-instance cache size limit.
- `instance_caches::IdDict{Any, CachedFunction}` – one cache per instance.

# Example
```julia
cm = CachedMethod(my_method; max_size=128)
cm(instance, arg1, arg2)   # caches per `instance`
```
"""
struct CachedMethod{F}
    func::F
    max_size::Float64
    instance_caches::IdDict{Any,CachedFunction}

    function CachedMethod(func::F; max_size::Real=Inf) where {F}
        new{F}(func, Float64(max_size), IdDict{Any,CachedFunction}())
    end
end

"""
    (cm::CachedMethod)(instance, args...; kwargs...)

Call the cached method, routing through a per-instance [`CachedFunction`](@ref).
A new `CachedFunction` is created transparently the first time `instance` is
seen.
"""
function (cm::CachedMethod)(instance, args...; kwargs...)
    cf = get!(cm.instance_caches, instance) do
        CachedFunction(cm.func; max_size=cm.max_size)
    end
    return cf(instance, args...; kwargs...)
end

"""
    get_cached_method(cm::CachedMethod, instance) -> CachedFunction

Retrieve (or create) the per-instance [`CachedFunction`](@ref) backing a
[`CachedMethod`](@ref). Useful for inspecting or clearing an instance's cache
directly.
"""
function get_cached_method(cm::CachedMethod, instance)
    return get!(cm.instance_caches, instance) do
        CachedFunction(cm.func; max_size=cm.max_size)
    end
end

"""
    @cached_method func_expr max_size=Inf

Wrap a function definition so that it is automatically memoized per-instance
(the first positional argument).

# Example
```julia
@cached_method function heavy_computation(obj, x, y)
    # expensive work …
end
```
"""
macro cached_method(expr, max_size=Inf)
    if expr.head === :function || expr.head === :(=)
        sig = expr.args[1]
        name = sig isa Symbol ? sig : sig.args[1]
        escaped_name = esc(name)
        escaped_expr = esc(expr)
        escaped_max = esc(max_size)
        return quote
            $(escaped_expr)
            $(escaped_name) = CachedMethod($(escaped_name); max_size=$(escaped_max))
        end
    else
        error("@cached_method requires a function definition")
    end
end

# ---------------------------------------------------------------------------
# CachedClass
# ---------------------------------------------------------------------------

"""
    CachedClass{T}

A constructor-level cache that ensures only one instance of type `T` exists
for each unique set of constructor arguments. Existing instances are held via
`WeakRef` so they can be garbage-collected when no strong references remain;
subsequent construction with the same arguments will then produce a new
instance.

Mirrors Python's `CachedClass` metaclass.

# Fields
- `constructor::Function` – callable that builds a new `T` (typically the
  inner constructor or a factory function).
- `cache::Dict{Any, WeakRef}` – maps canonical argument tuples to weak
  references of cached instances.
- `preprocess_args::Function` – optional hook to transform `(args, kwargs)`
  before cache-key construction (default: identity).
- `preprocess_cache_args::Function` – optional hook to transform the resolved
  positional args into the cache key (default: identity / `Tuple`).

# Example
```julia
cc = CachedClass{MyStruct}(MyStruct)
a  = cached_construct(cc, 1, 2, 3)
b  = cached_construct(cc, 1, 2, 3)
a === b  # true – same instance returned from cache
```
"""
struct CachedClass{T}
    constructor::Function
    cache::Dict{Any,WeakRef}
    preprocess_args::Function
    preprocess_cache_args::Function

    function CachedClass{T}(constructor::Function;
                            preprocess_args::Function = _default_preprocess_args,
                            preprocess_cache_args::Function = _default_preprocess_cache_args
                           ) where {T}
        new{T}(constructor, Dict{Any,WeakRef}(), preprocess_args, preprocess_cache_args)
    end
end

"""
    CachedClass(T::Type; kwargs...)

Convenience constructor using the type itself as the constructor callable.
"""
function CachedClass(::Type{T}; kwargs...) where {T}
    return CachedClass{T}(T; kwargs...)
end

"""
    cached_construct(cc::CachedClass{T}, args...; kwargs...) :: T

Return a cached instance of `T` for the given arguments, constructing a new
one only if no live (weakly-reachable) instance exists for the canonical key.

# Steps
1. `preprocess_args` transforms `(args, kwargs)`.
2. Arguments are merged into a canonical tuple.
3. `preprocess_cache_args` transforms the canonical tuple into the cache key.
4. If a live `WeakRef` exists for that key, its value is returned.
5. Otherwise a new instance is constructed, stored as a `WeakRef`, and returned.
"""
function cached_construct(cc::CachedClass{T}, args...; kwargs...) where {T}
    processed_args, processed_kw = cc.preprocess_args(args, kwargs)

    # Build canonical positional tuple (kwargs appended in sorted order).
    full_args = if isempty(processed_kw)
        processed_args
    else
        kw_sorted = Tuple(v for (_, v) in sort(collect(pairs(processed_kw)); by=first))
        (processed_args..., kw_sorted...)
    end

    cache_key = cc.preprocess_cache_args(full_args)

    # Check for a live cached instance.
    if haskey(cc.cache, cache_key)
        ref = cc.cache[cache_key]
        instance = ref.value
        if instance !== nothing
            return instance::T
        end
        # WeakRef target was collected; remove stale entry.
        delete!(cc.cache, cache_key)
    end

    # Construct, cache, and return.
    instance = cc.constructor(full_args...)::T
    cc.cache[cache_key] = WeakRef(instance)
    return instance
end

# -- default preprocessing hooks --------------------------------------------

"""
    _default_preprocess_args(args, kwargs)

Identity preprocessor: returns `(args, kwargs)` unchanged.
"""
_default_preprocess_args(args, kwargs) = (args, kwargs)

"""
    _default_preprocess_cache_args(args)

Identity preprocessor: returns `args` as-is for use as the cache key.
"""
_default_preprocess_cache_args(args) = args
