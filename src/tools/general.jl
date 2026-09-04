"""
General-purpose utilities for Dedalus.jl.

Translated from dedalus/tools/general.py. Provides ordered sets, natural sorting,
reverse enumeration, oscillating iterators, unification helpers, deferred tuples,
element replacement iterators, and dtype classification.
"""

using OrderedCollections: OrderedDict

# ---------------------------------------------------------------------------
# OrderedSet
# ---------------------------------------------------------------------------

"""
    OrderedSet{T}

An ordered set backed by an `OrderedDict{T, Nothing}`. Elements are unique and
maintain insertion order. Mirrors Python's `OrderedSet` (which subclasses
`OrderedDict`).

# Examples
```julia
s = OrderedSet{Int}()
push!(s, 1)
push!(s, 2)
push!(s, 1)        # no-op, already present
collect(s)          # [1, 2]
2 in s              # true
```
"""
struct OrderedSet{T}
    dict::OrderedDict{T, Nothing}
end

"""
    OrderedSet{T}()

Create an empty `OrderedSet` with element type `T`.
"""
OrderedSet{T}() where {T} = OrderedSet{T}(OrderedDict{T, Nothing}())

"""
    OrderedSet()

Create an empty `OrderedSet{Any}`.
"""
OrderedSet() = OrderedSet{Any}()

"""
    ordered_set_from(::Type{T}, iter)

Create an `OrderedSet{T}` populated with elements from `iter`.
"""
function ordered_set_from(::Type{T}, iter) where {T}
    s = OrderedSet{T}()
    for item in iter
        push!(s, item)
    end
    return s
end

"""
    OrderedSet(iter)

Create an `OrderedSet` whose element type is inferred from `iter`.
If `iter` is empty, falls back to `OrderedSet{Any}`.
"""
function OrderedSet(iter)
    items = collect(iter)
    if isempty(items)
        return OrderedSet{Any}()
    end
    T = eltype(items)
    s = OrderedSet{T}()
    for item in items
        push!(s, item)
    end
    return s
end

Base.length(s::OrderedSet) = length(s.dict)
Base.isempty(s::OrderedSet) = isempty(s.dict)
Base.in(item, s::OrderedSet) = haskey(s.dict, item)
Base.eltype(::Type{OrderedSet{T}}) where {T} = T

function Base.push!(s::OrderedSet{T}, item) where {T}
    s.dict[convert(T, item)] = nothing
    return s
end

function Base.delete!(s::OrderedSet{T}, item) where {T}
    delete!(s.dict, convert(T, item))
    return s
end

function Base.union!(s::OrderedSet{T}, iter) where {T}
    for item in iter
        push!(s, item)
    end
    return s
end

function Base.union!(s::OrderedSet{T}, iters...) where {T}
    for iter in iters
        union!(s, iter)
    end
    return s
end

Base.iterate(s::OrderedSet) = _os_iterate(s, iterate(s.dict))
Base.iterate(s::OrderedSet, state) = _os_iterate(s, iterate(s.dict, state))

function _os_iterate(::OrderedSet, iter_result)
    iter_result === nothing && return nothing
    (pair, state) = iter_result
    return (pair.first, state)
end

Base.collect(s::OrderedSet{T}) where {T} = collect(T, keys(s.dict))

function Base.show(io::IO, s::OrderedSet{T}) where {T}
    print(io, "OrderedSet{", T, "}(")
    print(io, collect(s))
    print(io, ")")
end

Base.empty!(s::OrderedSet) = (empty!(s.dict); s)
Base.copy(s::OrderedSet{T}) where {T} = OrderedSet{T}(copy(s.dict))
Base.:(==)(a::OrderedSet, b::OrderedSet) = keys(a.dict) == keys(b.dict)

Base.first(s::OrderedSet) = first(keys(s.dict))
Base.last(s::OrderedSet) = last(keys(s.dict))

# ---------------------------------------------------------------------------
# rev_enumerate
# ---------------------------------------------------------------------------

"""
    rev_enumerate(sequence)

Return an iterator of `(index, element)` pairs in reverse order, using 1-based
indexing. Mirrors Python's reversed `enumerate` but with Julia's 1-based convention.

# Examples
```julia
for (i, v) in rev_enumerate([10, 20, 30])
    println(i, " => ", v)
end
# 3 => 30
# 2 => 20
# 1 => 10
```
"""
function rev_enumerate(sequence)
    n = length(sequence)
    return ((n - i + 1, sequence[n - i + 1]) for i in 1:n)
end

# ---------------------------------------------------------------------------
# natural_sort
# ---------------------------------------------------------------------------

"""
    natural_sort(iterable; reverse=false)

Sort alphanumeric strings naturally so that, e.g., `"item2"` sorts before
`"item10"`. Non-digit substrings are compared case-insensitively; digit
substrings are compared numerically.

# Examples
```julia
natural_sort(["item10", "item2", "item1"])  # ["item1", "item2", "item10"]
```
"""
function natural_sort(iterable; reverse::Bool=false)
    function _natural_key(item)
        parts = split(string(item), r"([0-9]+)"; keepempty=false)
        raw = split(string(item), r"[^0-9]+"; keepempty=false)
        # Rebuild interleaved key: split on digit groups, convert digits to Int.
        key = Any[]
        for m in eachmatch(r"([0-9]+)|([^0-9]+)", string(item))
            sub = m.match
            if all(isdigit, sub)
                push!(key, (0, parse(Int, sub), ""))
            else
                push!(key, (1, 0, lowercase(sub)))
            end
        end
        return key
    end
    return sort(collect(iterable); by=_natural_key, rev=reverse)
end

# ---------------------------------------------------------------------------
# oscillate
# ---------------------------------------------------------------------------

"""
    oscillate(iterable; max_passes=Inf)

Return a `Channel`-based iterator that oscillates forward and backward through
`iterable`. One full forward + backward sweep counts as one pass. Uses 1-based
indexing.

# Examples
```julia
for x in Iterators.take(oscillate([1, 2, 3]), 9)
    print(x, " ")
end
# 1 2 3 2 1 2 3 2 1
```
"""
function oscillate(iterable; max_passes::Real=Inf)
    items = collect(iterable)
    n = length(items)
    return Channel{eltype(items)}(; ctype=eltype(items), csize=0) do ch
        passes = 0
        while true
            # Forward pass
            for i in 1:n
                put!(ch, items[i])
            end
            # Backward pass (exclude first and last to avoid repeats)
            for i in (n-1):-1:2
                put!(ch, items[i])
            end
            passes += 1
            if passes >= max_passes
                return
            end
        end
    end
end

# ---------------------------------------------------------------------------
# unify
# ---------------------------------------------------------------------------

"""
    unify(objects)

Check that all elements of `objects` are equal. If so, return the first element.
If not, throw a `ValueError`.

# Examples
```julia
unify([1, 1, 1])   # 1
unify([1, 2])       # throws ValueError
```
"""
function unify(objects)
    first_val = nothing
    first_set = false
    for (i, obj) in enumerate(objects)
        if !first_set
            first_val = obj
            first_set = true
        else
            if obj != first_val
                throw(ArgumentError("Objects are not all equal."))
            end
        end
    end
    if !first_set
        throw(ArgumentError("Cannot unify an empty collection."))
    end
    return first_val
end

# ---------------------------------------------------------------------------
# unify_attributes
# ---------------------------------------------------------------------------

"""
    unify_attributes(objects, attr::Symbol; require::Bool=true)

Collect the attribute (field or property) named `attr` from each object in
`objects`, then [`unify`](@ref) them. If `require` is `true` (default), an
error is thrown when an object lacks the attribute; otherwise the object is
silently skipped.

# Examples
```julia
struct Foo; x::Int; end
unify_attributes([Foo(1), Foo(1)], :x)  # 1
```
"""
function unify_attributes(objects, attr::Symbol; require::Bool=true)
    attrs = Any[]
    for obj in objects
        try
            push!(attrs, getproperty(obj, attr))
        catch e
            if e isa ErrorException || e isa UndefRefError
                if require
                    rethrow()
                end
                # else skip
            else
                rethrow()
            end
        end
    end
    return unify(attrs)
end

# ---------------------------------------------------------------------------
# replace_iter
# ---------------------------------------------------------------------------

"""
    replace_iter(data, selectors, replacement)

Return an iterator that yields `replacement` when the corresponding element in
`selectors` is truthy, or the original element from `data` otherwise.

Named `replace_iter` to avoid conflict with `Base.replace`.

# Examples
```julia
collect(replace_iter([1, 2, 3], [false, true, false], 0))  # [1, 0, 3]
```
"""
function replace_iter(data, selectors, replacement)
    return (s ? replacement : d for (d, s) in zip(data, selectors))
end

# ---------------------------------------------------------------------------
# DeferredTuple
# ---------------------------------------------------------------------------

"""
    DeferredTuple{T}

A lazy, fixed-size, read-only collection whose entries are computed on demand
via `entry_function(index)`. Uses 1-based indexing. Mirrors Python's
`DeferredTuple`.

# Fields
- `entry_function::Function` -- callable that receives a 1-based index and
  returns the element.
- `size::Int` -- total number of elements.

# Examples
```julia
dt = DeferredTuple(i -> i^2, 5)
dt[3]       # 9
length(dt)  # 5
```
"""
struct DeferredTuple{T}
    entry_function::Function
    size::Int
end

"""
    DeferredTuple(entry_function, size::Integer)

Convenience constructor with element type `Any`.
"""
DeferredTuple(entry_function::Function, size::Integer) =
    DeferredTuple{Any}(entry_function, Int(size))

Base.length(dt::DeferredTuple) = dt.size

function Base.getindex(dt::DeferredTuple, key::Integer)
    idx = key < 0 ? key + length(dt) + 1 : key
    if idx < 1 || idx > length(dt)
        throw(BoundsError(dt, key))
    end
    return dt.entry_function(idx)
end

Base.firstindex(dt::DeferredTuple) = 1
Base.lastindex(dt::DeferredTuple) = dt.size
Base.size(dt::DeferredTuple) = (dt.size,)

function Base.iterate(dt::DeferredTuple, state::Int=1)
    state > dt.size && return nothing
    return (dt.entry_function(state), state + 1)
end

# ---------------------------------------------------------------------------
# is_real_dtype / is_complex_dtype
# ---------------------------------------------------------------------------

"""
    is_real_dtype(T::Type) -> Bool

Return `true` if `T` is a real numeric type (e.g. `Float64`, `Float32`, `Int`).
Mirrors Python's `is_real_dtype`.

# Examples
```julia
is_real_dtype(Float64)     # true
is_real_dtype(ComplexF64)  # false
```
"""
function is_real_dtype(::Type{T}) where {T<:Number}
    return T <: Real
end

function is_real_dtype(::Type{T}) where {T}
    return false
end

"""
    is_complex_dtype(T::Type) -> Bool

Return `true` if `T` is a complex numeric type (e.g. `ComplexF64`, `ComplexF32`).
Mirrors Python's `is_complex_dtype`.

# Examples
```julia
is_complex_dtype(ComplexF64)  # true
is_complex_dtype(Float64)     # false
```
"""
function is_complex_dtype(::Type{T}) where {T<:Number}
    return T <: Complex
end

function is_complex_dtype(::Type{T}) where {T}
    return false
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export OrderedSet,
       rev_enumerate,
       natural_sort,
       oscillate,
       unify,
       unify_attributes,
       replace_iter,
       DeferredTuple,
       is_real_dtype,
       is_complex_dtype
