"""
Deterministic random array utilities for the Dedalus framework.

Translated from dedalus/tools/random_arrays.py. Provides chunked RNG streams,
element-wise access into those streams, and array-like objects whose entries are
deterministically random when indexed.

Uses Julia's `Random` standard library. Distribution names are mapped from
NumPy conventions:
  - `"uniform"`         -> `rand`
  - `"standard_normal"` -> `randn`
"""

using Random

# ---------------------------------------------------------------------------
# Distribution dispatch
# ---------------------------------------------------------------------------

"""
    _generate_chunk(rng::AbstractRNG, distribution::AbstractString, chunk_size::Int) -> Vector{Float64}

Generate a chunk of random numbers from the named distribution using `rng`.
"""
function _generate_chunk(rng::AbstractRNG, distribution::AbstractString, chunk_size::Int)
    dist = lowercase(distribution)
    if dist == "uniform"
        return rand(rng, chunk_size)
    elseif dist in ("standard_normal", "normal", "randn")
        return randn(rng, chunk_size)
    else
        error("Unsupported distribution: $distribution. Supported: \"uniform\", \"standard_normal\".")
    end
end

# ---------------------------------------------------------------------------
# chunked_rng  (Channel-based generator)
# ---------------------------------------------------------------------------

"""
    chunked_rng(seed, chunk_size::Int, distribution::AbstractString) -> Channel{Tuple{Int, Vector{Float64}}}

Return a `Channel` that yields `(chunk_index, chunk_data)` tuples, where
`chunk_index` is 0-based (matching the Python original) and `chunk_data` is a
`Vector{Float64}` of length `chunk_size`. The RNG is seeded with `seed` so the
stream is fully deterministic.

# Examples
```julia
ch = chunked_rng(42, 1024, "uniform")
idx, data = take!(ch)   # (0, [...])
```
"""
function chunked_rng(seed, chunk_size::Int, distribution::AbstractString)
    return Channel{Tuple{Int, Vector{Float64}}}(; csize=1) do ch
        rng = Xoshiro(seed)
        chunk_index = 0
        while true
            chunk_data = _generate_chunk(rng, distribution, chunk_size)
            put!(ch, (chunk_index, chunk_data))
            chunk_index += 1
        end
    end
end

# ---------------------------------------------------------------------------
# rng_element
# ---------------------------------------------------------------------------

"""
    rng_element(index::Int, seed, chunk_size::Int, distribution::AbstractString) -> Float64

Get element at the given 0-based `index` from a deterministic RNG stream.
The stream is chunked so that only the necessary chunks are generated.

# Examples
```julia
rng_element(5, 42, 1024, "uniform")
```
"""
function rng_element(index::Int, seed, chunk_size::Int, distribution::AbstractString)
    cs = min(1 + index, chunk_size)
    rng = chunked_rng(seed, cs, distribution)
    d, m = divrem(index, cs)
    local data
    for (chunk, chunk_data) in rng
        if chunk == d
            data = chunk_data
            break
        end
    end
    close(rng)
    return data[m + 1]  # 1-based array indexing
end

# ---------------------------------------------------------------------------
# rng_elements
# ---------------------------------------------------------------------------

"""
    rng_elements(indices::AbstractArray{Int}, seed, chunk_size::Int, distribution::AbstractString) -> Array{Float64}

Get an array of elements from a deterministic RNG stream. The `indices` array
uses 0-based element indices (matching the Python convention for stream
positions). The returned array has the same shape as `indices`.

# Examples
```julia
rng_elements([0, 5, 10], 42, 1024, "uniform")
```
"""
function rng_elements(indices::AbstractArray{Int}, seed, chunk_size::Int, distribution::AbstractString)
    if isempty(indices)
        # Probe the dtype by generating one element
        sample = rng_element(0, seed, chunk_size, distribution)
        return zeros(typeof(sample), size(indices))
    end
    cs = min(1 + maximum(indices), chunk_size)
    rng = chunked_rng(seed, cs, distribution)
    d_arr = div.(indices, cs)
    m_arr = rem.(indices, cs)
    div_set = Set(unique(d_arr))
    max_div = maximum(div_set)
    values = zeros(Float64, size(indices))
    for (chunk, data) in rng
        if chunk == 0
            values = zeros(eltype(data), size(indices))
        end
        if chunk in div_set
            selected = (d_arr .== chunk)
            # For each selected position, grab the corresponding element
            for ci in eachindex(indices)
                if selected[ci]
                    values[ci] = data[m_arr[ci] + 1]  # 1-based array indexing
                end
            end
        end
        if chunk == max_div
            break
        end
    end
    close(rng)
    return values
end

# ---------------------------------------------------------------------------
# IndexArray
# ---------------------------------------------------------------------------

"""
    IndexArray{N}

An `mgrid`-like object that returns an array of flat (linear) indices when
sliced. Supports both column-major (`'F'`, Julia default) and row-major
(`'C'`, NumPy default) ordering.

Indices are 0-based to match the Python original's usage in RNG stream
positioning.

# Fields
- `shape::NTuple{N, Int}` -- dimensions of the logical array
- `order::Char` -- `'C'` for row-major or `'F'` for column-major

# Examples
```julia
ia = IndexArray((3, 4))
ia[1:2, 1:3]  # 2x3 array of flat indices
```
"""
struct IndexArray{N}
    shape::NTuple{N, Int}
    order::Char
end

"""
    IndexArray(shape::NTuple{N,Int}; order::Char='C')

Create an `IndexArray` with the given shape and index ordering.
"""
IndexArray(shape::NTuple{N, Int}; order::Char='C') where {N} =
    IndexArray{N}(shape, order)

"""
    _linear_index(ia::IndexArray{N}, cartesian::NTuple{N, Int}) -> Int

Compute the 0-based flat index for a 0-based Cartesian index tuple.
"""
function _linear_index(ia::IndexArray{N}, cartesian::NTuple{N, Int}) where {N}
    if ia.order == 'C'
        # Row-major: last dimension varies fastest
        idx = 0
        for d in 1:N
            idx = idx * ia.shape[d] + cartesian[d]
        end
        return idx
    else
        # Column-major: first dimension varies fastest
        idx = 0
        for d in N:-1:1
            idx = idx * ia.shape[d] + cartesian[d]
        end
        return idx
    end
end

"""
    Base.getindex(ia::IndexArray{N}, key::Vararg{UnitRange{Int}, N}) where N

Slice the `IndexArray` with 1-based `UnitRange`s and return an `Array{Int}` of
0-based flat indices.
"""
function Base.getindex(ia::IndexArray{N}, key::Vararg{UnitRange{Int}, N}) where {N}
    # Convert 1-based ranges to 0-based ranges
    ranges_0 = Tuple(0:(last(r)-first(r)) .+ (first(r) - 1) for r in key)
    # Build output shape
    out_shape = Tuple(length(r) for r in key)
    result = Array{Int}(undef, out_shape)
    for ci in CartesianIndices(out_shape)
        # ci is 1-based; convert to 0-based indices in the original array
        cart_0 = Tuple(ranges_0[d][ci[d]] for d in 1:N)
        result[ci] = _linear_index(ia, cart_0)
    end
    return result
end

function Base.getindex(ia::IndexArray{N}, key::Vararg{Union{Int, UnitRange{Int}, Colon}, M}) where {N, M}
    # Expand the key to N dimensions, filling missing dims with Colon
    expanded = Vector{Union{Int, UnitRange{Int}}}(undef, N)
    if M > N
        error("Too many selections.")
    end
    for d in 1:N
        if d <= M
            k = key[d]
            if k isa Colon
                expanded[d] = 1:ia.shape[d]
            elseif k isa Int
                expanded[d] = k:k
            else
                expanded[d] = k
            end
        else
            expanded[d] = 1:ia.shape[d]
        end
    end
    ranges = Tuple(expanded[d] for d in 1:N)
    return getindex(ia, ranges...)
end

# ---------------------------------------------------------------------------
# ChunkedRandomArray
# ---------------------------------------------------------------------------

"""
    ChunkedRandomArray{N} <: AbstractArray{Float64, N}

Random array with deterministic elements when indexed. Each element is drawn
from a chunked RNG stream, so the value at any given index is fully
reproducible.

# Fields
- `index_array::IndexArray{N}` -- maps Cartesian slices to flat stream indices
- `seed` -- RNG seed
- `chunk_size::Int` -- chunk size for the RNG stream
- `distribution::String` -- distribution name (`"uniform"`, `"standard_normal"`)

# Examples
```julia
cra = ChunkedRandomArray((100, 100); seed=42, distribution="uniform")
cra[1:5, 1:5]  # deterministic 5x5 random matrix
```
"""
struct ChunkedRandomArray{N} <: AbstractArray{Float64, N}
    index_array::IndexArray{N}
    seed::Any
    chunk_size::Int
    distribution::String
end

"""
    ChunkedRandomArray(shape::NTuple{N,Int}; seed=nothing, chunk_size=2^20,
                       distribution="uniform", order='C')

Create a `ChunkedRandomArray` with the given shape and RNG parameters.
"""
function ChunkedRandomArray(shape::NTuple{N, Int};
                            seed=nothing,
                            chunk_size::Int=2^20,
                            distribution::AbstractString="uniform",
                            order::Char='C') where {N}
    ia = IndexArray(shape; order=order)
    return ChunkedRandomArray{N}(ia, seed, chunk_size, String(distribution))
end

Base.size(cra::ChunkedRandomArray) = cra.index_array.shape

function Base.getindex(cra::ChunkedRandomArray{N}, key::Vararg{Union{Int, UnitRange{Int}, Colon}, M}) where {N, M}
    indices = getindex(cra.index_array, key...)
    return rng_elements(vec(indices), cra.seed, cra.chunk_size, cra.distribution)
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export chunked_rng,
       rng_element,
       rng_elements,
       IndexArray,
       ChunkedRandomArray
