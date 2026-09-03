"""
    CoeffSystem and FieldSystem types for managing contiguous coefficient buffers.

Translates Python dedalus/core/system.py.
"""

# ============================================================
# CoeffSystem
# ============================================================

"""
    CoeffSystem

Contiguous buffer for coefficient data across subproblems/subsystems.
Provides views into the buffer for efficient pencil and group manipulation.
"""
mutable struct CoeffSystem{T}
    data::Vector{T}
    views::Dict{Any, Dict{Any, AbstractArray{T}}}
end

function CoeffSystem(subproblems, dtype::Type{T}) where {T}
    total_size = sum(sp -> size(sp.LHS, 2) * length(sp.subsystems), subproblems)
    data = zeros(T, total_size)
    views = Dict{Any, Dict{Any, AbstractArray{T}}}()
    i0 = 1
    for sp in subproblems
        views_sp = Dict{Any, AbstractArray{T}}()
        views[sp] = views_sp
        i_start = i0
        ncols = size(sp.LHS, 2)
        for ss in sp.subsystems
            i1 = i0 + ncols - 1
            views_sp[ss] = @view data[i0:i1]
            i0 = i1 + 1
        end
        i_end = i0 - 1
        span = i_end - i_start + 1
        if span > 0
            nss = length(sp.subsystems)
            views_sp[nothing] = reshape(@view(data[i_start:i_end]), ncols, nss)
        else
            views_sp[nothing] = reshape(@view(data[1:0]), 0, 0)
        end
    end
    CoeffSystem{T}(data, views)
end

"""
    CoeffSystem(subproblems; dtype=Float64)

Keyword-argument convenience constructor that delegates to the positional
form `CoeffSystem(subproblems, dtype)`.
"""
function CoeffSystem(subproblems; dtype::Type{T}=Float64) where {T}
    CoeffSystem(subproblems, dtype)
end

"""
    get_subdata(cs, sp; ss=nothing)

Return view of coefficient data for the given subproblem and subsystem.
"""
function get_subdata(cs::CoeffSystem, sp; ss=nothing)
    return cs.views[sp][ss]
end

# ============================================================
# FieldSystem
# ============================================================

"""
    FieldSystem

Collection of fields alongside a CoeffSystem buffer for efficient
pencil and group manipulation. Stores a permutation matrix to map
between field-ordered and group-ordered coefficient data.
"""
mutable struct FieldSystem{T}
    fields::Vector{Any}
    data::Vector{T}
    array_views::Vector{Any}
    perm::Any
    buffer::Vector{T}
    group_buffer::Any
    field_buffer::Any
end

function FieldSystem(fields, subproblems, coeff_layout)
    T = fields[1].dtype
    dim = fields[1].dist.dim

    array_shapes = [local_shape(coeff_layout, f.domain, ntuple(_ -> 1, dim)) for f in fields]
    array_sizes = [prod(shape) for shape in array_shapes]
    buffer_size = sum(array_sizes)
    starts = cumsum(array_sizes) .- array_sizes
    buffer = zeros(T, buffer_size)

    flat_views = [@view(buffer[s+1:s+sz]) for (s, sz) in zip(starts, array_sizes)]
    array_views = [reshape(flat, shape) for (flat, shape) in zip(flat_views, array_shapes)]

    # Build permutation — placeholder identity for serial mode
    n = buffer_size
    perm = sparse(collect(1:n), collect(1:n), ones(T, n), n, n)

    group_buffer = @view buffer[1:min(size(perm, 1), n)]
    field_buffer = @view buffer[1:min(size(perm, 2), n)]

    FieldSystem{T}(collect(fields), buffer, array_views, perm, buffer, group_buffer, field_buffer)
end

"""
    gather!(fs::FieldSystem)

Copy fields into system buffer.
"""
function gather!(fs::FieldSystem)
    for (field, av) in zip(fs.fields, fs.array_views)
        av .= field["c"]
    end
    fs.group_buffer .= fs.perm * fs.field_buffer
end

"""
    scatter!(fs::FieldSystem)

Extract fields from system buffer.
"""
function scatter!(fs::FieldSystem)
    fs.field_buffer .= transpose(fs.perm) * fs.group_buffer
    for (field, av) in zip(fs.fields, fs.array_views)
        field["c"] = av
    end
end

# ============================================================
# Exports
# ============================================================

export CoeffSystem,
       FieldSystem,
       get_subdata,
       gather!,
       scatter!
