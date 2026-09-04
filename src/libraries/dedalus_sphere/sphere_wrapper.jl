"""
Sphere wrapper module for Dedalus.jl.

Translated from dedalus/libraries/dedalus_sphere/sphere_wrapper.py.
Provides the SphereWrapper struct that precomputes quadrature grids, transform
matrices, and operators for spin-weighted spherical harmonic transforms.

This module defines:
- SphereWrapper: main struct holding grids, weights, and transform matrices
- forward_spin / backward_spin: grid-to-coefficient and coefficient-to-grid transforms
- op: cached operator matrix access
- tensor_index: compute tensor index ranges for multi-component fields
- forward / backward: full tensor transforms
- grad: gradient in coefficient space
"""

using SparseArrays
using LinearAlgebra

# ============================================================================
# SphereWrapper
# ============================================================================

"""
    SphereWrapper

Precomputes and caches quadrature grids, SWSH transform matrices, and operators
for spin-weighted spherical harmonic transforms on the sphere.

Named SphereWrapper (not Sphere) to avoid conflict with other types.

Fields
------
- L_max    : maximum spherical-harmonic degree
- S_max    : maximum spin weight
- N_theta  : number of theta grid points
- cos_grid : cos(theta) quadrature nodes
- weights  : quadrature weights
- grid     : theta = arccos(cos_grid)
- sin_grid : sin(theta)
- pushY    : Dict of transform matrices for forward (grid -> coeff) transform
- pullY    : Dict of transform matrices for backward (coeff -> grid) transform
- _op_cache     : Dict for caching operator matrices
- _tidx_cache   : Dict for caching tensor_index results
- _unitary_cache : Dict for caching unitary matrices
"""
mutable struct SphereWrapper
    L_max::Int
    S_max::Int
    N_theta::Int
    cos_grid::Vector{Float64}
    weights::Vector{Float64}
    grid::Vector{Float64}
    sin_grid::Vector{Float64}
    pushY::Dict{Tuple{Int,Int}, Matrix{Float64}}
    pullY::Dict{Tuple{Int,Int}, Matrix{Float64}}
    _op_cache::Dict{Tuple{String,Int,Int}, Any}
    _tidx_cache::Dict{Tuple{Int,Int}, Any}
    _unitary_cache::Dict{Tuple{Int,Bool}, Any}

    function SphereWrapper(L_max::Int; S_max::Int=0, N_theta::Union{Nothing,Int}=nothing,
                           m_min::Union{Nothing,Int}=nothing, m_max::Union{Nothing,Int}=nothing)
        if N_theta === nothing
            N_theta = L_max + 1
        end
        if m_min === nothing
            m_min = -L_max
        end
        if m_max === nothing
            m_max = L_max
        end

        # Compute quadrature grid and weights
        cos_grid, weights = sphere_quadrature(N_theta - 1)
        grid_theta = acos.(cos_grid)
        sin_grid_vals = sqrt.(1.0 .- cos_grid .^ 2)

        pushY = Dict{Tuple{Int,Int}, Matrix{Float64}}()
        pullY = Dict{Tuple{Int,Int}, Matrix{Float64}}()

        for s in -S_max:S_max
            for m in m_min:m_max
                Y = sphere_harmonics(L_max, m, s, cos_grid)
                # pushY: weights * Y for forward transform (grid -> coeff)
                # Y has shape (n_modes, n_theta), weights is n_theta vector
                # pushY[(m,s)] = (weights .* Y') transposed appropriately
                # In Python: pushY = (weights * Y).astype(float64)
                # weights is (n_theta,), Y is (n_modes, n_theta)
                # broadcasting: weights * Y multiplies each column of Y by corresponding weight
                pushY[(m, s)] = Float64.(Y .* weights')
                # pullY: Y^T for backward transform (coeff -> grid)
                pullY[(m, s)] = Float64.(Y')
            end
        end

        # Downcast grids to Float64
        cos_grid = Float64.(cos_grid)
        weights = Float64.(weights)
        grid_theta = Float64.(grid_theta)
        sin_grid_vals = Float64.(sin_grid_vals)

        new(L_max, S_max, N_theta, cos_grid, weights, grid_theta, sin_grid_vals,
            pushY, pullY,
            Dict{Tuple{String,Int,Int}, Any}(),
            Dict{Tuple{Int,Int}, Any}(),
            Dict{Tuple{Int,Bool}, Any}())
    end
end

# ============================================================================
# op - cached operator matrix access
# ============================================================================

"""
    op(wrapper::SphereWrapper, op_name::String, m::Int, s::Int)

Get the operator matrix for the given operator name and (m, s) parameters.
Results are cached for repeated calls.
"""
function op(wrapper::SphereWrapper, op_name::String, m::Int, s::Int)
    key = (op_name, m, s)
    if haskey(wrapper._op_cache, key)
        return wrapper._op_cache[key]
    end
    result = sphere_op(op_name, wrapper.L_max, m, s)
    # Convert to Float64 dense or sparse as appropriate
    result = Float64.(result)
    wrapper._op_cache[key] = result
    return result
end

# ============================================================================
# L_min - minimum ell
# ============================================================================

"""
    L_min(wrapper::SphereWrapper, m::Int, s::Int)

Minimum spherical harmonic degree for given (m, s).
"""
function L_min(wrapper::SphereWrapper, m::Int, s::Int)
    return sphere_L_min(m, s)
end

# ============================================================================
# zeros
# ============================================================================

"""
    sphere_wrapper_zeros(wrapper::SphereWrapper, m::Int, s_out::Int, s_in::Int)

Create a zero matrix of appropriate dimensions for an operator mapping
from spin s_in to spin s_out.
"""
function sphere_wrapper_zeros(wrapper::SphereWrapper, m::Int, s_out::Int, s_in::Int)
    return sphere_zeros(wrapper.L_max, m, s_out, s_in)
end

# ============================================================================
# forward_spin / backward_spin
# ============================================================================

"""
    forward_spin(wrapper::SphereWrapper, m::Int, s::Int, data)

Transform from grid space to coefficient space for a single spin component.
"""
function forward_spin(wrapper::SphereWrapper, m::Int, s::Int, data)
    return wrapper.pushY[(m, s)] * data
end

"""
    backward_spin(wrapper::SphereWrapper, m::Int, s::Int, data)

Transform from coefficient space to grid space for a single spin component.
"""
function backward_spin(wrapper::SphereWrapper, m::Int, s::Int, data)
    return wrapper.pullY[(m, s)] * data
end

# ============================================================================
# tensor_index
# ============================================================================

"""
    tensor_index(wrapper::SphereWrapper, m::Int, rank::Int)

Compute tensor index ranges for multi-component fields.

Returns (start_index, end_index, spin) where:
- start_index : 1-based starting indices for each spin component
- end_index   : 1-based ending indices for each spin component
- spin        : spin values for each component
"""
function tensor_index(wrapper::SphereWrapper, m::Int, rank::Int)
    key = (m, rank)
    if haskey(wrapper._tidx_cache, key)
        return wrapper._tidx_cache[key]
    end

    num = collect(0:(2^rank - 1))
    spin = (-1) .^ num
    for k in 2:rank
        spin .+= (-1) .^ (num .÷ 2^(k-1))
    end

    if rank == 0
        spin = [0]
    end

    # Use 1-based indexing for start/end indices
    start_index = Int[1]
    end_index = Int[]
    for k in 1:(2^rank)
        push!(end_index, start_index[k] + wrapper.L_max - sphere_L_min(m, spin[k]))
        if k < 2^rank
            push!(start_index, end_index[k] + 1)
        end
    end

    result = (start_index, end_index, spin)
    wrapper._tidx_cache[key] = result
    return result
end

# ============================================================================
# unitary
# ============================================================================

"""
    unitary(wrapper::SphereWrapper; rank::Int=1, adjoint::Bool=false)

Get the unitary transformation matrix between spin and regularity bases.
Results are cached.
"""
function unitary(wrapper::SphereWrapper; rank::Int=1, adjoint::Bool=false)
    key = (rank, adjoint)
    if haskey(wrapper._unitary_cache, key)
        return wrapper._unitary_cache[key]
    end
    result = sphere_unitary(; rank=rank, adjoint=adjoint)
    wrapper._unitary_cache[key] = result
    return result
end

# ============================================================================
# forward - full tensor transform
# ============================================================================

"""
    forward(wrapper::SphereWrapper, m::Int, rank::Int, data; unitary_mat=nothing)

Full tensor transform from grid space to coefficient space.

For rank 0, this is just forward_spin with spin=0.
For rank > 0, applies the unitary transformation to rotate from Cartesian
to spin basis, then transforms each spin component separately.
"""
function forward(wrapper::SphereWrapper, m::Int, rank::Int, data; unitary_mat=nothing)
    if rank == 0
        return forward_spin(wrapper, m, 0, data)
    end

    (si, ei, sp) = tensor_index(wrapper, m, rank)

    if unitary_mat === nothing
        unitary_mat = unitary(wrapper; rank=rank, adjoint=true)
    end

    # Apply unitary transformation: data_rot[i,:] = sum_j unitary[i,j] * data[j,:]
    # data has first dimension 2^rank, remaining dimensions are spatial
    if ndims(data) == 1
        data_rot = unitary_mat * data
    else
        # einsum "ij,j...->i..." equivalent
        n_comp = size(data, 1)
        rest_shape = size(data)[2:end]
        data_2d = reshape(data, n_comp, :)
        rot_2d = unitary_mat * data_2d
        data_rot = reshape(rot_2d, size(unitary_mat, 1), rest_shape...)
    end

    # Allocate output coefficient array
    total_coeffs = ei[end]
    if ndims(data) <= 2
        if ndims(data) == 1
            data_c = zeros(ComplexF64, total_coeffs)
        else
            rest_shape = size(data)[2:end]
            data_c = zeros(ComplexF64, total_coeffs, rest_shape[2:end]...)
        end
    else
        rest_shape = size(data)[2:end]
        out_shape = (total_coeffs, rest_shape[2:end]...)
        data_c = zeros(ComplexF64, out_shape...)
    end

    for i in 1:(2^rank)
        if ndims(data) == 1
            data_c[si[i]:ei[i]] = forward_spin(wrapper, m, sp[i], data_rot[i, :])
        elseif ndims(data) == 2
            data_c[si[i]:ei[i], :] = forward_spin(wrapper, m, sp[i],
                selectdim(data_rot, 1, i) isa AbstractVector ?
                reshape(selectdim(data_rot, 1, i), 1, :) :
                selectdim(data_rot, 1, i))
            # Simpler: treat each component as its row
            # forward_spin returns (n_modes, ...) from (n_theta, ...)
        else
            # General multi-dimensional case
            src = selectdim(data_rot, 1, i)
            coeffs = forward_spin(wrapper, m, sp[i], src)
            selectdim(data_c, 1, si[i]:ei[i]) .= coeffs
        end
    end

    return data_c
end

# ============================================================================
# backward - full tensor transform
# ============================================================================

"""
    backward(wrapper::SphereWrapper, m::Int, rank::Int, data; unitary_mat=nothing)

Full tensor transform from coefficient space to grid space.

For rank 0, this is just backward_spin with spin=0.
For rank > 0, transforms each spin component separately, then applies the
unitary transformation to rotate from spin back to Cartesian basis.
"""
function backward(wrapper::SphereWrapper, m::Int, rank::Int, data; unitary_mat=nothing)
    if rank == 0
        return backward_spin(wrapper, m, 0, data)
    end

    (si, ei, sp) = tensor_index(wrapper, m, rank)

    if unitary_mat === nothing
        unitary_mat = unitary(wrapper; rank=rank, adjoint=false)
    end

    # Allocate grid-space array with shape (2^rank, N_theta, ...)
    if ndims(data) == 1
        data_g = zeros(ComplexF64, 2^rank, wrapper.N_theta)
    else
        rest_shape = size(data)[2:end]
        data_g = zeros(ComplexF64, 2^rank, wrapper.N_theta, rest_shape...)
    end

    for i in 1:(2^rank)
        if ndims(data) == 1
            data_g[i, :] = backward_spin(wrapper, m, sp[i], data[si[i]:ei[i]])
        else
            selectdim(data_g, 1, i) .= backward_spin(wrapper, m, sp[i],
                data[si[i]:ei[i], ntuple(_ -> Colon(), ndims(data) - 1)...])
        end
    end

    # Apply unitary transformation
    if ndims(data_g) == 2
        result = unitary_mat * data_g
    else
        n_comp = size(data_g, 1)
        rest_shape = size(data_g)[2:end]
        g_2d = reshape(data_g, n_comp, :)
        rot_2d = unitary_mat * g_2d
        result = reshape(rot_2d, size(unitary_mat, 1), rest_shape...)
    end

    return result
end

# ============================================================================
# grad - gradient in coefficient space
# ============================================================================

"""
    grad(wrapper::SphereWrapper, m::Int, rank_in::Int, data, data_out)

Compute the gradient in coefficient space.

Transforms data from rank_in to rank_in+1 using the spin-raising and
spin-lowering operators (k+ and k-).

Parameters
----------
- wrapper  : SphereWrapper
- m        : azimuthal wavenumber
- rank_in  : input tensor rank
- data     : input coefficient data
- data_out : output array (modified in-place)
"""
function grad(wrapper::SphereWrapper, m::Int, rank_in::Int, data, data_out)
    (si_in, ei_in, sp_in) = tensor_index(wrapper, m, rank_in)
    rank_out = rank_in + 1
    (si_out, ei_out, sp_out) = tensor_index(wrapper, m, rank_out)

    half = 2^(rank_out - 1)
    for i in 1:(2^rank_out)
        if (i - 1) ÷ half == 0
            operator_mat = op(wrapper, "k+", m, sp_in[((i - 1) % half) + 1])
        else
            operator_mat = op(wrapper, "k-", m, sp_in[((i - 1) % half) + 1])
        end

        # Source index (1-based)
        src_idx = ((i - 1) % half) + 1
        src = data[si_in[src_idx]:ei_in[src_idx]]
        dst_range = si_out[i]:ei_out[i]
        data_out[dst_range] .= operator_mat * src
    end
end
