"""
    Linear algebra routines for CSR sparse matrix operations.

Translated from the Cython `dedalus/tools/linalg.pyx`. Provides CSR
(Compressed Sparse Row) matrix storage and two core operations applied along
arbitrary axes of multidimensional arrays:

- [`apply_csr!`](@ref)  -- CSR matrix-vector product along a specified axis.
- [`solve_upper_csr!`](@ref) -- backward substitution for an upper-triangular
  CSR system along a specified axis.

Julia natively uses CSC (Compressed Sparse Column) via `SparseMatrixCSC`, so
we define a [`CSRMatrix`](@ref) struct that stores the CSR representation
explicitly. A convenience constructor converts from `SparseMatrixCSC`.

# Axis convention

All axis arguments are **1-based** (Julia convention). Internally, the
dispatcher reshapes high-dimensional arrays into at most 3 dimensions and
delegates to one of four specialised kernels:

- `_vec`   -- 1-D vector (no reshape needed).
- `_first` -- operation along the first axis of a 2-D view.
- `_last`  -- operation along the last axis of a 2-D view.
- `_mid`   -- operation along the middle axis of a 3-D view (general case).
"""

using SparseArrays: SparseMatrixCSC, sparse

export CSRMatrix, apply_csr!, solve_upper_csr!

# ============================================================================
# CSRMatrix
# ============================================================================

"""
    CSRMatrix{T}

Compressed Sparse Row matrix representation.

# Fields
- `m::Int`              -- number of rows.
- `n::Int`              -- number of columns.
- `indptr::Vector{Int}` -- row pointer array of length `m + 1` (1-based).
- `indices::Vector{Int}` -- column indices of non-zero entries (1-based).
- `nzval::Vector{T}`    -- non-zero values corresponding to `indices`.

The row `i` has non-zero entries at positions `indptr[i] : indptr[i+1]-1` in
the `indices` and `nzval` arrays.

# Constructors

    CSRMatrix(m, n, indptr, indices, nzval)
    CSRMatrix(A::SparseMatrixCSC)

The `SparseMatrixCSC` constructor transposes the internal CSC storage to
obtain CSR arrays directly (CSR of A == CSC of A^T).
"""
struct CSRMatrix{T}
    m::Int
    n::Int
    indptr::Vector{Int}
    indices::Vector{Int}
    nzval::Vector{T}
end

"""
    CSRMatrix(A::SparseMatrixCSC{T}) where T

Construct a [`CSRMatrix`](@ref) from a `SparseMatrixCSC`.

The CSR representation of A is equivalent to the CSC representation of A^T,
so we transpose and read off `colptr`, `rowval`, `nzval` directly.
"""
function CSRMatrix(A::SparseMatrixCSC{T}) where {T}
    At = sparse(transpose(A))  # SparseMatrixCSC with CSC of A^T = CSR of A
    return CSRMatrix{T}(A.m, A.n, copy(At.colptr), copy(At.rowval), copy(At.nzval))
end

Base.size(C::CSRMatrix) = (C.m, C.n)
Base.size(C::CSRMatrix, d::Integer) = d == 1 ? C.m : d == 2 ? C.n : 1
Base.eltype(::CSRMatrix{T}) where {T} = T

# ============================================================================
# apply_csr!  --  CSR matrix-vector product along an axis
# ============================================================================

"""
    apply_csr!(indptr, indices, entries, x, y, axis)

Apply a CSR matrix (given by raw `indptr`, `indices`, `entries` arrays) to the
multidimensional array `x` along dimension `axis`, writing the result into `y`.

`y` must be pre-allocated with the appropriate shape (the size along `axis`
equals the number of rows of the CSR matrix, which is `length(indptr) - 1`).

`x` and `y` may share all dimensions except `axis`, where `x` has size equal
to the number of columns and `y` has size equal to the number of rows.

# Arguments
- `indptr::AbstractVector{Int}`  -- CSR row pointers (length `n_row + 1`, 1-based).
- `indices::AbstractVector{Int}` -- CSR column indices (1-based).
- `entries::AbstractVector`      -- CSR non-zero values.
- `x::AbstractArray`             -- input array.
- `y::AbstractArray`             -- output array (mutated in place).
- `axis::Int`                    -- 1-based axis along which to apply the matrix.
"""
function apply_csr!(indptr::AbstractVector{Int},
                    indices::AbstractVector{Int},
                    entries::AbstractVector,
                    x::AbstractArray,
                    y::AbstractArray,
                    axis::Int)
    N = ndims(x)
    n_row = length(indptr) - 1

    if N == 1
        _apply_csr_vec!(indptr, indices, entries, x, y, n_row)
    elseif axis == 1
        # First axis: reshape trailing dimensions into a single trailing dim.
        n_after = div(length(x), size(x, 1))
        xr = reshape(x, size(x, 1), n_after)
        yr = reshape(y, n_row, n_after)
        _apply_csr_first!(indptr, indices, entries, xr, yr, n_row, n_after)
    elseif axis == N
        # Last axis: reshape leading dimensions into a single leading dim.
        n_before = div(length(x), size(x, N))
        xr = reshape(x, n_before, size(x, N))
        yr = reshape(y, n_before, n_row)
        _apply_csr_last!(indptr, indices, entries, xr, yr, n_row, n_before)
    else
        # Middle axis: collapse dims before and after into single dims -> 3-D.
        n_before = prod(size(x, d) for d in 1:axis-1)
        n_after  = prod(size(x, d) for d in axis+1:N)
        xr = reshape(x, n_before, size(x, axis), n_after)
        yr = reshape(y, n_before, n_row, n_after)
        _apply_csr_mid!(indptr, indices, entries, xr, yr, n_row, n_before, n_after)
    end
    return y
end

# -- 1-D vector kernel -------------------------------------------------------

function _apply_csr_vec!(indptr, indices, entries, x, y, n_row)
    Threads.@threads for i in 1:n_row
        s = zero(eltype(y))
        @inbounds for jj in indptr[i]:indptr[i+1]-1
            s += entries[jj] * x[indices[jj]]
        end
        @inbounds y[i] = s
    end
    return nothing
end

# -- first-axis kernel (axis == 1 of 2-D) ------------------------------------

function _apply_csr_first!(indptr, indices, entries, x, y, n_row, n_after)
    Threads.@threads for i in 1:n_row
        @inbounds for k in 1:n_after
            y[i, k] = zero(eltype(y))
        end
        @inbounds for jj in indptr[i]:indptr[i+1]-1
            j = indices[jj]
            a = entries[jj]
            for k in 1:n_after
                y[i, k] += a * x[j, k]
            end
        end
    end
    return nothing
end

# -- last-axis kernel (axis == last of 2-D) ----------------------------------

function _apply_csr_last!(indptr, indices, entries, x, y, n_row, n_before)
    total = n_before * n_row
    Threads.@threads for hi in 1:total
        h = div(hi - 1, n_row) + 1
        i = mod(hi - 1, n_row) + 1
        s = zero(eltype(y))
        @inbounds for jj in indptr[i]:indptr[i+1]-1
            s += entries[jj] * x[h, indices[jj]]
        end
        @inbounds y[h, i] = s
    end
    return nothing
end

# -- middle-axis kernel (general 3-D case) -----------------------------------

function _apply_csr_mid!(indptr, indices, entries, x, y, n_row, n_before, n_after)
    total = n_before * n_row
    Threads.@threads for hi in 1:total
        h = div(hi - 1, n_row) + 1
        i = mod(hi - 1, n_row) + 1
        @inbounds for k in 1:n_after
            y[h, i, k] = zero(eltype(y))
        end
        @inbounds for jj in indptr[i]:indptr[i+1]-1
            j = indices[jj]
            a = entries[jj]
            for k in 1:n_after
                y[h, i, k] += a * x[h, j, k]
            end
        end
    end
    return nothing
end

# ============================================================================
# solve_upper_csr!  --  backward substitution for upper-triangular CSR
# ============================================================================

"""
    solve_upper_csr!(indptr, indices, entries, x, axis)

Solve an upper-triangular CSR system in place along dimension `axis` of the
array `x`. The diagonal element for row `i` is assumed to be the *first*
non-zero in that row (i.e., `entries[indptr[i]]`), and the remaining entries
in the row are the strictly upper-triangular part.

After this call, `x` contains the solution vector(s).

# Arguments
- `indptr::AbstractVector{Int}`  -- CSR row pointers (length `n_row + 1`, 1-based).
- `indices::AbstractVector{Int}` -- CSR column indices (1-based).
- `entries::AbstractVector`      -- CSR non-zero values.
- `x::AbstractArray`             -- right-hand side on entry, solution on exit (mutated).
- `axis::Int`                    -- 1-based axis along which to solve.
"""
function solve_upper_csr!(indptr::AbstractVector{Int},
                          indices::AbstractVector{Int},
                          entries::AbstractVector,
                          x::AbstractArray,
                          axis::Int)
    N = ndims(x)
    n_row = length(indptr) - 1

    if N == 1
        _solve_upper_csr_vec!(indptr, indices, entries, x, n_row)
    elseif axis == 1
        n_after = div(length(x), size(x, 1))
        xr = reshape(x, size(x, 1), n_after)
        _solve_upper_csr_first!(indptr, indices, entries, xr, n_row, n_after)
    elseif axis == N
        n_before = div(length(x), size(x, N))
        xr = reshape(x, n_before, size(x, N))
        _solve_upper_csr_last!(indptr, indices, entries, xr, n_row, n_before)
    else
        n_before = prod(size(x, d) for d in 1:axis-1)
        n_after  = prod(size(x, d) for d in axis+1:N)
        xr = reshape(x, n_before, size(x, axis), n_after)
        _solve_upper_csr_mid!(indptr, indices, entries, xr, n_row, n_before, n_after)
    end
    return x
end

# -- 1-D vector kernel -------------------------------------------------------

function _solve_upper_csr_vec!(indptr, indices, entries, x, n_row)
    @inbounds for i in n_row:-1:1
        s = x[i]
        for jj in indptr[i+1]-1:-1:indptr[i]+1
            s -= entries[jj] * x[indices[jj]]
        end
        x[i] = s / entries[indptr[i]]
    end
    return nothing
end

# -- first-axis kernel (axis == 1 of 2-D) ------------------------------------

function _solve_upper_csr_first!(indptr, indices, entries, x, n_row, n_after)
    @inbounds for i in n_row:-1:1
        for jj in indptr[i+1]-1:-1:indptr[i]+1
            j = indices[jj]
            a = entries[jj]
            for k in 1:n_after
                x[i, k] -= a * x[j, k]
            end
        end
        a = entries[indptr[i]]
        for k in 1:n_after
            x[i, k] /= a
        end
    end
    return nothing
end

# -- last-axis kernel (axis == last of 2-D) ----------------------------------

function _solve_upper_csr_last!(indptr, indices, entries, x, n_row, n_before)
    Threads.@threads for h in 1:n_before
        @inbounds for i in n_row:-1:1
            s = x[h, i]
            for jj in indptr[i+1]-1:-1:indptr[i]+1
                s -= entries[jj] * x[h, indices[jj]]
            end
            x[h, i] = s / entries[indptr[i]]
        end
    end
    return nothing
end

# -- middle-axis kernel (general 3-D case) -----------------------------------

function _solve_upper_csr_mid!(indptr, indices, entries, x, n_row, n_before, n_after)
    Threads.@threads for h in 1:n_before
        @inbounds for i in n_row:-1:1
            for jj in indptr[i+1]-1:-1:indptr[i]+1
                j = indices[jj]
                a = entries[jj]
                for k in 1:n_after
                    x[h, i, k] -= a * x[h, j, k]
                end
            end
            a = entries[indptr[i]]
            for k in 1:n_after
                x[h, i, k] /= a
            end
        end
    end
    return nothing
end
