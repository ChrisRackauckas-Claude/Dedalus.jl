"""
    Array and sparse-matrix utilities for Dedalus.jl

Julia translation of the Python Dedalus `array.py` module. Provides helper
functions for multidimensional array manipulation, sparse matrix construction
and application, eigenvalue solvers, and permutation matrices.

Key differences from the Python original:
- Julia is 1-indexed and column-major; all `axis` parameters are 1-based.
- Julia's default sparse format is CSC (`SparseMatrixCSC`), not CSR.
- `interleaved_view` uses `reinterpret` instead of numpy's buffer trick.
- `apply_sparse` / `solve_upper_sparse` forward to linalg module helpers
  (`apply_csc!`, `solve_upper_csc!`) using the native CSC format.
"""

using SparseArrays
using LinearAlgebra

# config.jl is included by the main module before this file

# ---------------------------------------------------------------------------
# Module-level configuration flags
# ---------------------------------------------------------------------------

const SPLIT_CSR_MATVECS = get_config_bool("linear_algebra", "SPLIT_CSR_MATVECS")
const OLD_CSR_MATVECS   = get_config_bool("linear_algebra", "OLD_CSR_MATVECS")

# ---------------------------------------------------------------------------
# interleaved_view
# ---------------------------------------------------------------------------

"""
    interleaved_view(data::AbstractArray{ComplexF64}) -> AbstractArray{Float64}

View an n-dimensional `ComplexF64` array as an (n+1)-dimensional `Float64`
array whose *first* axis (length 2) separates the real and imaginary parts.

This is the Julia analogue of viewing the raw memory; it uses `reinterpret`
so no data is copied.

# Examples
```julia
z = ComplexF64[1+2im, 3+4im]
iv = interleaved_view(z)  # 2x2 Float64 matrix: [1.0 3.0; 2.0 4.0]
```
"""
function interleaved_view(data::AbstractArray{ComplexF64})
    # reinterpret gives a flat Float64 view; reshape to (2, original_shape...)
    flat = reinterpret(Float64, data)
    return reshape(flat, 2, size(data)...)
end

# ---------------------------------------------------------------------------
# reshape_vector
# ---------------------------------------------------------------------------

"""
    reshape_vector(data::AbstractVector, dim::Int=2, axis::Int=dim) -> AbstractArray

Reshape a 1-dimensional array into a multidimensional array of `dim`
dimensions with all singleton dimensions except along `axis`.

`axis` is 1-based (Julia convention). The default `axis=dim` corresponds to
the Python default of `axis=-1`.

# Examples
```julia
v = [1.0, 2.0, 3.0]
reshape_vector(v, 3, 2)  # size (1, 3, 1)
```
"""
function reshape_vector(data::AbstractVector, dim::Int=2, axis::Int=dim)
    shape = ones(Int, dim)
    shape[axis] = length(data)
    return reshape(data, Tuple(shape))
end

# ---------------------------------------------------------------------------
# axindex / axslice
# ---------------------------------------------------------------------------

"""
    axindex(axis::Int, index) -> Tuple

Build an indexing tuple that selects `index` along `axis` and `:` (all)
along every other leading axis. `axis` must be >= 1 (1-based).

# Examples
```julia
A = rand(3, 4, 5)
A[axindex(2, 3)...]  # equivalent to A[:, 3, :]  -- but note this gives A[:, 3]
```
"""
function axindex(axis::Int, index)
    if axis < 1
        throw(ArgumentError("`axis` must be >= 1 (1-based)"))
    end
    return ntuple(i -> i == axis ? index : Colon(), axis)
end

"""
    axslice(axis::Int, start, stop, step=nothing) -> Tuple

Slice an array along a specified `axis` (1-based). Constructs the
appropriate range from `start:step:stop` (or `start:stop` when `step` is
`nothing`).

# Examples
```julia
A = rand(10, 10)
A[axslice(1, 2, 5)...]  # rows 2:5, all columns
```
"""
function axslice(axis::Int, start, stop, step=nothing)
    if step === nothing
        return axindex(axis, start:stop)
    else
        return axindex(axis, start:step:stop)
    end
end

# ---------------------------------------------------------------------------
# zeros_with_pattern
# ---------------------------------------------------------------------------

"""
    zeros_with_pattern(matrices::AbstractSparseMatrix...) -> SparseMatrixCSC

Create a sparse matrix whose sparsity pattern is the union of the patterns of
the given sparse matrices, but with all values set to zero.
"""
function zeros_with_pattern(matrices::AbstractSparseMatrix...)
    result = spzeros(eltype(first(matrices)), size(first(matrices))...)
    for A in matrices
        I, J, V = findnz(A)
        result += sparse(I, J, zero(V), size(A)...)
    end
    return result
end

# ---------------------------------------------------------------------------
# expand_pattern
# ---------------------------------------------------------------------------

"""
    expand_pattern(input::AbstractSparseMatrix, pattern::AbstractSparseMatrix) -> SparseMatrixCSC

Return a copy of `input` whose sparsity pattern has been expanded to also
include all the structural nonzeros of `pattern` (with zero values for the
new entries).
"""
function expand_pattern(input::AbstractSparseMatrix, pattern::AbstractSparseMatrix)
    Ai, Aj, Av = findnz(input)
    Pi, Pj, _  = findnz(pattern)
    rows = vcat(Ai, Pi)
    cols = vcat(Aj, Pj)
    data = vcat(Av, zeros(eltype(Av), length(Pi)))
    return sparse(rows, cols, data, size(input)...)
end

# ---------------------------------------------------------------------------
# apply_matrix
# ---------------------------------------------------------------------------

"""
    apply_matrix(matrix, array::AbstractArray, axis::Int; kw...)

Apply `matrix` (dense or sparse) to `array` along dimension `axis` (1-based).
Dispatches to [`apply_dense`](@ref) or [`apply_sparse`](@ref) depending on
the type of `matrix`.
"""
function apply_matrix(matrix, array::AbstractArray, axis::Int; kw...)
    if issparse(matrix)
        return apply_sparse(matrix, array, axis; kw...)
    else
        return apply_dense(matrix, array, axis; kw...)
    end
end

# ---------------------------------------------------------------------------
# move_single_axis
# ---------------------------------------------------------------------------

"""
    move_single_axis(a::AbstractArray, source::Int, destination::Int) -> AbstractArray

Move dimension `source` of `a` so that it becomes dimension `destination`,
shifting the remaining dimensions accordingly. Similar to `numpy.moveaxis`
for a single axis. All indices are 1-based.
"""
function move_single_axis(a::AbstractArray, source::Int, destination::Int)
    order = [n for n in 1:ndims(a) if n != source]
    insert!(order, destination, source)
    return permutedims(a, order)
end

# ---------------------------------------------------------------------------
# apply_dense
# ---------------------------------------------------------------------------

"""
    apply_dense(matrix::AbstractMatrix, array::AbstractArray, axis::Int;
                out::Union{AbstractArray,Nothing}=nothing) -> AbstractArray

Apply a dense matrix along dimension `axis` (1-based) of `array`.

If `out` is provided, the result is written into `out` and returned;
otherwise a new array is allocated.
"""
function apply_dense(matrix::AbstractMatrix, array::AbstractArray, axis::Int;
                     out::Union{AbstractArray,Nothing}=nothing)
    dim = ndims(array)
    # Normalise negative-style axis (though Julia conventionally uses positive)
    axis = mod1(axis, dim)
    if axis != 1
        array = move_single_axis(array, axis, 1)
    end
    local array_shape
    if dim > 2
        array_shape = size(array)
        array = reshape(array, size(array, 1), :)
    end
    temp = matrix * array
    if dim > 2
        temp = reshape(temp, size(temp, 1), array_shape[2:end]...)
    end
    if axis != 1
        temp = move_single_axis(temp, 1, axis)
    end
    if out === nothing
        return temp
    else
        copyto!(out, temp)
        return out
    end
end

# ---------------------------------------------------------------------------
# apply_sparse
# ---------------------------------------------------------------------------

"""
    apply_sparse(matrix::SparseMatrixCSC, array::AbstractArray, axis::Int;
                 out::Union{AbstractArray,Nothing}=nothing,
                 check_shapes::Bool=false) -> AbstractArray

Apply a sparse CSC matrix along dimension `axis` (1-based) of `array`.
The operation is always out-of-place; passing `out === array` is an error.

When the `linalg` module is available, this calls `apply_csc!` for
optimised multidimensional sparse matvec. The fallback uses dense slicing.

# Arguments
- `matrix`: Sparse matrix in CSC format.
- `array`: Input multidimensional array.
- `axis`: Dimension along which to apply the matrix (1-based).
- `out`: Pre-allocated output array (optional).
- `check_shapes`: If `true`, validate dimension compatibility.
"""
function apply_sparse(matrix::SparseMatrixCSC, array::AbstractArray, axis::Int;
                      out::Union{AbstractArray,Nothing}=nothing,
                      check_shapes::Bool=false)
    if out === nothing
        out_shape = collect(size(array))
        out_shape[axis] = size(matrix, 1)
        out = similar(array, Tuple(out_shape))
    elseif out === array
        throw(ArgumentError("Cannot apply sparse matrix in place"))
    end
    if check_shapes
        if !(1 <= axis <= ndims(array))
            throw(ArgumentError("Axis out of bounds."))
        end
        if size(matrix, 2) != size(array, axis) || size(matrix, 1) != size(out, axis)
            throw(ArgumentError("Matrix shape mismatch."))
        end
    end
    # Fallback implementation: move target axis to first position, multiply, move back
    dim = ndims(array)
    a = axis != 1 ? move_single_axis(array, axis, 1) : array
    a_shape = size(a)
    if dim > 2
        a = reshape(a, size(a, 1), :)
    elseif dim == 1
        a = reshape(a, size(a, 1), 1)
    end
    temp = matrix * a
    if dim > 2
        temp = reshape(temp, size(temp, 1), a_shape[2:end]...)
    elseif dim == 1
        temp = vec(temp)
    end
    if axis != 1
        temp = move_single_axis(temp, 1, axis)
    end
    copyto!(out, temp)
    return out
end

# ---------------------------------------------------------------------------
# solve_upper_sparse
# ---------------------------------------------------------------------------

"""
    solve_upper_sparse(matrix::SparseMatrixCSC, rhs::AbstractArray, axis::Int;
                       out::Union{AbstractArray,Nothing}=nothing,
                       check_shapes::Bool=false)

Solve an upper-triangular sparse system along dimension `axis` (1-based) of
`rhs`. The solution is computed in-place in `out` (which defaults to a copy
of `rhs`).

When the `linalg` module is available, this calls `solve_upper_csc!`.
The fallback uses Julia's built-in upper-triangular solver.
"""
function solve_upper_sparse(matrix::SparseMatrixCSC, rhs::AbstractArray, axis::Int;
                            out::Union{AbstractArray,Nothing}=nothing,
                            check_shapes::Bool=false)
    if out === nothing
        out = copy(rhs)
    elseif out !== rhs
        copyto!(out, rhs)
    end
    if check_shapes
        if !(1 <= axis <= ndims(rhs))
            throw(ArgumentError("Axis out of bounds."))
        end
        if !(size(matrix, 1) == size(matrix, 2) == size(rhs, axis))
            throw(ArgumentError("Matrix shape mismatch."))
        end
    end
    U = UpperTriangular(matrix)
    # Solve along the specified axis by moving it to the first position
    dim = ndims(out)
    if axis != 1
        out_perm = move_single_axis(out, axis, 1)
    else
        out_perm = out
    end
    perm_shape = size(out_perm)
    if dim > 2
        out_flat = reshape(out_perm, size(out_perm, 1), :)
    elseif dim == 1
        out_flat = reshape(out_perm, size(out_perm, 1), 1)
    else
        out_flat = out_perm
    end
    ldiv!(U, out_flat)
    # Reshape and permute back into `out`
    if dim > 2
        result = reshape(out_flat, perm_shape)
    elseif dim == 1
        result = vec(out_flat)
    else
        result = out_flat
    end
    if axis != 1
        result = move_single_axis(result, 1, axis)
        copyto!(out, result)
    end
    return out
end

# ---------------------------------------------------------------------------
# add_sparse
# ---------------------------------------------------------------------------

"""
    add_sparse(A, B)

Add sparse matrices, promoting scalars to multiples of the identity matrix
when one operand is scalar and the other is a matrix.

# Examples
```julia
M = sprand(5, 5, 0.3)
add_sparse(2.0, M)   # 2*I + M
add_sparse(M, 3.0)   # M + 3*I
add_sparse(1.0, 2.0) # 3.0
```
"""
function add_sparse(A, B)
    A_is_scalar = isa(A, Number)
    B_is_scalar = isa(B, Number)
    if A_is_scalar && B_is_scalar
        return A + B
    elseif A_is_scalar
        n = size(B, 1)
        I_mat = sparse(one(eltype(B)) * I, n, n)
        return A * I_mat + B
    elseif B_is_scalar
        n = size(A, 1)
        I_mat = sparse(one(eltype(A)) * I, n, n)
        return A + B * I_mat
    else
        return A + B
    end
end

# ---------------------------------------------------------------------------
# sparse_block_diag
# ---------------------------------------------------------------------------

"""
    sparse_block_diag(blocks; shape::Union{Tuple{Int,Int},Nothing}=nothing) -> SparseMatrixCSC

Build a block-diagonal sparse matrix from a collection of sparse matrix blocks.
Unlike `blockdiag` from SparseArrays, this correctly handles blocks of size 0.

# Arguments
- `blocks`: Iterable of sparse matrices.
- `shape`: Optional overall shape `(m, n)`. Defaults to the sum of block sizes.
"""
function sparse_block_diag(blocks; shape::Union{Tuple{Int,Int},Nothing}=nothing)
    all_rows = Int[]
    all_cols = Int[]
    all_data = eltype(first(blocks))[]
    i0, j0 = 0, 0
    for block in blocks
        blk = sparse(block)
        I_idx, J_idx, V = findnz(blk)
        if !isempty(V)
            append!(all_data, V)
            append!(all_rows, I_idx .+ i0)
            append!(all_cols, J_idx .+ j0)
        end
        i0 += size(blk, 1)
        j0 += size(blk, 2)
    end
    if shape === nothing
        shape = (i0, j0)
    end
    if !isempty(all_data)
        return sparse(all_rows, all_cols, all_data, shape...)
    else
        return spzeros(eltype(first(blocks)), shape...)
    end
end

# ---------------------------------------------------------------------------
# kron / nkron
# ---------------------------------------------------------------------------

"""
    kronecker(factors...) -> AbstractMatrix

Compute the Kronecker product of a sequence of matrices. Returns a 1x1
identity matrix when called with no arguments.

Named `kronecker` to avoid conflict with `Base.kron` / `SparseArrays.kron`.

# Examples
```julia
kronecker(A, B, C)  # kron(kron(A, B), C)
kronecker()          # [1.0;;]
```
"""
function kronecker(factors...)
    if isempty(factors)
        return Matrix{Float64}(I, 1, 1)
    end
    out = factors[1]
    for f in factors[2:end]
        out = kron(out, f)
    end
    return out
end

"""
    nkron(factor, n::Int) -> AbstractMatrix

Compute the `n`-fold Kronecker product of `factor` with itself.

# Examples
```julia
nkron(A, 3)  # kron(kron(A, A), A)
```
"""
function nkron(factor, n::Int)
    return kronecker(fill(factor, n)...)
end

# ---------------------------------------------------------------------------
# permute_axis
# ---------------------------------------------------------------------------

"""
    permute_axis(array::AbstractArray, axis::Int, permutation;
                 out::Union{AbstractArray,Nothing}=nothing) -> AbstractArray

Permute the entries of `array` along dimension `axis` according to
`permutation` (a vector of indices, 1-based).

# Examples
```julia
A = [10 20 30; 40 50 60]
permute_axis(A, 2, [3, 1, 2])  # columns reordered to [30 10 20; 60 40 50]
```
"""
function permute_axis(array::AbstractArray, axis::Int, permutation;
                      out::Union{AbstractArray,Nothing}=nothing)
    idx = [Colon() for _ in 1:ndims(array)]
    idx[axis] = permutation
    perm = array[idx...]
    if out === nothing
        return perm
    else
        copyto!(out, perm)
        return out
    end
end

# ---------------------------------------------------------------------------
# copyto (thin wrapper)
# ---------------------------------------------------------------------------

"""
    dedalus_copyto!(dest::AbstractArray, src::AbstractArray)

Copy `src` into `dest`, equivalent to `dest .= src`. Thin wrapper for
compatibility with the Python Dedalus `copyto` function.
"""
function dedalus_copyto!(dest::AbstractArray, src::AbstractArray)
    dest .= src
    return dest
end

# ---------------------------------------------------------------------------
# perm_matrix
# ---------------------------------------------------------------------------

"""
    perm_matrix(perm::AbstractVector{<:Integer};
                M::Union{Int,Nothing}=nothing,
                source_index::Bool=false,
                make_sparse::Bool=true) -> AbstractMatrix{Int}

Build a permutation matrix from a permutation vector (1-based indexing).

# Arguments
- `perm`: Permutation vector of length `N`.
- `M`: Number of rows in the output matrix. Defaults to `N`.
- `source_index`: If `true`, `perm[i]` gives the source column for row `i`.
  If `false` (default), `perm[i]` gives the destination row for column `i`.
- `make_sparse`: Return a `SparseMatrixCSC` (default `true`) or a dense
  `Matrix{Int}`.

# Examples
```julia
perm_matrix([2, 3, 1])  # 3x3 sparse permutation matrix
```
"""
function perm_matrix(perm::AbstractVector{<:Integer};
                     M::Union{Int,Nothing}=nothing,
                     source_index::Bool=false,
                     make_sparse::Bool=true)
    N = length(perm)
    if M === nothing
        M = N
    end
    if source_index
        rows = collect(1:N)
        cols = collect(perm)
    else
        rows = collect(perm)
        cols = collect(1:N)
    end
    if make_sparse
        data = ones(Int, N)
        return sparse(rows, cols, data, M, N)
    else
        output = zeros(Int, M, N)
        for k in 1:N
            output[rows[k], cols[k]] = 1
        end
        return output
    end
end

# ---------------------------------------------------------------------------
# drop_empty_rows
# ---------------------------------------------------------------------------

"""
    drop_empty_rows(mat::AbstractSparseMatrix) -> SparseMatrixCSC

Return a copy of `mat` with all-zero rows removed.
"""
function drop_empty_rows(mat::AbstractSparseMatrix)
    m = sparse(mat)
    # Identify rows that have at least one structural nonzero
    nonempty = falses(size(m, 1))
    rows, _, _ = findnz(m)
    for r in rows
        nonempty[r] = true
    end
    return m[nonempty, :]
end

# ---------------------------------------------------------------------------
# scipy_sparse_eigs  (targeted eigenvalue search via shift-invert)
# ---------------------------------------------------------------------------

"""
    scipy_sparse_eigs(A, B, left::Bool, N::Int, target, matsolver; kw...)

Perform a targeted eigenmode search using the shift-invert method.

Finds the `N` eigenvalues of the generalised eigenproblem `A x = lambda B x`
that are closest to `target`.

# Arguments
- `A`, `B`: Square sparse matrices defining the generalised eigenproblem.
- `left`: If `true`, also compute left eigenvectors.
- `N`: Number of eigenvalues to find.
- `target`: Shift target in the complex plane.
- `matsolver`: Callable that, given a matrix `C`, returns a solver object
  with `solve(solver, v)` (and `solve_H(solver, v)` for left eigenvectors).
- `kw...`: Additional keyword arguments forwarded to `Arpack.eigs`.

# Returns
- `(evals, evecs)` when `left == false`.
- `(evals, evecs, left_evals, left_evecs)` when `left == true`.

!!! note
    Requires the `Arpack` package. If `Arpack` is not loaded this function
    will throw an error with installation instructions.
"""
function scipy_sparse_eigs(A, B, left::Bool, N::Int, target, matsolver; kw...)
    # Arpack is not in Project.toml yet; require it at call time.
    if !isdefined(Main, :Arpack)
        try
            @eval Main using Arpack
        catch
            error("Arpack.jl is required for scipy_sparse_eigs. " *
                  "Install it with: ] add Arpack")
        end
    end
    arpack_eigs = Main.Arpack.eigs

    C = A - target * B
    solver = matsolver(C)

    # Build a linear map: x -> solver.solve(B * x)
    matvec(x) = solver.solve(B * x)
    n = size(A, 1)

    # Use Arpack.eigs with the shift-invert linear operator
    # We construct a wrapper matrix type to use with Arpack
    evals, evecs = _arpack_eigs_via_matvec(arpack_eigs, matvec, n, N,
                                            eltype(A); kw...)
    evals .= 1 ./ evals .+ target

    if left
        matvec_left(x) = solver.solve_H(conj(B)' * x)
        left_evals, left_evecs = _arpack_eigs_via_matvec(arpack_eigs, matvec_left, n, N,
                                                          eltype(A); kw...)
        left_evals .= 1 ./ left_evals .+ conj(target)
        return evals, evecs, left_evals, left_evecs
    else
        return evals, evecs
    end
end

"""
    _arpack_eigs_via_matvec(eigs_fn, matvec, n, nev, T; kw...)

Internal helper that calls Arpack.eigs using a `LinearMap`-style wrapper.
"""
function _arpack_eigs_via_matvec(eigs_fn, matvec, n::Int, nev::Int, ::Type{T}; kw...) where T
    # Build an operator wrapper that Arpack can use
    # Arpack.eigs accepts any object that supports mul!
    op = _MatvecOperator(matvec, n, T)
    return eigs_fn(op; nev=nev, which=:LM, kw...)
end

"""
    _MatvecOperator

Lightweight wrapper that presents a `matvec` function as a matrix-like
object for Arpack.eigs (supports `size`, `eltype`, and `*`).
"""
struct _MatvecOperator{F,T}
    matvec::F
    n::Int
end

function _MatvecOperator(matvec::F, n::Int, ::Type{T}) where {F,T}
    return _MatvecOperator{F,T}(matvec, n)
end

Base.size(op::_MatvecOperator) = (op.n, op.n)
Base.size(op::_MatvecOperator, d::Int) = op.n
Base.eltype(::_MatvecOperator{F,T}) where {F,T} = T

function Base.:*(op::_MatvecOperator, x::AbstractVector)
    return op.matvec(x)
end

function LinearAlgebra.mul!(y::AbstractVector, op::_MatvecOperator, x::AbstractVector)
    result = op.matvec(x)
    copyto!(y, result)
    return y
end

# ---------------------------------------------------------------------------
# interleave_matrices
# ---------------------------------------------------------------------------

"""
    interleave_matrices(matrices::AbstractVector{<:AbstractSparseMatrix}) -> SparseMatrixCSC

Interleave a collection of sparse matrices so that the `i`-th matrix
contributes to every `N`-th row/column starting at offset `i`, where
`N = length(matrices)`.

This is equivalent to `sum_i kron(matrix_i, e_i * e_i')` where `e_i` is the
`i`-th standard basis vector.

# Examples
```julia
A = sparse([1.0 2.0; 3.0 4.0])
B = sparse([5.0 6.0; 7.0 8.0])
interleave_matrices([A, B])  # 4x4 interleaved matrix
```
"""
function interleave_matrices(matrices::AbstractVector{<:AbstractSparseMatrix})
    N = length(matrices)
    if N == 1
        return matrices[1]
    end
    total = spzeros(eltype(first(matrices)), 0, 0)
    P = spzeros(Float64, N, N)
    for (i, matrix) in enumerate(matrices)
        P[i, i] = 1.0
        contribution = kron(matrix, P)
        if i == 1
            total = contribution
        else
            total = total + contribution
        end
        P[i, i] = 0.0
    end
    return total
end

# ---------------------------------------------------------------------------
# sparse_allclose
# ---------------------------------------------------------------------------

"""
    sparse_allclose(A::AbstractSparseMatrix, B::AbstractSparseMatrix;
                    atol::Real=0, rtol::Real=sqrt(eps())) -> Bool

Test whether two sparse matrices are element-wise approximately equal.
Compares the CSC internal arrays (`colptr`, `rowval`, `nzval`) directly for
speed when both matrices have the same sparsity structure.
"""
function sparse_allclose(A::AbstractSparseMatrix, B::AbstractSparseMatrix;
                         atol::Real=0, rtol::Real=sqrt(eps()))
    Ac = sparse(A)
    Bc = sparse(B)
    return (isapprox(nonzeros(Ac), nonzeros(Bc); atol=atol, rtol=rtol) &&
            rowvals(Ac) == rowvals(Bc) &&
            Ac.colptr == Bc.colptr)
end

# ---------------------------------------------------------------------------
# assert_sparse_pinv
# ---------------------------------------------------------------------------

"""
    assert_sparse_pinv(A::AbstractSparseMatrix, B::AbstractSparseMatrix)

Assert the four Moore-Penrose conditions for `B` being a pseudoinverse of `A`:
1. `A * B * A ≈ A`
2. `B * A * B ≈ B`
3. `(A * B)' ≈ A * B`  (Hermitian)
4. `(B * A)' ≈ B * A`  (Hermitian)

Throws an `AssertionError` if any condition fails.
"""
function assert_sparse_pinv(A::AbstractSparseMatrix, B::AbstractSparseMatrix)
    if !sparse_allclose(A * B * A, A)
        throw(AssertionError("Not a pseudoinverse (condition 1: A*B*A ≈ A)"))
    end
    if !sparse_allclose(B * A * B, B)
        throw(AssertionError("Not a pseudoinverse (condition 2: B*A*B ≈ B)"))
    end
    AB = A * B
    if !sparse_allclose(sparse(AB'), AB)
        throw(AssertionError("Not a pseudoinverse (condition 3: (A*B)' ≈ A*B)"))
    end
    BA = B * A
    if !sparse_allclose(sparse(BA'), BA)
        throw(AssertionError("Not a pseudoinverse (condition 4: (B*A)' ≈ B*A)"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export interleaved_view,
       reshape_vector,
       axindex,
       axslice,
       zeros_with_pattern,
       expand_pattern,
       apply_matrix,
       move_single_axis,
       apply_dense,
       apply_sparse,
       solve_upper_sparse,
       add_sparse,
       sparse_block_diag,
       kronecker,
       nkron,
       permute_axis,
       dedalus_copyto!,
       perm_matrix,
       drop_empty_rows,
       scipy_sparse_eigs,
       interleave_matrices,
       sparse_allclose,
       assert_sparse_pinv,
       SPLIT_CSR_MATVECS,
       OLD_CSR_MATVECS
