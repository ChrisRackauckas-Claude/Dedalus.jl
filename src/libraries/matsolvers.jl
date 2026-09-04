"""
Matrix solver wrappers.

Provides a pluggable registry of matrix solver types for solving linear systems
arising in spectral PDE discretizations. Each solver wraps a particular
factorization or direct-solve strategy and exposes a uniform `solve(solver, b)`
interface.

Translated from dedalus/libraries/matsolvers.py.
"""

using SparseArrays
using LinearAlgebra

# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------

"""Global registry mapping lowercase solver name to solver type."""
const MATSOLVER_REGISTRY = Dict{String, Any}()

"""
    register_solver!(T::Type)

Register solver type `T` in the global `MATSOLVER_REGISTRY` under its
lowercase type name.  Analogous to the Python `@add_solver` decorator.
"""
function register_solver!(T::Type)
    MATSOLVER_REGISTRY[lowercase(string(nameof(T)))] = T
    return T
end

"""
    register_solver!(name::AbstractString, T::Type)

Register solver type `T` under an explicit `name`.
"""
function register_solver!(name::AbstractString, T::Type)
    MATSOLVER_REGISTRY[lowercase(name)] = T
    return T
end

# ---------------------------------------------------------------------------
# Abstract base types
# ---------------------------------------------------------------------------

"""
    AbstractMatSolver

Abstract base type for all matrix solvers.

Every concrete subtype must support:

    solve(solver, vector) -> solution vector

and may optionally support:

    solve_H(solver, vector) -> conjugate-transpose solve
"""
abstract type AbstractMatSolver end

"""
    solve(::AbstractMatSolver, vector)

Solve the linear system represented by the solver for the given right-hand side.
Must be implemented by every concrete solver type.
"""
function solve end

"""
    solve!(result, solver::AbstractMatSolver, vector)

In-place solve: write the solution into `result`, avoiding allocation of the
output vector.  Falls back to `copyto!(result, solve(solver, vector))` for
solvers that do not provide a specialized method.
"""
function solve!(result, solver::AbstractMatSolver, vector::AbstractVecOrMat)
    copyto!(result, solve(solver, vector))
    return result
end

"""
    solve_H(solver::AbstractMatSolver, vector)

Solve the conjugate-transpose (Hermitian adjoint) system.  Not all solvers
support this; the default throws an error.
"""
function solve_H(solver::AbstractMatSolver, ::AbstractVecOrMat)
    error("$(typeof(solver)) has not implemented 'solve_H' method")
end

# Intermediate abstract types mirroring the Python class hierarchy
"""Abstract type for sparse-matrix solvers (`sparse=true, banded=false`)."""
abstract type AbstractSparseSolver <: AbstractMatSolver end

"""Abstract type for banded-matrix solvers (`sparse=false, banded=true`)."""
abstract type AbstractBandedSolver <: AbstractMatSolver end

"""Abstract type for dense-matrix solvers (`sparse=false, banded=false`)."""
abstract type AbstractDenseSolver <: AbstractMatSolver end

# Traits -------------------------------------------------------------------

"""Return `true` if the solver operates on sparse matrices."""
is_sparse(::AbstractSparseSolver) = true
is_sparse(::AbstractBandedSolver) = false
is_sparse(::AbstractDenseSolver)  = false

"""Return `true` if the solver operates on banded matrices."""
is_banded(::AbstractSparseSolver) = false
is_banded(::AbstractBandedSolver) = true
is_banded(::AbstractDenseSolver)  = false

# ---------------------------------------------------------------------------
# Utility: sparse → banded conversion
# ---------------------------------------------------------------------------

"""
    sparse_to_banded(matrix; u=nothing, l=nothing) -> (l, u), ab

Convert a sparse matrix to LAPACK-style banded storage (`ab`).

`u` and `l` are the number of upper and lower diagonals respectively.  When
omitted they are inferred from the sparsity pattern.

Returns a tuple `((l, u), ab)` where `ab` has size `(l + u + 1, n)`.
"""
function sparse_to_banded(matrix::AbstractSparseMatrix; u::Union{Nothing,Int}=nothing, l::Union{Nothing,Int}=nothing)
    n = size(matrix, 2)
    rows = rowvals(matrix)
    vals = nonzeros(matrix)

    # Determine bandwidth from non-zero entries
    if u === nothing || l === nothing
        max_upper = 0
        max_lower = 0
        for col in 1:n
            for idx in nzrange(matrix, col)
                row = rows[idx]
                d = col - row          # positive = upper, negative = lower
                max_upper = max(max_upper, d)
                max_lower = max(max_lower, -d)
            end
        end
        u = u === nothing ? max_upper : u
        l = l === nothing ? max_lower : l
    end

    ab = zeros(eltype(matrix), l + u + 1, n)

    # Fill banded storage:  ab[u + 1 + (row - col), col] = A[row, col]
    # This matches LAPACK's banded storage convention.
    for col in 1:n
        for idx in nzrange(matrix, col)
            row = rows[idx]
            ab[u + 1 + (row - col), col] = vals[idx]
        end
    end

    return (l, u), ab
end

# ---------------------------------------------------------------------------
# DummySolver
# ---------------------------------------------------------------------------

"""
    DummySolver(matrix)

Dummy solver that returns zeros for testing purposes.
"""
struct DummySolver <: AbstractMatSolver end

DummySolver(matrix, solver=nothing) = DummySolver()

function solve(s::DummySolver, vector::AbstractVecOrMat)
    return zero(vector)
end

register_solver!(DummySolver)

# ---------------------------------------------------------------------------
# UmfpackSpsolve  –  direct sparse solve via UMFPACK (Julia default `\`)
# ---------------------------------------------------------------------------

"""
    UmfpackSpsolve(matrix)

Direct sparse solve using UMFPACK (Julia's built-in sparse `\\` operator).
No pre-factorization; each `solve` call performs a fresh factorisation + solve.
"""
struct UmfpackSpsolve{Tv, Ti} <: AbstractSparseSolver
    matrix::SparseMatrixCSC{Tv, Ti}
end

function UmfpackSpsolve(matrix::AbstractSparseMatrix, solver=nothing)
    UmfpackSpsolve(convert(SparseMatrixCSC, copy(matrix)))
end

function solve(s::UmfpackSpsolve, vector::AbstractVecOrMat)
    return s.matrix \ vector
end

register_solver!(UmfpackSpsolve)

# ---------------------------------------------------------------------------
# SuperluNaturalSpsolve  –  mapped to UMFPACK direct solve in Julia
# ---------------------------------------------------------------------------

"""
    SuperluNaturalSpsolve(matrix)

SuperLU spsolve with 'NATURAL' column permutation.  In Julia this is mapped to
UMFPACK direct sparse solve (equivalent functionality via SuiteSparse).
"""
struct SuperluNaturalSpsolve{Tv, Ti} <: AbstractSparseSolver
    matrix::SparseMatrixCSC{Tv, Ti}
end

function SuperluNaturalSpsolve(matrix::AbstractSparseMatrix, solver=nothing)
    SuperluNaturalSpsolve(convert(SparseMatrixCSC, copy(matrix)))
end

function solve(s::SuperluNaturalSpsolve, vector::AbstractVecOrMat)
    return s.matrix \ vector
end

register_solver!(SuperluNaturalSpsolve)

# ---------------------------------------------------------------------------
# SuperluColamdSpsolve  –  mapped to UMFPACK direct solve in Julia
# ---------------------------------------------------------------------------

"""
    SuperluColamdSpsolve(matrix)

SuperLU spsolve with 'COLAMD' column permutation.  In Julia this is mapped to
UMFPACK direct sparse solve (equivalent functionality via SuiteSparse).
"""
struct SuperluColamdSpsolve{Tv, Ti} <: AbstractSparseSolver
    matrix::SparseMatrixCSC{Tv, Ti}
end

function SuperluColamdSpsolve(matrix::AbstractSparseMatrix, solver=nothing)
    SuperluColamdSpsolve(convert(SparseMatrixCSC, copy(matrix)))
end

function solve(s::SuperluColamdSpsolve, vector::AbstractVecOrMat)
    return s.matrix \ vector
end

register_solver!(SuperluColamdSpsolve)

# ---------------------------------------------------------------------------
# UmfpackFactorized  –  pre-factorized LU solve
# ---------------------------------------------------------------------------

"""
    UmfpackFactorized(matrix)

UMFPACK LU-factorized solve.  The matrix is factorized once at construction
time; subsequent `solve` calls reuse the factorization.
"""
struct UmfpackFactorized{F} <: AbstractSparseSolver
    LU::F
end

function UmfpackFactorized(matrix::AbstractSparseMatrix, solver=nothing)
    F = lu(convert(SparseMatrixCSC, matrix))
    UmfpackFactorized(F)
end

function solve(s::UmfpackFactorized, vector::AbstractVecOrMat)
    result = similar(vector)
    ldiv!(result, s.LU, vector)
    return result
end

function solve!(result, s::UmfpackFactorized, vector::AbstractVecOrMat)
    ldiv!(result, s.LU, vector)
    return result
end

register_solver!(UmfpackFactorized)

# ---------------------------------------------------------------------------
# Helper: Factorized solver with configurable transpose mode
# ---------------------------------------------------------------------------
# Python's _SuperluFactorizedBase accepts a `trans` flag ("N", "T", "H") and
# optionally transposes the matrix before factorisation.  We mirror this with
# a single parametric struct and thin wrappers.

"""
    FactorizedTransposeSolver(matrix; trans=:N)

Internal factorized solver that supports normal, transpose, and
conjugate-transpose solves.

`trans` can be `:N` (normal), `:T` (transpose), or `:H` (conjugate transpose).
When `trans` is `:T` or `:H`, the matrix is appropriately transposed *before*
factorisation so that a normal forward-solve on the factorisation is equivalent
to the desired transpose solve on the original matrix.
"""
struct FactorizedTransposeSolver{F} <: AbstractSparseSolver
    LU::F
    trans::Symbol   # :N, :T, or :H
end

function FactorizedTransposeSolver(matrix::AbstractSparseMatrix; trans::Symbol=:N, solver=nothing)
    M = convert(SparseMatrixCSC, matrix)
    if trans == :T
        M = copy(transpose(M))   # materialise transpose
    elseif trans == :H
        M = copy(adjoint(M))     # materialise conjugate-transpose
    end
    F = lu(M)
    FactorizedTransposeSolver(F, trans)
end

function solve(s::FactorizedTransposeSolver, vector::AbstractVecOrMat)
    result = similar(vector)
    if s.trans == :N
        ldiv!(result, s.LU, vector)
    elseif s.trans == :T
        ldiv!(result, transpose(s.LU), vector)
    else  # :H
        ldiv!(result, adjoint(s.LU), vector)
    end
    return result
end

function solve!(result, s::FactorizedTransposeSolver, vector::AbstractVecOrMat)
    if s.trans == :N
        ldiv!(result, s.LU, vector)
    elseif s.trans == :T
        ldiv!(result, transpose(s.LU), vector)
    else  # :H
        ldiv!(result, adjoint(s.LU), vector)
    end
    return result
end

function solve_H(s::FactorizedTransposeSolver, vector::AbstractVecOrMat)
    result = similar(vector)
    if s.trans == :N
        ldiv!(result, adjoint(s.LU), vector)
    elseif s.trans == :H
        ldiv!(result, s.LU, vector)
    else  # :T
        conj_vec = conj.(vector)
        ldiv!(result, s.LU, conj_vec)
        result .= conj.(result)
    end
    return result
end

# ---------------------------------------------------------------------------
# SuperluNaturalFactorized
# ---------------------------------------------------------------------------

"""
    SuperluNaturalFactorized(matrix)

SuperLU factorized solve with 'NATURAL' column permutation.
Mapped to UMFPACK LU factorization in Julia.
"""
struct SuperluNaturalFactorized{F} <: AbstractSparseSolver
    inner::FactorizedTransposeSolver{F}
end

function SuperluNaturalFactorized(matrix::AbstractSparseMatrix, solver=nothing)
    inner = FactorizedTransposeSolver(matrix; trans=:N, solver=solver)
    SuperluNaturalFactorized(inner)
end

function solve(s::SuperluNaturalFactorized, vector::AbstractVecOrMat)
    return solve(s.inner, vector)
end

function solve!(result, s::SuperluNaturalFactorized, vector::AbstractVecOrMat)
    return solve!(result, s.inner, vector)
end

function solve_H(s::SuperluNaturalFactorized, vector::AbstractVecOrMat)
    return solve_H(s.inner, vector)
end

register_solver!(SuperluNaturalFactorized)

# ---------------------------------------------------------------------------
# SuperluNaturalFactorizedTranspose
# ---------------------------------------------------------------------------

"""
    SuperluNaturalFactorizedTranspose(matrix)

SuperLU factorized transpose solve with 'NATURAL' column permutation.
Mapped to UMFPACK LU factorization (of the transposed matrix) in Julia.
"""
struct SuperluNaturalFactorizedTranspose{F} <: AbstractSparseSolver
    inner::FactorizedTransposeSolver{F}
end

function SuperluNaturalFactorizedTranspose(matrix::AbstractSparseMatrix, solver=nothing)
    inner = FactorizedTransposeSolver(matrix; trans=:T, solver=solver)
    SuperluNaturalFactorizedTranspose(inner)
end

function solve(s::SuperluNaturalFactorizedTranspose, vector::AbstractVecOrMat)
    return solve(s.inner, vector)
end

function solve!(result, s::SuperluNaturalFactorizedTranspose, vector::AbstractVecOrMat)
    return solve!(result, s.inner, vector)
end

function solve_H(s::SuperluNaturalFactorizedTranspose, vector::AbstractVecOrMat)
    return solve_H(s.inner, vector)
end

register_solver!(SuperluNaturalFactorizedTranspose)

# ---------------------------------------------------------------------------
# SuperluColamdFactorized
# ---------------------------------------------------------------------------

"""
    SuperluColamdFactorized(matrix)

SuperLU factorized solve with 'COLAMD' column permutation.
Mapped to UMFPACK LU factorization in Julia.
"""
struct SuperluColamdFactorized{F} <: AbstractSparseSolver
    inner::FactorizedTransposeSolver{F}
end

function SuperluColamdFactorized(matrix::AbstractSparseMatrix, solver=nothing)
    inner = FactorizedTransposeSolver(matrix; trans=:N, solver=solver)
    SuperluColamdFactorized(inner)
end

function solve(s::SuperluColamdFactorized, vector::AbstractVecOrMat)
    return solve(s.inner, vector)
end

function solve!(result, s::SuperluColamdFactorized, vector::AbstractVecOrMat)
    return solve!(result, s.inner, vector)
end

function solve_H(s::SuperluColamdFactorized, vector::AbstractVecOrMat)
    return solve_H(s.inner, vector)
end

register_solver!(SuperluColamdFactorized)

# ---------------------------------------------------------------------------
# SuperluColamdFactorizedTranspose
# ---------------------------------------------------------------------------

"""
    SuperluColamdFactorizedTranspose(matrix)

SuperLU factorized transpose solve with 'COLAMD' column permutation.
Mapped to UMFPACK LU factorization (of the transposed matrix) in Julia.
"""
struct SuperluColamdFactorizedTranspose{F} <: AbstractSparseSolver
    inner::FactorizedTransposeSolver{F}
end

function SuperluColamdFactorizedTranspose(matrix::AbstractSparseMatrix, solver=nothing)
    inner = FactorizedTransposeSolver(matrix; trans=:T, solver=solver)
    SuperluColamdFactorizedTranspose(inner)
end

function solve(s::SuperluColamdFactorizedTranspose, vector::AbstractVecOrMat)
    return solve(s.inner, vector)
end

function solve!(result, s::SuperluColamdFactorizedTranspose, vector::AbstractVecOrMat)
    return solve!(result, s.inner, vector)
end

function solve_H(s::SuperluColamdFactorizedTranspose, vector::AbstractVecOrMat)
    return solve_H(s.inner, vector)
end

register_solver!(SuperluColamdFactorizedTranspose)

# ---------------------------------------------------------------------------
# UmfpackFactorizedTranspose  –  UMFPACK factorized transpose solve
# ---------------------------------------------------------------------------

"""
    UmfpackFactorizedTranspose(matrix)

UMFPACK-based factorized transpose solve.  The transposed matrix is LU-factorized
at construction time.
"""
struct UmfpackFactorizedTranspose{F} <: AbstractSparseSolver
    inner::FactorizedTransposeSolver{F}
end

function UmfpackFactorizedTranspose(matrix::AbstractSparseMatrix, solver=nothing)
    inner = FactorizedTransposeSolver(matrix; trans=:T, solver=solver)
    UmfpackFactorizedTranspose(inner)
end

function solve(s::UmfpackFactorizedTranspose, vector::AbstractVecOrMat)
    return solve(s.inner, vector)
end

function solve!(result, s::UmfpackFactorizedTranspose, vector::AbstractVecOrMat)
    return solve!(result, s.inner, vector)
end

function solve_H(s::UmfpackFactorizedTranspose, vector::AbstractVecOrMat)
    return solve_H(s.inner, vector)
end

register_solver!(UmfpackFactorizedTranspose)

# ---------------------------------------------------------------------------
# ScipyBanded → BandedLAPACK  –  banded solve via LAPACK gbtrf!/gbtrs!
# ---------------------------------------------------------------------------

"""
    BandedLAPACK(matrix)

Banded LU-factorized solve using LAPACK routines `gbtrf!`/`gbtrs!`.
Equivalent to Python's `ScipyBanded` solver.

The sparse input matrix is converted to LAPACK banded storage at construction
time, then factorised with `gbtrf!`.  Subsequent solves use `gbtrs!`.
"""
mutable struct BandedLAPACK{T} <: AbstractBandedSolver
    kl::Int                  # number of lower diagonals
    ku::Int                  # number of upper diagonals
    AB::Matrix{T}            # LAPACK banded storage (2*kl + ku + 1, n) after gbtrf!
    ipiv::Vector{Int}        # pivot indices from gbtrf!
end

function BandedLAPACK(matrix::AbstractSparseMatrix, solver=nothing)
    (kl, ku), ab_narrow = sparse_to_banded(matrix)
    n = size(ab_narrow, 2)
    T = eltype(ab_narrow)

    # LAPACK gbtrf! needs (2*kl + ku + 1) rows: kl extra rows at the top for
    # fill-in during factorisation.
    AB = zeros(T, 2*kl + ku + 1, n)
    AB[kl+1:end, :] .= ab_narrow   # place the band data after the fill-in rows

    AB, ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, n, AB)
    BandedLAPACK{T}(kl, ku, AB, ipiv)
end

function solve(s::BandedLAPACK, vector::AbstractVecOrMat)
    b = copy(convert(Matrix, reshape(vector, size(vector, 1), :)))
    LinearAlgebra.LAPACK.gbtrs!('N', s.kl, s.ku, size(s.AB, 2), s.AB, s.ipiv, b)
    return ndims(vector) == 1 ? vec(b) : b
end

function solve!(result, s::BandedLAPACK, vector::AbstractVecOrMat)
    copyto!(result, solve(s, vector))
    return result
end

register_solver!(BandedLAPACK)
# Also register under the Python name for compatibility
register_solver!("scipybanded", BandedLAPACK)

# ---------------------------------------------------------------------------
# SPQRSolve  –  SuiteSparseQR solve
# ---------------------------------------------------------------------------

"""
    SPQRSolve(matrix)

SuiteSparse QR solve.  Uses Julia's built-in `qr` factorization on a sparse
matrix and solves via `\\`.
"""
struct SPQRSolve{F} <: AbstractSparseSolver
    QR::F
end

function SPQRSolve(matrix::AbstractSparseMatrix, solver=nothing)
    F = qr(convert(SparseMatrixCSC, copy(matrix)))
    SPQRSolve(F)
end

function solve(s::SPQRSolve, vector::AbstractVecOrMat)
    # SuiteSparseQR does not support ldiv! with separate output; use \ here
    return s.QR \ vector
end

register_solver!(SPQRSolve)
# Also register under the Python name for compatibility
register_solver!("spqr_solve", SPQRSolve)

# ---------------------------------------------------------------------------
# SparseInverse  –  explicit sparse inverse
# ---------------------------------------------------------------------------

"""
    SparseInverse(matrix)

Sparse inversion solve.  Computes and stores the explicit sparse inverse of the
matrix.  Solves are then simple matrix-vector multiplications.

**Warning**: forming the explicit inverse is expensive and generally not
recommended for large systems.
"""
struct SparseInverse{Tv, Ti} <: AbstractSparseSolver
    matrix_inverse::SparseMatrixCSC{Tv, Ti}
    # Private inner constructor to avoid ambiguity with the outer constructor
    function SparseInverse{Tv, Ti}(mi::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
        new{Tv, Ti}(mi)
    end
end

function SparseInverse(matrix::AbstractSparseMatrix, solver=nothing)
    M = convert(SparseMatrixCSC, matrix)
    # Compute inverse via LU factorisation and identity solves
    n = size(M, 1)
    F = lu(M)
    Minv = sparse(F \ Matrix{eltype(M)}(I, n, n))
    SparseInverse{eltype(Minv), eltype(rowvals(Minv))}(Minv)
end

function solve(s::SparseInverse, vector::AbstractVecOrMat)
    return s.matrix_inverse * vector
end

function solve!(result, s::SparseInverse, vector::AbstractVecOrMat)
    mul!(result, s.matrix_inverse, vector)
    return result
end

register_solver!(SparseInverse)

# ---------------------------------------------------------------------------
# DenseInverse  –  explicit dense inverse
# ---------------------------------------------------------------------------

"""
    DenseInverse(matrix)

Dense inversion solve.  Converts the sparse matrix to dense form, computes its
inverse, and stores it.  Solves are dense matrix-vector multiplications.
"""
struct DenseInverse{T} <: AbstractDenseSolver
    matrix_inverse::Matrix{T}
end

function DenseInverse(matrix::AbstractSparseMatrix, solver=nothing)
    M = Matrix(matrix)
    DenseInverse(inv(M))
end

function solve(s::DenseInverse, vector::AbstractVecOrMat)
    return s.matrix_inverse * vector
end

function solve!(result, s::DenseInverse, vector::AbstractVecOrMat)
    mul!(result, s.matrix_inverse, vector)
    return result
end

register_solver!(DenseInverse)

# ---------------------------------------------------------------------------
# BlockInverse  –  block-diagonal inverse
# ---------------------------------------------------------------------------

"""
    BlockInverse(matrix, solver)

Block inversion solve for uncoupled (block-diagonal) problems.

Requires the solver's domain to have an uncoupled last basis.  For scalar
problems (block size 1), reduces to a diagonal solve.  Otherwise, inverts
each diagonal block independently.
"""
struct BlockInverse{T} <: AbstractBandedSolver
    matrix_inverse::Union{SparseMatrixCSC{T}, Nothing}
    inv_diagonal::Union{Vector{T}, Nothing}
    use_diagonal::Bool
end

function BlockInverse(matrix::AbstractSparseMatrix, solver_obj)
    # Check separability
    if hasproperty(solver_obj, :domain) &&
       hasproperty(solver_obj.domain, :bases) &&
       length(solver_obj.domain.bases) > 0 &&
       hasproperty(last(solver_obj.domain.bases), :coupled) &&
       last(solver_obj.domain.bases).coupled
        error("Block solver requires uncoupled problems.")
    end

    block_size = if hasproperty(solver_obj, :problem) && hasproperty(solver_obj.problem, :variables)
        length(solver_obj.problem.variables)
    else
        1
    end

    T = eltype(matrix)

    if block_size == 1
        # Special-case: diagonal matrix
        d = diag(matrix)
        inv_diag = one(T) ./ d
        return BlockInverse{T}(nothing, inv_diag, true)
    else
        # General block-diagonal: extract blocks, invert, reassemble
        n = size(matrix, 1)
        b = block_size
        nblocks = div(n, b)
        M = Matrix(matrix)
        inv_blocks = [inv(M[(i-1)*b+1:i*b, (i-1)*b+1:i*b]) for i in 1:nblocks]
        # Reassemble as sparse block-diagonal
        Minv = blockdiag([sparse(blk) for blk in inv_blocks]...)
        return BlockInverse{T}(Minv, nothing, false)
    end
end

function solve(s::BlockInverse, vector::AbstractVecOrMat)
    if s.use_diagonal
        return s.inv_diagonal .* vector
    else
        return s.matrix_inverse * vector
    end
end

function solve!(result, s::BlockInverse, vector::AbstractVecOrMat)
    if s.use_diagonal
        result .= s.inv_diagonal .* vector
    else
        mul!(result, s.matrix_inverse, vector)
    end
    return result
end

register_solver!(BlockInverse)

# ---------------------------------------------------------------------------
# DenseLU  –  dense LU factorized solve (maps Python's ScipyDenseLU)
# ---------------------------------------------------------------------------

"""
    DenseLU(matrix)

Dense LU-factorized solve.  Converts the sparse matrix to dense form and
computes an LU factorization.  Equivalent to Python's `ScipyDenseLU`.
"""
struct DenseLU{F} <: AbstractDenseSolver
    LU::F
end

function DenseLU(matrix::AbstractSparseMatrix, solver=nothing)
    DenseLU(lu(Matrix(matrix)))
end

function solve(s::DenseLU, vector::AbstractVecOrMat)
    result = copy(convert(Array, vector))
    ldiv!(s.LU, result)
    return result
end

function solve!(result, s::DenseLU, vector::AbstractVecOrMat)
    copyto!(result, vector)
    ldiv!(s.LU, result)
    return result
end

register_solver!(DenseLU)
# Also register under the Python name for compatibility
register_solver!("scipydenselu", DenseLU)

# ---------------------------------------------------------------------------
# Woodbury  –  bordered matrix solve via Woodbury formula
# ---------------------------------------------------------------------------

"""
    Woodbury(matrix, subproblem, matsolver_type)

Solve a top-and-right-bordered matrix using the Woodbury matrix identity.

Decomposes `A = A₀ + U V` where `A₀` is the matrix with borders removed,
then applies the Woodbury formula:

    A⁻¹ = A₀⁻¹ − A₀⁻¹ U (I + V A₀⁻¹ U)⁻¹ V A₀⁻¹

The config dict `bc_top = true` indicates the border structure.
"""
struct Woodbury{MS<:AbstractMatSolver, T} <: AbstractSparseSolver
    A_matsolver::MS
    Ainv_U::Matrix{T}
    Sinv::Matrix{T}
    V::Matrix{T}
end

function Woodbury(matrix::AbstractSparseMatrix, subproblem, matsolver_type)
    R = subproblem.update_rank
    n = size(matrix, 1)
    T = eltype(matrix)

    # Form Woodbury factors U and V
    U = zeros(T, n, 2*R)
    V = zeros(T, 2*R, n)

    # Remove top border, leaving upper left subblock
    U[1:R, 1:R] = Matrix{T}(I, R, R)
    V[1:R, R+1:end] = Matrix(matrix[1:R, R+1:end])

    # Remove right border, leaving upper right and lower right subblocks
    U[R+1:end-R, R+1:2*R] = Matrix(matrix[R+1:end-R, end-R+1:end])
    V[R+1:2*R, end-R+1:end] = Matrix{T}(I, R, R)

    # A₀ = matrix - U * V
    A = matrix - sparse(U) * sparse(V)

    # Solve A₀ using specified matsolver
    A_ms = matsolver_type(A)
    Ainv_U = Matrix{T}(undef, n, 2*R)
    for j in 1:2*R
        Ainv_U[:, j] = solve(A_ms, U[:, j])
    end

    # Schur complement:  S = I + V * A₀⁻¹ U
    S = Matrix{T}(I, 2*R, 2*R) + V * Ainv_U
    Sinv = inv(S)

    Woodbury(A_ms, Ainv_U, Sinv, V)
end

function solve(s::Woodbury, vector::AbstractVecOrMat)
    Ainv_Y = solve(s.A_matsolver, vector)
    return Ainv_Y - s.Ainv_U * (s.Sinv * (s.V * Ainv_Y))
end

function solve!(result, s::Woodbury, vector::AbstractVecOrMat)
    solve!(result, s.A_matsolver, vector)
    # result now holds Ainv_Y; compute correction in-place
    result .-= s.Ainv_U * (s.Sinv * (s.V * result))
    return result
end

# ---------------------------------------------------------------------------
# Generate Woodbury-wrapped variants for every registered solver
# ---------------------------------------------------------------------------

"""
    build_woodbury_registry!()

For each solver in `MATSOLVER_REGISTRY`, create a Woodbury-wrapped variant
registered as `"woodbury<name>"`.  Call this after all base solvers have been
registered.
"""
function build_woodbury_registry!()
    base_names = collect(keys(MATSOLVER_REGISTRY))
    for name in base_names
        ms = MATSOLVER_REGISTRY[name]
        woodbury_name = "woodbury" * name
        if !haskey(MATSOLVER_REGISTRY, woodbury_name)
            # Store a closure that, given (matrix, subproblem), produces a
            # Woodbury solver wrapping `ms`.
            MATSOLVER_REGISTRY[woodbury_name] = (matrix, subproblem) -> Woodbury(matrix, subproblem, ms)
        end
    end
end

# Build the Woodbury variants now that all base solvers are registered
build_woodbury_registry!()

# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

"""
    get_solver(name::AbstractString) -> Type or callable

Look up a registered solver by (case-insensitive) name.
"""
function get_solver(name::AbstractString)
    key = lowercase(name)
    haskey(MATSOLVER_REGISTRY, key) || error("Unknown matsolver: $name. Available: $(join(sort(collect(keys(MATSOLVER_REGISTRY))), ", "))")
    return MATSOLVER_REGISTRY[key]
end

export AbstractMatSolver,
       MATSOLVER_REGISTRY,
       get_solver
