"""
Clenshaw summation algorithms for Dedalus.jl.

Translated from dedalus/tools/clenshaw.py. Provides Clenshaw recurrence
evaluation of polynomial series in scalar, matrix, and Kronecker-product
forms, as well as Jacobi recursion coefficient builders.

Uses SparseArrays for sparse matrix operations and the `DeferredTuple` type
from general.jl for lazy coefficient computation.
"""

using SparseArrays
using LinearAlgebra: I

# ---------------------------------------------------------------------------
# scalar_clenshaw
# ---------------------------------------------------------------------------

"""
    scalar_clenshaw(c, A, B, f0)

Clenshaw algorithm on scalar coefficients with array argument.

Evaluates `S(x) = sum_n c[n] f_n(x)` where the `f_n` satisfy a three-term
recurrence defined by `A` and `B`. Both `A` and `B` are indexable sequences
(1-based) with `A` of length >= `N` and `B` of length >= `N+1`.

# Arguments
- `c`  -- coefficient vector of length `N`.
- `A`  -- recurrence coefficients (indexed 1..N).
- `B`  -- recurrence coefficients (indexed 1..N+1).
- `f0` -- initial function value `f_0(x)`.

# Returns
The evaluated sum `S(x)`.
"""
function scalar_clenshaw(c, A, B, f0)
    N = length(c)
    # Clenshaw recurrence (backward sweep)
    b0 = zero(f0)
    b1 = zero(f0)
    for n in N:-1:1
        b2 = b1
        b1 = b0
        # Python uses 0-based indexing: A[n], B[n+1] -> Julia 1-based: A[n], B[n+1]
        b0 = c[n] .+ (A[n] .* b1) .+ (B[n+1] .* b2)
    end
    return b0 .* f0
end

# ---------------------------------------------------------------------------
# matrix_clenshaw
# ---------------------------------------------------------------------------

"""
    matrix_clenshaw(c, A, B, f0, cutoff)

Clenshaw algorithm on scalar coefficients with matrix argument.

Evaluates `S(X) = sum_n c[n] f_n(X)` where `X` is a matrix and the `f_n`
satisfy a three-term recurrence. Operations use sparse matrix arithmetic.

# Arguments
- `c`      -- coefficient vector of length `N`.
- `A`      -- recurrence coefficient matrices (indexed 1..N).
- `B`      -- recurrence coefficient matrices (indexed 1..N+1).
- `f0`     -- initial function matrix `f_0(X)`.
- `cutoff` -- absolute threshold below which a coefficient `c[n]` is
              treated as zero (its identity contribution is skipped).

# Returns
The evaluated matrix sum `f0 * b0`.
"""
function matrix_clenshaw(c, A, B, f0, cutoff)
    N = length(c)
    m = size(f0, 1)
    Imat = sparse(1.0 * I, m, m)
    # Clenshaw recurrence (backward sweep)
    b0 = 0 * Imat
    b1 = 0 * Imat
    for n in N:-1:1
        b2 = b1
        b1 = b0
        if abs(c[n]) > cutoff
            b0 = (c[n] * Imat) + (A[n] * b1) + (B[n+1] * b2)
        else
            b0 = (A[n] * b1) + (B[n+1] * b2)
        end
    end
    return f0 * b0
end

# ---------------------------------------------------------------------------
# kronecker_clenshaw
# ---------------------------------------------------------------------------

"""
    kronecker_clenshaw(val_c, norm_c, A, B, f0, cutoff; coeffs_left=true)

Clenshaw algorithm on matrix coefficients with matrix argument, using
Kronecker products.

Evaluates `S(X) = sum_n kron(f_n(X), c_n)` (or `kron(c_n, f_n(X))` when
`coeffs_left` is `true`).

If `val_c[1]` is a scalar, falls back to [`matrix_clenshaw`](@ref).

# Arguments
- `val_c`       -- coefficient matrices (or scalars) indexed 1..N.
- `norm_c`      -- scalar norms of each coefficient for cutoff comparison.
- `A`           -- recurrence coefficient matrices (indexed 1..N).
- `B`           -- recurrence coefficient matrices (indexed 1..N+1).
- `f0`          -- initial function matrix `f_0(X)`.
- `cutoff`      -- absolute threshold below which a norm is treated as zero.
- `coeffs_left` -- if `true` (default), Kronecker product puts coefficients
                   on the left: `kron(C, X)`.

# Returns
The evaluated Kronecker-product sum.
"""
function kronecker_clenshaw(val_c, norm_c, A, B, f0, cutoff; coeffs_left::Bool=true)
    function _kron(X, C)
        if coeffs_left
            return kron(C, X)
        else
            return kron(X, C)
        end
    end

    # Fall back to matrix_clenshaw for scalar coefficients
    if val_c[1] isa Number
        return matrix_clenshaw(val_c, A, B, f0, cutoff)
    end

    N = length(norm_c)
    m0 = size(f0, 1)
    m1 = size(val_c[1], 1)
    I0 = sparse(1.0 * I, m0, m0)
    I1 = sparse(1.0 * I, m1, m1)

    # Clenshaw recurrence (backward sweep)
    b0 = 0 * _kron(I0, val_c[1])
    b1 = 0 * _kron(I0, val_c[1])
    for n in N:-1:1
        b2 = b1
        b1 = b0
        b0 = (_kron(A[n], I1) * b1) + (_kron(B[n+1], I1) * b2)
        if norm_c[n] > cutoff
            b0 = b0 + _kron(I0, val_c[n])
        end
    end
    return _kron(f0, I1) * b0
end

# ---------------------------------------------------------------------------
# jacobi_recursion
# ---------------------------------------------------------------------------

"""
    jacobi_recursion(N, a, b, X)

Build Clenshaw recurrence coefficients `(A, B)` for Jacobi polynomials
P_n^{(a,b)}.

The Jacobi three-term recurrence is:
    J[n,n-1]*f[n-1] + J[n,n]*f[n] + J[n,n+1]*f[n+1] = X*f[n]
so:
    f[n+1] = (X - J[n,n])/J[n,n+1]*f[n] - J[n,n-1]/J[n,n+1]*f[n-1]

The Clenshaw coefficients are:
    A[n] = (X - J[n,n]) / J[n,n+1]
    B[n] = -J[n,n-1] / J[n,n+1]

Both `A` and `B` are returned as [`DeferredTuple`](@ref) objects that lazily
compute entries on demand.

# Arguments
- `N` -- number of recurrence terms.
- `a` -- first Jacobi parameter (alpha).
- `b` -- second Jacobi parameter (beta).
- `X` -- scalar or sparse matrix argument.

# Returns
- `(A, B)` tuple of `DeferredTuple` objects of length `N+1`.

# Notes
Requires `jacobi_matrix` from this module (which is currently a stub that
depends on the `dedalus_sphere` library).
"""
function jacobi_recursion(N::Integer, a, b, X)
    # Build the Jacobi matrix (N x N tridiagonal)
    J = jacobi_matrix(N, a, b)

    # Identity element (scalar 1 or sparse identity)
    if X isa Number
        Ident = one(X)
        X_val = X
    else
        # Convert to CSC if needed
        X_val = sparse(X)
        Ident = sparse(1.0 * I, size(X_val, 1), size(X_val, 1))
    end

    # Clenshaw coefficients as deferred tuples (1-based indexing)
    # Python used 0-based: A[n] for 0 <= n < N, B[n] for 0 <= n <= N
    # In Julia 1-based: A[n] for 1 <= n <= N+1, B[n] for 1 <= n <= N+1
    # Python's A[n] accesses J[n,n] and J[n,n+1] with 0-based indexing.
    # Julia's J is 1-based, so Python's J[n,n] -> Julia's J[n+1, n+1] when
    # the DeferredTuple index maps n (1-based) -> n-1 (0-based Python index).
    function compute_A(n)
        # n is 1-based DeferredTuple index, maps to Python's 0-based index (n-1)
        idx = n - 1  # 0-based index into Python's recursion
        if 0 <= idx < (N - 1)
            # J is 1-based Julia matrix: J[idx+1, idx+1] = diagonal, J[idx+1, idx+2] = super-diagonal
            return (X_val - J[idx+1, idx+1] * Ident) / J[idx+1, idx+2]
        else
            return 0 * Ident
        end
    end

    function compute_B(n)
        # n is 1-based DeferredTuple index, maps to Python's 0-based index (n-1)
        idx = n - 1  # 0-based index into Python's recursion
        if 0 < idx < (N - 1)
            return (-J[idx+1, idx] / J[idx+1, idx+2]) * Ident
        else
            return 0 * Ident
        end
    end

    A = DeferredTuple(compute_A, N + 1)
    B = DeferredTuple(compute_B, N + 1)
    return (A, B)
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export scalar_clenshaw,
       matrix_clenshaw,
       kronecker_clenshaw,
       jacobi_recursion
