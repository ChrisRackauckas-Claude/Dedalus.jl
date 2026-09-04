"""
Clenshaw summation within the dedalus_sphere framework.

Translated from dedalus/libraries/dedalus_sphere/clenshaw.py.
NOTE: This is DISTINCT from tools/clenshaw.jl — it provides NCC matrix
construction and Clenshaw summation using the dedalus_sphere Jacobi operators.

Provides:
- DSCDeferredTuple: deferred evaluation container (named to avoid conflict
  with tools/general.jl DeferredTuple)
- dsc_ncc_matrix: NCC matrix via Clenshaw algorithm
- dsc_jacobi_recursion: Clenshaw recurrence coefficients
- dsc_matrix_clenshaw: Clenshaw summation on matrices
- dsc_jacobi_matrix: Jacobi tridiagonal matrix
"""

using SparseArrays
using LinearAlgebra: I as eye_I

# ============================================================================
# DSCDeferredTuple
# ============================================================================

"""
    DSCDeferredTuple

Deferred evaluation container for the dedalus_sphere Clenshaw module.
Named to avoid conflict with DeferredTuple from tools/general.jl.

Uses 1-based indexing.
"""
struct DSCDeferredTuple
    entry_function::Function
    size::Int
end

Base.length(dt::DSCDeferredTuple) = dt.size

function Base.getindex(dt::DSCDeferredTuple, key::Integer)
    idx = key
    if idx < 0
        idx += length(dt) + 1
    end
    if idx < 1 || idx > length(dt)
        throw(BoundsError(dt, key))
    end
    return dt.entry_function(idx)
end

Base.firstindex(dt::DSCDeferredTuple) = 1
Base.lastindex(dt::DSCDeferredTuple) = dt.size

# ============================================================================
# dsc_jacobi_matrix
# ============================================================================

"""
    dsc_jacobi_matrix(N, a, b)

Jacobi tridiagonal matrix of size N x N, using the Jacobi operator framework.
Delegates to tools/jacobi.jl's `jacobi_matrix` and returns a sparse CSR-like matrix.
"""
function dsc_jacobi_matrix(N::Int, a, b)
    J = jacobi_matrix(N, Float64(a), Float64(b))
    return sparse(J)
end

# ============================================================================
# dsc_jacobi_recursion
# ============================================================================

"""
    dsc_jacobi_recursion(N, a, b, X)

Build Clenshaw recurrence coefficients for Jacobi polynomials.

Jacobi matrix recursion:
    J[n,n-1]*f[n-1] + J[n,n]*f[n] + J[n,n+1]*f[n+1] = X*f[n]
    f[n+1] = (X - J[n,n])/J[n,n+1]*f[n] - J[n,n-1]/J[n,n+1]*f[n-1]

Clenshaw coefficients (1-based indexing, n is 1-based here):
    A[n] = (X - J[n,n])/J[n,n+1]
    B[n] = - J[n,n-1]/J[n,n+1]

Returns (A, B) as DSCDeferredTuples of size N+1.
"""
function dsc_jacobi_recursion(N::Int, a, b, X)
    # Jacobi matrix (N x N)
    J = dsc_jacobi_matrix(N, a, b)
    JA = Matrix(J)

    # Identity element: scalar or sparse identity depending on X
    if isa(X, Number)
        Id = 1
    else
        Id = sparse(one(Float64) * eye_I, size(X, 1), size(X, 1))
    end

    # Clenshaw coefficients (1-based indexing)
    # Python n ranges 0..N-1 for valid, here Julia n ranges 1..N
    function compute_A(n)
        # n is 1-based; Python n was 0-based
        # Valid when 1 <= n <= N-1 (Python: 0 <= n < N-1)
        n0 = n - 1  # convert to 0-based for matrix indexing comparison
        if 0 <= n0 < (N - 1)
            # JA is 1-based: JA[n, n] is diagonal, JA[n, n+1] is superdiagonal
            return (X - JA[n, n] * Id) / JA[n, n + 1]
        else
            return 0 * Id
        end
    end

    function compute_B(n)
        # n is 1-based; Python n was 0-based
        # Valid when 1 < n0 < N-1, i.e., n0 > 0 (Python: 0 < n < N-1)
        n0 = n - 1
        if 0 < n0 < (N - 1)
            return (-J[n, n - 1] / J[n, n + 1]) * Id
        else
            return 0 * Id
        end
    end

    A = DSCDeferredTuple(compute_A, N + 1)
    B = DSCDeferredTuple(compute_B, N + 1)
    return A, B
end

# ============================================================================
# dsc_matrix_clenshaw
# ============================================================================

"""
    dsc_matrix_clenshaw(c, A, B, f0; cutoff=1e-6)

Clenshaw algorithm on scalar coefficients, matrix argument:
    S(X) = sum_n c_n f_n(X)

Returns (n_terms, max_term, result_matrix).
"""
function dsc_matrix_clenshaw(c, A, B, f0; cutoff::Float64=1e-6)
    N = length(c)
    Id = sparse(one(Float64) * eye_I, size(f0, 1), size(f0, 1))

    # Clenshaw iteration
    b0 = 0 * Id
    b1 = 0 * Id
    n_terms = 0
    max_term = 0

    # Reverse iteration: Python range(N) reversed is N-1, N-2, ..., 0
    # In Julia with 1-based indexing on c and A/B:
    # c[n+1] in Julia corresponds to c[n] in Python (0-based)
    # A[n+1] in Julia corresponds to A[n] in Python
    # B[n+2] in Julia corresponds to B[n+1] in Python
    for n in reverse(0:(N - 1))
        b2 = b1
        b1 = b0
        # n is 0-based coefficient index; c is 1-based array
        # A and B are 1-based DSCDeferredTuples
        if abs(c[n + 1]) > cutoff
            b0 = (c[n + 1] * Id) + (A[n + 1] * b1) + (B[n + 2] * b2)
            n_terms += 1
            if max_term == 0
                # reversed range, so first term is max_term
                max_term = n
            end
        else
            b0 = (A[n + 1] * b1) + (B[n + 2] * b2)
        end
    end

    return n_terms, max_term, (b0 * f0)
end

# ============================================================================
# dsc_ncc_matrix
# ============================================================================

"""
    dsc_ncc_matrix(N, a_ncc, b_ncc, a_arg, b_arg, coeffs; cutoff=1e-6)

Build NCC matrix via Clenshaw algorithm.

Called from basis of NCC (i.e., r). A, B are from basis that the NCC is in;
arg_basis is the basis of the thing we're multiplying by (i.e., if we are
doing u.grad X, then arg_basis is the basis of u).
"""
function dsc_ncc_matrix(N::Int, a_ncc, b_ncc, a_arg, b_arg, coeffs; cutoff::Float64=1e-6)
    # Kronecker Clenshaw on argument Jacobi matrix
    J = dsc_jacobi_matrix(N, a_arg, b_arg)
    A, B = dsc_jacobi_recursion(N, a_ncc, b_ncc, J)
    f0 = (1.0 / sqrt(jacobi_mass(a_ncc, b_ncc))) * sparse(one(Float64) * eye_I, N, N)
    total = dsc_matrix_clenshaw(coeffs, A, B, f0; cutoff=cutoff)
    return total
end
