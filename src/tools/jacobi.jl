"""
Jacobi polynomial tools for Dedalus.jl.

Translated from dedalus/tools/jacobi.py. Provides the Jacobi mass function
(implemented directly) and stub interfaces to dedalus_sphere.jacobi library
functions (grid construction, polynomial evaluation, differentiation,
conversion, and integration).
"""

using SpecialFunctions: loggamma
using LinearAlgebra: eigen, Symmetric, SymTridiagonal, diagm, diag, I as eye_I

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

"""Output element type for all Jacobi polynomial operations."""
const OUTPUT_DTYPE = Float64

# ---------------------------------------------------------------------------
# mass
# ---------------------------------------------------------------------------

"""
    mass(a, b)

Compute the mass (L2 norm squared) of the Jacobi weight function:

    mass(a, b) = integral from -1 to 1 of (1-x)^a (1+x)^b dx
               = 2^(a+b+1) * Gamma(a+1) * Gamma(b+1) / Gamma(a+b+2)

Uses the log-gamma function for numerical stability.

# Examples
```julia
mass(0, 0)    # 2.0
mass(0.5, 0.5)  # approximately pi/2
```
"""
function mass(a, b)
    return exp((a + b + 1) * log(2) + loggamma(a + 1) + loggamma(b + 1) - loggamma(a + b + 2))
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _jacobi_norm_sq(n, a, b) -> Float64

Squared norm of the classical Jacobi polynomial P_n^{(a,b)} under the weight
(1-x)^a * (1+x)^b on [-1,1]:
    h_n = 2^{a+b+1}/(2n+a+b+1) * Gamma(n+a+1)*Gamma(n+b+1)/(n!*Gamma(n+a+b+1))
"""
function _jacobi_norm_sq(n::Integer, a, b)
    if n == 0
        return mass(a, b)
    end
    return exp(
        (a + b + 1) * log(2.0) - log(2.0 * n + a + b + 1) +
        loggamma(n + a + 1) + loggamma(n + b + 1) -
        loggamma(Float64(n + 1)) - loggamma(n + a + b + 1)
    )
end

"""
    _norm_ratio(dn, da, db, n, a, b) -> Float64

Compute sqrt(h_{n+dn}^{(a+da,b+db)} / h_n^{(a,b)}) where h_n is the classical
Jacobi polynomial squared norm. Used for normalizing operators.
"""
function _norm_ratio(dn::Integer, da::Integer, db::Integer, n::Integer, a, b)
    return sqrt(_jacobi_norm_sq(n + dn, a + da, b + db) / _jacobi_norm_sq(n, a, b))
end

# ---------------------------------------------------------------------------
# jacobi_matrix
# ---------------------------------------------------------------------------

"""
    jacobi_matrix(N, a, b) -> Matrix{Float64}

Build the tridiagonal Jacobi matrix (three-term recurrence matrix) for
unit-normalized Jacobi polynomials P^{(a,b)}. This is the N x N symmetric
tridiagonal matrix whose eigenvalues are the Gauss-Jacobi quadrature nodes.

The recurrence is: x * p_n = e_{n-1} * p_{n-1} + d_n * p_n + e_n * p_{n+1}
where d_n = (b^2-a^2)/((2n+a+b)(2n+a+b+2)) is the diagonal entry and
e_n = sqrt(4(n+1)(n+a+1)(n+b+1)(n+a+b+1)/((2n+a+b+2)^2*((2n+a+b+2)^2-1)))
is the off-diagonal entry.

# Arguments
- `N::Integer` -- number of modes (matrix will be `N x N`).
- `a` -- first Jacobi parameter.
- `b` -- second Jacobi parameter.

# Returns
- `Matrix{Float64}` of size `(N, N)`.
"""
function jacobi_matrix(N::Integer, a, b)
    if N == 0
        return Matrix{OUTPUT_DTYPE}(undef, 0, 0)
    end
    d = zeros(OUTPUT_DTYPE, N)
    e = zeros(OUTPUT_DTYPE, max(N - 1, 0))

    # Diagonal entries
    for n in 0:(N - 1)
        s = 2.0 * n + a + b
        if abs(s) < 1e-15
            # When 2n+a+b = 0, use L'Hopital: (b^2-a^2)/(s*(s+2)) -> (b-a)/(s+2)
            d[n + 1] = (b - a) / (s + 2)
        else
            d[n + 1] = (b^2 - a^2) / (s * (s + 2))
        end
    end

    # Off-diagonal entries
    for n in 0:(N - 2)
        s = 2.0 * n + a + b + 2.0
        num = 4.0 * (n + 1) * (n + a + 1) * (n + b + 1) * (n + a + b + 1)
        den = s^2 * (s^2 - 1)
        if n == 0 && abs(a + b + 1) < 1e-15
            # Special case: a+b=-1, n=0. Both num and den -> 0.
            # Limit: 2*(a+1)*(b+1)
            e[n + 1] = sqrt(abs(2.0 * (a + 1) * (b + 1)))
        elseif abs(den) < 1e-30
            e[n + 1] = 0.0
        else
            e[n + 1] = sqrt(abs(num / den))
        end
    end

    return Matrix{OUTPUT_DTYPE}(SymTridiagonal(d, e))
end

# ---------------------------------------------------------------------------
# build_grid
# ---------------------------------------------------------------------------

"""
    build_grid(N, a, b) -> Vector{Float64}

Build the Gauss-Jacobi quadrature grid of `N` points for parameters `(a, b)`.
Returns only the grid points (not the weights). Computed via eigendecomposition
of the Jacobi matrix (Golub-Welsch algorithm).

# Arguments
- `N::Integer` -- number of quadrature points.
- `a` -- first Jacobi parameter.
- `b` -- second Jacobi parameter.

# Returns
- `Vector{Float64}` of length `N` containing the quadrature nodes sorted ascending.
"""
function build_grid(N::Integer, a, b)
    if N == 0
        return Vector{OUTPUT_DTYPE}()
    end
    J = jacobi_matrix(N, a, b)
    F = eigen(Symmetric(J))
    idx = sortperm(F.values)
    return OUTPUT_DTYPE.(F.values[idx])
end

# ---------------------------------------------------------------------------
# build_weights
# ---------------------------------------------------------------------------

"""
    build_weights(N, a, b) -> Vector{Float64}

Build the Gauss-Jacobi quadrature weights for `N` points with parameters `(a, b)`.
Returns only the weights (not the grid points). Computed via the eigenvectors
of the Jacobi matrix: weights = mass(a,b) * v[1,:]^2 where v are the
normalized eigenvectors.

# Arguments
- `N::Integer` -- number of quadrature points.
- `a` -- first Jacobi parameter.
- `b` -- second Jacobi parameter.

# Returns
- `Vector{Float64}` of length `N` containing the quadrature weights.
"""
function build_weights(N::Integer, a, b)
    if N == 0
        return Vector{OUTPUT_DTYPE}()
    end
    J = jacobi_matrix(N, a, b)
    F = eigen(Symmetric(J))
    idx = sortperm(F.values)
    m = mass(a, b)
    return OUTPUT_DTYPE[m * F.vectors[1, i]^2 for i in idx]
end

# ---------------------------------------------------------------------------
# build_polynomials
# ---------------------------------------------------------------------------

"""
    build_polynomials(M, a, b, grid) -> Matrix{Float64}

Evaluate the first `M` unit-normalized Jacobi polynomials P_n^{(a,b)} on the
given `grid`. Returns an `M x length(grid)` matrix where row `n` contains
P_{n-1}^{(a,b)} evaluated at each grid point.

Uses the three-term recurrence derived from the Jacobi matrix entries.

# Arguments
- `M::Integer` -- number of polynomials to evaluate (modes 0 through M-1).
- `a` -- first Jacobi parameter.
- `b` -- second Jacobi parameter.
- `grid` -- vector of points at which to evaluate the polynomials.

# Returns
- `Matrix{Float64}` of size `(M, length(grid))`.
"""
function build_polynomials(M::Integer, a, b, grid)
    Ng = length(grid)
    P = zeros(OUTPUT_DTYPE, M, Ng)
    if M == 0
        return P
    end
    # Normalized P_0 = 1 / sqrt(mass(a,b))
    P[1, :] .= 1.0 / sqrt(mass(a, b))
    if M == 1
        return P
    end
    # Use the Jacobi matrix for the three-term recurrence:
    # x * P_n = e_{n-1} * P_{n-1} + d_n * P_n + e_n * P_{n+1}
    # => P_{n+1} = ((x - d_n) * P_n - e_{n-1} * P_{n-1}) / e_n
    J = jacobi_matrix(M, a, b)
    d = diag(J)         # diagonal entries, length M
    e = diag(J, 1)      # superdiagonal entries, length M-1
    for n in 1:(M - 1)
        for j in 1:Ng
            x = grid[j]
            if n == 1
                P[n + 1, j] = (x - d[n]) * P[n, j] / e[n]
            else
                P[n + 1, j] = ((x - d[n]) * P[n, j] - e[n - 1] * P[n - 1, j]) / e[n]
            end
        end
    end
    return P
end

# ---------------------------------------------------------------------------
# conversion_matrix
# ---------------------------------------------------------------------------

"""
    conversion_matrix(N, a0, b0, a1, b1) -> Matrix{Float64}

Build the conversion matrix that maps coefficients of unit-normalized Jacobi
polynomials P^{(a0, b0)} to coefficients of P^{(a1, b1)}.

The parameters must satisfy:
- `a1 - a0` must be a non-negative integer (number of 'A' raising steps).
- `b1 - b0` must be a non-negative integer (number of 'B' raising steps).

The conversion is built by composing the elementary raising operators
A(+1) and B(+1), matching the Python dedalus_sphere jacobi.operator convention.

# Arguments
- `N::Integer` -- number of modes (matrix will be `N x N`).
- `a0` -- source alpha parameter.
- `b0` -- source beta parameter.
- `a1` -- target alpha parameter.
- `b1` -- target beta parameter.

# Returns
- `Matrix{Float64}` of size `(N, N)`.

# Throws
- `ArgumentError` if `a1 - a0` or `b1 - b0` is not a non-negative integer.
"""
function conversion_matrix(N::Integer, a0, b0, a1, b1)
    da = a1 - a0
    db = b1 - b0
    if da != round(Int, da) || da < 0
        throw(ArgumentError("a1 - a0 must be a non-negative integer, got $da"))
    end
    if db != round(Int, db) || db < 0
        throw(ArgumentError("b1 - b0 must be a non-negative integer, got $db"))
    end
    da_int = round(Int, da)
    db_int = round(Int, db)

    result = Matrix{OUTPUT_DTYPE}(eye_I, N, N)

    # Apply B-raising operator db times first: P^{(a0,b0)} -> P^{(a0,b0+db)}
    # Then apply A-raising operator da times: P^{(a0,b0+db)} -> P^{(a0+da,b0+db)}
    # This matches the Python: conv = A**da @ B**db, which first applies B, then A.
    cur_a = Float64(a0)
    cur_b = Float64(b0)
    for _ in 1:db_int
        result = _raising_B_normalized(N, cur_a, cur_b) * result
        cur_b += 1
    end
    for _ in 1:da_int
        result = _raising_A_normalized(N, cur_a, cur_b) * result
        cur_a += 1
    end

    return result
end

"""
    _raising_A_normalized(N, a, b) -> Matrix{Float64}

Build the A(+1) operator for unit-normalized Jacobi polynomials.
Maps P^{(a,b)} coefficients to P^{(a+1,b)} coefficients.

For unit-normalized polynomials tilde{P}_n = P_n / sqrt(h_n), the relation
P_n^{(a,b)} = c_{diag} P_n^{(a+1,b)} + c_{sup} P_{n-1}^{(a+1,b)} becomes:
tilde{P}_n^{(a,b)} = (c_{diag} * r_diag) tilde{P}_n^{(a+1,b)}
                    + (c_{sup} * r_sup) tilde{P}_{n-1}^{(a+1,b)}
where r_diag = sqrt(h_n^{(a+1,b)}/h_n^{(a,b)}) and r_sup = sqrt(h_{n-1}^{(a+1,b)}/h_n^{(a,b)}).
"""
function _raising_A_normalized(N::Integer, a, b)
    M = zeros(OUTPUT_DTYPE, N, N)
    for n in 0:(N - 1)
        s = 2.0 * n + a + b
        # Classical diagonal coefficient
        if n == 0 && abs(a + b + 1) < 1e-15
            c_diag = 1.0
        elseif n == 0
            c_diag = (n + a + b + 1) / (a + b + 1)
        else
            c_diag = (n + a + b + 1) / (s + 1)
        end
        # Normalized diagonal: A[n+1, n+1]
        M[n + 1, n + 1] = c_diag * _norm_ratio(0, 1, 0, n, a, b)
        # Superdiagonal: A[n, n+1] for n >= 1 (0-indexed)
        if n >= 1
            c_sup = -(n + b) / (s + 1)
            M[n, n + 1] = c_sup * _norm_ratio(-1, 1, 0, n, a, b)
        end
    end
    return M
end

"""
    _raising_B_normalized(N, a, b) -> Matrix{Float64}

Build the B(+1) operator for unit-normalized Jacobi polynomials.
Maps P^{(a,b)} coefficients to P^{(a,b+1)} coefficients.
Computed via: B(+1) = Pi * A(+1)(b,a) * Pi, where Pi = diag((-1)^n).
"""
function _raising_B_normalized(N::Integer, a, b)
    Pi = diagm([(-1.0)^n for n in 0:(N - 1)])
    Ap = _raising_A_normalized(N, b, a)
    return Pi * Ap * Pi
end

# ---------------------------------------------------------------------------
# differentiation_matrix
# ---------------------------------------------------------------------------

"""
    differentiation_matrix(N, a, b) -> Matrix{Float64}

Build the spectral differentiation matrix for unit-normalized Jacobi
polynomials P^{(a,b)}. The derivative of P_n^{(a,b)} lies in the P^{(a+1,b+1)}
basis, so this maps N coefficients of P^{(a,b)} to N coefficients of
P^{(a+1,b+1)}.

For classical polynomials: d/dx P_n^{(a,b)} = (n+a+b+1)/2 * P_{n-1}^{(a+1,b+1)}.
For unit-normalized polynomials, each entry is scaled by the appropriate norm ratio.

# Arguments
- `N::Integer` -- number of modes (matrix will be `N x N`).
- `a` -- first Jacobi parameter.
- `b` -- second Jacobi parameter.

# Returns
- `Matrix{Float64}` of size `(N, N)`.
"""
function differentiation_matrix(N::Integer, a, b)
    D = zeros(OUTPUT_DTYPE, N, N)
    # Superdiagonal: D[n, n+1] for n = 0..N-2 (0-indexed n maps to 1-indexed n+1)
    # Classical: d/dx P_n^{(a,b)} = (n+a+b+1)/2 * P_{n-1}^{(a+1,b+1)}
    # For unit-normalized polynomials, entry at position (n-1, n) in 0-indexed,
    # i.e., (n, n+1) in 1-indexed, is:
    # (n+a+b+1)/2 * sqrt(h_{n-1}^{(a+1,b+1)} / h_n^{(a,b)})
    for n in 1:(N - 1)
        c = (n + a + b + 1) / 2.0
        D[n, n + 1] = c * _norm_ratio(-1, 1, 1, n, a, b)
    end
    return D
end

# ---------------------------------------------------------------------------
# integration_vector
# ---------------------------------------------------------------------------

"""
    integration_vector(N, a, b) -> Vector{Float64}

Build the spectral integration vector for unit-normalized Jacobi polynomials
P^{(a,b)}. The vector `v` satisfies `v . c = integral from -1 to 1 of f(x) dx`
where `c` are the Jacobi expansion coefficients of `f`.

The implementation uses Gauss-Legendre quadrature on the interpolated
polynomial, normalized by `2 / mass(0, 0)`. Values below machine resolution
are zeroed out.

# Arguments
- `N::Integer` -- number of modes.
- `a` -- first Jacobi parameter.
- `b` -- second Jacobi parameter.

# Returns
- `Vector{Float64}` of length `N`.
"""
function integration_vector(N::Integer, a, b)
    if N == 0
        return Vector{OUTPUT_DTYPE}()
    end
    # Build Gauss-Legendre quadrature (a=0, b=0)
    leg_grid = build_grid(N, 0, 0)
    leg_weights = build_weights(N, 0, 0)
    # Evaluate Jacobi polynomials P^{(a,b)} at Legendre grid points
    interp = build_polynomials(N, a, b, leg_grid)  # N x Ng matrix
    # Compute integration vector: v_n = sum_j w_j * P_n(x_j) * (2 / mass(0,0))
    # interp is N x Ng, leg_weights is Ng-vector
    # interp * leg_weights gives N-vector
    integ = interp * leg_weights * (2.0 / mass(0.0, 0.0))
    # Zero out entries below machine resolution (matches Python: eps * N threshold)
    cutoff = eps(OUTPUT_DTYPE) * N
    for i in 1:N
        if abs(integ[i]) < cutoff
            integ[i] = 0.0
        end
    end
    return integ
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export OUTPUT_DTYPE,
       mass,
       build_grid,
       build_weights,
       build_polynomials,
       conversion_matrix,
       differentiation_matrix,
       jacobi_matrix,
       integration_vector
