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

    mass(a, b) = ∫₋₁¹ (1-x)^a (1+x)^b dx
               = 2^(a+b+1) * Γ(a+1) * Γ(b+1) / Γ(a+b+2)

Uses the log-gamma function for numerical stability.

# Examples
```julia
mass(0, 0)    # 2.0
mass(0.5, 0.5)  # ≈ π/2
```
"""
function mass(a, b)
    return exp((a + b + 1) * log(2) + loggamma(a + 1) + loggamma(b + 1) - loggamma(a + b + 2))
end

# ---------------------------------------------------------------------------
# build_grid (stub)
# ---------------------------------------------------------------------------

"""
    build_grid(N, a, b) -> Vector{Float64}

Build the Gauss-Jacobi quadrature grid of `N` points for parameters `(a, b)`.
Returns only the grid points (not the weights).

Delegates to `jacobi.quadrature(N, a, b)` from the `dedalus_sphere` library.

# Arguments
- `N::Integer` -- number of quadrature points.
- `a` -- first Jacobi parameter (α).
- `b` -- second Jacobi parameter (β).

# Returns
- `Vector{Float64}` of length `N` containing the quadrature nodes.
"""
function build_grid(N::Integer, a, b)
    if N == 0
        return Vector{OUTPUT_DTYPE}()
    end
    J = jacobi_matrix(N, a, b)
    F = eigen(Symmetric(J))
    # Eigenvalues are the quadrature nodes, sorted ascending
    idx = sortperm(F.values)
    return OUTPUT_DTYPE.(F.values[idx])
end

# ---------------------------------------------------------------------------
# build_weights (stub)
# ---------------------------------------------------------------------------

"""
    build_weights(N, a, b) -> Vector{Float64}

Build the Gauss-Jacobi quadrature weights for `N` points with parameters `(a, b)`.
Returns only the weights (not the grid points).

Delegates to `jacobi.quadrature(N, a, b)` from the `dedalus_sphere` library.

# Arguments
- `N::Integer` -- number of quadrature points.
- `a` -- first Jacobi parameter (α).
- `b` -- second Jacobi parameter (β).

# Returns
- `Vector{Float64}` of length `N` containing the quadrature weights.
"""
function build_weights(N::Integer, a, b)
    if N == 0
        return Vector{OUTPUT_DTYPE}()
    end
    J = jacobi_matrix(N, a, b)
    F = eigen(Symmetric(J))
    # Sort by eigenvalues to match build_grid ordering
    idx = sortperm(F.values)
    # Weights = mass(a, b) * (first component of each eigenvector)^2
    m = mass(a, b)
    w = OUTPUT_DTYPE[m * F.vectors[1, i]^2 for i in idx]
    return w
end

# ---------------------------------------------------------------------------
# build_polynomials (stub)
# ---------------------------------------------------------------------------

"""
    build_polynomials(M, a, b, grid) -> Matrix{Float64}

Evaluate the first `M` Jacobi polynomials P_n^{(a,b)} on the given `grid`.
Returns an `M × length(grid)` matrix where row `n` contains P_{n-1}^{(a,b)}
evaluated at each grid point.

Delegates to `jacobi.polynomials(M, a, b, grid)` from the `dedalus_sphere` library.

# Arguments
- `M::Integer` -- number of polynomials to evaluate (modes 0 through M-1).
- `a` -- first Jacobi parameter (α).
- `b` -- second Jacobi parameter (β).
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
    # Normalized initial value: P_0 = 1 / sqrt(mass(a,b))
    P[1, :] .= 1.0 / sqrt(mass(a, b))
    if M == 1
        return P
    end
    # Use the Jacobi matrix for the three-term recurrence
    J = jacobi_matrix(M, a, b)
    d = diag(J)          # diagonal entries, length M
    e = diag(J, 1)       # superdiagonal entries, length M-1
    # Recurrence: e[n] * P_{n+1}(x) = (x - d[n]) * P_n(x) - e[n-1] * P_{n-1}(x)
    # (with e[0] = 0 for the first step)
    for n in 1:(M - 1)
        for j in 1:Ng
            x = grid[j]
            P[n + 1, j] = (x - d[n]) * P[n, j] / e[n]
            if n > 1
                P[n + 1, j] -= e[n - 1] * P[n - 1, j] / e[n]
            end
        end
    end
    return P
end

# ---------------------------------------------------------------------------
# conversion_matrix (stub)
# ---------------------------------------------------------------------------

"""
    conversion_matrix(N, a0, b0, a1, b1) -> Matrix{Float64}

Build the conversion matrix that maps coefficients of Jacobi polynomials
P^{(a0, b0)} to coefficients of P^{(a1, b1)}.

The parameters must satisfy:
- `a1 - a0` must be a non-negative integer (number of 'A' raising steps).
- `b1 - b0` must be a non-negative integer (number of 'B' raising steps).

The conversion is built by composing the elementary raising operators
`jacobi.operator('A')(+1)` and `jacobi.operator('B')(+1)` from the
`dedalus_sphere` library.

Delegates to `jacobi.operator` from the `dedalus_sphere` library.

# Arguments
- `N::Integer` -- number of modes (matrix will be `N × N`).
- `a0` -- source α parameter.
- `b0` -- source β parameter.
- `a1` -- target α parameter.
- `b1` -- target β parameter.

# Returns
- `Matrix{Float64}` of size `(N, N)`.

# Throws
- `ValueError` if `a1 - a0` or `b1 - b0` is not a non-negative integer.
"""
function conversion_matrix(N::Integer, a0, b0, a1, b1)
    # Validate integer separation of parameters
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

    # Apply A-raising operator da times: P^{(a,b)} -> P^{(a+1,b)}
    ac = Float64(a0)
    bc = Float64(b0)
    for _ in 1:da_int
        result = _raising_A(N, ac, bc) * result
        ac += 1
    end

    # Apply B-raising operator db times: P^{(a,b)} -> P^{(a,b+1)}
    for _ in 1:db_int
        result = _raising_B(N, ac, bc) * result
        bc += 1
    end

    return result
end

"""
    _raising_A(N, a, b) -> Matrix{Float64}

Elementary raising operator A(+1): maps P^{(a,b)} coefficients to P^{(a+1,b)} coefficients.
Matrix layout follows the Python dedalus_sphere convention: diagonal + subdiagonal.
"""
function _raising_A(N::Integer, a, b)
    M = zeros(OUTPUT_DTYPE, N, N)
    for n in 0:(N - 1)
        s = 2.0 * n + a + b
        # Diagonal: (n + a + b + 1) / (s + 2)
        denom = s + 2
        if abs(denom) > 1e-30
            M[n + 1, n + 1] = (n + a + b + 1) / denom
        end
        # Subdiagonal: M[n+1, n] = (n + b) / s  for n >= 1 (0-indexed)
        if n >= 1
            if abs(s) > 1e-30
                M[n + 1, n] = (n + b) / s
            else
                M[n + 1, n] = 0.5
            end
        end
    end
    return M
end

"""
    _raising_B(N, a, b) -> Matrix{Float64}

Elementary raising operator B(+1): maps P^{(a,b)} coefficients to P^{(a,b+1)} coefficients.
Matrix layout follows the Python dedalus_sphere convention: diagonal + subdiagonal.
"""
function _raising_B(N::Integer, a, b)
    M = zeros(OUTPUT_DTYPE, N, N)
    for n in 0:(N - 1)
        s = 2.0 * n + a + b
        # Diagonal: (n + a + b + 1) / (s + 2)
        denom = s + 2
        if abs(denom) > 1e-30
            M[n + 1, n + 1] = (n + a + b + 1) / denom
        end
        # Subdiagonal: M[n+1, n] = -(n + a) / s  for n >= 1 (0-indexed)
        if n >= 1
            if abs(s) > 1e-30
                M[n + 1, n] = -(n + a) / s
            else
                M[n + 1, n] = -0.5
            end
        end
    end
    return M
end

# ---------------------------------------------------------------------------
# differentiation_matrix (stub)
# ---------------------------------------------------------------------------

"""
    differentiation_matrix(N, a, b) -> Matrix{Float64}

Build the spectral differentiation matrix for Jacobi polynomials P^{(a,b)}.
The derivative raises both parameters by 1, so this maps N coefficients of
P^{(a,b)} to N coefficients of P^{(a+1,b+1)}.

Delegates to `jacobi.operator('D')(+1)` from the `dedalus_sphere` library.

# Arguments
- `N::Integer` -- number of modes (matrix will be `N × N`).
- `a` -- first Jacobi parameter (α).
- `b` -- second Jacobi parameter (β).

# Returns
- `Matrix{Float64}` of size `(N, N)`.
"""
function differentiation_matrix(N::Integer, a, b)
    M = zeros(OUTPUT_DTYPE, N, N)
    # Superdiagonal: M[i, i+1] = (n + a + b + 1) / 2 where n = i-1 (0-indexed)
    # Python: coeffs[n] = (n + a + b + 1) / 2, placed on superdiagonal k=1
    # In 1-indexed Julia: M[i, i+1] = ((i-1) + a + b + 1) / 2 = (i + a + b) / 2
    for i in 1:(N - 1)
        M[i, i + 1] = (i - 1 + a + b + 1) / 2.0
    end
    return M
end

# ---------------------------------------------------------------------------
# jacobi_matrix (stub)
# ---------------------------------------------------------------------------

"""
    jacobi_matrix(N, a, b) -> Matrix{Float64}

Build the tridiagonal Jacobi matrix (three-term recurrence matrix) for
Jacobi polynomials P^{(a,b)}. This is the `N × N` symmetric tridiagonal
matrix whose eigenvalues are the Gauss-Jacobi quadrature nodes.

Delegates to `jacobi.operator('Z')` from the `dedalus_sphere` library.

# Arguments
- `N::Integer` -- number of modes (matrix will be `N × N`).
- `a` -- first Jacobi parameter (α).
- `b` -- second Jacobi parameter (β).

# Returns
- `Matrix{Float64}` of size `(N, N)`.
"""
function jacobi_matrix(N::Integer, a, b)
    if N == 0
        return Matrix{OUTPUT_DTYPE}(undef, 0, 0)
    end
    if N == 1
        # For n=0, diagonal is (b^2-a^2)/((a+b)(a+b+2)), but handle a+b=0 or a+b=-1 specially
        if a + b == 0
            d0 = (b - a) / 2.0
        elseif a + b == -1
            d0 = 0.0  # (b^2-a^2) / ((2*0+a+b)*(2*0+a+b+2)) = (b-a)(b+a)/(-1*1) but a+b=-1 so numerator is (b-a)*(-1)
            # Actually for a+b=-1: 2n+a+b = -1 for n=0, formula has 0/0 issue.
            # From the Python code: the Z operator for N=1 is just 0 when a=b, and (b-a)/(a+b+2) otherwise?
            # Let me use the standard formula: diagonal[n] = (b^2-a^2)/((2n+a+b)(2n+a+b+2))
            # For n=0, a+b=-1: (b^2-a^2)/((-1)(1)) = -(b^2-a^2) = a^2-b^2
            d0 = (a^2 - b^2) / 1.0  # (b^2-a^2)/((a+b)*(a+b+2)) = (b^2-a^2)/((-1)*(1))
        else
            d0 = (b^2 - a^2) / ((a + b) * (a + b + 2))
        end
        return OUTPUT_DTYPE[d0;;]
    end
    # Build N×N symmetric tridiagonal Jacobi matrix for unit-normalized Jacobi polynomials
    # The recurrence: x * p_n = off_{n-1} * p_{n-1} + diag_n * p_n + off_n * p_{n+1}
    # diag_n = (b^2 - a^2) / ((2n+a+b)(2n+a+b+2))
    # off_n = 2/(2n+a+b+2) * sqrt((n+1)(n+a+1)(n+b+1)(n+a+b+1)/((2n+a+b+1)(2n+a+b+3)))
    diag = zeros(OUTPUT_DTYPE, N)
    offdiag = zeros(OUTPUT_DTYPE, N - 1)
    for n in 0:(N-1)
        s = 2 * n + a + b
        if s == 0 && n == 0
            # Special case: a+b=0, n=0
            # diag[0] = (b-a)/2 from L'Hopital or direct computation
            diag[n+1] = (b - a) / 2.0
        elseif abs(s) < 1e-15 && abs(s + 2) < 1e-15
            diag[n+1] = 0.0
        else
            diag[n+1] = (b^2 - a^2) / (s * (s + 2))
        end
    end
    for n in 0:(N-2)
        s = 2 * n + a + b + 2
        num = (n + 1) * (n + a + 1) * (n + b + 1) * (n + a + b + 1)
        den = (s - 1) * (s + 1)
        if abs(den) < 1e-30
            offdiag[n+1] = 0.0
        else
            offdiag[n+1] = 2.0 / s * sqrt(abs(num / den))
        end
    end
    J = Matrix{OUTPUT_DTYPE}(SymTridiagonal(diag, offdiag))
    return J
end

# ---------------------------------------------------------------------------
# integration_vector (stub)
# ---------------------------------------------------------------------------

"""
    integration_vector(N, a, b) -> Vector{Float64}

Build the spectral integration vector for Jacobi polynomials P^{(a,b)}.
The vector `v` satisfies `v ⋅ c ≈ ∫₋₁¹ f(x) dx` where `c` are the Jacobi
expansion coefficients of `f`.

The implementation uses Gauss-Legendre quadrature on the interpolated
polynomial, normalized by `2 / mass(0, 0)`. Values below machine resolution
are zeroed out.

Delegates to `jacobi.quadrature` and `build_polynomials` from this module.

# Arguments
- `N::Integer` -- number of modes.
- `a` -- first Jacobi parameter (α).
- `b` -- second Jacobi parameter (β).

# Returns
- `Vector{Float64}` of length `N`.
"""
function integration_vector(N::Integer, a, b)
    if N == 0
        return Vector{OUTPUT_DTYPE}()
    end
    # Build Gauss-Legendre quadrature (a=0, b=0)
    x = build_grid(N, 0, 0)
    w = build_weights(N, 0, 0)
    # Evaluate Jacobi polynomials P^{(a,b)}_n at Legendre grid points
    P = build_polynomials(N, a, b, x)  # M x Ng matrix
    # Integration vector: v[n] = sum_j w[j] * P[n, j]
    v = P * w
    # Zero out values below machine epsilon threshold
    threshold = eps(OUTPUT_DTYPE) * N
    v[abs.(v) .< threshold] .= 0.0
    return v
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
