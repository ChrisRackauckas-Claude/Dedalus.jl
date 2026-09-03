"""
Jacobi polynomial tools for Dedalus.jl.

Translated from dedalus/tools/jacobi.py. Provides the Jacobi mass function
(implemented directly) and stub interfaces to dedalus_sphere.jacobi library
functions (grid construction, polynomial evaluation, differentiation,
conversion, and integration).
"""

using SpecialFunctions: loggamma

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
    error("Not implemented: requires dedalus_sphere library (jacobi.quadrature)")
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
    error("Not implemented: requires dedalus_sphere library (jacobi.quadrature)")
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
    error("Not implemented: requires dedalus_sphere library (jacobi.polynomials)")
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
    error("Not implemented: requires dedalus_sphere library (jacobi.operator)")
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
    error("Not implemented: requires dedalus_sphere library (jacobi.operator('D'))")
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
    error("Not implemented: requires dedalus_sphere library (jacobi.operator('Z'))")
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
    error("Not implemented: requires dedalus_sphere library (jacobi.quadrature)")
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
