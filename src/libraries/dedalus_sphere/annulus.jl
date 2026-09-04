"""
Annulus operator algebra for Dedalus.jl.

Translated from dedalus/libraries/dedalus_sphere/annulus.py.
Provides quadrature, trial functions, and operator construction
for annular domains built on the Jacobi framework.
"""

using SparseArrays

# ============================================================================
# Constants
# ============================================================================

"""Default Jacobi parameters for annulus."""
const annulus_alpha = (-0.5, -0.5)

# ============================================================================
# Helper: evaluate a Jacobi operator at given parameters, returning a matrix
# ============================================================================

"""
    _annulus_jacobi_op(name, n, a, b)

Evaluate a named Jacobi operator at (n, a, b), returning an InfiniteCSC matrix.

Supported names: "0", "I", "J", "A+", "B+", "D+", "z=-1", "z=+1".
"""
function _annulus_jacobi_op(name::String, n::Int, a, b)
    if name == "0"
        return InfiniteCSC(spzeros(Float64, max(n + 1, 0), max(n + 1, 0)))
    elseif name == "I"
        return jacobi_operator("Id")(n + 1, a, b)
    elseif name == "J"
        return jacobi_operator("Z")(n + 1, a, b)
    elseif name == "A+"
        return jacobi_operator("A")(+1)(n + 1, a, b)
    elseif name == "B+"
        return jacobi_operator("B")(+1)(n + 1, a, b)
    elseif name == "D+"
        return jacobi_operator("D")(+1)(n + 1, a, b)
    elseif name == "z=-1"
        # Evaluation at z = -1: row vector [P_0(-1), P_1(-1), ..., P_{n}(-1)]
        # For unit-normalised Jacobi: P_k(-1) = (-1)^k * sqrt(h_k(a,b)) * ...
        # Use polynomial evaluation directly
        z_arr = [-1.0]
        P = jacobi_polynomials(n + 1, a, b, z_arr)
        # P is (n+1) x 1, transpose to get 1 x (n+1) row vector
        return InfiniteCSC(sparse(P'))
    elseif name == "z=+1"
        # Evaluation at z = +1
        z_arr = [1.0]
        P = jacobi_polynomials(n + 1, a, b, z_arr)
        return InfiniteCSC(sparse(P'))
    else
        error("Unknown annulus Jacobi operator name: $name")
    end
end

# ============================================================================
# annulus_quadrature
# ============================================================================

"""
    annulus_quadrature(Nmax; alpha=annulus_alpha, kwargs...)

Compute Jacobi quadrature for the annulus domain.
"""
function annulus_quadrature(Nmax; alpha=annulus_alpha, kwargs...)
    return jacobi_quadrature(Nmax, alpha[1], alpha[2]; kwargs...)
end

# ============================================================================
# annulus_trial_functions
# ============================================================================

"""
    annulus_trial_functions(Nmax, z; alpha=annulus_alpha)

Evaluate trial functions for the annulus via Jacobi polynomial recursion.
"""
function annulus_trial_functions(Nmax, z; alpha=annulus_alpha)
    init = 1.0 / sqrt(jacobi_mass(alpha[1], alpha[2])) .+ 0.0 .* z
    return jacobi_polynomials(Nmax, alpha[1], alpha[2], z; init=init)
end

# ============================================================================
# annulus_operator
# ============================================================================

"""
    annulus_operator(dimension, op, Nmax, k, ell, radii; pad=0, alpha=annulus_alpha)

Operator factory for annulus domain.

Supported operators: "0", "I", "R", "E", "D+", "D-", "r=Ri", "r=Ro".

Parameters:
- dimension: spatial dimension
- op: operator name string
- Nmax: maximum polynomial degree
- k: regularity parameter
- ell: azimuthal mode number
- radii: (Ri, Ro) inner and outer radii tuple
- pad: extra padding for matrix size
- alpha: base Jacobi parameters
"""
function annulus_operator(dimension, op::String, Nmax, k, ell, radii; pad::Int=0, alpha=annulus_alpha)
    if radii[2] <= radii[1]
        throw(ArgumentError("Inner radius must be greater than outer radius."))
    end

    gapwidth = radii[2] - radii[1]
    aspectratio = (radii[2] + radii[1]) / gapwidth

    a = k + alpha[1]
    b = k + alpha[2]
    N = Nmax + pad

    # zeros
    if op == "0"
        return _annulus_jacobi_op("0", N, a, b)
    end

    # identity
    if op == "I"
        return _annulus_jacobi_op("I", N, a, b)
    end

    Z = aspectratio * _annulus_jacobi_op("I", N + 2, a, b) + _annulus_jacobi_op("J", N + 2, a, b)

    # r multiplication
    if op == "R"
        return (gapwidth / 2) * Z[1:(N+2), 1:(N+2)]
    end

    E_mat = _annulus_jacobi_op("A+", N + 2, a, b + 1) * _annulus_jacobi_op("B+", N + 2, a, b)

    # conversion
    if op == "E"
        return 0.5 * (E_mat * Z)[1:(N+2), 1:(N+2)]
    end

    D_mat = _annulus_jacobi_op("D+", N + 2, a, b) * Z

    # derivatives
    # Use + with negated scalar for InfiniteCSC row-padding compatibility (no - defined).
    # Use (1/gapwidth) * instead of / gapwidth (no / defined for InfiniteCSC).
    if op == "D+"
        return (1.0 / gapwidth) * (D_mat + (-(ell + k + 1)) * E_mat)[1:(N+2), 1:(N+2)]
    end
    if op == "D-"
        return (1.0 / gapwidth) * (D_mat + (ell - k + dimension - 3) * E_mat)[1:(N+2), 1:(N+2)]
    end

    # restriction
    if op == "r=Ri"
        return _annulus_jacobi_op("z=-1", N, a, b)
    end
    if op == "r=Ro"
        return _annulus_jacobi_op("z=+1", N, a, b)
    end

    error("Unknown annulus operator: $op")
end
