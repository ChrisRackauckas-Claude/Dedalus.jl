"""
Ball wrapper module for Dedalus.jl.

Translated from dedalus/libraries/dedalus_sphere/ball_wrapper.py.
Provides the BallWrapper struct that precomputes quadrature grids, transform
matrices, and operators for 3D ball geometry spectral transforms.

This module relies on ball128 functions implemented here as ball_* wrappers
around the existing Zernike/Jacobi framework (dimension=3), and on the
SphereWrapper from sphere_wrapper.jl for angular transforms.

This module defines:
- Ball-specific radial functions: ball_quadrature, ball_polynomial, ball_N_min,
  ball_operator, ball_zeros, ball_xi, ball_unitary3D, ball_spins, ball_bar,
  ball_get_element, ball_replace_index, ball_Q_normalization, ball_delta,
  ball_k, ball_R_helper, ball_recurseQ, ball_Q
- BallWrapper: main struct holding grids, weights, transforms, operators
- grid, weight: grid/weight access methods
- ball_op: cached operator matrix access
- ball_xi_method: xi method on BallWrapper
- forward_angle, backward_angle: angular transforms
- forward_component, backward_component: radial component transforms
- radial_forward, radial_backward: full radial transforms
- ball_ncc_matrix: NCC matrix construction
- ball_grad, ball_curl: gradient and curl in coefficient space
- ball_div, ball_div_grad: divergence and Laplacian
- ball_cross_grid, ball_dot_grid: grid-space cross/dot products
- ball_unpack, ball_pack, ball_rank: utility functions
- TensorField, TensorField2D, TensorField3D: tensor field types
"""

using SparseArrays
using LinearAlgebra

# ============================================================================
# Ball-specific radial functions (ball128 equivalents)
# ============================================================================

"""Default Jacobi parameter for ball (same as Zernike alpha)."""
const ball_alpha = -0.5

"""
    ball_quadrature(N_max; a=ball_alpha, niter=3, report_error=true)

Compute Gauss-Jacobi quadrature nodes and weights for the ball radial coordinate.
Wraps jacobi_quadrature with parameters (a, 1/2) and rescales weights by 1/sqrt(32).
"""
function ball_quadrature(N_max::Int; a=ball_alpha, niter::Int=3, report_error::Bool=true)
    z, w = jacobi_quadrature(N_max, a, 0.5; days=niter)
    return z, w ./ sqrt(32)
end

"""
    ball_polynomial(N, k, ell, z; a=ball_alpha)

Evaluate ball radial basis polynomials at grid points z.
Parameters: N is max degree, k is derivative order shift, ell is spherical harmonic degree.
"""
function ball_polynomial(N::Int, k::Int, ell::Int, z; a=ball_alpha)
    q = k + a
    m = ell + 0.5
    # init = sqrt(2^(k+5/2)) * jacobi_envelope(q, m, q, 1/2, z)
    # jacobi_envelope computes the measure ratio, equivalent to:
    # init = sqrt(2^(k+5/2)) * sqrt(measure(q,m,z) / measure(q,1/2,z))
    # We use the Zernike-style initialization with jacobi_measure and jacobi_mass
    # The envelope in ball128 is: jacobi.envelope(q, m, q, 1/2, z)
    # which equals: sqrt(measure(q, m, z) / mass(q, m)) * sqrt(mass(q, 1/2) / measure(q, 1/2, z))
    # In the zernike framework, init uses jacobi_measure and jacobi_mass similarly.

    # Direct approach: compute init as in ball128.py
    # init = sqrt(2^(k+5/2)) * envelope(q, m, q, 1/2, z)
    # envelope(a, b, c, d, z) = sqrt(measure(a,b,z)/mass(a,b)) / sqrt(measure(c,d,z)/mass(c,d))
    #                         = sqrt(measure(a,b,z)*mass(c,d) / (mass(a,b)*measure(c,d,z)))
    # For our case: a=q, b=m, c=q, d=1/2
    # envelope = sqrt(measure(q,m,z)/mass(q,m)) / sqrt(measure(q,1/2,z)/mass(q,1/2))
    #          = sqrt(measure(q,m,z)*mass(q,1/2) / (mass(q,m)*measure(q,1/2,z)))

    log_init = jacobi_measure(0, ell, z; log=true, probability=false)
    # This gives log of (1-z)^0 * (1+z)^ell on the grid points
    # Actually for ball128: init = sqrt(2^(k+5/2)) * envelope
    # Let's follow the zernike pattern more closely but for dimension=3:
    # zernike_polynomials(3, n, k, l, z):
    #   b = l + 3/2 - 1 = l + 1/2
    #   init = exp(0.5 * (jacobi_measure(0, l, z; log=true, probability=false)
    #           - jacobi_mass(k, b; log=true) + log(2)*(k + 3/2 + 1)))
    # This is exactly what we need, since dimension=3 for the ball
    b = ell + 0.5  # = m
    log_init_vals = jacobi_measure(0, ell, z; log=true, probability=false)
    log_init_vals .-= jacobi_mass(k, b; log=true) .- log(2) * (k + 3/2 + 1)
    init = exp.(0.5 .* log_init_vals)

    return jacobi_polynomials(N, k, b, z; init=init)
end

"""
    ball_N_min(ell)

Minimum polynomial degree for given spherical harmonic degree ell.
"""
function ball_N_min(ell::Int)
    return max(fld(ell, 2), 0)
end

"""
    ball_operator(op_name, N, k, ell; a=ball_alpha, dtype=Float64)

Return sparse matrix operators for the ball radial basis.
Maps ball operator names to Jacobi operators with appropriate parameters.
"""
function ball_operator(op_name::String, N::Int, k::Int, ell::Int; a=ball_alpha, dtype::Type=Float64)
    q = k + a
    m = ell + 0.5

    if op_name == "0"
        op = jacobi_operator("Id")
        return dtype.(sparse(op(N, q, m))) .* 0
    elseif op_name == "I"
        op = jacobi_operator("Id")
        return dtype.(sparse(op(N, q, m)))
    elseif op_name == "E"
        # Conversion operator: A+ with rescale sqrt(0.5)
        op = jacobi_operator("A")
        mat = sparse(op(+1)(N, q, m))
        return dtype.(sqrt(0.5) .* mat)
    elseif op_name == "D-"
        # Derivative operator: C+ with rescale 2.0
        op = jacobi_operator("C")
        mat = sparse(op(+1)(N, q, m))
        return dtype.(2.0 .* mat)
    elseif op_name == "D+"
        # Derivative operator: D+ with rescale 2.0
        op = jacobi_operator("D")
        mat = sparse(op(+1)(N, q, m))
        return dtype.(2.0 .* mat)
    elseif op_name == "R-"
        # r-multiplication lowering: B- with rescale sqrt(0.5)
        op = jacobi_operator("B")
        mat = sparse(op(-1)(N, q, m))
        return dtype.(sqrt(0.5) .* mat)
    elseif op_name == "R+"
        # r-multiplication raising: B+ with rescale sqrt(0.5)
        op = jacobi_operator("B")
        mat = sparse(op(+1)(N, q, m))
        return dtype.(sqrt(0.5) .* mat)
    elseif op_name == "Z"
        # z = 2r^2-1 multiplication: J operator
        op = jacobi_operator("Z")
        mat = sparse(op(N, q, m))
        return dtype.(mat)
    elseif op_name == "r=1"
        # Boundary evaluation at r=1: z=+1 with rescale sqrt(2.0)
        op = jacobi_operator("A")
        # z=+1 evaluation uses the sum of all polynomial values at z=1
        # In jacobi128: operator('z=+1', N, q, m, rescale=sqrt(2.0))
        # This is typically a row vector that evaluates the polynomial at z=+1
        error("ball_operator 'r=1' not yet implemented")
    else
        error("Unknown ball operator name: $op_name")
    end
end

"""
    ball_zeros(N, ell, deg_out, deg_in)

Return a sparse zero matrix of appropriate dimensions for an operator
mapping between ball radial bases.
"""
function ball_zeros(N::Int, ell::Int, deg_out::Int, deg_in::Int)
    nrows = N + 1 - ball_N_min(ell + deg_out)
    ncols = N + 1 - ball_N_min(ell + deg_in)
    return spzeros(Float64, max(nrows, 0), max(ncols, 0))
end

"""
    ball_xi(mu, ell)

Normalized derivative scale factors for ball geometry.
For scalar mu: xi(-1,ell) = sqrt(ell/(2*ell+1)), xi(+1,ell) = sqrt((ell+1)/(2*ell+1)).
"""
function ball_xi(mu::Int, ell::Int)
    if mu == -1
        return sqrt(ell / (2 * ell + 1))
    elseif mu == +1
        return sqrt((ell + 1) / (2 * ell + 1))
    else
        return 0.0
    end
end

"""
    ball_xi(mu::Vector{Int}, ell::Int)

When mu is a list like [-1,+1], returns a tuple of xi values.
"""
function ball_xi(mu::Vector{Int}, ell::Int)
    return Tuple(ball_xi(m, ell) for m in mu)
end

"""
    ball_unitary3D(; rank=1, adjoint=false)

Unitary transformation matrix between Cartesian (r, theta, phi) and spin (-, 0, +) components.
For rank > 1, uses Kronecker products.
"""
function ball_unitary3D(; rank::Int=1, adjoint::Bool=false)
    if adjoint
        U = sqrt(0.5) * [0.0+0.0im 1.0+0.0im -1.0im;
                          sqrt(2.0)+0.0im 0.0+0.0im 0.0+0.0im;
                          0.0+0.0im 1.0+0.0im 1.0im]
    else
        U = sqrt(0.5) * [0.0+0.0im sqrt(2.0)+0.0im 0.0+0.0im;
                          1.0+0.0im 0.0+0.0im 1.0+0.0im;
                          1.0im 0.0+0.0im -1.0im]
    end

    unitary_mat = U
    for _ in 1:(rank - 1)
        unitary_mat = kron(U, unitary_mat)
    end
    return unitary_mat
end

"""
    ball_get_element(nu, element_rank)

Extract a single spin index from a packed multi-index nu at position element_rank (0-based).
Returns value in {-1, 0, +1}.
"""
function ball_get_element(nu::Int, element_rank::Int)
    nu_shifted = fld(nu, 3^element_rank)
    return (nu_shifted % 3) - 1
end

"""
    ball_bar(mu, rank)

Compute the total spin ("bar") of a packed tensor index mu
by summing individual spin elements across all rank positions.
"""
function ball_bar(mu::Int, rank::Int)
    mubar = 0
    for i in 0:(rank - 1)
        mubar += ball_get_element(mu, i)
    end
    return mubar
end

"""
    ball_spins(rank)

Returns an array of total spin values for each component of a tensor of given rank.
Array has length 3^rank.
"""
function ball_spins(rank::Int)
    n = 3^rank
    spin = zeros(Float64, n)
    for i in 0:(n - 1)
        spin[i + 1] = ball_bar(i, rank)
    end
    return spin
end

"""
    ball_replace_index(nu, nup, i)

In multi-index nu, replace the i-th element (0-based) with nup.
"""
function ball_replace_index(nu::Int, nup::Int, i::Int)
    nui = ball_get_element(nu, i)
    nu -= (nui + 1) * (3^i)
    return nu + (nup + 1) * (3^i)
end

"""
    ball_delta(mu, nu)

Kronecker delta: returns 1.0 if mu == nu, else 0.0.
"""
function ball_delta(mu::Int, nu::Int)
    return mu == nu ? 1.0 : 0.0
end

"""
    ball_k(mu, ell, s)

Angular derivative scale factors.
k(mu, ell, s) = -mu * sqrt((ell - mu*s) * (ell + mu*s + 1) / 2)
Returns Inf when out of range.
"""
function ball_k(mu::Int, ell::Int, s)
    if (ell < mu * s) || (ell < -mu * s - 1)
        return Inf
    end
    return -mu * sqrt((ell - mu * s) * (ell + mu * s + 1) / 2)
end

"""
    ball_Q_normalization(ell, mu)

Normalization factor for Q-matrix elements.
Returns nothing (equivalent to Python None) when the normalization is undefined.
"""
function ball_Q_normalization(ell::Int, mu::Int)
    if ell > 0
        if mu == 0
            return sqrt((ell + 1) / ell)
        else
            # 1 / xi(mu, ell)
            return 1.0 / ball_xi(mu, ell)
        end
    elseif ell == 0 && mu == 1
        return 1.0
    end
    return nothing
end

"""
    ball_R_helper(tau, mu, nu, Q_old, rank)

Helper for Q-matrix recurrence. Computes spin-coupling terms.
"""
function ball_R_helper(tau::Int, mu::Int, nu::Int, Q_old::AbstractMatrix, rank::Int)
    R = 0.0
    if rank == 0 || mu == 0
        return R
    end
    for i in 0:(rank - 1)
        nui = ball_get_element(nu, i)
        if (nui == +1 && mu == -1) || (nui == -1 && mu == +1)
            # Q_old uses 1-based indexing
            R -= Q_old[ball_replace_index(nu, 0, i) + 1, tau + 1]
        elseif nui == 0 && mu == -1
            R += Q_old[ball_replace_index(nu, -1, i) + 1, tau + 1]
        elseif nui == 0 && mu == +1
            R += Q_old[ball_replace_index(nu, +1, i) + 1, tau + 1]
        end
    end
    return R
end

"""
    ball_recurseQ(Q_old, ell, rank)

Build the regularity (Q) matrix for tensors of given rank at spherical harmonic
degree ell, by recursing from the (rank-1) Q-matrix.
"""
function ball_recurseQ(Q_old::AbstractMatrix, ell::Int, rank::Int)
    n = 3^rank
    Q = zeros(Float64, n, n)
    for i in 0:(n - 1)
        for j in 0:(n - 1)
            # Decode i into leading spin mu and trailing multi-index nu
            mu = fld(i, 3^(rank - 1)) - 1
            nu = i % (3^(rank - 1))
            # Decode j into leading spin sigma and trailing multi-index tau
            sigma = fld(j, 3^(rank - 1)) - 1
            tau = j % (3^(rank - 1))

            nubar = ball_bar(nu, rank - 1)
            deg = ell + ball_bar(tau, rank - 1)
            k_ang = ball_k(mu, ell, nubar)
            Qnorm = ball_Q_normalization(deg, sigma)
            S = ball_R_helper(tau, mu, nu, Q_old, rank - 1)

            if Qnorm !== nothing && !isinf(k_ang)
                if sigma == -1
                    Q[i + 1, j + 1] = Qnorm * (Q_old[nu + 1, tau + 1] * (deg * ball_delta(mu, 0) + k_ang) - S) / (2 * deg + 1)
                elseif sigma == 0
                    Q[i + 1, j + 1] = Qnorm * (-mu * Q_old[nu + 1, tau + 1] * k_ang + mu * S) / (deg + 1)
                elseif sigma == 1
                    Q[i + 1, j + 1] = Qnorm * (Q_old[nu + 1, tau + 1] * ((deg + 1) * ball_delta(mu, 0) - k_ang) + S) / (2 * deg + 1)
                end
            end
        end
    end
    return Q
end

"""
    ball_Q(ell, order)

Convenience wrapper: builds the full Q-matrix for degree ell and tensor order
by starting from Q = [[1]] and applying ball_recurseQ iteratively.
"""
function ball_Q(ell::Int, order::Int)
    Q = ones(Float64, 1, 1)
    for rank in 1:order
        Q = ball_recurseQ(Q, ell, rank)
    end
    return Q
end

# ============================================================================
# BallWrapper
# ============================================================================

"""
    BallWrapper

Precomputes and caches quadrature grids, transform matrices, and operators
for spectral transforms on the 3D ball domain.

Fields
------
- N_max      : maximum polynomial degree
- L_max      : maximum spherical harmonic degree
- R_max      : maximum tensor rank
- a          : Jacobi parameter (default -0.5)
- N_r        : number of radial grid points
- ell_min    : minimum ell
- ell_max    : maximum ell
- m_min      : minimum m
- m_max      : maximum m
- S          : SphereWrapper for angular transforms
- theta      : colatitude grid
- cos_theta  : cos(theta) grid
- sin_theta  : sin(theta) grid
- radius     : radial grid r = sqrt((z+1)/2)
- dV         : volume integration matrix
- pushW      : Dict of forward (grid->coeff) radial transform matrices
- pullW      : Dict of backward (coeff->grid) radial transform matrices
- Q          : Dict of regularity (Q) matrices
- LU_grad_initialized : list tracking grad LU initialization
- LU_grad    : list of cached grad LU factorizations
- LU_curl_initialized : list tracking curl LU initialization
- LU_curl    : list of cached curl LU factorizations
- _op_cache  : Dict for caching operator matrices
- _unitary_cache : Dict for caching unitary matrices
- _spins_cache : Dict for caching spins arrays
- _N_min_cache : Dict for caching N_min values
- store_lu   : whether to store LU factorizations
"""
mutable struct BallWrapper
    N_max::Int
    L_max::Int
    R_max::Int
    a::Float64
    N_r::Int
    ell_min::Int
    ell_max::Int
    m_min::Int
    m_max::Int
    S::SphereWrapper
    theta::Vector{Float64}
    cos_theta::Vector{Float64}
    sin_theta::Vector{Float64}
    radius::Vector{Float64}
    dV::Vector{Float64}
    pushW::Dict{Int, Matrix{Float64}}
    pullW::Dict{Int, Matrix{Float64}}
    Q::Dict{Tuple{Int, Int}, Matrix{Float64}}
    LU_grad_initialized::Vector{Vector{Bool}}
    LU_grad::Vector{Vector{Any}}
    LU_curl_initialized::Vector{Vector{Bool}}
    LU_curl::Vector{Vector{Any}}
    _op_cache::Dict{Tuple{String, Int, Int, Int, DataType}, Any}
    _unitary_cache::Dict{Tuple{Int, Bool}, Any}
    _spins_cache::Dict{Int, Vector{Float64}}
    _N_min_cache::Dict{Int, Int}
    store_lu::Bool

    function BallWrapper(N_max::Int, L_max::Int;
                         R_max::Int=0, a::Real=0, N_r::Union{Nothing, Int}=nothing,
                         N_theta::Union{Nothing, Int}=nothing,
                         ell_min::Union{Nothing, Int}=nothing,
                         ell_max::Union{Nothing, Int}=nothing,
                         m_min::Union{Nothing, Int}=nothing,
                         m_max::Union{Nothing, Int}=nothing,
                         store_lu::Bool=false)
        if N_r === nothing
            N_r = N_max + 1
        end
        a_float = Float64(a)

        if ell_min === nothing; ell_min = 0; end
        if ell_max === nothing; ell_max = L_max; end
        if m_min === nothing; m_min = 0; end
        if m_max === nothing; m_max = L_max; end

        # Spherical Harmonic Transforms
        S = SphereWrapper(L_max; S_max=R_max, N_theta=N_theta, m_min=m_min, m_max=m_max)

        theta = S.grid
        cos_theta = S.cos_grid
        sin_theta = S.sin_grid

        # Grid and weights for the radial transforms
        z_projection, weights_projection = ball_quadrature(N_r - 1; niter=3, a=a_float, report_error=false)

        # Grid and weights for radial integral using volume measure
        z0, weights0 = ball_quadrature(N_r - 1; a=0.0)

        Q0 = ball_polynomial(N_r - 1, 0, 0, z0; a=a_float)
        Q_projection = ball_polynomial(N_r - 1, 0, 0, z_projection; a=a_float)

        # dV computation: volume integration weights
        # Python: dV = ((Q0.dot(weights0)).T).dot(weights_projection*Q_projection)
        # Q0 is (N_poly, N_z0), weights0 is (N_z0,) -> Q0*weights0 is (N_poly,)
        # weights_projection is (N_zp,), Q_projection is (N_poly, N_zp)
        # weights_projection*Q_projection broadcasts to (N_poly, N_zp)
        # (N_poly,).dot((N_poly, N_zp)) = (N_zp,)  -> 1D vector
        dV_coeffs = Float64.(Q0 * weights0)  # (N_poly,)
        wQ = weights_projection' .* Q_projection  # (N_poly, N_zp)
        dV_result = Float64.(vec(dV_coeffs' * wQ))  # (N_zp,)

        pushW_dict = Dict{Int, Matrix{Float64}}()
        pullW_dict = Dict{Int, Matrix{Float64}}()

        for ell in max(ell_min - R_max, 0):(ell_max + R_max)
            W = ball_polynomial(N_max + R_max - ball_N_min(ell), 0, ell, z_projection; a=a_float)
            pushW_dict[ell] = Float64.(weights_projection' .* W)
            pullW_dict[ell] = Float64.(W')
        end

        Q_dict = Dict{Tuple{Int, Int}, Matrix{Float64}}()
        for ell in ell_min:ell_max
            Q_dict[(ell, 0)] = ones(Float64, 1, 1)
            for deg in 1:R_max
                Q_dict[(ell, deg)] = ball_recurseQ(Q_dict[(ell, deg - 1)], ell, deg)
            end
        end

        # Downcast radius to double precision
        radius = Float64.(sqrt.((z_projection .+ 1) ./ 2))

        # Initialize LU factorization storage
        n_ell = ell_max - ell_min + 1
        LU_grad_initialized = [fill(false, 2) for _ in 1:n_ell]
        LU_grad = [Vector{Any}(nothing, 2) for _ in 1:n_ell]
        LU_curl_initialized = [fill(false, 2) for _ in 1:n_ell]
        LU_curl = [Vector{Any}(nothing, 2) for _ in 1:n_ell]

        new(N_max, L_max, R_max, a_float, N_r,
            ell_min, ell_max, m_min, m_max,
            S, theta, cos_theta, sin_theta, radius,
            dV_result,
            pushW_dict, pullW_dict, Q_dict,
            LU_grad_initialized, LU_grad,
            LU_curl_initialized, LU_curl,
            Dict{Tuple{String, Int, Int, Int, DataType}, Any}(),
            Dict{Tuple{Int, Bool}, Any}(),
            Dict{Int, Vector{Float64}}(),
            Dict{Int, Int}(),
            store_lu)
    end
end

# ============================================================================
# grid - grid access method
# ============================================================================

"""
    ball_grid(B::BallWrapper, axis::Int; dimensions::Int=2)

Get the grid for the specified axis, reshaped for the given number of dimensions.
Note: axis is 0-based (matching the Python convention) and gets converted to
1-based for reshape_vector.

In Python:
- axis=0, dim=2 -> theta
- axis=1, dim=2 -> radius
- axis=1, dim=3 -> theta
- axis=2, dim=3 -> radius
"""
function ball_grid(B::BallWrapper, axis::Int; dimensions::Int=2)
    if axis == 0 && dimensions == 2
        grid = B.theta
    elseif axis == 1 && dimensions == 2
        grid = B.radius
    elseif axis == 1 && dimensions == 3
        grid = B.theta
    elseif axis == 2 && dimensions == 3
        grid = B.radius
    else
        error("Invalid axis=$axis for dimensions=$dimensions")
    end
    # Convert 0-based axis to 1-based for reshape_vector
    return reshape_vector(grid, dimensions, axis + 1)
end

# ============================================================================
# weight - weight access method
# ============================================================================

"""
    ball_weight(B::BallWrapper, axis::Int; dimensions::Int=2)

Get the quadrature weights for the specified axis, reshaped for the given number
of dimensions. axis is 0-based.

In Python:
- axis=0, dim=2 -> S.weights
- axis=1, dim=2 -> dV
- axis=1, dim=3 -> S.weights
- axis=2, dim=3 -> dV
"""
function ball_weight(B::BallWrapper, axis::Int; dimensions::Int=2)
    if axis == 0 && dimensions == 2
        weight = B.S.weights
    elseif axis == 1 && dimensions == 2
        weight = B.dV
    elseif axis == 1 && dimensions == 3
        weight = B.S.weights
    elseif axis == 2 && dimensions == 3
        weight = B.dV
    else
        error("Invalid axis=$axis for dimensions=$dimensions")
    end
    return reshape_vector(weight, dimensions, axis + 1)
end

# ============================================================================
# ball_op - cached operator matrix access
# ============================================================================

"""
    ball_op(B::BallWrapper, op_name::String, N::Int, k::Int, ell::Int;
            dtype::Type=Float64, a::Union{Nothing, Real}=nothing)

Get the operator matrix for the given operator name and parameters.
Results are cached for repeated calls.
"""
function ball_op(B::BallWrapper, op_name::String, N::Int, k::Int, ell::Int;
                 dtype::Type=Float64, a::Union{Nothing, Real}=nothing)
    if a === nothing
        a = B.a
    end
    key = (op_name, N, k, ell, dtype)
    if haskey(B._op_cache, key)
        return B._op_cache[key]
    end
    result = ball_operator(op_name, N, k, ell; a=a, dtype=dtype)
    B._op_cache[key] = result
    return result
end

# ============================================================================
# ball_xi_method - xi method on BallWrapper
# ============================================================================

"""
    ball_xi_method(B::BallWrapper, mu, ell)

Returns xi for ell > 0 or ell == 0 and mu == +1.
Otherwise returns 0.
"""
function ball_xi_method(B::BallWrapper, mu::Int, ell::Int)
    if (ell > 0) || (ell == 0 && mu == 1)
        return ball_xi(mu, ell)
    end
    return 0.0
end

"""
    ball_xi_method(B::BallWrapper, mu::Vector{Int}, ell::Int)

When mu is a vector like [-1,+1], returns a tuple of xi values.
"""
function ball_xi_method(B::BallWrapper, mu::Vector{Int}, ell::Int)
    return Tuple(ball_xi_method(B, m, ell) for m in mu)
end

# ============================================================================
# Cached accessors: unitary3D, spins, N_min
# ============================================================================

"""
    ball_wrapper_unitary3D(B::BallWrapper; rank::Int=1, adjoint::Bool=false)

Cached accessor for ball_unitary3D.
"""
function ball_wrapper_unitary3D(B::BallWrapper; rank::Int=1, adjoint::Bool=false)
    key = (rank, adjoint)
    if haskey(B._unitary_cache, key)
        return B._unitary_cache[key]
    end
    result = ball_unitary3D(; rank=rank, adjoint=adjoint)
    B._unitary_cache[key] = result
    return result
end

"""
    ball_wrapper_spins(B::BallWrapper, rank::Int)

Cached accessor for ball_spins.
"""
function ball_wrapper_spins(B::BallWrapper, rank::Int)
    if haskey(B._spins_cache, rank)
        return B._spins_cache[rank]
    end
    result = ball_spins(rank)
    B._spins_cache[rank] = result
    return result
end

"""
    ball_wrapper_N_min(B::BallWrapper, ell::Int)

Cached accessor for ball_N_min.
"""
function ball_wrapper_N_min(B::BallWrapper, ell::Int)
    if haskey(B._N_min_cache, ell)
        return B._N_min_cache[ell]
    end
    result = ball_N_min(ell)
    B._N_min_cache[ell] = result
    return result
end

# ============================================================================
# Forward/backward angular transforms
# ============================================================================

"""
    forward_angle(B::BallWrapper, m::Int, rank::Int, data_in, data_out)

Forward angular transform (grid -> coefficients) for BallWrapper.
Applies unitary rotation for rank > 0 before transforming each spin component.
"""
function forward_angle(B::BallWrapper, m::Int, rank::Int, data_in, data_out)
    if rank == 0
        l_min = Int(L_min(B.S, m, 0))
        # 1-based: l_min+1 is the first index, data_out[1] is component 0
        data_out[1, (l_min + 1):end] .= forward_spin(B.S, m, 0, data_in[1, :])
        return
    end

    spins_arr = ball_wrapper_spins(B, rank)
    unitary_mat = ball_wrapper_unitary3D(B; rank=rank, adjoint=true)

    # Apply unitary transformation: einsum "ij,j...->i..."
    n_comp = size(data_in, 1)
    if ndims(data_in) == 1
        data_rot = unitary_mat * data_in
    else
        rest_dims = size(data_in)[2:end]
        data_2d = reshape(data_in, n_comp, :)
        rot_2d = unitary_mat * data_2d
        data_rot = reshape(rot_2d, size(unitary_mat, 1), rest_dims...)
    end

    for i in 1:(3^rank)
        s = Int(spins_arr[i])
        l_min = Int(L_min(B.S, m, s))
        # In Python: data_out[i, int(S.L_min(m, spins[i])):] = S.forward_spin(m, spins[i], data_in[i])
        # Python i is 0-based, Julia i is 1-based
        data_out[i, (l_min + 1):end] .= forward_spin(B.S, m, s, data_rot[i, :])
    end
end

"""
    backward_angle(B::BallWrapper, m::Int, rank::Int, data_in, data_out)

Backward angular transform (coefficients -> grid) for BallWrapper.
"""
function backward_angle(B::BallWrapper, m::Int, rank::Int, data_in, data_out)
    if rank == 0
        l_min = Int(L_min(B.S, m, 0))
        data_out[1, :] .= backward_spin(B.S, m, 0, data_in[1, (l_min + 1):end])
        return
    end

    spins_arr = ball_wrapper_spins(B, rank)

    for i in 1:(3^rank)
        s = Int(spins_arr[i])
        l_min = Int(L_min(B.S, m, s))
        data_out[i, :] .= backward_spin(B.S, m, s, data_in[i, (l_min + 1):end])
    end
end

# ============================================================================
# NCC matrix
# ============================================================================

"""
    ball_ncc_matrix(B::BallWrapper, N::Int, k::Int, ell::Int, deg_in::Int, deg_out::Int,
                    data; cutoff::Float64=1e-6, name::String="")

Build NCC (non-constant coefficient) matrix for the ball domain.
"""
function ball_ncc_matrix(B::BallWrapper, N::Int, k::Int, ell::Int, deg_in::Int, deg_out::Int,
                         data; cutoff::Float64=1e-6, name::String="")
    q_in = B.a
    m_in = deg_in + 0.5
    q_out = k + B.a
    m_out = ell + deg_out + 0.5
    n_terms, max_term, matrix = dsc_ncc_matrix(N, q_in, m_in, q_out, m_out, data; cutoff=cutoff)
    matrix ./= 0.5^(3 / 4)
    @debug "Expanded NCC $name to mode $max_term with $n_terms terms."
    return matrix
end

# ============================================================================
# Forward/backward radial component transforms
# ============================================================================

"""
    forward_component(B::BallWrapper, ell::Int, deg::Int, data)

Forward radial component transform: grid -> coefficients for a single component.
"""
function forward_component(B::BallWrapper, ell::Int, deg::Int, data)
    N = B.N_max - ball_wrapper_N_min(B, ell - B.R_max) + 1
    if ell + deg >= 0
        return B.pushW[ell + deg][1:N, :] * data
    else
        shape = collect(size(data))
        shape[1] = N
        return zeros(eltype(data), Tuple(shape))
    end
end

"""
    backward_component(B::BallWrapper, ell::Int, deg::Int, data)

Backward radial component transform: coefficients -> grid for a single component.
"""
function backward_component(B::BallWrapper, ell::Int, deg::Int, data)
    N = B.N_max - ball_wrapper_N_min(B, ell - B.R_max) + 1
    if ell + deg >= 0
        return B.pullW[ell + deg][:, 1:N] * data
    else
        shape = collect(size(data))
        shape[1] = B.N_r
        return zeros(eltype(data), Tuple(shape))
    end
end

# ============================================================================
# Full radial transforms
# ============================================================================

"""
    radial_forward(B::BallWrapper, ell::Int, rank::Int, data_in, data_out)

Forward radial transform: grid -> coefficients, handling multi-component fields.
For rank 0, transforms a single component.
For rank > 0, applies Q-matrix rotation and transforms each spin component.
"""
function radial_forward(B::BallWrapper, ell::Int, rank::Int, data_in, data_out)
    if rank == 0
        data_out .= forward_component(B, ell, 0, data_in[1, :])
        return
    end

    degs = ball_wrapper_spins(B, rank)
    N = B.N_max - ball_wrapper_N_min(B, ell - B.R_max) + 1
    Q_mat = B.Q[(ell, rank)]

    # Apply Q^T to data_in: einsum "ij,j...->i..."
    # data_in is indexed by component then radial points
    n_comp = size(data_in, 1)
    if ndims(data_in) == 1
        data_rot = Q_mat' * data_in
    else
        rest_dims = size(data_in)[2:end]
        data_2d = reshape(data_in, n_comp, :)
        rot_2d = Q_mat' * data_2d
        data_rot = reshape(rot_2d, size(Q_mat, 2), rest_dims...)
    end

    for i in 1:(3^rank)
        deg = Int(degs[i])
        fc = forward_component(B, ell, deg, data_rot[i, :])
        # data_out is a 1D array packed as [comp1; comp2; comp3; ...]
        # Python: data_out[i*N:(i+1)*N] (0-based) -> Julia: data_out[(i-1)*N+1:i*N]
        data_out[((i - 1) * N + 1):(i * N)] .= fc
    end
end

"""
    radial_backward(B::BallWrapper, ell::Int, rank::Int, data_in, data_out)

Backward radial transform: coefficients -> grid, handling multi-component fields.
"""
function radial_backward(B::BallWrapper, ell::Int, rank::Int, data_in, data_out)
    if rank == 0
        data_out[1, :] .= backward_component(B, ell, 0, data_in)
        return
    end

    degs = ball_wrapper_spins(B, rank)
    N = B.N_max - ball_wrapper_N_min(B, ell - B.R_max) + 1

    for i in 1:(3^rank)
        deg = Int(degs[i])
        # Python: data_in[i*N:(i+1)*N] (0-based) -> Julia: data_in[(i-1)*N+1:i*N]
        data_out[i, :] .= backward_component(B, ell, deg, data_in[((i - 1) * N + 1):(i * N)])
    end
end

# ============================================================================
# Utility functions: unpack, pack, rank
# ============================================================================

"""
    ball_unpack(B::BallWrapper, ell::Int, rank::Int, data_in)

Unpack a 1D coefficient array into a list of per-component arrays.
"""
function ball_unpack(B::BallWrapper, ell::Int, rank::Int, data_in)
    N = B.N_max + 1 - ball_wrapper_N_min(B, ell - B.R_max)
    data_out = Vector{Any}(undef, 3^rank)
    for i in 1:(3^rank)
        data_out[i] = data_in[((i - 1) * N + 1):(i * N)]
    end
    return data_out
end

"""
    ball_pack(B::BallWrapper, ell::Int, rank::Int, data_in, data_out)

Pack a list of per-component arrays into a single 1D coefficient array.
"""
function ball_pack(B::BallWrapper, ell::Int, rank::Int, data_in, data_out)
    N = B.N_max + 1 - ball_wrapper_N_min(B, ell - B.R_max)
    for i in 1:(3^rank)
        data_out[((i - 1) * N + 1):(i * N)] .= data_in[i]
    end
end

"""
    ball_rank(B::BallWrapper, length::Int)

Determine tensor rank from the number of components.
"""
function ball_rank(B::BallWrapper, len::Int)
    if len == 1
        return 0
    else
        return 1 + ball_rank(B, fld(len, 3))
    end
end

# ============================================================================
# Gradient
# ============================================================================

"""
    ball_grad(B::BallWrapper, ell::Int, rank::Int, data_in)

Compute the gradient in coefficient space.
Transforms data from rank to rank+1.
Returns a new array with 3x the size of data_in.
"""
function ball_grad(B::BallWrapper, ell::Int, rank::Int, data_in)
    shape = collect(size(data_in))
    shape[1] *= 3
    data_dtype = eltype(data_in)
    data_out = zeros(data_dtype, Tuple(shape))

    if B.store_lu
        i_LU = ell - B.ell_min + 1  # 1-based index
        if !B.LU_grad_initialized[i_LU][rank + 1]
            @debug "LU_grad not initialized l=$ell, rank=$rank"
            B.LU_grad[i_LU][rank + 1] = Vector{Any}(nothing, 4 * (3^rank))
        end
    end

    for i in 0:(3^rank - 1)
        tau_bar = ball_bar(i, rank)
        N = B.N_max - ball_wrapper_N_min(B, ell - B.R_max)

        if ell + tau_bar >= 1
            Cm = ball_op(B, "E", N, 0, ell + tau_bar - 1, dtype=data_dtype)
            Dm = ball_op(B, "D-", N, 0, ell + tau_bar, dtype=data_dtype)
            xim = ball_xi_method(B, -1, ell + tau_bar)
            index = i  # 0-based
            if B.store_lu
                if !B.LU_grad_initialized[i_LU][rank + 1]
                    B.LU_grad[i_LU][rank + 1][index + 1] = lu(Matrix(Cm))
                end
                # Python: data_out[i*(N+1):(i+1)*(N+1)] = LU.solve(xim*Dm.dot(data_in[...]))
                src = data_in[(i * (N + 1) + 1):((i + 1) * (N + 1))]
                data_out[(i * (N + 1) + 1):((i + 1) * (N + 1))] .= B.LU_grad[i_LU][rank + 1][index + 1] \ (xim * Dm * src)
            else
                src = data_in[(i * (N + 1) + 1):((i + 1) * (N + 1))]
                data_out[(i * (N + 1) + 1):((i + 1) * (N + 1))] .= Matrix(Cm) \ (xim * Dm * src)
            end
        end

        if ell + tau_bar >= 0
            Cp = ball_op(B, "E", N, 0, ell + tau_bar + 1, dtype=data_dtype)
            Dp = ball_op(B, "D+", N, 0, ell + tau_bar, dtype=data_dtype)
            xip = ball_xi_method(B, +1, ell + tau_bar)
            index = i + 2 * (3^rank)  # 0-based
            if B.store_lu
                if !B.LU_grad_initialized[i_LU][rank + 1]
                    B.LU_grad[i_LU][rank + 1][index + 1] = lu(Matrix(Cp))
                end
                src = data_in[(i * (N + 1) + 1):((i + 1) * (N + 1))]
                data_out[(index * (N + 1) + 1):((index + 1) * (N + 1))] .= B.LU_grad[i_LU][rank + 1][index + 1] \ (xip * Dp * src)
            else
                src = data_in[(i * (N + 1) + 1):((i + 1) * (N + 1))]
                data_out[(index * (N + 1) + 1):((index + 1) * (N + 1))] .= Matrix(Cp) \ (xip * Dp * src)
            end
        end
    end

    if B.store_lu && !B.LU_grad_initialized[i_LU][rank + 1]
        B.LU_grad_initialized[i_LU][rank + 1] = true
    end

    return data_out
end

# ============================================================================
# Curl
# ============================================================================

"""
    ball_curl(B::BallWrapper, ell::Int, rank::Int, data_in, data_out)

Compute the curl in coefficient space.
Operates on rank-1 fields (vectors).
"""
function ball_curl(B::BallWrapper, ell::Int, rank::Int, data_in, data_out)
    data_dtype = eltype(data_in)

    if B.store_lu
        i_LU = ell - B.ell_min + 1  # 1-based
        if !B.LU_curl_initialized[i_LU][rank + 1]
            @debug "LU_curl not initialized l=$ell, rank=$rank"
            B.LU_curl[i_LU][rank + 1] = Vector{Any}(nothing, 3)
        end
    end

    N = B.N_max - ball_wrapper_N_min(B, ell - B.R_max)
    xim = ball_xi_method(B, -1, ell)
    xip = ball_xi_method(B, +1, ell)

    # Component 0 (data_out[1:N+1])
    if ell >= 1
        Cm = ball_op(B, "E", N, 0, ell - 1, dtype=data_dtype)
        Dm = ball_op(B, "D-", N, 0, ell, dtype=data_dtype)
        src = data_in[(N + 2):(2 * (N + 1))]  # Python: data_in[(N+1):2*(N+1)]
        rhs = -1im * xip * Dm * src
        if B.store_lu
            index = 1
            if !B.LU_curl_initialized[i_LU][rank + 1]
                B.LU_curl[i_LU][rank + 1][index] = lu(Matrix(Cm))
            end
            data_out[1:(N + 1)] .= B.LU_curl[i_LU][rank + 1][index] \ rhs
        else
            data_out[1:(N + 1)] .= Matrix(Cm) \ rhs
        end
    else
        data_out[1:(N + 1)] .= 0.0
    end

    # Component 1 (data_out[N+2:2*(N+1)])
    C0 = ball_op(B, "E", N, 0, ell, dtype=data_dtype)
    Dm_1 = ball_op(B, "D-", N, 0, ell + 1, dtype=data_dtype)

    if B.store_lu
        index = 2
        if !B.LU_curl_initialized[i_LU][rank + 1]
            B.LU_curl[i_LU][rank + 1][index] = lu(Matrix(C0))
        end
    end

    if ell >= 1
        Dp_1 = ball_op(B, "D+", N, 0, ell - 1, dtype=data_dtype)
        src_top = data_in[1:(N + 1)]
        src_bot = data_in[(2 * (N + 1) + 1):end]
        rhs = 1im * xim * Dm_1 * src_bot - 1im * xip * Dp_1 * src_top
        if B.store_lu
            data_out[(N + 2):(2 * (N + 1))] .= B.LU_curl[i_LU][rank + 1][index] \ rhs
        else
            data_out[(N + 2):(2 * (N + 1))] .= Matrix(C0) \ rhs
        end
    else
        src_bot = data_in[(2 * (N + 1) + 1):end]
        rhs = 1im * xim * Dm_1 * src_bot
        if B.store_lu
            data_out[(N + 2):(2 * (N + 1))] .= B.LU_curl[i_LU][rank + 1][index] \ rhs
        else
            data_out[(N + 2):(2 * (N + 1))] .= Matrix(C0) \ rhs
        end
    end

    # Component 2 (data_out[2*(N+1)+1:end])
    Cp = ball_op(B, "E", N, 0, ell + 1, dtype=data_dtype)
    Dp_2 = ball_op(B, "D+", N, 0, ell, dtype=data_dtype)
    src_mid = data_in[(N + 2):(2 * (N + 1))]
    rhs = 1im * xim * Dp_2 * src_mid

    if B.store_lu
        index = 3
        if !B.LU_curl_initialized[i_LU][rank + 1]
            B.LU_curl[i_LU][rank + 1][index] = lu(Matrix(Cp))
        end
        data_out[(2 * (N + 1) + 1):end] .= B.LU_curl[i_LU][rank + 1][index] \ rhs
    else
        data_out[(2 * (N + 1) + 1):end] .= Matrix(Cp) \ rhs
    end

    if B.store_lu && !B.LU_curl_initialized[i_LU][rank + 1]
        B.LU_curl_initialized[i_LU][rank + 1] = true
    end
end

# ============================================================================
# Divergence
# ============================================================================

"""
    ball_div(B::BallWrapper, data_in)

Compute the divergence in coefficient space.
data_in is a vector of per-ell coefficient arrays.
Returns a vector of per-ell result arrays.
"""
function ball_div(B::BallWrapper, data_in)
    rank = ball_rank(B, length(data_in))
    data_dtype = eltype(data_in[1])

    data_out = Vector{Any}(undef, 3^(rank - 1))

    for i in 0:(3^(rank - 1) - 1)
        tau_bar = ball_bar(i, rank - 1)
        m_tau_bar = -1 + tau_bar
        p_tau_bar = 1 + tau_bar
        # Initialize arrays
        data_out[i + 1] = []
        for ell in 0:B.L_max
            N = B.N_max - ball_wrapper_N_min(B, ell - B.R_max)

            if ell + tau_bar == 0
                C = ball_op(B, "E", N, 0, ell + tau_bar, dtype=data_dtype)
                Dm = ball_op(B, "D-", N, 0, ell + p_tau_bar, dtype=data_dtype)
                xip = ball_xi_method(B, +1, ell + tau_bar)

                # Python: data_in[i+2*(3**(rank-1))][ell], with 0-based i and ell as list index
                result = Matrix(C) \ (xip * Dm * data_in[i + 2 * 3^(rank - 1) + 1][ell + 1])
                push!(data_out[i + 1], result)

            elseif ell + tau_bar > 0
                C = ball_op(B, "E", N, 0, ell + tau_bar, dtype=data_dtype)
                Dm = ball_op(B, "D-", N, 0, ell + p_tau_bar, dtype=data_dtype)
                Dp = ball_op(B, "D+", N, 0, ell + m_tau_bar, dtype=data_dtype)
                xim, xip = ball_xi_method(B, [-1, +1], ell + tau_bar)

                # Python: data_in[i+2*(3**(rank-1))][ell] and data_in[i][ell]
                result = Matrix(C) \ (xip * Dm * data_in[i + 2 * 3^(rank - 1) + 1][ell + 1] +
                                      xim * Dp * data_in[i + 1][ell + 1])
                push!(data_out[i + 1], result)

            else
                push!(data_out[i + 1], 0 * data_in[i + 1][ell + 1])
            end
        end
    end

    return data_out
end

# ============================================================================
# Divergence of Gradient (Laplacian)
# ============================================================================

"""
    ball_div_grad(B::BallWrapper, data_in; ell_start::Int=0, ell_end::Union{Nothing, Int}=nothing)

Compute the Laplacian (div(grad)) in coefficient space.
"""
function ball_div_grad(B::BallWrapper, data_in; ell_start::Int=0, ell_end::Union{Nothing, Int}=nothing)
    if ell_end === nothing
        ell_end = B.L_max
    end

    rank = ball_rank(B, length(data_in))

    data_out = Vector{Any}(undef, 3^rank)

    for i in 0:(3^rank - 1)
        tau_bar = ball_bar(i, rank)
        # Initialize arrays
        data_out[i + 1] = []
        for ell in ell_start:ell_end
            ell_local = ell - ell_start + 1  # 1-based

            N = B.N_max - ball_wrapper_N_min(B, ell - B.R_max)

            if ell + tau_bar >= 0
                CC = ball_op(B, "E", N, 1, ell + tau_bar) * ball_op(B, "E", N, 0, ell + tau_bar)
                DD = ball_op(B, "D-", N, 1, ell + tau_bar + 1) * ball_op(B, "D+", N, 0, ell + tau_bar)
                Lap = Matrix(CC) \ (DD * data_in[i + 1][ell_local])
                push!(data_out[i + 1], Lap)
            else
                push!(data_out[i + 1], 0 * data_in[i + 1][ell_local])
            end
        end
    end

    return data_out
end

# ============================================================================
# Grid-space cross and dot products
# ============================================================================

"""
    ball_cross_grid(B::BallWrapper, a, b)

Compute cross product of two vector fields in grid space.
a and b are 3-component arrays.
"""
function ball_cross_grid(B::BallWrapper, a, b)
    return [a[2] .* b[3] .- a[3] .* b[2],
            a[3] .* b[1] .- a[1] .* b[3],
            a[1] .* b[2] .- a[2] .* b[1]]
end

"""
    ball_dot_grid(B::BallWrapper, a, b)

Compute dot product of two vector fields in grid space.
"""
function ball_dot_grid(B::BallWrapper, a, b)
    return a[1] .* b[1] .+ a[2] .* b[2] .+ a[3] .* b[3]
end

# ============================================================================
# TensorField abstract type
# ============================================================================

"""
    AbstractBallTensorField

Abstract type for tensor fields on the ball domain.
"""
abstract type AbstractBallTensorField end

# ============================================================================
# BallTensorField (base type matching Python TensorField)
# ============================================================================

"""
    BallTensorField

Base type for tensor fields on the ball, storing rank, BallWrapper reference,
domain, and ell range.
"""
mutable struct BallTensorField <: AbstractBallTensorField
    rank::Int
    B::BallWrapper
    domain::Any
    ell_min::Int
    ell_max::Int
    _layout::Union{Char, Int}
    data::Any

    function BallTensorField(rank::Int, B::BallWrapper, domain)
        new(rank, B, domain, B.ell_min, B.ell_max, 'g', nothing)
    end
end

"""
    require_layout(tf::BallTensorField, layout)

Ensure the tensor field is in the requested layout.
"""
function require_layout(tf::BallTensorField, layout)
    if layout == 'g' && tf._layout == 'c'
        require_grid_space(tf)
    elseif layout == 'c' && tf._layout == 'g'
        require_coeff_space(tf)
    end
end

function set_layout!(tf::BallTensorField, layout)
    tf._layout = layout
end

# ============================================================================
# BallTensorField2D
# ============================================================================

"""
    BallTensorField2D

2D tensor field on the ball (single m value), with grid, coefficient,
and intermediate data storage.
"""
mutable struct BallTensorField2D <: AbstractBallTensorField
    rank::Int
    B::BallWrapper
    domain::Any
    m::Int
    ell_min::Int
    ell_max::Int
    ell_r_layout::Any
    r_ell_layout::Any
    grid_data::Array{ComplexF64}
    ellr_data::Array{ComplexF64}
    rell_data::Array{ComplexF64}
    fields::Any
    coeff_data::Vector{Vector{ComplexF64}}
    _layout::Union{Char, Int}
    data::Any

    function BallTensorField2D(rank::Int, m::Int, B::BallWrapper, domain)
        ell_min = B.ell_min
        ell_max = B.ell_max

        mesh = domain.distributor.mesh
        if length(mesh) == 0  # serial
            ell_r_layout = domain.distributor.layouts[2]  # 1-based
            r_ell_layout = domain.distributor.layouts[2]
        else
            ell_r_layout = domain.distributor.layouts[3]
            r_ell_layout = domain.distributor.layouts[2]
        end

        local_grid_shape = ell_r_layout.local_shape(scales=1)
        local_grid_shape = (Int(domain.dealias[1] * local_grid_shape[1]),
                            Int(domain.dealias[2] * local_grid_shape[2]))
        local_ellr_shape = ell_r_layout.local_shape(scales=domain.dealias)
        local_rell_shape = r_ell_layout.local_shape(scales=domain.dealias)

        grid_data = zeros(ComplexF64, 3^rank, local_grid_shape...)
        ellr_data = zeros(ComplexF64, 3^rank, local_ellr_shape...)
        rell_data = zeros(ComplexF64, 3^rank, local_rell_shape...)

        fields = domain.new_fields(3^rank)
        for field in fields
            field.preset_scales(domain.dealias)
        end

        coeff_data = Vector{Vector{ComplexF64}}()
        for ell in ell_min:ell_max
            N = B.N_max - ball_wrapper_N_min(B, ell - B.R_max) + 1
            push!(coeff_data, zeros(ComplexF64, N * (3^rank)))
        end

        new(rank, B, domain, m, ell_min, ell_max,
            ell_r_layout, r_ell_layout,
            grid_data, ellr_data, rell_data,
            fields, coeff_data,
            'g', grid_data)
    end
end

"""
    require_coeff_space(tf::BallTensorField2D)

Transform from grid space to coefficient space for 2D tensor field.
"""
function require_coeff_space(tf::BallTensorField2D)
    rank = tf.rank
    B = tf.B

    forward_angle(B, tf.m, rank, tf.grid_data, tf.ellr_data)

    for (i, field) in enumerate(tf.fields)
        field.layout = tf.ell_r_layout
        field.data = tf.ellr_data[i, :, :]
        require_layout(field, tf.r_ell_layout)
    end

    for ell in tf.ell_min:tf.ell_max
        ell_local = ell - tf.ell_min + 1  # 1-based
        radial_input = [tf.fields[i].data[ell_local, :] for i in 1:(3^rank)]
        radial_forward(B, ell, rank, hcat(radial_input...)', tf.coeff_data[ell_local])
    end

    tf.data = tf.coeff_data
    tf._layout = 'c'
end

"""
    require_grid_space(tf::BallTensorField2D)

Transform from coefficient space to grid space for 2D tensor field.
"""
function require_grid_space(tf::BallTensorField2D)
    rank = tf.rank
    B = tf.B

    for ell in tf.ell_min:tf.ell_max
        ell_local = ell - tf.ell_min + 1  # 1-based
        radial_backward(B, ell, rank, tf.coeff_data[ell_local], view(tf.rell_data, :, ell_local, :))
        Q_mat = B.Q[(ell, rank)]
        # einsum "ij,j...->i..."
        n_comp = size(tf.rell_data, 1)
        tmp = reshape(tf.rell_data[:, ell_local, :], n_comp, :)
        rot = Q_mat * tmp
        tf.rell_data[:, ell_local, :] .= reshape(rot, n_comp, :)
    end

    for (i, field) in enumerate(tf.fields)
        field.layout = tf.r_ell_layout
        field.data = tf.rell_data[i, :, :]
        require_layout(field, tf.ell_r_layout)
    end

    angle_input = zeros(ComplexF64, 3^rank, size(tf.fields[1].data)...)
    for i in 1:(3^rank)
        angle_input[i, :, :] .= tf.fields[i].data
    end

    backward_angle(B, tf.m, rank, angle_input, tf.grid_data)
    if rank > 0
        unitary_mat = ball_wrapper_unitary3D(B; rank=rank, adjoint=false)
        n_comp = size(tf.grid_data, 1)
        rest_dims = size(tf.grid_data)[2:end]
        data_2d = reshape(tf.grid_data, n_comp, :)
        rot_2d = unitary_mat * data_2d
        tf.grid_data .= reshape(rot_2d, n_comp, rest_dims...)
    end

    tf.data = tf.grid_data
    tf._layout = 'g'
end

# ============================================================================
# BallTensorField3D
# ============================================================================

"""
    BallTensorField3D

3D tensor field on the ball, with grid, coefficient, and intermediate
data storage for multi-dimensional domain decomposition.
"""
mutable struct BallTensorField3D <: AbstractBallTensorField
    rank::Int
    B::BallWrapper
    domain::Any
    ell_min::Int
    ell_max::Int
    phi_layout::Any
    th_m_layout::Any
    ell_r_layout::Any
    r_ell_layout::Any
    grid_data::Array{Float64}
    mlr_ell_data::Array{ComplexF64}
    mlr_r_data::Array{ComplexF64}
    rlm_data::Array{ComplexF64}
    mthr_data::Array{ComplexF64}
    fields::Any
    coeff_data::Vector{Matrix{ComplexF64}}
    _layout::Union{Char, Int}
    data::Any

    function BallTensorField3D(rank::Int, B::BallWrapper, domain)
        ell_min = B.ell_min
        ell_max = B.ell_max

        mesh = domain.distributor.mesh

        if length(mesh) == 0  # serial
            phi_layout = domain.distributor.layouts[4]
            th_m_layout = domain.distributor.layouts[3]
            ell_r_layout = domain.distributor.layouts[2]
            r_ell_layout = domain.distributor.layouts[2]
        elseif length(mesh) == 1  # 1D domain decomposition
            phi_layout = domain.distributor.layouts[5]
            th_m_layout = domain.distributor.layouts[3]
            ell_r_layout = domain.distributor.layouts[2]
            r_ell_layout = domain.distributor.layouts[2]
        elseif length(mesh) == 2  # 2D domain decomposition
            phi_layout = domain.distributor.layouts[6]
            th_m_layout = domain.distributor.layouts[4]
            ell_r_layout = domain.distributor.layouts[3]
            r_ell_layout = domain.distributor.layouts[2]
        else
            error("Unsupported mesh size: $(length(mesh))")
        end

        # Allocating arrays
        local_grid_shape = phi_layout.local_shape(scales=domain.dealias)
        grid_data = zeros(Float64, 3^rank, local_grid_shape...)

        scales = (1, 1, domain.dealias[3])
        local_ellr_shape = ell_r_layout.local_shape(scales=scales)
        mlr_ell_data = zeros(ComplexF64, 3^rank, local_ellr_shape...)

        local_rell_shape = r_ell_layout.local_shape(scales=scales)
        mlr_r_data = zeros(ComplexF64, 3^rank, local_rell_shape...)
        rlm_data = zeros(ComplexF64, 3^rank, reverse(local_rell_shape)...)

        scales = (1, domain.dealias[2], domain.dealias[3])
        local_mthr_shape = th_m_layout.local_shape(scales=scales)
        mthr_data = zeros(ComplexF64, 3^rank, local_mthr_shape...)

        fields = domain.new_fields(3^rank)
        for field in fields
            field.preset_scales(domain.dealias)
        end

        m_size = B.m_max - B.m_min + 1
        coeff_data = Vector{Matrix{ComplexF64}}()
        for ell in ell_min:ell_max
            N = B.N_max - ball_wrapper_N_min(B, ell - B.R_max) + 1
            push!(coeff_data, zeros(ComplexF64, N * (3^rank), m_size))
        end

        new(rank, B, domain, ell_min, ell_max,
            phi_layout, th_m_layout, ell_r_layout, r_ell_layout,
            grid_data, mlr_ell_data, mlr_r_data, rlm_data, mthr_data,
            fields, coeff_data,
            'g', grid_data)
    end
end

"""
    require_coeff_space(tf::BallTensorField3D)

Transform from grid space to coefficient space for 3D tensor field.
Decrements layout step by step until reaching coefficient space.
"""
function require_coeff_space(tf::BallTensorField3D)
    while tf._layout != 'c'
        decrement_layout(tf)
    end
end

"""
    decrement_layout(tf::BallTensorField3D)

Decrement the layout by one step toward coefficient space.
"""
function decrement_layout(tf::BallTensorField3D)
    rank = tf.rank
    B = tf.B

    if tf._layout == 'g'
        for (i, field) in enumerate(tf.fields)
            field.layout = tf.phi_layout
            copyto!(field.data, tf.data[i, ntuple(_ -> Colon(), ndims(tf.data) - 1)...])
            require_layout(field, tf.th_m_layout)
            copyto!(tf.mthr_data[i, ntuple(_ -> Colon(), ndims(tf.mthr_data) - 1)...], field.data)
        end
        tf._layout = 3
    elseif tf._layout == 3
        for m in B.m_min:B.m_max
            m_local = m - B.m_min + 1  # 1-based
            forward_angle(B, m, rank,
                          tf.mthr_data[:, m_local, :],
                          view(tf.mlr_ell_data, :, m_local, :, :))
        end
        tf._layout = 2
    elseif tf._layout == 2
        if tf.ell_r_layout !== tf.r_ell_layout
            for (i, field) in enumerate(tf.fields)
                field.layout = tf.ell_r_layout
                copyto!(field.data, tf.mlr_ell_data[i, ntuple(_ -> Colon(), ndims(tf.mlr_ell_data) - 1)...])
                tf.domain.distributor.paths[2].decrement([field])
                tf.rlm_data[i, ntuple(_ -> Colon(), ndims(tf.rlm_data) - 1)...] .= permutedims(field.data)
            end
        else
            for (i, _field) in enumerate(tf.fields)
                tf.rlm_data[i, ntuple(_ -> Colon(), ndims(tf.rlm_data) - 1)...] .= permutedims(tf.mlr_ell_data[i, ntuple(_ -> Colon(), ndims(tf.mlr_ell_data) - 1)...])
            end
        end
        tf._layout = 1
    elseif tf._layout == 1
        for ell in B.ell_min:B.ell_max
            ell_local = ell - B.ell_min + 1  # 1-based
            radial_forward(B, ell, rank,
                           tf.rlm_data[:, :, ell_local, :],
                           tf.coeff_data[ell_local])
        end
        tf.data = tf.coeff_data
        tf._layout = 'c'
    end
end

"""
    require_grid_space(tf::BallTensorField3D)

Transform from coefficient space to grid space for 3D tensor field.
Increments layout step by step until reaching grid space.
"""
function require_grid_space(tf::BallTensorField3D)
    while tf._layout != 'g'
        increment_layout(tf)
    end
end

"""
    increment_layout(tf::BallTensorField3D)

Increment the layout by one step toward grid space.
"""
function increment_layout(tf::BallTensorField3D)
    rank = tf.rank
    B = tf.B

    if tf._layout == 'c'
        for ell in B.ell_min:B.ell_max
            ell_local = ell - B.ell_min + 1  # 1-based
            radial_backward(B, ell, rank,
                            tf.data[ell_local],
                            view(tf.rlm_data, :, :, ell_local, :))
            Q_mat = B.Q[(ell, rank)]
            # einsum "ij,j...->i..."
            n_comp = size(tf.rlm_data, 1)
            slice = tf.rlm_data[:, :, ell_local, :]
            data_2d = reshape(slice, n_comp, :)
            rot_2d = Q_mat * data_2d
            tf.rlm_data[:, :, ell_local, :] .= reshape(rot_2d, n_comp, size(slice)[2:end]...)
        end
        # transpose (0,3,2,1) in Python -> (1,4,3,2) in Julia
        # rlm_data shape: (ncomp, r, ell, m) -> mlr_r_data shape: (ncomp, m, ell, r)
        # But the Python does transpose(0,3,2,1) on (ncomp, r, ell, m) -> (ncomp, m, ell, r)
        # In Julia (1-based): permutedims(data, (1, 4, 3, 2))
        copyto!(tf.mlr_r_data, permutedims(tf.rlm_data, (1, 4, 3, 2)))
        tf._layout = 1
    elseif tf._layout == 1
        if tf.ell_r_layout !== tf.r_ell_layout
            for (i, field) in enumerate(tf.fields)
                field.layout = tf.r_ell_layout
                copyto!(field.data, tf.mlr_r_data[i, ntuple(_ -> Colon(), ndims(tf.mlr_r_data) - 1)...])
                tf.domain.distributor.paths[2].increment([field])
                copyto!(tf.mlr_ell_data[i, ntuple(_ -> Colon(), ndims(tf.mlr_ell_data) - 1)...], field.data)
            end
        else
            for (i, _field) in enumerate(tf.fields)
                copyto!(tf.mlr_ell_data[i, ntuple(_ -> Colon(), ndims(tf.mlr_ell_data) - 1)...],
                        tf.mlr_r_data[i, ntuple(_ -> Colon(), ndims(tf.mlr_r_data) - 1)...])
            end
        end
        tf._layout = 2
    elseif tf._layout == 2
        for m in B.m_min:B.m_max
            m_local = m - B.m_min + 1  # 1-based
            backward_angle(B, m, rank,
                           tf.mlr_ell_data[:, m_local, :, :],
                           view(tf.mthr_data, :, m_local, :, :))
        end
        if rank > 0
            unitary_mat = ball_wrapper_unitary3D(B; rank=rank, adjoint=false)
            n_comp = size(tf.mthr_data, 1)
            rest_dims = size(tf.mthr_data)[2:end]
            data_2d = reshape(tf.mthr_data, n_comp, :)
            rot_2d = unitary_mat * data_2d
            tf.mthr_data .= reshape(rot_2d, n_comp, rest_dims...)
        end
        tf._layout = 3
    elseif tf._layout == 3
        for (i, field) in enumerate(tf.fields)
            field.layout = tf.th_m_layout
            copyto!(field.data, tf.mthr_data[i, ntuple(_ -> Colon(), ndims(tf.mthr_data) - 1)...])
            require_layout(field, tf.phi_layout)
            copyto!(tf.grid_data[i, ntuple(_ -> Colon(), ndims(tf.grid_data) - 1)...], field.data)
        end
        tf.data = tf.grid_data
        tf._layout = 'g'
    end
end
