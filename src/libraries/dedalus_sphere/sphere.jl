"""
Sphere module for Dedalus.jl.

Translated from dedalus/libraries/dedalus_sphere/sphere.py.
Provides spin-weighted spherical harmonics (SWSH), sphere quadrature,
and sphere operators built on top of the Jacobi polynomial framework.

This module defines:
- SphereCodomain: codomain for sphere operators tracking (dL, dm, ds, pi)
- spin2Jacobi: convert sphere (L,m,s) to Jacobi (n,a,b) parameters
- sphere_harmonics: evaluate SWSH using Jacobi polynomials with measure envelope
- sphere_quadrature: Gauss quadrature on the sphere
- sphere_operator: factory for sphere operators (Id, Pi, L, M, S, Cos, D, Sin)
- SphereOperator: factory struct producing Operators for a given spin shift ds
- sphere_L_min: minimum ell for given (m,s) = max(|m|, |s|)
- sphere_op: convenience function to get operator matrix for given (name, Lmax, m, s)
- sphere_zeros: zero matrix for operator between spin spaces
- sphere_unitary: unitary transformation matrix from spin to regularity basis
"""

using SparseArrays

# ============================================================================
# SphereCodomain
# ============================================================================

"""
    SphereCodomain

Codomain for sphere operators, tracking shifts in (L, m, s) space with
an optional parity flag.

    codomain = SphereCodomain(dL, dm, ds, pi)
    L', m', s' = codomain(L, m, s)

If pi == 0: L' = L+dL, m' = m+dm, s' = s+ds
If pi == 1: L' = L+dL, m' = m+dm, s' = -(s+ds)  (spin parity flip)

This is analogous to JacobiCodomain but operates on sphere parameters.
Since Julia cannot subclass concrete types, SphereCodomain is a standalone struct
with the same interface.
"""
struct SphereCodomain
    arrow::NTuple{4, Int}

    function SphereCodomain(dL::Int=0, dm::Int=0, ds::Int=0, pi::Int=0)
        new((dL, dm, ds, pi))
    end
end

Base.getindex(c::SphereCodomain, i::Int) = c.arrow[i]
Base.getindex(c::SphereCodomain, ::Colon) = c.arrow
Base.getindex(c::SphereCodomain, r::UnitRange) = c.arrow[r]

Base.length(::SphereCodomain) = 4
Base.iterate(c::SphereCodomain, state...) = iterate(c.arrow, state...)

function Base.show(io::IO, c::SphereCodomain)
    s = "(L->L+$(c[1]),m->m+$(c[2]),s->s+$(c[3]))"
    if c[4] != 0
        s = replace(s, "s->s" => "s->-s")
    end
    s = replace(s, "+0" => "")
    s = replace(s, "+-" => "-")
    print(io, s)
end

function (c::SphereCodomain)(args...; evaluate::Bool=true)
    L, m, s = args[1], args[2], args[3]
    if c[4] != 0
        s *= -1
    end
    return c[1] + L, c[2] + m, c[3] + s
end

function Base.:-(c::SphereCodomain)
    m, s = -c[2], -c[3]
    if c[4] != 0
        s *= -1
    end
    return SphereCodomain(-c[1], m, s, c[4])
end

function Base.:+(a::SphereCodomain, b::SphereCodomain)
    # Compose: apply a to b's (L,m,s) shifts
    L, m, s = a(b[1], b[2], b[3]; evaluate=false)
    return SphereCodomain(L, m, s, xor(a[4], b[4]))
end

function Base.:(==)(a::SphereCodomain, b::SphereCodomain)
    return a[2] == b[2] && a[3] == b[3] && a[4] == b[4]
end

Base.hash(c::SphereCodomain, h::UInt) = hash((c[2], c[3], c[4]), hash(:SphereCodomain, h))

function Base.:|(a::SphereCodomain, b::SphereCodomain)
    if a != b
        throw(TypeError("operators have incompatible codomains."))
    end
    if a[1] >= b[1]
        return a
    end
    return b
end

function Base.:*(c::SphereCodomain, other::Int)
    if other == 0
        return SphereCodomain(0, 0, 0, 0)
    end
    if other < 0
        return -c + (other + 1) * c
    end
    result = c
    for _ in 2:other
        result = result + c
    end
    return result
end

Base.:*(other::Int, c::SphereCodomain) = c * other

function Base.:-(a::SphereCodomain, b::SphereCodomain)
    a + (-b)
end

"""Convert a SphereCodomain to a base Codomain for use with Operator."""
function _sphere_to_codomain(sc::SphereCodomain)
    Codomain(sc.arrow...; Output=Codomain)
end

# ============================================================================
# spin2Jacobi
# ============================================================================

"""
    spin2Jacobi(Lmax, m, s; ds=nothing, dm=nothing)

Convert sphere parameters (L, m, s) to Jacobi parameters (n, a, b).

If ds and dm are both nothing, returns (n, a, b).
Otherwise, also returns the shifts (dn, da, db) resulting from ds and dm.

Parameters
----------
- Lmax : maximum spherical-harmonic degree
- m, s : azimuthal and spin quantum numbers
- ds   : spin shift (optional)
- dm   : azimuthal shift (optional)
"""
function spin2Jacobi(Lmax::Int, m::Int, s::Int; ds=nothing, dm=nothing)
    n = Lmax + 1 - max(abs(m), abs(s))
    a, b = abs(m + s), abs(m - s)

    if ds === nothing && dm === nothing
        return n, a, b
    end

    ds_val = ds === nothing ? 0 : ds
    dm_val = dm === nothing ? 0 : dm

    m2 = m + dm_val
    s2 = s + ds_val

    dn = Lmax + 1 - max(abs(m2), abs(s2)) - n
    da = abs(m2 + s2) - a
    db = abs(m2 - s2) - b

    return n, a, b, dn, da, db
end

# ============================================================================
# sphere_harmonics
# ============================================================================

"""
    sphere_harmonics(Lmax, m, s, cos_theta; kwargs...)

Evaluate spin-weighted spherical harmonic functions on a grid.

Returns an array with shape (Lmax - L_min(m,s) + 1, length(cos_theta))
or (Lmax - L_min(m,s) + 1,) if cos_theta is a single point.

Parameters
----------
- Lmax      : maximum spherical-harmonic degree
- m, s      : spherical harmonic parameters
- cos_theta : grid of cos(theta) values
"""
function sphere_harmonics(Lmax::Int, m::Int, s::Int, cos_theta; kwargs...)
    n, a, b = spin2Jacobi(Lmax, m, s)

    # Compute envelope: exp(0.5 * log_measure) * (-1)^max(m,-s)
    log_mu = jacobi_measure(a, b, cos_theta; log=true)
    init = exp.(0.5 .* log_mu)
    init .*= (-1.0)^max(m, -s)

    return jacobi_polynomials(n, a, b, cos_theta; init=init, kwargs...)
end

# ============================================================================
# sphere_quadrature
# ============================================================================

"""
    sphere_quadrature(Lmax; dtype=Float64)

Generate the Gauss quadrature grid and weights for spherical harmonics transform.

Returns (cos_theta, weights) that exactly integrate polynomials on (-1,+1)
up to degree 2*Lmax+1.

Parameters
----------
- Lmax : int >= 0, spherical-harmonic degree
"""
function sphere_quadrature(Lmax::Int; dtype::Type=Float64)
    return jacobi_quadrature(Lmax + 1, 0, 0; dtype=dtype)
end

# ============================================================================
# SphereOperator
# ============================================================================

"""
    SphereOperator

Factory struct that produces Operators for sphere operations with a given
spin shift ds. Supports derivative (D) and sine (Sin) operators.

Fields
------
- name    : operator name ("D" or "Sin")
- radius  : sphere radius (default 1)
- dtype   : output element type
"""
struct SphereOperator
    name::String
    radius::Float64
    dtype::Type

    function SphereOperator(name::String; radius::Real=1, dtype::Type=Float64)
        new(name, Float64(radius), dtype)
    end
end

"""
    (so::SphereOperator)(ds)

Call the SphereOperator factory with a given spin shift ds.
Returns an Operator.
"""
function (so::SphereOperator)(ds::Int)
    if so.name == "D"
        return _sphere_D(so, ds)
    elseif so.name == "Sin"
        return _sphere_Sin(so, ds)
    else
        error("Unknown SphereOperator name: $(so.name)")
    end
end

function _sphere_D(so::SphereOperator, ds::Int)
    function D(Lmax, m, s)
        n, a, b, dn, da, db = spin2Jacobi(Lmax, m, s; ds=ds)

        op_name = (da + db == 0) ? "C" : "D"
        D_op = jacobi_operator(op_name; dtype=so.dtype)(da)

        return (-ds * sqrt(0.5) / so.radius) * D_op(n, a, b)
    end
    return Operator(D, _sphere_to_codomain(SphereCodomain(0, 0, ds, 0)); Output=Operator)
end

function _sphere_Sin(so::SphereOperator, ds::Int)
    function Sin(Lmax, m, s)
        n, a, b, dn, da, db = spin2Jacobi(Lmax, m, s; ds=ds)

        S = jacobi_operator("A"; dtype=so.dtype)(da)
        S = compose(S, jacobi_operator("B"; dtype=so.dtype)(db))

        return (da * ds) * S(n, a, b)
    end
    return Operator(Sin, _sphere_to_codomain(SphereCodomain(1, 0, ds, 0)); Output=Operator)
end

# ============================================================================
# sphere_operator - factory
# ============================================================================

"""
    sphere_operator(name; dtype=Float64)

Interface to sphere operators.

Supported names:
- "Id"  : identity operator
- "Pi"  : parity operator
- "L"   : spherical harmonic degree (diagonal)
- "M"   : azimuthal number (diagonal)
- "S"   : spin number (diagonal)
- "Cos" : cosine operator (multiplication by cos(theta))
- "D"   : returns SphereOperator factory for derivative
- "Sin" : returns SphereOperator factory for sine
"""
function sphere_operator(name::String; dtype::Type=Float64)
    if name == "Id"
        return _sphere_identity(; dtype=dtype)
    end

    if name == "Pi"
        return _sphere_parity(; dtype=dtype)
    end

    if name == "L"
        return _sphere_L(; dtype=dtype)
    end

    if name == "M"
        return _sphere_M(; dtype=dtype)
    end

    if name == "S"
        return _sphere_S(; dtype=dtype)
    end

    if name == "Cos"
        function Cos(Lmax, m, s)
            return jacobi_operator("Z"; dtype=dtype)(spin2Jacobi(Lmax, m, s)...)
        end
        return Operator(Cos, _sphere_to_codomain(SphereCodomain(1, 0, 0, 0)); Output=Operator)
    end

    return SphereOperator(name; dtype=dtype)
end

# ============================================================================
# Static sphere operators
# ============================================================================

function _sphere_identity(; dtype::Type=Float64)
    function I(Lmax, m, s)
        n = spin2Jacobi(Lmax, m, s)[1]
        N_vec = ones(dtype, max(n, 0))
        mat = _spdiag(N_vec, 0, max(n, 0), max(n, 0))
        return InfiniteCSC(mat)
    end
    return Operator(I, _sphere_to_codomain(SphereCodomain(0, 0, 0, 0)); Output=Operator)
end

function _sphere_parity(; dtype::Type=Float64)
    function Pi(Lmax, m, s)
        return jacobi_operator("Pi"; dtype=dtype)(spin2Jacobi(Lmax, m, s)...)
    end
    return Operator(Pi, _sphere_to_codomain(SphereCodomain(0, 0, 0, 1)); Output=Operator)
end

function _sphere_L(; dtype::Type=Float64)
    function L(Lmax, m, s)
        n = spin2Jacobi(Lmax, m, s)[1]
        # Eigenvalues: L values from L_min to Lmax
        N_vec = collect(dtype.(Lmax + 1 - n : Lmax))
        mat = _spdiag(N_vec, 0, max(n, 0), max(n, 0))
        return InfiniteCSC(mat)
    end
    return Operator(L, _sphere_to_codomain(SphereCodomain(0, 0, 0, 0)); Output=Operator)
end

function _sphere_M(; dtype::Type=Float64)
    function M(Lmax, m, s)
        n = spin2Jacobi(Lmax, m, s)[1]
        N_vec = fill(dtype(m), max(n, 0))
        mat = _spdiag(N_vec, 0, max(n, 0), max(n, 0))
        return InfiniteCSC(mat)
    end
    return Operator(M, _sphere_to_codomain(SphereCodomain(0, 0, 0, 0)); Output=Operator)
end

function _sphere_S(; dtype::Type=Float64)
    function S(Lmax, m, s)
        n = spin2Jacobi(Lmax, m, s)[1]
        N_vec = fill(dtype(abs(s)), max(n, 0))
        mat = _spdiag(N_vec, 0, max(n, 0), max(n, 0))
        return InfiniteCSC(mat)
    end
    return Operator(S, _sphere_to_codomain(SphereCodomain(0, 0, 0, 0)); Output=Operator)
end

# ============================================================================
# Convenience functions (sphere128 interface)
# ============================================================================

"""
    sphere_L_min(m, s)

Minimum spherical harmonic degree for given (m, s).
"""
sphere_L_min(m::Int, s::Int) = max(abs(m), abs(s))

"""
    sphere_op(name, Lmax, m, s; dtype=Float64)

Convenience function: get the operator matrix for the given operator name
and sphere parameters. This is the sphere128-style interface where the
operator is fully evaluated to a matrix.

For simple operators (Id, Pi, L, M, S, Cos), returns the matrix directly.
For compound operators like "k+" and "k-" (spin raising/lowering),
constructs the appropriate combination.
"""
function sphere_op(name::String, Lmax::Int, m::Int, s::Int; dtype::Type=Float64)
    if name == "k+" || name == "k-"
        ds = name == "k+" ? +1 : -1
        # k+/k- = D(ds) + Sin(ds), spin raising/lowering
        # Evaluate each operator separately and add the resulting InfiniteCSC matrices.
        # Cannot use Operator algebra (+) because D and Sin have different codomains
        # (different dL), but their matrices can be added via InfiniteCSC.
        D_op = sphere_operator("D"; dtype=dtype)(ds)
        Sin_op = sphere_operator("Sin"; dtype=dtype)(ds)
        D_mat = D_op(Lmax, m, s)
        Sin_mat = Sin_op(Lmax, m, s)
        result = D_mat + Sin_mat
        # Resize to the correct output dimensions:
        # Input dimension: n_in = Lmax + 1 - max(|m|, |s|)
        # Output dimension: n_out = Lmax + 1 - max(|m|, |s + ds|)
        n_in = spin2Jacobi(Lmax, m, s)[1]
        n_out = spin2Jacobi(Lmax, m, s + ds)[1]
        n_in = max(n_in, 0)
        n_out = max(n_out, 0)
        result_sp = sparse(Float64.(result))
        # Truncate or pad to (n_out, n_in)
        r, c = size(result_sp)
        if r == n_out && c == n_in
            return result_sp
        end
        return resize_matrix(result_sp, n_out, n_in)
    else
        op_obj = sphere_operator(name; dtype=dtype)
        if op_obj isa Operator
            return sparse(Float64.(op_obj(Lmax, m, s)))
        else
            # SphereOperator: needs a spin shift argument
            error("Operator '$name' requires a spin shift argument. Use sphere_operator(\"$name\")(ds) instead.")
        end
    end
end

"""
    sphere_zeros(Lmax, m, s_out, s_in)

Create a zero matrix of appropriate dimensions for an operator mapping
from spin s_in to spin s_out coefficient space.
"""
function sphere_zeros(Lmax::Int, m::Int, s_out::Int, s_in::Int)
    n_out = spin2Jacobi(Lmax, m, s_out)[1]
    n_in = spin2Jacobi(Lmax, m, s_in)[1]
    return spzeros(Float64, max(n_out, 0), max(n_in, 0))
end

"""
    sphere_unitary(; rank=1, adjoint=false)

Compute the unitary transformation matrix between spin and regularity bases
for the given tensor rank. This uses the Intertwiner from spin_operators.

For rank 1, this is the 2x2 unitary matrix that maps from regularity (-1,+1)
basis to spin (-1,+1) basis.

If adjoint=true, returns the conjugate transpose.
"""
function sphere_unitary(; rank::Int=1, adjoint::Bool=false)
    # The unitary matrix for 2D spin operations uses a 2-element indexing (-1, +1)
    indexing = (-1, 1)
    n = length(indexing)^rank
    Q = zeros(ComplexF64, n, n)

    # For rank 0, it's just [1]
    if rank == 0
        return ones(Float64, 1, 1)
    end

    # Build the unitary from the Intertwiner evaluated at L=1 (minimum nontrivial L)
    # The Intertwiner Q(L) maps from regularity to spin basis
    # For the sphere wrapper, the 2D indexing is (-1, +1)
    # We use a fixed L value that gives a well-defined rotation
    Q_mat = 1.0 / sqrt(2.0) * [1.0 1.0; -1.0im 1.0im]

    if rank == 1
        if adjoint
            return collect(Q_mat')
        else
            return collect(Q_mat)
        end
    end

    # For higher ranks, take the tensor product
    result = Q_mat
    for _ in 2:rank
        result = kron(result, Q_mat)
    end

    if adjoint
        return collect(result')
    else
        return collect(result)
    end
end
