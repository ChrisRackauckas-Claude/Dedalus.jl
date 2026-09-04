"""
Zernike polynomial operator algebra for Dedalus.jl.

Translated from dedalus/libraries/dedalus_sphere/zernike.py.
Provides Zernike polynomial mass, quadrature, polynomial evaluation,
and operator construction built on top of the Jacobi framework.
"""

using SparseArrays

# ============================================================================
# ZernikeCodomain
# ============================================================================

"""
    ZernikeCodomain

Codomain for Zernike polynomial operators, tracking shifts in (n, k, l) space.
Analogous to JacobiCodomain but with Zernike-specific parameter semantics.

    codomain = ZernikeCodomain(dn, dk, dl, pi)
"""
struct ZernikeCodomain
    arrow::NTuple{4, Int}
    Output::Type

    function ZernikeCodomain(dn::Int=0, dk::Int=0, dl::Int=0, pi::Int=0; Output::Type=ZernikeCodomain)
        new((dn, dk, dl, pi), Output)
    end
end

Base.getindex(c::ZernikeCodomain, i::Int) = c.arrow[i]
Base.getindex(c::ZernikeCodomain, ::Colon) = c.arrow
Base.getindex(c::ZernikeCodomain, r::UnitRange) = c.arrow[r]

Base.length(::ZernikeCodomain) = 3

Base.iterate(c::ZernikeCodomain, state...) = iterate(c.arrow, state...)

function Base.show(io::IO, c::ZernikeCodomain)
    s = "(n->n+$(c[1]),k->k+$(c[2]),l->l+$(c[3]))"
    s = replace(s, "+0" => "")
    s = replace(s, "+-" => "-")
    print(io, s)
end

function Base.:+(a::ZernikeCodomain, b::ZernikeCodomain)
    # Compose: additive arrows, XOR parity
    dn = a[1] + b[1]
    dk = a[2] + b[2]
    dl = a[3] + b[3]
    pi = xor(a[4], b[4])
    return a.Output(dn, dk, dl, pi; Output=a.Output)
end

function (c::ZernikeCodomain)(args...)
    n, k, l = args[1], args[2], args[3]
    return (c[1] + n, c[2] + k, c[3] + l)
end

function Base.:(==)(a::ZernikeCodomain, b::ZernikeCodomain)
    # Compare (dk, dl, pi), i.e. indices 2,3,4 (all but dn)
    return a[2] == b[2] && a[3] == b[3] && a[4] == b[4]
end

Base.hash(c::ZernikeCodomain, h::UInt) = hash((c[2], c[3], c[4]), hash(:ZernikeCodomain, h))

function Base.:|(a::ZernikeCodomain, b::ZernikeCodomain)
    if a != b
        throw(TypeError("operators have incompatible codomains."))
    end
    if a[1] >= b[1]
        return a
    end
    return b
end

function Base.:-(c::ZernikeCodomain)
    return c.Output(-c[1], -c[2], -c[3], c[4]; Output=c.Output)
end

function Base.:*(c::ZernikeCodomain, other::Int)
    if other == 0
        return c.Output(0, 0, 0, 0; Output=c.Output)
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

Base.:*(other::Int, c::ZernikeCodomain) = c * other

function Base.:-(a::ZernikeCodomain, b::ZernikeCodomain)
    a + (-b)
end

"""Convert a ZernikeCodomain to a base Codomain for use with Operator."""
function _zernike_to_codomain(zc::ZernikeCodomain)
    Codomain(zc.arrow...; Output=Codomain)
end

# ============================================================================
# Constants
# ============================================================================

"""Default Jacobi parameter for Zernike polynomials."""
const zernike_alpha = 0

# ============================================================================
# zernike_mass
# ============================================================================

"""
    zernike_mass(dimension; k=0)

Compute the Zernike mass for given dimension and regularity parameter k.
"""
function zernike_mass(dimension; k::Int=zernike_alpha)
    return jacobi_mass(k, dimension / 2 - 1) / 2^(k + dimension / 2 + 1)
end

# ============================================================================
# zernike_quadrature
# ============================================================================

"""
    zernike_quadrature(dimension, n; k=0)

Weights associated with dV = (1-r*r)^k * r^(dimension-1) dr, where 0 <= r <= 1.
"""
function zernike_quadrature(dimension, n; k::Int=zernike_alpha)
    z, w = jacobi_quadrature(n, k, dimension / 2 - 1)
    w ./= 2^(k + dimension / 2 + 1)
    return z, w
end

# ============================================================================
# zernike_min_degree
# ============================================================================

"""
    zernike_min_degree(l)

Minimum polynomial degree for given azimuthal number l.
"""
function zernike_min_degree(l)
    return max(l ÷ 2, 0)
end

# ============================================================================
# zernike_polynomials
# ============================================================================

"""
    zernike_polynomials(dimension, n, k, l, z)

Unit normalised Zernike polynomials: integral(Q^2 dV) = 1.
"""
function zernike_polynomials(dimension, n, k, l, z)
    b = l + dimension / 2 - 1

    init = jacobi_measure(0, l, z; log=true, probability=false)
    init .-= jacobi_mass(k, b; log=true) .- log(2) * (k + dimension / 2 + 1)
    init = exp.(0.5 .* init)

    return jacobi_polynomials(n, k, b, z; init=init)
end

# ============================================================================
# ZernikeOperator
# ============================================================================

"""
    ZernikeOperator

Factory producing Operators for D, E, R Zernike operations.
Each call with a parameter p returns an Operator.
"""
struct ZernikeOperator
    _func::Function
    _dimension::Number
    _radius::Number
end

function (zo::ZernikeOperator)(p)
    func, cod = zo._func(p)
    return Operator(func, _zernike_to_codomain(cod); Output=Operator)
end

function _zernike_b(dimension, l)
    return l + dimension / 2 - 1
end

function _zernike_D_method(dimension, radius)
    function _D(dl)
        function D(n, k, l)
            D_op = jacobi_operator(dl > 0 ? "D" : "C")(+1)
            return (2 / radius) * D_op(n, k, _zernike_b(dimension, l))
        end
        return D, ZernikeCodomain(fld(-(1 + dl), 2), 1, dl)
    end
    return _D
end

function _zernike_E_method(dimension, _radius)
    function _E(dk)
        function E(n, k, l)
            E_op = jacobi_operator("A")(dk)
            return sqrt(0.5) * E_op(n, k, _zernike_b(dimension, l))
        end
        return E, ZernikeCodomain(fld(1 - dk, 2), dk, 0)
    end
    return _E
end

function _zernike_R_method(dimension, radius)
    function _R(dl)
        function R(n, k, l)
            R_op = jacobi_operator("B")(dl)
            return (sqrt(0.5) * radius) * R_op(n, k, _zernike_b(dimension, l))
        end
        return R, ZernikeCodomain(fld(1 - dl, 2), 0, dl)
    end
    return _R
end

# ============================================================================
# zernike_operator
# ============================================================================

"""
    zernike_operator(dimension, name; radius=1)

Interface to base ZernikeOperator class.

For "Id" and "Z": returns an Operator directly.
For "D", "E", "R": returns a ZernikeOperator factory.
"""
function zernike_operator(dimension, name::String; radius=1)
    if name == "Id"
        function I_func(n, k, l)
            return jacobi_operator("Id")(n, k, l + dimension / 2 - 1)
        end
        return Operator(I_func, _zernike_to_codomain(ZernikeCodomain(0, 0, 0)); Output=Operator)
    end

    if name == "Z"
        function Z_func(n, k, l)
            return jacobi_operator("Z")(n, k, l + dimension / 2 - 1)
        end
        return Operator(Z_func, _zernike_to_codomain(ZernikeCodomain(1, 0, 0)); Output=Operator)
    end

    if name == "D"
        return ZernikeOperator(_zernike_D_method(dimension, radius), dimension, radius)
    elseif name == "E"
        return ZernikeOperator(_zernike_E_method(dimension, radius), dimension, radius)
    elseif name == "R"
        return ZernikeOperator(_zernike_R_method(dimension, radius), dimension, radius)
    else
        error("Unknown Zernike operator name: $name")
    end
end
