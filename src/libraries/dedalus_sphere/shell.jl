"""
Shell operator algebra for Dedalus.jl.

Translated from dedalus/libraries/dedalus_sphere/shell.py.
Provides codomain and operator construction for spherical shell domains
built on the Jacobi framework.
"""

using SparseArrays

# ============================================================================
# ShellCodomain
# ============================================================================

"""
    ShellCodomain

Codomain for shell operators, tracking shifts in (n, k) space.
"""
struct ShellCodomain
    arrow::NTuple{2, Int}
    Output::Type

    function ShellCodomain(dn::Int=0, dk::Int=0; Output::Type=ShellCodomain)
        new((dn, dk), Output)
    end
end

Base.getindex(c::ShellCodomain, i::Int) = c.arrow[i]
Base.getindex(c::ShellCodomain, ::Colon) = c.arrow
Base.getindex(c::ShellCodomain, r::UnitRange) = c.arrow[r]

Base.length(::ShellCodomain) = 2

Base.iterate(c::ShellCodomain, state...) = iterate(c.arrow, state...)

function Base.show(io::IO, c::ShellCodomain)
    s = "(n->n+$(c[1]),k->k+$(c[2]))"
    s = replace(s, "+0" => "")
    s = replace(s, "+-" => "-")
    print(io, s)
end

function Base.:+(a::ShellCodomain, b::ShellCodomain)
    return a.Output(a[1] + b[1], a[2] + b[2]; Output=a.Output)
end

function (c::ShellCodomain)(args...)
    n, k = args[1], args[2]
    return (c[1] + n, c[2] + k)
end

function Base.:(==)(a::ShellCodomain, b::ShellCodomain)
    return a[2] == b[2]
end

Base.hash(c::ShellCodomain, h::UInt) = hash(c[2], hash(:ShellCodomain, h))

function Base.:|(a::ShellCodomain, b::ShellCodomain)
    if a != b
        throw(TypeError("operators have incompatible codomains."))
    end
    if a[1] >= b[1]
        return a
    end
    return b
end

function Base.:-(c::ShellCodomain)
    return c.Output(-c[1], -c[2]; Output=c.Output)
end

function Base.:*(c::ShellCodomain, other::Int)
    if other == 0
        return c.Output(0, 0; Output=c.Output)
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

Base.:*(other::Int, c::ShellCodomain) = c * other

function Base.:-(a::ShellCodomain, b::ShellCodomain)
    a + (-b)
end

"""Convert a ShellCodomain to a base Codomain for use with Operator."""
function _shell_to_codomain(sc::ShellCodomain)
    Codomain(sc.arrow...; Output=Codomain)
end

# ============================================================================
# Constants
# ============================================================================

"""Default Jacobi parameters for shell."""
const shell_alpha = (-0.5, -0.5)

# ============================================================================
# shell_operator
# ============================================================================

"""
    shell_operator(dimension, radii, name; alpha=(-0.5, -0.5))

Shell operator factory.

Parameters:
- dimension: spatial dimension
- radii: (Ri, Ro) inner and outer radii tuple
- name: operator name string ("Z", "Id", "R", "AB", "E", "D")
- alpha: base Jacobi parameters

For most operators, returns an Operator directly.
For "D", returns a callable that takes (dl, l) and returns an Operator.
"""
function shell_operator(dimension, radii, name::String; alpha=shell_alpha)
    width = radii[2] - radii[1]

    if name == "Z"
        function Z_func(n, k)
            return jacobi_operator("Z")(n, k + alpha[1], k + alpha[2])
        end
        return Operator(Z_func, _shell_to_codomain(ShellCodomain(0, 0)); Output=Operator)
    end

    # Compose Z = (radii[2] + radii[1]) / width + jacobi_operator("Z")
    # This is: aspectratio * Id + Jacobi_Z
    # As a lazy Operator composition:
    Z_op = (radii[2] + radii[1]) / width + jacobi_operator("Z")

    if name == "Id"
        function I_func(n, k)
            return jacobi_operator("Id")(n, k + alpha[1], k + alpha[2])
        end
        return Operator(I_func, _shell_to_codomain(ShellCodomain(0, 0)); Output=Operator)
    end

    if name == "R"
        function R_func(n, k)
            return (0.5 * width) * Z_op(n, k + alpha[1], k + alpha[2])
        end
        return Operator(R_func, _shell_to_codomain(ShellCodomain(1, 0)); Output=Operator)
    end

    # AB = A(+1) @ B(+1) — composed Jacobi operators
    AB_op = compose(jacobi_operator("A")(+1), jacobi_operator("B")(+1))

    if name == "AB"
        function AB_func(n, k)
            return AB_op(n, k + alpha[1], k + alpha[2])
        end
        return Operator(AB_func, _shell_to_codomain(ShellCodomain(0, 1)); Output=Operator)
    end

    if name == "E"
        # E = 0.5 * (AB @ Z) — compose AB with Z_op
        E_comp = compose(AB_op, Z_op)
        function E_func(n, k)
            return 0.5 * E_comp(n, k + alpha[1], k + alpha[2])
        end
        return Operator(E_func, _shell_to_codomain(ShellCodomain(1, 1)); Output=Operator)
    end

    if name == "D"
        function D_factory(dl, l)
            function D_func(n, k)
                # D = D(+1) @ Z
                D_composed = compose(jacobi_operator("D")(+1), Z_op)

                # K = A(0) - alpha[1]
                # K += dl*l + (dl == -1)*(2-dimension)
                K_op = jacobi_operator("A")(0) - alpha[1]
                K_op = K_op + (dl * l + (dl == -1 ? 1 : 0) * (2 - dimension))

                # D = (D - K @ AB) / width
                D_result = (D_composed - compose(K_op, AB_op)) / width

                return D_result(n, k + alpha[1], k + alpha[2])
            end
            return Operator(D_func, _shell_to_codomain(ShellCodomain(0, 1)); Output=Operator)
        end
        return D_factory
    end

    error("Unknown shell operator name: $name")
end
