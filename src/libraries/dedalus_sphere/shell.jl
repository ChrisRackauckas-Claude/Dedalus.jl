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
    aspectratio = (radii[2] + radii[1]) / width

    # Jacobi operator handles for reuse
    _J_Id = jacobi_operator("Id")
    _J_Z = jacobi_operator("Z")
    _A_plus = jacobi_operator("A")(+1)
    _B_plus = jacobi_operator("B")(+1)
    _D_plus = jacobi_operator("D")(+1)

    # Z_shell(n, a, b) = aspectratio * I(n, a, b) + Z(n, a, b)
    # Built at the matrix level to avoid codomain compatibility issues:
    # JacobiCodomain == only checks (da, db, pi), but base Codomain == checks all elements.
    # The Jacobi Z operator has codomain (1,0,0,0) which is incompatible with (0,0,0,0)
    # under the base Codomain equality check used by the + operator on Operators.
    function _Z_shell(n, a, b)
        return aspectratio * _J_Id(n, a, b) + _J_Z(n, a, b)
    end

    # AB_compose(n, a, b) = A(+1)(B_cod(n,a,b)) * B(+1)(n,a,b)
    # B(+1) codomain = (0, 0, +1), so shifted = (n, a, b+1)
    function _AB_compose(n, a, b)
        B_mat = _B_plus(n, a, b)
        A_mat = _A_plus(n, a, b + 1)
        return A_mat * B_mat
    end

    if name == "Z"
        function Z_func(n, k)
            return _J_Z(n, k + alpha[1], k + alpha[2])
        end
        return Operator(Z_func, _shell_to_codomain(ShellCodomain(0, 0)); Output=Operator)
    end

    if name == "Id"
        function I_func(n, k)
            return _J_Id(n, k + alpha[1], k + alpha[2])
        end
        return Operator(I_func, _shell_to_codomain(ShellCodomain(0, 0)); Output=Operator)
    end

    if name == "R"
        function R_func(n, k)
            return (0.5 * width) * _Z_shell(n, k + alpha[1], k + alpha[2])
        end
        return Operator(R_func, _shell_to_codomain(ShellCodomain(1, 0)); Output=Operator)
    end

    if name == "AB"
        function AB_func(n, k)
            return _AB_compose(n, k + alpha[1], k + alpha[2])
        end
        return Operator(AB_func, _shell_to_codomain(ShellCodomain(0, 1)); Output=Operator)
    end

    if name == "E"
        function E_func(n, k)
            a_k = k + alpha[1]
            b_k = k + alpha[2]
            # E = 0.5 * (AB @ Z_shell)
            # compose(AB, Z_shell)(n, a, b):
            #   Z_mat = Z_shell(n, a, b)
            #   Z_shell codomain = (1, 0, 0) (dn=1 from Jacobi Z dominating)
            #   AB evaluated at shifted point (n+1, a, b)
            Z_mat = _Z_shell(n, a_k, b_k)
            AB_mat = _AB_compose(n + 1, a_k, b_k)
            return 0.5 * (AB_mat * Z_mat)
        end
        return Operator(E_func, _shell_to_codomain(ShellCodomain(1, 1)); Output=Operator)
    end

    if name == "D"
        function D_factory(dl, l)
            function D_func(n, k)
                a_k = k + alpha[1]
                b_k = k + alpha[2]

                # D_composed = D(+1) @ Z_shell
                # compose(D(+1), Z_shell)(n, a, b):
                #   Z_mat = Z_shell(n, a, b)
                #   Z_shell codomain = (1, 0, 0), shifted = (n+1, a, b)
                #   D(+1) evaluated at (n+1, a, b)
                Z_mat = _Z_shell(n, a_k, b_k)
                D_mat = _D_plus(n + 1, a_k, b_k) * Z_mat

                # K = A(0) - alpha[1] + dl*l + (dl==-1)*(2-dimension)
                # K @ AB = compose(K, AB)(n, a, b)
                # AB codomain = (0, 1, 1), so K evaluated at (n, a+1, b+1)
                # A(0) at (n, a+1, b+1) gives (a+1)*I where a is the first Jacobi param
                # a = a_k, so A(0) gives (a_k+1)*I
                # K value = (a_k+1) - alpha[1] + dl*l + (dl==-1)*(2-dimension)
                K_val = (a_k + 1) - alpha[1] + dl * l + (dl == -1 ? 1 : 0) * (2 - dimension)

                # K @ AB: K is scalar * I, so K_val * AB_mat
                # Use + with negation for InfiniteCSC row-padding compatibility
                # Use (1/width) * instead of / width since InfiniteCSC lacks / Number
                AB_mat = _AB_compose(n, a_k, b_k)
                result = (1 / width) * (D_mat + (-K_val) * AB_mat)

                return result
            end
            return Operator(D_func, _shell_to_codomain(ShellCodomain(0, 1)); Output=Operator)
        end
        return D_factory
    end

    error("Unknown shell operator name: $name")
end
