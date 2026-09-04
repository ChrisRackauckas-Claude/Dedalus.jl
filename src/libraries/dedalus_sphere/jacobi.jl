"""
Jacobi polynomial operator algebra for Dedalus.jl.

Translated from dedalus/libraries/dedalus_sphere/jacobi.py.
Provides an algebraic operator framework for composing Jacobi polynomial
transformations lazily, building on the Operator/Codomain system from operators.jl.

This module defines:
- JacobiCodomain: tracks (dn, da, db, pi) parameter shifts with parity support
- JacobiOperator: factory for primary Jacobi recurrence operators (A, B, C, D, etc.)
- Quadrature, polynomial evaluation, measure, mass, norm_ratio, etc.

Note: This module is self-contained and does NOT depend on tools/jacobi.jl.
"""

using SparseArrays
using LinearAlgebra: eigvals, SymTridiagonal, diag, diagm
using SpecialFunctions: beta as _beta, logbeta as _logbeta

# ============================================================================
# JacobiCodomain
# ============================================================================

"""
    JacobiCodomain

Codomain for Jacobi polynomial operators, tracking shifts in (n, a, b) space
with an optional parity flag.

    codomain = JacobiCodomain(dn, da, db, pi)
    n', a', b' = codomain(n, a, b)

If pi == 0: n', a', b' = n+dn, a+da, b+db
If pi == 1: n', a', b' = n+dn, b+da, a+db  (a,b swap before shift)

pi_0 + pi_1 = pi_0 XOR pi_1
"""
struct JacobiCodomain
    arrow::NTuple{4, Int}
    Output::Type

    function JacobiCodomain(dn::Int=0, da::Int=0, db::Int=0, pi::Int=0; Output::Type=JacobiCodomain)
        new((dn, da, db, pi), Output)
    end
end

Base.getindex(c::JacobiCodomain, i::Int) = c.arrow[i]
Base.getindex(c::JacobiCodomain, ::Colon) = c.arrow
Base.getindex(c::JacobiCodomain, r::UnitRange) = c.arrow[r]

Base.length(::JacobiCodomain) = 3

Base.iterate(c::JacobiCodomain, state...) = iterate(c.arrow, state...)

function Base.show(io::IO, c::JacobiCodomain)
    s = "(n->n+$(c[1]),a->a+$(c[2]),b->b+$(c[3]))"
    if c[4] != 0
        s = replace(s, "a->a" => "a->b")
        s = replace(s, "b->b" => "b->a")
    end
    s = replace(s, "+0" => "")
    s = replace(s, "+-" => "-")
    print(io, s)
end

function Base.:+(a::JacobiCodomain, b::JacobiCodomain)
    n, alpha, beta = _jc_call(a, b[1], b[2], b[3]; evaluate=false)
    return a.Output(n, alpha, beta, xor(a[4], b[4]); Output=a.Output)
end

function _jc_call(c::JacobiCodomain, n, a, b; evaluate::Bool=true)
    if c[4] != 0
        a, b = b, a
    end
    n_out = c[1] + n
    a_out = c[2] + a
    b_out = c[3] + b
    if evaluate && (a_out <= -1 || b_out <= -1)
        throw(ArgumentError("invalid Jacobi parameter."))
    end
    return n_out, a_out, b_out
end

function (c::JacobiCodomain)(args...; evaluate::Bool=true)
    n, a, b = args[1], args[2], args[3]
    return _jc_call(c, n, a, b; evaluate=evaluate)
end

function Base.:-(c::JacobiCodomain)
    a, b = -c[2], -c[3]
    if c[4] != 0
        a, b = b, a
    end
    return c.Output(-c[1], a, b, c[4]; Output=c.Output)
end

function Base.:(==)(a::JacobiCodomain, b::JacobiCodomain)
    return a[2] == b[2] && a[3] == b[3] && a[4] == b[4]
end

Base.hash(c::JacobiCodomain, h::UInt) = hash((c[2], c[3], c[4]), hash(:JacobiCodomain, h))

function Base.:|(a::JacobiCodomain, b::JacobiCodomain)
    if a != b
        throw(TypeError("operators have incompatible codomains."))
    end
    if a[1] >= b[1]
        return a
    end
    return b
end

function Base.:*(c::JacobiCodomain, other::Int)
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

Base.:*(other::Int, c::JacobiCodomain) = c * other

function Base.:-(a::JacobiCodomain, b::JacobiCodomain)
    a + (-b)
end

# ============================================================================
# Bridge
# ============================================================================

function _to_codomain(jc::JacobiCodomain)
    Codomain(jc.arrow...; Output=Codomain)
end

# ============================================================================
# Helper: build banded sparse matrix from diagonals
# ============================================================================

function _spdiag(bands::Vector{Vector{T}}, offsets::Vector{Int}, rows::Int, cols::Int) where {T}
    if rows == 0 || cols == 0
        return sparse(Int[], Int[], T[], rows, cols)
    end
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]
    for (band, offset) in zip(bands, offsets)
        for j in 1:cols
            i = j - offset
            if i >= 1 && i <= rows && j <= length(band)
                val = band[j]
                if val != zero(T)
                    push!(I_idx, i)
                    push!(J_idx, j)
                    push!(V_val, val)
                end
            end
        end
    end
    return sparse(I_idx, J_idx, V_val, rows, cols)
end

function _spdiag(band::Vector{T}, offset::Int, rows::Int, cols::Int) where {T}
    _spdiag([band], [offset], rows, cols)
end

# ============================================================================
# mass
# ============================================================================

"""
    jacobi_mass(a, b; log=false)

Compute 2^(a+b+1) * Beta(a+1, b+1) = integral from -1 to 1 of (1-z)^a * (1+z)^b dz.
"""
function jacobi_mass(a, b; log::Bool=false)
    if !log
        return 2^(a + b + 1) * _beta(a + 1, b + 1)
    end
    return (a + b + 1) * Base.log(2) + _logbeta(a + 1, b + 1)
end

# ============================================================================
# norm_ratio
# ============================================================================

"""
    jacobi_norm_ratio(dn, da, db, n, a, b; squared=false)

Ratio of classical Jacobi normalisation:
    sqrt(N(n+dn, a+da, b+db) / N(n, a, b))
"""
function jacobi_norm_ratio(dn::Int, da::Int, db::Int, n, a, b; squared::Bool=false)
    function tricky(n_val, a_val, b_val)
        if a_val + b_val != -1
            return (2 * n_val + a_val + b_val + 1) / (n_val + a_val + b_val + 1)
        end
        return 2 - (n_val == 0 ? 1 : 0)
    end

    function n_ratio(d, n_val, a_val, b_val)
        if d < 0
            return 1.0 ./ n_ratio(-d, n_val .+ d, a_val, b_val)
        end
        if d == 0
            return 1.0 .+ 0 .* n_val
        end
        if d == 1
            return ((n_val .+ a_val .+ 1) .* (n_val .+ b_val .+ 1) ./
                    ((n_val .+ 1) .* (2 .* n_val .+ a_val .+ b_val .+ 3))) .*
                   tricky.(n_val, a_val, b_val)
        end
        return n_ratio(1, n_val .+ d .- 1, a_val, b_val) .* n_ratio(d - 1, n_val, a_val, b_val)
    end

    function ab_ratio(d, n_val, a_val, b_val)
        if d < 0
            return 1.0 ./ ab_ratio(-d, n_val, a_val .+ d, b_val)
        end
        if d == 0
            return 1.0 .+ 0 .* n_val
        end
        if d == 1
            return (2 .* (n_val .+ a_val .+ 1) ./ (2 .* n_val .+ a_val .+ b_val .+ 2)) .*
                   tricky.(n_val, a_val, b_val)
        end
        return ab_ratio(1, n_val, a_val .+ d .- 1, b_val) .* ab_ratio(d - 1, n_val, a_val, b_val)
    end

    ratio = n_ratio(dn, n, a + da, b + db) .* ab_ratio(da, n, a, b + db) .* ab_ratio(db, n, b, a)

    if !squared
        return sqrt.(ratio)
    end
    return ratio
end

# ============================================================================
# measure
# ============================================================================

"""
    jacobi_measure(a, b, z; probability=true, log=false)

Compute the Jacobi measure mu(a,b,z) = (1-z)^a * (1+z)^b.
"""
function jacobi_measure(a, b, z; probability::Bool=true, log::Bool=false)
    if !log
        w = ones(eltype(z), size(z))
        if a != 0
            w .*= (1 .- z) .^ a
        end
        if b != 0
            w .*= (1 .+ z) .^ b
        end
        if probability
            w ./= jacobi_mass(a, b)
        end
        return w
    end

    S = zeros(eltype(z), size(z))
    if a != 0
        S .+= a .* Base.log.(1 .- z)
    end
    if b != 0
        S .+= b .* Base.log.(1 .+ z)
    end
    if probability
        S .-= jacobi_mass(a, b; log=true)
    end
    return S
end

# ============================================================================
# JacobiOperator
# ============================================================================

struct JacobiOperator
    _func::Function
    normalised::Bool
    dtype::Type
end

function JacobiOperator(name::String; normalised::Bool=true, dtype::Type=Float64)
    func = _get_jacobi_method(name, normalised, dtype)
    JacobiOperator(func, normalised, dtype)
end

function (jo::JacobiOperator)(p::Int)
    func, cod = jo._func(p)
    return Operator(func, _to_codomain(cod); Output=Operator)
end

# ============================================================================
# Internal operator methods
# ============================================================================

function _get_jacobi_method(name::String, normalised::Bool, dtype::Type)
    if name == "A"
        return p -> _jacobi_A(p, normalised, dtype)
    elseif name == "B"
        return p -> _jacobi_B(p, normalised, dtype)
    elseif name == "C"
        return p -> _jacobi_C(p, normalised, dtype)
    elseif name == "D"
        return p -> _jacobi_D(p, normalised, dtype)
    else
        error("Unknown Jacobi operator name: $name")
    end
end

function _nan_to_zero!(arr)
    for i in eachindex(arr)
        if isnan(arr[i])
            arr[i] = zero(eltype(arr))
        end
    end
    return arr
end

function _jacobi_A(p::Int, normalised::Bool, dtype::Type)
    if p == 0
        function A0(n, a, b)
            N_vec = fill(dtype(a), max(n, 0))
            mat = _spdiag(N_vec, 0, max(n, 0), max(n, 0))
            return InfiniteCSC(mat)
        end
        return A0, JacobiCodomain(0, 0, 0, 0)
    end

    function A_pm(n, a, b)
        N_arr = collect(dtype.(0:(n - 1)))

        if p == 1
            band0 = N_arr .+ (a + b + 1)
            band1 = -(N_arr .+ b)
        else
            band0 = 2 .* (N_arr .+ a)
            band1 = -2 .* (N_arr .+ 1)
        end

        if a + b == -1
            band0[1] = 1.0
            band1[1] = 1.0
        else
            band0[1] /= (a + b + 1)
            band1[1] /= (a + b + 1)
        end

        for k in 2:n
            denom = 2 * N_arr[k] + a + b + 1
            band0[k] /= denom
            band1[k] /= denom
        end

        if normalised
            nr0 = jacobi_norm_ratio(0, p, 0, N_arr, a, b)
            band0 .*= nr0
            start_idx = div(1 + p, 2) + 1
            if start_idx >= 1 && start_idx <= n
                nr1 = jacobi_norm_ratio(-p, p, 0, N_arr[start_idx:end], a, b)
                band1[start_idx:end] .*= nr1
            end
        end

        _nan_to_zero!(band0)
        _nan_to_zero!(band1)

        nrows = max(n + div(1 - p, 2), 0)
        ncols = max(n, 0)
        mat = _spdiag([band0, band1], [0, p], nrows, ncols)
        return InfiniteCSC(mat)
    end

    return A_pm, JacobiCodomain(div(1 - p, 2), p, 0, 0)
end

function _jacobi_B(p::Int, normalised::Bool, dtype::Type)
    function B_func(n, a, b)
        N_arr = collect(dtype.(0:(n - 1)))
        parity_vec = (-1.0) .^ N_arr

        A_func, _ = _jacobi_A(p, normalised, dtype)
        A_mat = sparse(A_func(n, b, a))

        a_rows = size(A_mat, 1)
        Pi_left_vec = (-1.0) .^ collect(dtype.(0:(a_rows - 1)))
        Pi_left = _spdiag(Pi_left_vec, 0, a_rows, a_rows)
        Pi_right = _spdiag(parity_vec, 0, max(n, 0), max(n, 0))

        result = Pi_left * A_mat * Pi_right
        return InfiniteCSC(result)
    end
    return B_func, JacobiCodomain(div(1 - p, 2), 0, p, 0)
end

function _jacobi_C(p::Int, normalised::Bool, dtype::Type)
    function C_func(n, a, b)
        N_arr = collect(dtype.(0:(n - 1)))
        band0 = p == 1 ? (N_arr .+ b) : (N_arr .+ a)

        if normalised
            nr = jacobi_norm_ratio(0, p, -p, N_arr, a, b)
            band0 .*= nr
        end

        _nan_to_zero!(band0)

        nrows = max(n, 0)
        ncols = max(n, 0)
        mat = _spdiag(band0, 0, nrows, ncols)
        return InfiniteCSC(mat)
    end
    return C_func, JacobiCodomain(0, p, -p, 0)
end

function _jacobi_D(p::Int, normalised::Bool, dtype::Type)
    function D_func(n, a, b)
        N_arr = collect(dtype.(0:(n - 1)))
        coeff = p == 1 ? (N_arr .+ (a + b + 1)) : (N_arr .+ 1)
        band0 = coeff .* 2.0^(-p)

        if normalised
            start_idx = div(1 + p, 2) + 1
            if start_idx >= 1 && start_idx <= n
                nr = jacobi_norm_ratio(-p, p, p, N_arr[start_idx:end], a, b)
                band0[start_idx:end] .*= nr
            end
        end

        _nan_to_zero!(band0)

        nrows = max(n - p, 0)
        ncols = max(n, 0)
        mat = _spdiag(band0, p, nrows, ncols)
        return InfiniteCSC(mat)
    end
    return D_func, JacobiCodomain(-p, p, p, 0)
end

# ============================================================================
# Static operator constructors: identity, parity, number
# ============================================================================

function _jacobi_identity(; dtype::Type=Float64)
    function I_func(n, a, b)
        N_vec = ones(dtype, max(n, 0))
        mat = _spdiag(N_vec, 0, max(n, 0), max(n, 0))
        return InfiniteCSC(mat)
    end
    return Operator(I_func, _to_codomain(JacobiCodomain(0, 0, 0, 0)); Output=Operator)
end

function _jacobi_parity(; dtype::Type=Float64)
    function P_func(n, a, b)
        N_arr = collect(dtype.(0:(n - 1)))
        N_vec = (-1.0) .^ N_arr
        mat = _spdiag(N_vec, 0, max(n, 0), max(n, 0))
        return InfiniteCSC(mat)
    end
    return Operator(P_func, _to_codomain(JacobiCodomain(0, 0, 0, 1)); Output=Operator)
end

function _jacobi_number(; dtype::Type=Float64)
    function N_func(n, a, b)
        N_vec = collect(dtype.(0:(n - 1)))
        mat = _spdiag(N_vec, 0, max(n, 0), max(n, 0))
        return InfiniteCSC(mat)
    end
    return Operator(N_func, _to_codomain(JacobiCodomain(0, 0, 0, 0)); Output=Operator)
end

# ============================================================================
# operator factory
# ============================================================================

"""
    jacobi_operator(name; normalised=true, dtype=Float64)

Interface to base JacobiOperator class.

Parameters:
- name: "A", "B", "C", "D", "Id", "Pi", "N", "Z" (Jacobi matrix)
- normalised: true -> unit-integral, false -> classical
- dtype: output dtype
"""
function jacobi_operator(name::String; normalised::Bool=true, dtype::Type=Float64)
    if name == "Id"
        return _jacobi_identity(; dtype=dtype)
    end
    if name == "Pi"
        return _jacobi_parity(; dtype=dtype)
    end
    if name == "N"
        return _jacobi_number(; dtype=dtype)
    end
    if name == "Z"
        A_jo = JacobiOperator("A"; normalised=normalised, dtype=dtype)
        B_jo = JacobiOperator("B"; normalised=normalised, dtype=dtype)
        return (compose(B_jo(-1), B_jo(1)) - compose(A_jo(-1), A_jo(1))) / 2
    end
    return JacobiOperator(name; normalised=normalised, dtype=dtype)
end

# ============================================================================
# _extract_z_bands: Extract dia_matrix band data from Z operator
# ============================================================================

function _extract_z_bands(n::Int, a, b; normalised::Bool=true, dtype::Type=Float64)
    Z_op = jacobi_operator("Z"; normalised=normalised, dtype=dtype)
    Z_icsc = Z_op(n + 1, a, b)
    Z_orig = Matrix(sparse(Z_icsc))
    Z_t = collect(transpose(Z_orig))

    nrows_t, ncols_t = size(Z_t)

    band_sub = zeros(Float64, ncols_t)
    band_main = zeros(Float64, ncols_t)
    band_sup = zeros(Float64, ncols_t)

    for j in 1:ncols_t
        if j + 1 >= 1 && j + 1 <= nrows_t
            band_sub[j] = Z_t[j + 1, j]
        end
        if j >= 1 && j <= nrows_t
            band_main[j] = Z_t[j, j]
        end
        if j - 1 >= 1 && j - 1 <= nrows_t
            band_sup[j] = Z_t[j - 1, j]
        end
    end

    has_main = any(x -> abs(x) > 1e-15, band_main)
    return band_sub, band_main, band_sup, has_main
end

# ============================================================================
# polynomials
# ============================================================================

"""
    jacobi_polynomials(n, a, b, z; init=nothing, Newton=false, normalised=true, dtype=Float64)

Jacobi polynomials P(n, a, b, z) of type (a, b) up to degree n-1.

Returns a matrix where row k contains P(k-1, a, b, z) evaluated at points z.

Newton=true: cubic-converging update of P(n-1, a, b, z) = 0.
"""
function jacobi_polynomials(n::Int, a, b, z;
                            init=nothing,
                            Newton::Bool=false,
                            normalised::Bool=true,
                            dtype::Type=Float64)

    z_arr = z isa AbstractVector ? Float64.(z) : Float64.([z])
    nz = length(z_arr)

    if n < 1
        return zeros(dtype, 0, nz)
    end

    if init === nothing
        init_vec = ones(Float64, nz)
        if normalised
            init_vec ./= sqrt(jacobi_mass(a, b))
        end
    else
        init_vec = init isa AbstractVector ? Float64.(init) : Float64.(init .+ 0 .* z_arr)
    end

    band_sub, band_main, band_sup, has_main = _extract_z_bands(n, a, b; normalised=normalised, dtype=Float64)

    P = zeros(Float64, n + 1, nz)
    P[1, :] .= init_vec

    if !has_main
        P[2, :] .= z_arr .* P[1, :] ./ band_sup[2]
        for k in 2:n
            P[k + 1, :] .= (z_arr .* P[k, :] .- band_sub[k - 1] .* P[k - 1, :]) ./ band_sup[k + 1]
        end
    else
        P[2, :] .= (z_arr .- band_main[1]) .* P[1, :] ./ band_sup[2]
        for k in 2:n
            P[k + 1, :] .= ((z_arr .- band_main[k]) .* P[k, :] .- band_sub[k - 1] .* P[k - 1, :]) ./ band_sup[k + 1]
        end
    end

    if Newton
        L = n + (a + b) / 2
        z_new = z_arr .+ (1 .- z_arr .^ 2) .* P[n, :] ./
                (L .* band_sup[n + 1] .* P[n + 1, :] .- (L - 1) .* band_sub[n - 1] .* P[n - 1, :])
        return z_new, P[1:n-1, :]
    end

    return dtype.(P[1:n, :])
end

# ============================================================================
# grid_guess
# ============================================================================

"""
    jacobi_grid_guess(n, a, b; dtype=Float64)

Approximate solution to P(n, a, b, z) = 0.
"""
function jacobi_grid_guess(n::Int, a, b; dtype::Type=Float64, quick::Bool=false)
    if a == b == -0.5
        quick = true
    end

    if n == 1
        Z_op = jacobi_operator("Z")
        Z_mat = Matrix(sparse(Z_op(n, a, b)))
        return dtype.([Z_mat[1, 1]])
    end

    if quick
        indices = collect(range(4 * n - 1, stop=3, step=-4))
        return dtype.(cos.(Float64(pi) .* (indices .+ 2 * a) ./ (4 * n + 2 * (a + b + 1))))
    end

    Z_op = jacobi_operator("Z")
    Z_mat = Matrix(sparse(Z_op(n, a, b)))
    Z_sq = Z_mat[1:n, 1:n]
    d = diag(Z_sq, 0)
    e = diag(Z_sq, 1)
    eigenvalues = eigvals(SymTridiagonal(d, e))
    return dtype.(eigenvalues)
end

# ============================================================================
# quadrature
# ============================================================================

"""
    jacobi_quadrature(n, a, b; days=3, probability=false, dtype=Float64)

Jacobi 'roots' grid and weights; solutions to P(n, a, b, z) = 0.

Returns (z, w) where z are the quadrature nodes and w are the weights.

sum(w .* f.(z)) approximates integral from -1 to 1 of (1-z)^a (1+z)^b f(z) dz,
exactly for degree(f) <= 2n - 1.
"""
function jacobi_quadrature(n::Int, a, b; days::Int=3, probability::Bool=false, dtype::Type=Float64)
    z = jacobi_grid_guess(n, a, b; dtype=Float64)

    if probability
        w_scale = 1.0
    else
        w_scale = jacobi_mass(a, b)
    end

    if a == b == -0.5
        return dtype.(z), dtype.(w_scale / n .+ 0 .* z)
    end

    local P
    if a == b == 0.5
        P = jacobi_polynomials(n + 1, a, b, z; dtype=Float64)[1:n, :]
    else
        for _ in 1:days
            z, P = jacobi_polynomials(n + 1, a, b, z; Newton=true)
        end
    end

    col_norms = sqrt.(sum(P .^ 2; dims=1))
    P[1, :] ./= vec(col_norms)
    w = w_scale .* P[1, :] .^ 2

    return dtype.(z), dtype.(w)
end

# ============================================================================
# coefficient_connection
# ============================================================================

"""
    jacobi_coefficient_connection(N, ab, cd; init_ab=1, init_cd=1)

The connection matrix between any bases coefficients:
    Pab(z) = Pcd(z) @ Cab2cd  -->  Acd = Cab2cd @ Aab

The output is always a dense matrix format.
"""
function jacobi_coefficient_connection(N::Int, ab::Tuple, cd::Tuple; init_ab=1, init_cd=1)
    a, b = ab
    c, d = cd

    zcd, wcd = jacobi_quadrature(N, c, d)

    wcd ./= sum(wcd)

    init_ab_vec = init_ab .+ 0 .* zcd
    init_cd_vec = init_cd .+ 0 .* zcd

    Pab = jacobi_polynomials(N, a, b, zcd; init=init_ab_vec)
    Pcd = jacobi_polynomials(N, c, d, zcd; init=init_cd_vec)

    return Pcd * (wcd .* Pab)'
end
