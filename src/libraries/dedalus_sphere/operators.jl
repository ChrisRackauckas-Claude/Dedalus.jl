using SparseArrays

# ============================================================================
# Codomain
# ============================================================================

"""
    Codomain

Base class for codomain objects that track how operator domains transform
under composition. A codomain is an arrow between any given domain and the
resulting codomain.

The arrow is stored as a tuple of integers, and composition follows additive rules:
- `codomain(A) + codomain(B) = codomain(A∘B)`
- `-codomain(A)` gives the inverse codomain
- `codomain(A) | codomain(B)` checks compatibility
"""
struct Codomain
    arrow::Tuple{Vararg{Int}}
    Output::Type

    function Codomain(arrow::Tuple{Vararg{Int}}; Output::Type=Codomain)
        new(arrow, Output)
    end
end

Codomain(args::Int...; Output::Type=Codomain) = Codomain(args; Output=Output)

Base.getindex(c::Codomain, i::Int) = c.arrow[i]
Base.getindex(c::Codomain, ::Colon) = c.arrow
function Base.getindex(c::Codomain, r::UnitRange)
    c.arrow[r]
end

Base.length(c::Codomain) = length(c.arrow)
Base.show(io::IO, c::Codomain) = print(io, string(c.arrow))
Base.iterate(c::Codomain, state...) = iterate(c.arrow, state...)

function Base.:+(a::Codomain, b::Codomain)
    a.Output(tuple((x + y for (x, y) in zip(a[:], b[:]))...); Output=a.Output)
end

function (c::Codomain)(args...)
    tuple((a + b for (a, b) in zip(c.arrow, args))...)
end

function Base.:(==)(a::Codomain, b::Codomain)
    return a[:] == b[:]
end

Base.hash(c::Codomain, h::UInt) = hash(c.arrow, hash(:Codomain, h))

function Base.:|(a::Codomain, b::Codomain)
    if a != b
        throw(TypeError("operators have incompatible codomains."))
    end
    a.Output(tuple((x | y for (x, y) in zip(a[:], b[:]))...); Output=a.Output)
end

function Base.:-(c::Codomain)
    c.Output(tuple((-a for a in c.arrow)...); Output=c.Output)
end

function Base.:*(c::Codomain, other::Int)
    if other == 0
        return c.Output(tuple(zeros(Int, length(c.arrow))...); Output=c.Output)
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

Base.:*(other::Int, c::Codomain) = c * other

function Base.:-(a::Codomain, b::Codomain)
    a + (-b)
end

# ============================================================================
# Operator
# ============================================================================

"""
    Operator

Lazy matrix-valued function between parameterised vector spaces.

An `Operator` stores a function that, when called with domain parameters,
returns a sparse matrix. The `codomain` tracks how domain parameters
transform under composition.

# Composition rules
- `A ∘ B` (matmul): `(A ∘ B)(domain) = A(codomain(B)(domain)) * B(domain)`
- `A + B`: `(A + B)(domain) = A(domain) + B(domain)` (requires compatible codomains)
- `a * A`: scalar multiplication
- `A.T`: transpose operator with negated codomain
"""
mutable struct Operator
    _function::Function
    _codomain::Codomain
    _Output::Type

    function Operator(func::Function, codomain::Codomain; Output::Type=Operator)
        new(func, codomain, Output)
    end
end

get_function(op::Operator) = op._function
get_codomain(op::Operator) = op._codomain
get_output_type(op::Operator) = op._Output

function (op::Operator)(args...)
    op._function(args...)
end

function compose(a::Operator, b::Operator)
    function func(args...)
        b_codomain_args = get_codomain(b)(args...)
        return a(b_codomain_args...) * b(args...)
    end
    return get_output_type(a)(func, get_codomain(a) + get_codomain(b); Output=get_output_type(a))
end

function op_transpose(op::Operator)
    cod = -get_codomain(op)
    function func(args...)
        return sparse(transpose(op(cod(args...)...)))
    end
    return get_output_type(op)(func, cod; Output=get_output_type(op))
end

function op_identity(op::Operator)
    function func(args...)
        m = op(args...)
        n = size(m, 2)
        return InfiniteCSC(sparse(one(Float64) * I, n, n))
    end
    return get_output_type(op)(func, 0 * get_codomain(op); Output=get_output_type(op))
end

function Base.:^(op::Operator, exponent::Int)
    if exponent < 0
        throw(TypeError("exponent must be a non-negative integer."))
    end
    if exponent == 0
        return op_identity(op)
    end
    return compose(op, op^(exponent - 1))
end

function Base.:+(a::Operator, b::Operator)
    cod = get_codomain(a) | get_codomain(b)
    function func(args...)
        return a(args...) + b(args...)
    end
    return get_output_type(a)(func, cod; Output=get_output_type(a))
end

function Base.:+(op::Operator, other::Number)
    if other == 0
        return op
    end
    return op + other * op_identity(op)
end

Base.:+(other::Number, op::Operator) = op + other

function Base.:*(op::Operator, other::Operator)
    return compose(op, other) - compose(other, op)
end

function Base.:*(op::Operator, other::Number)
    function func(args...)
        return other * op(args...)
    end
    return get_output_type(op)(func, get_codomain(op); Output=get_output_type(op))
end

Base.:*(other::Number, op::Operator) = op * other

Base.:/(op::Operator, other::Number) = op * (1 / other)

Base.:+(op::Operator) = op

Base.:-(op::Operator) = (-1) * op

function Base.:-(a::Operator, b::Operator)
    a + (-b)
end

Base.:-(op::Operator, other::Number) = op + (-other)
Base.:-(other::Number, op::Operator) = (-op) + other

# ============================================================================
# InfiniteCSC - row/column-extendable sparse matrix
# ============================================================================

"""
    InfiniteCSC

Sparse matrix wrapper supporting row-extendable addition and zero-padded slicing.

If `A` has shape `(j, n)` and `B` has shape `(k, n)`, `A + B` pads the smaller
matrix with zero rows to match dimensions.
"""
struct InfiniteCSC{T} <: AbstractMatrix{T}
    data::SparseMatrixCSC{T, Int}
end

InfiniteCSC(m::AbstractMatrix{T}) where {T} = InfiniteCSC{T}(sparse(m))
InfiniteCSC(m::SparseMatrixCSC{T}) where {T} = InfiniteCSC{T}(m)

Base.size(m::InfiniteCSC) = size(m.data)
Base.size(m::InfiniteCSC, d::Int) = size(m.data, d)
Base.getindex(m::InfiniteCSC, i::Int, j::Int) = getindex(m.data, i, j)

function Base.show(io::IO, ::MIME"text/plain", m::InfiniteCSC)
    r, c = size(m)
    nnzs = nnz(m.data)
    print(io, "InfiniteCSC{$(eltype(m))} ($r × $c) with $nnzs stored elements")
end

SparseArrays.sparse(m::InfiniteCSC) = m.data
SparseArrays.nnz(m::InfiniteCSC) = nnz(m.data)

function Base.transpose(m::InfiniteCSC)
    InfiniteCSC(sparse(transpose(m.data)))
end

function inf_identity(m::InfiniteCSC{T}) where {T}
    n = size(m, 2)
    InfiniteCSC(sparse(one(T) * I, n, n))
end

function square(m::InfiniteCSC)
    n = size(m, 2)
    r = size(m, 1)
    if r >= n
        return m.data[1:n, :]
    end
    result = spzeros(eltype(m), n, n)
    result[1:r, :] = m.data
    return result
end

function _pad_rows(m::InfiniteCSC{T}, new_rows::Int) where {T}
    r, c = size(m)
    if new_rows <= r
        return m
    end
    result = spzeros(T, new_rows, c)
    result[1:r, :] = m.data
    return InfiniteCSC(result)
end

function Base.getindex(m::InfiniteCSC, rows::UnitRange, cols::Union{Colon, UnitRange})
    max_row = last(rows)
    r = size(m, 1)
    if max_row <= r
        return InfiniteCSC(m.data[rows, cols isa Colon ? (1:size(m, 2)) : cols])
    end
    padded = _pad_rows(m, max_row)
    return InfiniteCSC(padded.data[rows, cols isa Colon ? (1:size(padded, 2)) : cols])
end

function Base.getindex(m::InfiniteCSC, row::Int, ::Colon)
    r = size(m, 1)
    if row <= r
        return m.data[row, :]
    end
    return spzeros(eltype(m), size(m, 2))
end

function Base.:+(a::InfiniteCSC{T}, b::InfiniteCSC{T}) where {T}
    ns, no = size(a, 1), size(b, 1)
    nc = max(size(a, 2), size(b, 2))
    if ns == no
        return InfiniteCSC(a.data + b.data)
    end
    n = max(ns, no)
    result = spzeros(T, n, nc)
    result[1:ns, 1:size(a, 2)] += a.data
    result[1:no, 1:size(b, 2)] += b.data
    return InfiniteCSC(result)
end

function Base.:+(a::InfiniteCSC, b::AbstractMatrix)
    a + InfiniteCSC(b)
end

function Base.:+(a::AbstractMatrix, b::InfiniteCSC)
    InfiniteCSC(a) + b
end

function Base.:*(a::Number, m::InfiniteCSC)
    InfiniteCSC(a * m.data)
end

Base.:*(m::InfiniteCSC, a::Number) = a * m

function Base.:*(a::InfiniteCSC, b::InfiniteCSC)
    na_cols = size(a, 2)
    nb_rows = size(b, 1)
    if na_cols == nb_rows
        return InfiniteCSC(a.data * b.data)
    end
    padded_a = na_cols < nb_rows ? _pad_rows(InfiniteCSC(sparse(transpose(_pad_rows(InfiniteCSC(sparse(transpose(a.data))), nb_rows).data))), size(a, 1)) : a
    padded_b = nb_rows < na_cols ? _pad_rows(b, na_cols) : b
    InfiniteCSC(padded_a.data * padded_b.data)
end

function Base.:*(a::InfiniteCSC, b::AbstractMatrix)
    a * InfiniteCSC(b)
end

function Base.:*(a::AbstractMatrix, b::InfiniteCSC)
    InfiniteCSC(a) * b
end

# ============================================================================
# resize_matrix utility
# ============================================================================

"""
    resize_matrix(matrix, new_rows, new_cols)

Resize a sparse matrix, zero-padding or truncating as needed.
"""
function resize_matrix(matrix::AbstractMatrix, new_rows::Int, new_cols::Int)
    old_rows, old_cols = size(matrix)
    T = eltype(matrix)
    result = spzeros(T, new_rows, new_cols)
    rr = min(old_rows, new_rows)
    cc = min(old_cols, new_cols)
    result[1:rr, 1:cc] = matrix[1:rr, 1:cc]
    return result
end

function resize_matrix(matrix::InfiniteCSC, new_rows::Int, new_cols::Int)
    InfiniteCSC(resize_matrix(matrix.data, new_rows, new_cols))
end
