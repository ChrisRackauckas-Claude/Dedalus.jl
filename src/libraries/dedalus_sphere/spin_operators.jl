# ============================================================================
# Constants
# ============================================================================

const SPIN_INDEXING = (-1, 0, 1)
const SPIN_THRESHOLD = 1e-12

# ============================================================================
# Helper: int2tuple for getindex
# ============================================================================
# In Python, the @int2tuple decorator converts integer arguments to singleton
# tuples.  In Julia we handle this via multiple dispatch on getindex.  When a
# user passes plain Ints we wrap them into tuples before forwarding.

"""
    _ensure_tuple(x)

Wrap a scalar `Int` in a 1-tuple; leave tuples untouched.
"""
_ensure_tuple(x::Int) = (x,)
_ensure_tuple(x::Tuple) = x

# ============================================================================
# TensorCodomain
# ============================================================================

"""
    TensorCodomain

Tracks rank changes produced by `TensorOperator` composition.

Wraps a `Codomain` internally (composition pattern instead of inheritance,
since Julia concrete structs cannot subtype one another).
"""
struct TensorCodomain
    _codomain::Codomain

    function TensorCodomain(rank_change::Int)
        new(Codomain(rank_change; Output=Codomain))
    end

    function TensorCodomain(cod::Codomain)
        new(cod)
    end
end

Base.getindex(tc::TensorCodomain, i::Int) = tc._codomain[i]
Base.getindex(tc::TensorCodomain, ::Colon) = tc._codomain[:]
Base.getindex(tc::TensorCodomain, r::UnitRange) = tc._codomain[r]
Base.length(tc::TensorCodomain) = length(tc._codomain)
Base.iterate(tc::TensorCodomain, state...) = iterate(tc._codomain, state...)

function Base.show(io::IO, tc::TensorCodomain)
    s = "(rank->rank+$(tc[1]))"
    s = replace(s, "+0" => "")
    s = replace(s, "+-" => "-")
    print(io, s)
end

function Base.:+(a::TensorCodomain, b::TensorCodomain)
    TensorCodomain(a._codomain + b._codomain)
end

function Base.:-(tc::TensorCodomain)
    TensorCodomain(-tc._codomain)
end

function Base.:-(a::TensorCodomain, b::TensorCodomain)
    a + (-b)
end

function Base.:(==)(a::TensorCodomain, b::TensorCodomain)
    a._codomain == b._codomain
end

Base.hash(tc::TensorCodomain, h::UInt) = hash(tc._codomain, h)

function Base.:|(a::TensorCodomain, b::TensorCodomain)
    TensorCodomain(a._codomain | b._codomain)
end

function (tc::TensorCodomain)(args...)
    tc._codomain(args...)
end

function Base.:*(tc::TensorCodomain, other::Int)
    TensorCodomain(tc._codomain * other)
end

Base.:*(other::Int, tc::TensorCodomain) = tc * other

# ============================================================================
# Abstract TensorOperator
# ============================================================================

"""
    AbstractTensorOperator

Abstract supertype for all tensor operators in spin/regularity space.

Concrete subtypes must implement:
- `tensor_getindex(op, sigma::Tuple, tau::Tuple)` — element access
- `get_codomain(op)` — returns a `TensorCodomain`
- `get_indexing(op)` — returns the indexing tuple (default `(-1,0,1)`)
- `get_threshold(op)` — returns the threshold (default `1e-12`)
"""
abstract type AbstractTensorOperator end

get_indexing(::AbstractTensorOperator) = SPIN_INDEXING
get_threshold(::AbstractTensorOperator) = SPIN_THRESHOLD

"""
    tensor_dimension(op)

Number of basis indices (length of the indexing tuple).
"""
tensor_dimension(op::AbstractTensorOperator) = length(get_indexing(op))

"""
    tensor_range(op, rank)

Generate all length-`rank` tuples from the indexing set (Cartesian product).
"""
function tensor_range(op::AbstractTensorOperator, rank::Int)
    if rank == 0
        return [()]
    end
    return vec(collect(Iterators.product(ntuple(_ -> get_indexing(op), rank)...)))
end

"""
    tensor_array(op, ranks)

Build the dense matrix representation of the tensor operator for the given
`(output_rank, input_rank)` pair.
"""
function tensor_array(op::AbstractTensorOperator, ranks)
    dims = tuple((tensor_dimension(op)^r for r in ranks)...)
    T = zeros(Float64, dims)
    idx = get_indexing(op)
    out_tuples = tensor_range(op, ranks[1])
    in_tuples  = tensor_range(op, ranks[2])
    for sigma in out_tuples
        for tau in in_tuples
            i = tuple2index(sigma, idx) + 1  # 1-based
            j = tuple2index(tau, idx) + 1    # 1-based
            T[i, j] = tensor_getindex(op, sigma, tau)
        end
    end
    return T
end

"""
    tensor_call(op, rank)

Evaluate the tensor operator — returns the dense array for the given rank.
Applies thresholding to send small values to zero.
"""
function tensor_call(op::AbstractTensorOperator, rank::Int)
    output = _tensor_eval(op, rank)
    # Apply threshold (matching Python: np.where(np.abs(output) < threshold, 0, output))
    thresh = get_threshold(op)
    output[abs.(output) .< thresh] .= 0.0
    return output
end

# Default getindex: dispatch on different argument forms
# Handles (op, (sigma, tau)), (op, Int, Int), (op, Tuple, Tuple), etc.

function Base.getindex(op::AbstractTensorOperator, sigma, tau)
    s = _ensure_tuple(sigma)
    t = _ensure_tuple(tau)
    return tensor_getindex(op, s, t)
end

function Base.getindex(op::AbstractTensorOperator, args::Tuple{Tuple, Tuple})
    return Base.getindex(op, args[1], args[2])
end

# ============================================================================
# TensorOperatorGeneric — wraps a function (used for composed/added operators)
# ============================================================================

"""
    TensorOperatorGeneric

Generic tensor operator wrapping a function (result of arithmetic operations).
"""
struct TensorOperatorGeneric <: AbstractTensorOperator
    _function::Function
    _codomain::TensorCodomain
    _indexing::Tuple
    _threshold::Float64
end

get_codomain(op::TensorOperatorGeneric) = op._codomain
get_indexing(op::TensorOperatorGeneric) = op._indexing
get_threshold(op::TensorOperatorGeneric) = op._threshold

function _tensor_eval(op::TensorOperatorGeneric, rank::Int)
    op._function(rank)
end

function tensor_getindex(op::TensorOperatorGeneric, sigma::Tuple, tau::Tuple)
    i = tuple2index(sigma, op._indexing)  # 0-based
    j = tuple2index(tau, op._indexing)    # 0-based
    return _tensor_eval(op, length(tau))[i+1, j+1]  # Julia 1-based array indexing
end

# ============================================================================
# TensorIdentity
# ============================================================================

"""
    TensorIdentity

Spin/regularity space identity transformation of arbitrary rank.

    op[sigma, tau] = 1 if sigma == tau else 0
"""
struct TensorIdentity <: AbstractTensorOperator
    _codomain::TensorCodomain
    _indexing::Tuple
    _threshold::Float64

    function TensorIdentity(; indexing=SPIN_INDEXING, threshold=SPIN_THRESHOLD)
        new(TensorCodomain(0), indexing, threshold)
    end
end

get_codomain(op::TensorIdentity) = op._codomain
get_indexing(op::TensorIdentity) = op._indexing
get_threshold(op::TensorIdentity) = op._threshold

function tensor_getindex(op::TensorIdentity, sigma::Tuple, tau::Tuple)
    return sigma == tau ? 1 : 0
end

function _tensor_eval(op::TensorIdentity, rank::Int)
    tensor_array(op, (rank, rank))
end

# ============================================================================
# Metric
# ============================================================================

"""
    Metric

Spin-space representation of the local Cartesian metric tensor.

    op[sigma, tau] = 1 if sigma == dual(tau) else 0

E.g.: Id = e(+)e(-) + e(0)e(0) + e(-)e(+)
"""
struct Metric <: AbstractTensorOperator
    _codomain::TensorCodomain
    _indexing::Tuple
    _threshold::Float64

    function Metric(; indexing=SPIN_INDEXING, threshold=SPIN_THRESHOLD)
        new(TensorCodomain(0), indexing, threshold)
    end
end

get_codomain(op::Metric) = op._codomain
get_indexing(op::Metric) = op._indexing
get_threshold(op::Metric) = op._threshold

function tensor_getindex(op::Metric, sigma::Tuple, tau::Tuple)
    return sigma == dual(tau) ? 1 : 0
end

function _tensor_eval(op::Metric, rank::Int)
    tensor_array(op, (rank, rank))
end

# ============================================================================
# TensorTranspose
# ============================================================================

"""
    TensorTranspose

Transpose operator for arbitrary rank tensors.

    T[i,j,...,k] -> T[permutation(i,j,...,k)]

Default transposes indices 0 <--> 1 (0-based).

The `permutation` tuple uses 0-based indexing (matching the Python convention
and the `apply` function from tuple_tools.jl).
"""
struct TensorTranspose <: AbstractTensorOperator
    _codomain::TensorCodomain
    _permutation::Tuple
    _indexing::Tuple
    _threshold::Float64

    function TensorTranspose(permutation::Tuple=(1, 0);
                             indexing=SPIN_INDEXING, threshold=SPIN_THRESHOLD)
        new(TensorCodomain(0), permutation, indexing, threshold)
    end
end

get_codomain(op::TensorTranspose) = op._codomain
get_indexing(op::TensorTranspose) = op._indexing
get_threshold(op::TensorTranspose) = op._threshold

"""
    get_permutation(op::TensorTranspose)

Return the permutation tuple (0-based indices).
"""
get_permutation(op::TensorTranspose) = op._permutation

function tensor_getindex(op::TensorTranspose, sigma::Tuple, tau::Tuple)
    return sigma == apply(op._permutation)(tau) ? 1 : 0
end

function _tensor_eval(op::TensorTranspose, rank::Int)
    tensor_array(op, (rank, rank))
end

# ============================================================================
# TensorTrace
# ============================================================================

"""
    TensorTrace

Contraction operator that sums over specified indices.

    sum_(i+j=0) T[..i,..j,..]

The `indices` tuple specifies which positions to trace over (0-based).
"""
struct TensorTrace <: AbstractTensorOperator
    _codomain::TensorCodomain
    _indices::Tuple
    _indexing::Tuple
    _threshold::Float64

    function TensorTrace(indices; indexing=SPIN_INDEXING, threshold=SPIN_THRESHOLD)
        if isa(indices, Int)
            indices = (indices,)
        end
        new(TensorCodomain(-length(indices)), indices, indexing, threshold)
    end
end

get_codomain(op::TensorTrace) = op._codomain
get_indexing(op::TensorTrace) = op._indexing
get_threshold(op::TensorTrace) = op._threshold

"""
    get_trace_indices(op::TensorTrace)

Return the indices being traced over (0-based).
"""
get_trace_indices(op::TensorTrace) = op._indices

function tensor_getindex(op::TensorTrace, sigma::Tuple, tau::Tuple)
    cond1 = sigma == remove(op._indices)(tau)
    cond2 = sum_(op._indices)(tau) == 0
    return (cond1 && cond2) ? 1 : 0
end

function _tensor_eval(op::TensorTrace, rank::Int)
    tensor_array(op, (rank - length(op._indices), rank))
end

# ============================================================================
# TensorProduct
# ============================================================================

"""
    TensorProduct

Multiplication by a single spin-tensor basis element.

    e(kappa) ⊗ T  (action='left')  : sigma == (kappa..., tau...)
    T ⊗ e(kappa)  (action='right') : sigma == (tau..., kappa...)

The `element` is a tuple of spin indices, `action` is `"left"` or `"right"`.
"""
struct TensorProduct <: AbstractTensorOperator
    _codomain::TensorCodomain
    _element::Tuple
    _action::String
    _indexing::Tuple
    _threshold::Float64

    function TensorProduct(element, action::String="left";
                           indexing=SPIN_INDEXING, threshold=SPIN_THRESHOLD)
        if isa(element, Int)
            element = (element,)
        end
        new(TensorCodomain(length(element)), element, action, indexing, threshold)
    end
end

get_codomain(op::TensorProduct) = op._codomain
get_indexing(op::TensorProduct) = op._indexing
get_threshold(op::TensorProduct) = op._threshold

"""
    get_element(op::TensorProduct)

Return the basis element tuple.
"""
get_element(op::TensorProduct) = op._element

"""
    get_action(op::TensorProduct)

Return the action direction ("left" or "right").
"""
get_action(op::TensorProduct) = op._action

function tensor_getindex(op::TensorProduct, sigma::Tuple, tau::Tuple)
    if op._action == "left"
        return sigma == (op._element..., tau...) ? 1 : 0
    elseif op._action == "right"
        return sigma == (tau..., op._element...) ? 1 : 0
    end
end

function _tensor_eval(op::TensorProduct, rank::Int)
    tensor_array(op, (rank + length(op._element), rank))
end

# ============================================================================
# xi — normalised derivative scale factor
# ============================================================================

"""
    xi(mu, ell)

Normalised derivative scale factors.

    xi(-1, ell)^2 + xi(+1, ell)^2 = 1
    xi(0, ell) = 0

Parameters
----------
- `mu`  : regularity index; -1, +1, or 0.
- `ell` : spherical-harmonic degree.
"""
function xi(mu, ell)
    return abs(mu) * sqrt((1 + mu / (2 * ell + 1)) / 2)
end

# ============================================================================
# Intertwiner
# ============================================================================

"""
    Intertwiner

Regularity-to-spin map: Q(ell)[spin, regularity].

Recursively constructs coupling coefficients between spin and regularity
representations at spherical-harmonic degree `L`.

Fields
------
- `L` : spherical-harmonic degree
"""
struct Intertwiner <: AbstractTensorOperator
    _codomain::TensorCodomain
    L::Int
    _indexing::Tuple
    _threshold::Float64

    function Intertwiner(L::Int; indexing=SPIN_INDEXING, threshold=SPIN_THRESHOLD)
        new(TensorCodomain(0), L, indexing, threshold)
    end
end

get_codomain(op::Intertwiner) = op._codomain
get_indexing(op::Intertwiner) = op._indexing
get_threshold(op::Intertwiner) = op._threshold

"""
    intertwiner_k(op::Intertwiner, mu, s)

Angular spherical wavenumber helper.
"""
function intertwiner_k(op::Intertwiner, mu, s)
    return -mu * sqrt((op.L - s * mu) * (op.L + s * mu + 1) / 2)
end

"""
    forbidden_spin(op::Intertwiner, spin::Tuple)

Returns `true` if the spin component is forbidden (does not exist for this L).
"""
function forbidden_spin(op::Intertwiner, spin)
    sp = _ensure_tuple(spin)
    return op.L < abs(sum(sp; init=0))
end

"""
    forbidden_regularity(op::Intertwiner, regularity::Tuple)

Returns `true` if the regularity component is forbidden for this L.
"""
function forbidden_regularity(op::Intertwiner, regularity)
    reg = _ensure_tuple(regularity)
    # Fast return for clearly allowed cases
    if op.L >= length(reg)
        return false
    end
    # Check other cases
    walk = [op.L]
    for r in reverse(reg)
        push!(walk, walk[end] + r)
        if walk[end] < 0 || (length(walk) >= 2 && walk[end-1] == 0 && walk[end] == 0)
            return true
        end
    end
    return false
end

function tensor_getindex(op::Intertwiner, sigma::Tuple, tau::Tuple)
    spin = sigma
    regularity = tau

    if length(spin) == 0
        return 1
    end

    if forbidden_spin(op, spin) || forbidden_regularity(op, regularity)
        return 0
    end

    s = spin[1]       # sigma
    a = regularity[1] # a

    # tau = spin[2:end], b = regularity[2:end]
    tau_rest = spin[2:end]
    b = regularity[2:end]

    R = 0.0
    for (i, t) in enumerate(tau_rest)
        # i is 1-based in Julia; replace_at expects 0-based j
        idx = i - 1  # 0-based index into tau_rest
        if t + s == 0
            R -= tensor_getindex(op, replace_at(idx, 0)(tau_rest), b)
        end
        if t == 0
            R += tensor_getindex(op, replace_at(idx, s)(tau_rest), b)
        end
    end

    Q = Float64(tensor_getindex(op, tau_rest, b))
    R -= intertwiner_k(op, s, sum(tau_rest; init=0)) * Q
    J = op.L + sum(b; init=0)

    if s != 0
        Q = 0.0
    end

    if a == -1
        return (Q * J - R) / sqrt(J * (2 * J + 1))
    elseif a == 0
        return s * R / sqrt(J * (J + 1))
    elseif a == 1
        return (Q * (J + 1) + R) / sqrt((J + 1) * (2 * J + 1))
    end
end

function _tensor_eval(op::Intertwiner, rank::Int)
    tensor_array(op, (rank, rank))
end

# ============================================================================
# Arithmetic on AbstractTensorOperator
# ============================================================================

# Compose: A @ B  (Python __matmul__)
# (A ∘ B)[sigma, tau] = sum_rho A[sigma, rho] * B[rho, tau]
# As dense arrays: A(codomain(B)(rank)) * B(rank)

function tensor_compose(a::AbstractTensorOperator, b::AbstractTensorOperator)
    cod = get_codomain(a) + get_codomain(b)
    idx = get_indexing(a)
    thresh = get_threshold(a)
    function func(rank::Int)
        b_arr = _tensor_eval(b, rank)
        # codomain of b tells us the rank change
        a_rank = rank + get_codomain(b)[1]
        a_arr = _tensor_eval(a, a_rank)
        # For the composed operator, a_arr is (dim^(a_rank + cod_a), dim^a_rank)
        # and b_arr is (dim^a_rank, dim^rank), so matrix multiply works
        return a_arr * b_arr
    end
    return TensorOperatorGeneric(func, cod, idx, thresh)
end

# Add: A + B
function Base.:+(a::AbstractTensorOperator, b::AbstractTensorOperator)
    cod = get_codomain(a) | get_codomain(b)
    idx = get_indexing(a)
    thresh = get_threshold(a)
    function func(rank::Int)
        return _tensor_eval(a, rank) .+ _tensor_eval(b, rank)
    end
    return TensorOperatorGeneric(func, cod, idx, thresh)
end

# Scalar multiply: c * A
function Base.:*(c::Number, op::AbstractTensorOperator)
    cod = get_codomain(op)
    idx = get_indexing(op)
    thresh = get_threshold(op)
    function func(rank::Int)
        return c .* _tensor_eval(op, rank)
    end
    return TensorOperatorGeneric(func, cod, idx, thresh)
end

Base.:*(op::AbstractTensorOperator, c::Number) = c * op
Base.:/(op::AbstractTensorOperator, c::Number) = (1 / c) * op

# Unary minus
Base.:-(op::AbstractTensorOperator) = (-1) * op

# Subtraction
function Base.:-(a::AbstractTensorOperator, b::AbstractTensorOperator)
    a + (-b)
end

# Add with zero
function Base.:+(op::AbstractTensorOperator, other::Number)
    if other == 0
        return op
    end
    error("Cannot add non-zero scalar to TensorOperator")
end

Base.:+(other::Number, op::AbstractTensorOperator) = op + other
Base.:-(op::AbstractTensorOperator, other::Number) = op + (-other)
Base.:-(other::Number, op::AbstractTensorOperator) = (-op) + other

# Compose operator: use the ∘ operator (matching Python @)
Base.:∘(a::AbstractTensorOperator, b::AbstractTensorOperator) = tensor_compose(a, b)

# Python uses * for the Lie bracket [A, B] = A∘B - B∘A on Operators,
# but for TensorOperators the Python code inherits this from Operator.
# We match that behavior here.
function tensor_bracket(a::AbstractTensorOperator, b::AbstractTensorOperator)
    return tensor_compose(a, b) - tensor_compose(b, a)
end

# Power: A^n
function Base.:^(op::AbstractTensorOperator, exponent::Int)
    if exponent < 0
        error("exponent must be a non-negative integer.")
    end
    if exponent == 0
        return TensorIdentity(indexing=get_indexing(op), threshold=get_threshold(op))
    end
    result = op
    for _ in 2:exponent
        result = tensor_compose(result, op)
    end
    return result
end

# Transpose (op_transpose equivalent for tensor operators)
function tensor_op_transpose(op::AbstractTensorOperator)
    cod = -get_codomain(op)
    idx = get_indexing(op)
    thresh = get_threshold(op)
    function func(rank::Int)
        # Evaluate at the transposed rank
        neg_cod = cod
        adjusted_rank = rank + neg_cod[1]
        arr = _tensor_eval(op, adjusted_rank)
        return collect(transpose(arr))
    end
    return TensorOperatorGeneric(func, cod, idx, thresh)
end
