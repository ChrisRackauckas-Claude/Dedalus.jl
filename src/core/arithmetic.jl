"""
    Arithmetic operators for Dedalus.jl

Julia translation of `dedalus/core/arithmetic.py`. Provides lazy expression
tree node types for addition, multiplication (scalar-field and field-field),
dot product, and cross product operations.

## Type hierarchy

    AbstractOperand (abstract, from field.jl)
    └── AbstractFuture <: AbstractOperand (abstract, from future.jl)
        ├── Add (abstract)
        │   └── AddFields <: FutureField
        └── Product (abstract)
            ├── Multiply (abstract)
            │   ├── MultiplyFields <: FutureField
            │   └── MultiplyNumberField <: FutureField
            ├── DotProduct <: FutureField
            └── CrossProduct <: FutureField

## Key translation choices

- Python's `MultiClass` metaclass for `Add` and `Multiply` → Julia multiple
  dispatch via factory functions (`dedalus_add`, `dedalus_multiply`).
- Python `numexpr` → Julia `@.` broadcasting macro.
- Python `np.einsum` → manual contraction loops with `@views`.
- Python `isinstance` checks → Julia's type dispatch.
- Preprocessing (drop zeros, drop ones, cast) → performed in factory
  functions before struct construction.
"""

using SparseArrays
using LinearAlgebra

# ============================================================================
# Forward-reference stubs
# ============================================================================
#
# Types from field.jl, future.jl, domain.jl, basis.jl are assumed to be
# loaded before this file.  The `convert_operand` function from operators.jl
# may not yet be defined when this file loads; we define a stub that gets
# overwritten later.
#
# Required abstract types (from field.jl / future.jl):
#   AbstractOperand, AbstractCurrent, AbstractFuture, FutureField
# Required concrete types (from field.jl / domain.jl):
#   Field, Operand (alias for AbstractOperand), Domain
# Required functions (from future.jl):
#   evaluate, get_out, build_out, operate
# ============================================================================

"""
    convert_operand(arg, bases)

Convert an operand to the specified output bases.  This is a forward reference
to `operators.convert`; the real implementation is loaded from operators.jl.
The default stub simply returns the argument unchanged.
"""
function convert_operand end

# Default stub: no conversion
convert_operand(arg, bases) = arg

# Alphabet used for einsum string construction
const EINSUM_ALPHABET = "abcdefghijklmnopqrstuvwxy"

# ============================================================================
# Alias registry
# ============================================================================

"""
Global alias dictionary mapping string names to operator types/factories.
Populated by `@alias` calls and the alias registrations at the bottom of this
file.
"""
const ARITHMETIC_ALIASES = Dict{String, Any}()

"""
    register_alias!(name::String, op)

Register `op` under the given alias name.
"""
function register_alias!(name::String, op)
    ARITHMETIC_ALIASES[name] = op
    return op
end

# ============================================================================
# enum_indices
# ============================================================================

"""
    enum_indices(tensorsig)

Enumerate all multi-indices for a tensor with the given tensor signature.
Returns an iterator of `(flat_index, multi_index)` pairs where `flat_index`
is 1-based and `multi_index` is a tuple of 1-based component indices.

`tensorsig` is a tuple/vector of coordinate systems; each `cs` must support
`dim(cs)` returning the number of components.

# Examples
```julia
# For a rank-2 tensor in 3D:
for (i, idx) in enum_indices(tensorsig)
    # i ranges 1..9, idx is (1,1), (1,2), ..., (3,3)
end
```
"""
function enum_indices(tensorsig)
    if isempty(tensorsig)
        return [(1, ())]
    end
    shape = Tuple(cs_dim(cs) for cs in tensorsig)
    indices = CartesianIndices(shape)
    return [(LinearIndices(shape)[ci], Tuple(ci)) for ci in indices]
end

"""
    cs_dim(cs)

Return the dimension of a coordinate system. Dispatches to the `dim` field
or property of the coordinate system object.
"""
cs_dim(cs) = cs.dim

# ============================================================================
# Add — Abstract addition operator
# ============================================================================

"""
    Add

Abstract type for the addition operator in the expression tree. Concrete
subtypes handle specific operand combinations (e.g., `AddFields` for
field + field addition).
"""
abstract type Add <: AbstractFuture end

"""
    add_name(::Type{<:Add})

Return the string name for addition operators.
"""
add_name(::Type{<:Add}) = "Add"
operator_name(::Add) = "Add"

"""
    add_build_bases(args...)

Build output bases for an addition operation. For each coordinate, combines
the bases from all arguments.
"""
function add_build_bases(args...)
    dist = unify_attributes(args, :dist)
    bases = []
    bases_by_coord_first = args[1].domain.bases_by_coord
    for coord in keys(bases_by_coord_first)
        ax_bases = [get(arg.domain.bases_by_coord, coord, nothing) for arg in args]
        # All constant bases yields constant basis
        if all(b === nothing for b in ax_bases)
            push!(bases, nothing)
        # Combine any constant bases to avoid adding nothing to nothing
        elseif any(b === nothing for b in ax_bases)
            ax_bases_nonnull = [b for b in ax_bases if b !== nothing]
            push!(bases, reduce(basis_add, ax_bases_nonnull) + nothing)
        # Add all bases
        else
            push!(bases, reduce(basis_add, ax_bases))
        end
    end
    return Tuple(bases)
end

"""
    basis_add(a, b)

Add two bases together.  Mirrors Python's `np.sum(ax_bases)` which calls
`Basis.__add__`.  This is a placeholder that dispatches to the basis
addition protocol.
"""
basis_add(a, b) = a + b

"""
    Base.show(io::IO, op::Add)

Display an Add node as `arg1 + arg2 + ...`.
"""
function Base.show(io::IO, op::Add)
    str_args = [string(arg) for arg in op.args]
    print(io, join(str_args, " + "))
end

"""
    add_base(::Add)

Return the base Add type for reinstantiation.
"""
add_base(::Add) = Add

"""
    reinitialize(op::Add; kw...)

Reinitialize the Add with reinitialized arguments.
"""
function reinitialize(op::Add; kw...)
    arg0 = reinitialize(op.args[1]; kw...)
    arg1 = reinitialize(op.args[2]; kw...)
    return new_operands(op, arg0, arg1; kw...)
end

"""
    new_operands(op::Add, arg0, arg1; kw...)

Create a new Add with the given operands.
"""
function new_operands(op::Add, arg0, arg1; kw...)
    return dedalus_add(arg0, arg1; kw...)
end

"""
    split(op::Add, vars...)

Split into expressions containing and not containing specified operands/operators.
"""
function split(op::Add, vars...)
    # Sum over argument splittings
    split_results = [split(arg, vars...) for arg in op.args]
    # split_results is a list of (containing, not_containing) pairs
    containing = sum(s[1] for s in split_results)
    not_containing = sum(s[2] for s in split_results)
    return (containing, not_containing)
end

"""
    sym_diff(op::Add, var)

Symbolically differentiate with respect to specified operand.
"""
function sym_diff(op::Add, var)
    return sum(sym_diff(arg, var) for arg in op.args)
end

"""
    expand(op::Add, vars...)

Expand expression over specified variables.
"""
function expand(op::Add, vars...)
    if has(op, vars...)
        return sum(expand(arg, vars...) for arg in op.args)
    else
        return op
    end
end

"""
    require_linearity(op::Add, vars...; kw...)

Require expression to be linear in specified variables.
"""
function require_linearity(op::Add, vars...; kw...)
    for arg in op.args
        require_linearity(arg, vars...; kw...)
    end
end

"""
    require_first_order(op::Add, vars...; kw...)

Require expression to be maximally first order in specified operators.
"""
function require_first_order(op::Add, vars...; kw...)
    for arg in op.args
        require_first_order(arg, vars...; kw...)
    end
end

"""
    matrix_dependence(op::Add, vars...)

Determine dimension-by-dimension matrix dependence.
"""
function matrix_dependence(op::Add, vars...)
    deps = [matrix_dependence(arg, vars...) for arg in op.args]
    # Logical OR across all arguments, element-wise
    return reduce((a, b) -> a .| b, deps)
end

"""
    matrix_coupling(op::Add, vars...)

Determine dimension-by-dimension matrix coupling.
"""
function matrix_coupling(op::Add, vars...)
    couplings = [matrix_coupling(arg, vars...) for arg in op.args]
    return reduce((a, b) -> a .| b, couplings)
end

"""
    build_ncc_matrices(op::Add, separability, vars; kw...)

Precompute non-constant coefficients and build multiplication matrices.
"""
function build_ncc_matrices(op::Add, separability, vars; kw...)
    for arg in op.args
        build_ncc_matrices(arg, separability, vars; kw...)
    end
end

"""
    expression_matrices(op::Add, subproblem, vars; kw...)

Build expression matrices for a specific subproblem and variables.
"""
function expression_matrices(op::Add, subproblem, vars; kw...)
    # Intercept calls to compute matrices over expressions
    if op in vars
        size_val = field_size(subproblem, op)
        matrix = sparse(1.0I, size_val, size_val)
        return Dict(op => matrix)
    end
    matrices = Dict{Any, Any}()
    # Iteratively add argument expression matrices
    for arg in op.args
        arg_matrices = expression_matrices(arg, subproblem, vars; kw...)
        for var in keys(arg_matrices)
            if haskey(matrices, var)
                matrices[var] = matrices[var] + arg_matrices[var]
            else
                matrices[var] = arg_matrices[var]
            end
        end
    end
    return matrices
end

# ============================================================================
# AddFields — Concrete addition for field operands
# ============================================================================

"""
    AddFields

Addition operator for field operands. Both arguments must be Field or
FutureField instances.

Corresponds to Python's `AddFields(Add, FutureField)`.
"""
mutable struct AddFields <: Add
    args::Vector{Any}
    original_args::Vector{Any}
    out::Any
    dist::Any
    domain::Any
    tensorsig::Any
    dtype::DataType
    _bases::Tuple
    scales::Any
end

# Make AddFields also satisfy FutureField interface
is_future_field(::AddFields) = true

"""
    AddFields(args...; out=nothing, kw...)

Construct an AddFields node. Arguments are converted to the output bases
before storage.
"""
function AddFields(args...; out=nothing, kw...)
    _bases = add_build_bases(args...)
    # Convert arguments to output bases
    converted_args = [convert_operand(arg, _bases) for arg in args]
    arg_list = collect(Any, converted_args)
    original = copy(arg_list)
    dist = unify_attributes(converted_args, :dist)
    domain = Domain(dist, _bases)
    tensorsig = unify_attributes(converted_args, :tensorsig)
    dtype = promote_type([arg.dtype for arg in converted_args]...)
    return AddFields(arg_list, original, out, dist, domain, tensorsig, dtype, _bases, 1)
end

"""
    check_conditions(op::AddFields)

Check that arguments are in a proper layout.  All layouts must match.
"""
function check_conditions(op::AddFields)
    layouts = Set(arg.layout for arg in op.args)
    return length(layouts) == 1
end

"""
    enforce_conditions(op::AddFields)

Require arguments to be in a proper layout.
"""
function enforce_conditions(op::AddFields)
    layout = choose_layout(op)
    for arg in op.args
        change_layout(arg, layout)
    end
end

"""
    choose_layout(op::AddFields)

Determine the best target layout.  Currently picks the first argument's layout.
"""
function choose_layout(op::AddFields)
    return op.args[1].layout
end

"""
    operate(op::AddFields, out)

Perform the addition operation, writing results into `out`.
"""
function operate(op::AddFields, out)
    arg0, arg1 = op.args[1], op.args[2]
    # Set output layout
    preset_layout(out, arg0.layout)
    @. out.data = arg0.data + arg1.data
end

# ============================================================================
# Product — Abstract base for multiplication-like operations
# ============================================================================

"""
    Product

Abstract type for multiplication-like operations (scalar multiply, tensor
multiply, dot product, cross product).  Provides common expression tree
methods, NCC (non-constant coefficient) handling, and tensor contraction
helpers.
"""
abstract type Product <: AbstractFuture end

"""
    product_build_bases(arg0, arg1; ncc=false, ncc_vars=nothing, kw...)

Build output bases for a product operation.
"""
function product_build_bases(arg0, arg1; ncc::Bool=false, ncc_vars=nothing, kw...)
    bases = []
    arg0_bases = arg0.domain.bases_by_coord
    arg1_bases = arg1.domain.bases_by_coord
    for coord in keys(arg0_bases)
        b0 = arg0_bases[coord]
        b1 = arg1_bases[coord]
        # All constant bases yields constant basis
        if b0 === nothing && b1 === nothing
            continue
        # Multiply all bases
        elseif ncc && has(arg0, ncc_vars...)
            push!(bases, b1 * b0)  # matmul order: b1 @ b0
        elseif ncc && has(arg1, ncc_vars...)
            push!(bases, b0 * b1)  # matmul order: b0 @ b1
        else
            push!(bases, b0 * b1)
        end
    end
    return Tuple(bases)
end

"""
    reinitialize(op::Product; kw...)

Reinitialize the Product with reinitialized arguments.
"""
function reinitialize(op::Product; kw...)
    arg0 = reinitialize(op.args[1]; kw...)
    arg1 = reinitialize(op.args[2]; kw...)
    return new_operands(op, arg0, arg1; kw...)
end

"""
    new_operands(op::Product, args...; kw...)

Create a new Product node with the given operands. Must be overridden.
"""
function new_operands(op::Product, args...; kw...)
    error("$(typeof(op)) has not implemented new_operands method.")
end

"""
    split(op::Product, vars...)

Split into expressions containing and not containing specified operands/operators.
"""
function split(op::Product, vars...)
    # Take cartesian product of argument splittings
    split_args = []
    for arg in op.args
        if arg isa AbstractOperand
            push!(split_args, split(arg, vars...))
        else
            push!(split_args, (0, arg))
        end
    end
    # Cartesian product of splittings
    combos = [(a0, a1) for a0 in split_args[1] for a1 in split_args[2]]
    # Check if last combo should be dropped
    last_combo = combos[end]
    drop_last = !(last_combo[1] != 0 && last_combo[2] != 0)
    # Filter combos where both elements are nonzero
    filtered = [(a0, a1) for (a0, a1) in combos if (a0 != 0 && a1 != 0)]
    # Take product of each term
    split_ops = [new_operands(op, a0, a1) for (a0, a1) in filtered]
    # Append zero if last combo was dropped
    if drop_last
        push!(split_ops, 0)
    end
    # Last combo is all negative splittings, others contain at least one positive
    if length(split_ops) > 1
        return (sum(split_ops[1:end-1]), split_ops[end])
    else
        return (0, split_ops[1])
    end
end

"""
    sym_diff(op::Product, var)

Symbolically differentiate with respect to specified operand (product rule).
"""
function sym_diff(op::Product, var)
    args = op.args
    n = length(args)
    # Apply product rule to arguments
    terms = []
    for i in 1:n
        new_args = [j == i ? sym_diff(arg, var) : arg for (j, arg) in enumerate(args)]
        push!(terms, new_operands(op, new_args...))
    end
    return sum(terms)
end

"""
    expand(op::Product, vars...)

Expand expression over specified variables.
"""
function expand(op::Product, vars...)
    if has(op, vars...)
        # Expand arguments
        exp_args = [expand(arg, vars...) for arg in op.args]
        # Sum over cartesian product of sums including specified variables
        arg_sets = []
        for arg in exp_args
            if arg isa Add && has(arg, vars...)
                push!(arg_sets, arg.args)
            else
                push!(arg_sets, [arg])
            end
        end
        # Cartesian product
        return sum(new_operands(op, combo...) for combo in Iterators.product(arg_sets...))
    else
        return op
    end
end

"""
    require_linearity(op::Product, vars...; allow_affine=false, self_name=nothing,
                      vars_name=nothing, error_type=AssertionError, recurse=true)

Require expression to be linear in specified variables.
"""
function require_linearity(op::Product, vars...;
                           allow_affine::Bool=false,
                           self_name=nothing,
                           vars_name=nothing,
                           error_type::Type=ErrorException,
                           recurse::Bool=true)
    arg0, arg1 = op.args[1], op.args[2]
    op_arg0 = (arg0 isa AbstractOperand) && has(arg0, vars...)
    op_arg1 = (arg1 isa AbstractOperand) && has(arg1, vars...)
    if op_arg0 && op_arg1
        sn = self_name === nothing ? string(op) : self_name
        vn = vars_name === nothing ? [string(v) for v in vars] : vars_name
        throw(error_type("$sn is nonlinear in $vn."))
    elseif op_arg0 || op_arg1
        op_index = op_arg1 ? 2 : 1
        if recurse
            require_linearity(op.args[op_index], vars...;
                              allow_affine=allow_affine,
                              self_name=self_name,
                              vars_name=vars_name,
                              error_type=error_type)
        end
        return op_index
    elseif !allow_affine
        sn = self_name === nothing ? string(op) : self_name
        vn = vars_name === nothing ? [string(v) for v in vars] : vars_name
        throw(error_type("$sn must be strictly linear in $vn."))
    end
    return nothing
end

"""
    require_first_order(op::Product, args...; kw...)

Require expression to be maximally first order in specified operators.
"""
function require_first_order(op::Product, args...; kw...)
    for arg in op.args
        if arg isa AbstractOperand
            require_first_order(arg, args...; kw...)
        end
    end
end

"""
    prep_nccs(op::Product, vars)

Separate NCC factors. Identifies which argument contains the variables
and which is the NCC (non-constant coefficient).
"""
function prep_nccs(op::Product, vars)
    op._ncc_vars = vars
    op_index = require_linearity(op, vars...; recurse=false)
    op.ncc_first = (op_index == 2)
    op.operand = op.args[op_index]
    op.ncc = op.args[3 - op_index]  # Assumes 2 operands (1-based: other is 3-i)
    # Recurse
    prep_nccs(op.operand, vars)
end

"""
    gather_ncc_coeffs(op::Product)

Communicate NCC coeffs prior to matrix construction.
"""
function gather_ncc_coeffs(op::Product)
    # Recurse
    gather_ncc_coeffs(op.operand)
    # Evaluate NCC
    ncc = op.ncc
    if ncc isa AbstractFuture
        ncc = evaluate(ncc)
    end
    # Allgather NCC coefficients
    if ncc isa Field
        require_coeff_space(ncc)
        op._ncc_data = allgather_data(ncc)
    else
        op._ncc_data = ncc
    end
end

"""
    store_ncc_matrices(op::Product, vars, subproblems; kw...)

Store precomputed NCC matrices for all subproblems.
"""
function store_ncc_matrices(op::Product, vars, subproblems; kw...)
    prep_nccs(op, vars)
    gather_ncc_coeffs(op)
    op._ncc_matrices = Dict{Any, Any}()
    ncc_cutoff = get(kw, :ncc_cutoff, 1e-6)
    max_ncc_terms = get(kw, :max_ncc_terms, nothing)
    for subproblem in subproblems
        op._ncc_matrices[subproblem] = build_ncc_matrix(op, subproblem;
            ncc_cutoff=ncc_cutoff, max_ncc_terms=max_ncc_terms)
    end
end

"""
    evaluate_as_ncc(op::Product)

Evaluate using precomputed NCC matrices instead of point-wise multiplication.
"""
function evaluate_as_ncc(op::Product)
    op_result = evaluate(op.operand)
    out = get_out(op)
    out["c"] = 0
    for (subproblem, ncc_matrix) in op._ncc_matrices
        for subsystem in subproblem.subsystems
            op_ss = op_result["c"][field_slices(subsystem, op_result)...]
            out_ss = out["c"][field_slices(subsystem, out)...]
            out_ss[:] = reshape(ncc_matrix * vec(op_ss), size(out_ss))
        end
    end
    return out
end

"""
    build_ncc_matrix(op::Product, subproblem; ncc_cutoff=1e-6, max_ncc_terms=nothing)

Build NCC multiplication matrix for a given subproblem.
"""
function build_ncc_matrix(op::Product, subproblem; ncc_cutoff=1e-6, max_ncc_terms=nothing)
    if op.dist.single_coordsys !== nothing && !op.dist.single_coordsys.curvilinear
        return build_cartesian_ncc_matrix(op, subproblem; ncc_cutoff=ncc_cutoff, max_ncc_terms=max_ncc_terms)
    elseif length(op.domain.bases) > 0
        out_basis = op.domain.bases[end]
        return build_ncc_matrix(out_basis, op, subproblem, ncc_cutoff, max_ncc_terms)
    else
        return _last_axis_field_ncc_matrix(op, subproblem, 0, nothing, nothing, nothing,
            op._ncc_data, ncc_cutoff, max_ncc_terms)
    end
end

"""
    build_cartesian_ncc_matrix(op::Product, subproblem; ncc_cutoff=1e-6, max_ncc_terms=nothing)

Build NCC matrix for Cartesian coordinate systems (no intertwiners).
"""
function build_cartesian_ncc_matrix(op::Product, subproblem; ncc_cutoff=1e-6, max_ncc_terms=nothing)
    ncc = op.ncc
    arg = op.operand
    out = op
    # Build Gamma array
    if op.ncc_first
        G = GammaCoord(op, ncc.tensorsig, arg.tensorsig, out.tensorsig)
        G = permutedims(G, (3, 2, 1))
    else
        G = GammaCoord(op, arg.tensorsig, ncc.tensorsig, out.tensorsig)
        G = permutedims(G, (3, 1, 2))
    end
    # Loop over NCC modes
    out_size = field_size(subproblem, out)
    arg_size = field_size(subproblem, arg)
    shape = (out_size, arg_size)
    matrix = spzeros(op.dtype, shape...)
    subproblem_shape = coeff_shape(subproblem, out.domain)
    ncc_rank = length(ncc.tensorsig)
    if any(op._ncc_data .!= 0)
        ncc_data = op._ncc_data
        # Iterate over NCC mode indices (tensor components are first ncc_rank dims)
        ncc_shape = size(ncc_data)
        spatial_shape = ncc_shape[ncc_rank+1:end]
        for ncc_mode in CartesianIndices(spatial_shape)
            ncc_mode_tuple = Tuple(ncc_mode)
            # Extract tensor-valued coefficient at this mode
            select_idx = ntuple(i -> i <= ncc_rank ? Colon() : ncc_mode_tuple[i - ncc_rank], ndims(ncc_data))
            ncc_coeffs = ncc_data[select_idx...]
            if maximum(abs.(ncc_coeffs)) > ncc_cutoff
                mode_matrix = cartesian_mode_matrix(subproblem_shape,
                    ncc.domain, arg.domain, out.domain, ncc_mode_tuple)
                ncc_coeffs_flat = vec(ncc_coeffs)
                gamma_ncc = G * ncc_coeffs_flat  # matrix-vector product (contracts NCC index)
                # gamma_ncc is now shaped (out_dim, arg_dim), needs flattening for kron
                mode_matrix = kron(reshape(gamma_ncc, :, 1) * reshape(ones(Int, 1), 1, :),
                                   mode_matrix)
                # Actually: kron(Gamma . ncc_coeffs, mode_matrix)
                # G is (out_dim, arg_dim, ncc_dim), ncc_coeffs is (ncc_dim,)
                # Result should be (out_dim, arg_dim) matrix, then kron with mode_matrix
                G_contracted = reshape(G, :, size(G, 3)) * ncc_coeffs_flat
                G_matrix = reshape(G_contracted, size(G, 1), size(G, 2))
                mode_matrix = kron(G_matrix, mode_matrix)
                matrix = matrix + mode_matrix
            end
        end
    end
    return matrix
end

"""
    cartesian_mode_matrix(subproblem_shape, ncc_domain, arg_domain, out_domain, ncc_mode)

Build the spatial mode matrix for a single NCC mode in Cartesian coordinates.
"""
function cartesian_mode_matrix(subproblem_shape, ncc_domain, arg_domain, out_domain, ncc_mode)
    dim = out_domain.dist.dim
    matrix = nothing
    for axis in 1:dim
        ncc_basis = ncc_domain.full_bases[axis]
        arg_basis = arg_domain.full_bases[axis]
        out_basis = out_domain.full_bases[axis]
        if ncc_basis === nothing
            mode_mat = sparse(1.0I, subproblem_shape[axis], subproblem_shape[axis])
        else
            mode_mat = product_matrix(ncc_basis, arg_basis, out_basis, ncc_mode[axis])
        end
        if matrix === nothing
            matrix = mode_mat
        else
            matrix = kron(matrix, mode_mat)
        end
    end
    return matrix
end

"""
    expression_matrices(op::Product, subproblem, vars; kw...)

Build expression matrices for a specific subproblem and variables.
"""
function expression_matrices(op::Product, subproblem, vars; kw...)
    # Intercept calls to compute matrices over expressions
    if op in vars
        size_val = field_size(subproblem, op)
        matrix = sparse(1.0I, size_val, size_val)
        return Dict(op => matrix)
    end
    # Check vars vs. NCC prep
    if vars != op._ncc_vars
        throw(SymbolicParsingError("Must build NCC matrices with same variables."))
    end
    # Apply NCC matrix to operand matrices
    operand_mats = expression_matrices(op.operand, subproblem, vars; kw...)
    ncc_mat = build_ncc_matrix(op, subproblem; kw...)
    return Dict(var => ncc_mat * operand_mats[var] for var in keys(operand_mats))
end

"""
    matrix_dependence(op::Product, vars...)

Determine dimension-by-dimension matrix dependence for product operations.
"""
function matrix_dependence(op::Product, vars...)
    operand = op.operand
    operand_dep = matrix_dependence(operand, vars...)
    ncc_dep = operand.domain.mode_dependence
    return ncc_dep .| operand_dep
end

"""
    matrix_coupling(op::Product, vars...)

Determine dimension-by-dimension matrix coupling for product operations.
"""
function matrix_coupling(op::Product, vars...)
    operand = op.operand
    operand_coupling = matrix_coupling(operand, vars...)
    ncc = op.ncc
    ncc_coupling = ncc.domain.nonconstant
    return ncc_coupling .| operand_coupling
end

"""
    check_conditions(op::Product)

Check that arguments are in grid layout for pointwise operations.
"""
function check_conditions(op::Product)
    layout0 = op.args[1].layout
    layout1 = op.args[2].layout
    return all(layout0.grid_space) && (layout0 === layout1)
end

"""
    enforce_conditions(op::Product)

Require arguments to be in full grid space.
"""
function enforce_conditions(op::Product)
    for arg in op.args
        require_grid_space(arg)
    end
end

"""
    Gamma(op::Product, A_tensorsig, B_tensorsig, C_tensorsig,
          A_group, B_group, C_group, axis)

Compute Gamma(a,b,c) in components after intertwiners for specified axis.
Requires mode groups of previous axes, i.e. `length(group) == axis - 1`
(0-based in Python, 1-based axis here means axis-1 previous axes).
"""
function Gamma(op::Product, A_tensorsig, B_tensorsig, C_tensorsig,
               A_group, B_group, C_group, axis)
    # Base case
    if axis == 1
        return GammaCoord(op, A_tensorsig, B_tensorsig, C_tensorsig)
    end
    # Recurse
    G = Gamma(op, A_tensorsig, B_tensorsig, C_tensorsig,
              A_group, B_group, C_group, axis - 1)
    # Apply Q (intertwiner transforms)
    cs = get_coordsystem(op.dist, axis)
    cs_axis = get_axis(op.dist, cs)
    subaxis = axis - cs_axis
    QA = transpose(backward_intertwiner(cs, subaxis, length(A_tensorsig),
                                        A_group[cs_axis:end]))
    QB = transpose(backward_intertwiner(cs, subaxis, length(B_tensorsig),
                                        B_group[cs_axis:end]))
    QC = forward_intertwiner(cs, subaxis, length(C_tensorsig),
                             C_group[cs_axis:end])
    Q = kronecker(QA, QB, QC)
    G = reshape(Q * vec(G), size(G))
    return G
end

"""
    GammaCoord(op::Product, A_tensorsig, B_tensorsig, C_tensorsig)

Compute the tensor contraction Gamma array in coordinate components.
Must be implemented by each concrete Product subtype.
"""
function GammaCoord(op::Product, A_tensorsig, B_tensorsig, C_tensorsig)
    error("$(typeof(op)) has not implemented GammaCoord")
end

# ============================================================================
# DotProduct
# ============================================================================

"""
    DotProduct

Dot product (tensor contraction) of two fields along specified indices.
Contracts a pair of tensor indices, reducing the total rank by 2.

Registered as alias "dot".
"""
mutable struct DotProduct <: Product
    args::Vector{Any}
    original_args::Vector{Any}
    out::Any
    dist::Any
    domain::Any
    tensorsig::Any
    dtype::DataType
    indices::Tuple{Int, Int}
    gamma_args::Vector{Any}
    einsum_str::String
    arg0_ghost_broadcaster::Any
    arg1_ghost_broadcaster::Any
    ncc_method::String
    # NCC fields (lazily set)
    _ncc_vars::Any
    ncc_first::Bool
    operand::Any
    ncc::Any
    _ncc_data::Any
    _ncc_matrices::Any
    scales::Any

    function DotProduct(arg0, arg1; indices=(-1, 1), out=nothing, kw...)
        checked_indices = _check_dot_indices(arg0, arg1, indices)
        # Build output tensor signature
        arg0_ts = collect(arg0.tensorsig)
        arg0_ts_reduced = copy(arg0_ts)
        deleteat!(arg0_ts_reduced, checked_indices[1])
        arg1_ts = collect(arg1.tensorsig)
        arg1_ts_reduced = copy(arg1_ts)
        deleteat!(arg1_ts_reduced, checked_indices[2])
        tensorsig = Tuple(vcat(arg0_ts_reduced, arg1_ts_reduced))
        # FutureField requirements
        dist = unify_attributes((arg0, arg1), :dist)
        bases = product_build_bases(arg0, arg1; kw...)
        domain = Domain(dist, bases)
        dtype = promote_type(arg0.dtype, arg1.dtype)
        # Setup ghost broadcasting
        broadcast_dims = collect(domain.nonconstant)
        arg0_gb = GhostBroadcaster(arg0.domain, dist.grid_layout, broadcast_dims)
        arg1_gb = GhostBroadcaster(arg1.domain, dist.grid_layout, broadcast_dims)
        # Compose einsum string (for reference / potential use)
        rank0 = length(arg0.tensorsig)
        rank1 = length(arg1.tensorsig)
        arg1_str = EINSUM_ALPHABET[1:rank0]
        arg2_str = EINSUM_ALPHABET[rank0+1:rank0+rank1]
        # Replace contracted indices with 'z'
        arg1_chars = collect(arg1_str)
        arg1_chars[checked_indices[1]] = 'z'
        arg2_chars = collect(arg2_str)
        arg2_chars[checked_indices[2]] = 'z'
        arg1_str_mod = String(arg1_chars)
        arg2_str_mod = String(arg2_chars)
        out_str = replace(arg1_str_mod * arg2_str_mod, "z" => "")
        einsum_str = arg1_str_mod * "...," * arg2_str_mod * "...->" * out_str * "..."

        args_list = Any[arg0, arg1]
        new(args_list, copy(args_list), out, dist, domain, tensorsig, dtype,
            checked_indices, [checked_indices], einsum_str, arg0_gb, arg1_gb,
            "dot_product_ncc",
            nothing, false, nothing, nothing, nothing, nothing, 1)
    end
end

is_future_field(::DotProduct) = true
operator_name(::DotProduct) = "Dot"

# Register alias
register_alias!("dot", DotProduct)

"""
    _check_dot_indices(arg0, arg1, indices)

Validate and normalize dot product contraction indices (convert to 1-based positive).
"""
function _check_dot_indices(arg0, arg1, indices)
    if !(arg0 isa AbstractOperand) || !(arg1 isa AbstractOperand)
        throw(ArgumentError("Both arguments to DotProduct must be AbstractOperand"))
    end
    arg0_rank = length(arg0.tensorsig)
    arg1_rank = length(arg1.tensorsig)
    checked = collect(indices)
    ranks = (arg0_rank, arg1_rank)
    for i in 1:2
        idx = checked[i]
        rank = ranks[i]
        if idx > rank || idx < -rank
            throw(ArgumentError("index $idx out of range for field with rank $rank"))
        end
        if idx < 0
            idx += rank + 1  # Julia 1-based: -1 -> rank
        end
        checked[i] = idx
    end
    return Tuple(checked)
end

function Base.show(io::IO, op::DotProduct)
    function paren_str(arg)
        if arg isa Add
            return "(" * string(arg) * ")"
        else
            return string(arg)
        end
    end
    str_args = [paren_str(arg) for arg in op.args]
    print(io, join(str_args, "@"))
end

function new_operands(op::DotProduct, arg0, arg1; kw...)
    if arg0 == 0 || arg1 == 0
        return 0
    end
    return DotProduct(arg0, arg1; indices=op.indices, kw...)
end

function GammaCoord(op::DotProduct, A_tensorsig, B_tensorsig, C_tensorsig)
    A_dim = prod(cs_dim(cs) for cs in A_tensorsig; init=1)
    B_dim = prod(cs_dim(cs) for cs in B_tensorsig; init=1)
    C_dim = prod(cs_dim(cs) for cs in C_tensorsig; init=1)
    G = zeros(Int, A_dim, B_dim, C_dim)
    for (ia, a) in enum_indices(A_tensorsig)
        a_other = collect(a)
        a_dot = a_other[op.indices[1]]
        deleteat!(a_other, op.indices[1])
        for (ib, b) in enum_indices(B_tensorsig)
            b_other = collect(b)
            b_dot = b_other[op.indices[2]]
            deleteat!(b_other, op.indices[2])
            if a_dot == b_dot
                for (ic, c) in enum_indices(C_tensorsig)
                    if Tuple(vcat(a_other, b_other)) == c
                        G[ia, ib, ic] = 1
                    end
                end
            end
        end
    end
    return G
end

"""
    operate(op::DotProduct, out)

Perform the dot product operation using einsum-like contraction.
"""
function operate(op::DotProduct, out)
    arg0, arg1 = op.args[1], op.args[2]
    preset_layout(out, arg0.layout)
    # Broadcast
    arg0_data = ghost_cast(op.arg0_ghost_broadcaster, arg0)
    arg1_data = ghost_cast(op.arg1_ghost_broadcaster, arg1)
    # Perform contraction
    if length(out.data) > 0
        _einsum_contract!(out.data, arg0_data, arg1_data, op.indices,
                          length(arg0.tensorsig), length(arg1.tensorsig))
    end
end

"""
    _einsum_contract!(out_data, arg0_data, arg1_data, indices, rank0, rank1)

Perform tensor contraction (einsum-like) for dot product.
Contracts the `indices[1]`-th index of arg0 with the `indices[2]`-th index of arg1.
"""
function _einsum_contract!(out_data, arg0_data, arg1_data, indices, rank0, rank1)
    idx0 = indices[1]
    idx1 = indices[2]
    # Get the contraction dimension size
    contract_size = size(arg0_data, idx0)
    # Zero the output
    out_data .= 0
    # Sum over the contracted index
    for k in 1:contract_size
        # Build index tuples for arg0: all colons except idx0 = k
        a0_idx = ntuple(i -> i == idx0 ? k : Colon(), ndims(arg0_data))
        # Build index tuples for arg1: all colons except idx1+rank0-1 adjusted = k
        a1_idx = ntuple(i -> i == idx1 ? k : Colon(), ndims(arg1_data))
        # The output has rank0-1 + rank1-1 tensor indices plus spatial dims
        # We need to figure out the output slice
        # For simplicity, use views and broadcasting
        a0_slice = @view arg0_data[a0_idx...]
        a1_slice = @view arg1_data[a1_idx...]
        # The output tensor indices come from the remaining indices of arg0 then arg1
        # This requires reshaping for proper broadcasting
        # arg0 slice has shape: (dims except idx0-th)
        # arg1 slice has shape: (dims except idx1-th)
        out_rank = rank0 - 1 + rank1 - 1
        if out_rank == 0
            # Scalar output: just accumulate element-wise product
            @. out_data += a0_slice * a1_slice
        else
            # Need to expand dims for proper broadcasting
            # arg0 contributes (rank0-1) tensor dims, arg1 contributes (rank1-1) tensor dims
            a0_ndim = ndims(a0_slice)
            a1_ndim = ndims(a1_slice)
            spatial_dims = a0_ndim - (rank0 - 1)
            # Reshape arg0_slice: insert singleton dims for arg1's tensor positions
            a0_shape = size(a0_slice)
            a0_exp_shape = (a0_shape[1:rank0-1]..., ntuple(_ -> 1, rank1 - 1)..., a0_shape[rank0:end]...)
            # Reshape arg1_slice: insert singleton dims for arg0's tensor positions
            a1_shape = size(a1_slice)
            a1_exp_shape = (ntuple(_ -> 1, rank0 - 1)..., a1_shape[1:rank1-1]..., a1_shape[rank1:end]...)
            a0_exp = reshape(a0_slice, a0_exp_shape)
            a1_exp = reshape(a1_slice, a1_exp_shape)
            @. out_data += a0_exp * a1_exp
        end
    end
end

# ============================================================================
# CrossProduct
# ============================================================================

"""
    CrossProduct

Cross product of two 3D vector fields. Only supports rank-1 (vector) fields
with 3-component coordinate systems.

Registered as alias "cross".
"""
mutable struct CrossProduct <: Product
    args::Vector{Any}
    original_args::Vector{Any}
    out::Any
    dist::Any
    domain::Any
    tensorsig::Any
    dtype::DataType
    arg0_ghost_broadcaster::Any
    arg1_ghost_broadcaster::Any
    _operate_fn::Function  # either operate_right_handed or operate_left_handed
    # NCC fields (lazily set)
    _ncc_vars::Any
    ncc_first::Bool
    operand::Any
    ncc::Any
    _ncc_data::Any
    _ncc_matrices::Any
    scales::Any

    function CrossProduct(arg0, arg1; out=nothing, kw...)
        # Check that both fields are rank-1
        if length(arg0.tensorsig) != 1 || length(arg1.tensorsig) != 1
            throw(ErrorException("CrossProduct currently only implemented for vector fields."))
        end
        # Check that vector bundles are the same
        if arg0.tensorsig[1] !== arg1.tensorsig[1]
            throw(ArgumentError("CrossProduct requires identical vector bundles."))
        end
        # Check that vector bundles are 3D
        if cs_dim(arg0.tensorsig[1]) != 3
            throw(ArgumentError("CrossProduct requires 3-component vector fields."))
        end
        # FutureField requirements
        dist = unify_attributes((arg0, arg1), :dist)
        bases = product_build_bases(arg0, arg1; kw...)
        domain = Domain(dist, bases)
        tensorsig = arg0.tensorsig
        dtype = promote_type(arg0.dtype, arg1.dtype)
        # Setup ghost broadcasting
        broadcast_dims = collect(domain.nonconstant)
        arg0_gb = GhostBroadcaster(arg0.domain, dist.grid_layout, broadcast_dims)
        arg1_gb = GhostBroadcaster(arg1.domain, dist.grid_layout, broadcast_dims)
        # Pick operate method based on coordsys handedness
        if tensorsig[1].right_handed
            _operate = _operate_right_handed
        else
            _operate = _operate_left_handed
        end
        args_list = Any[arg0, arg1]
        new(args_list, copy(args_list), out, dist, domain, tensorsig, dtype,
            arg0_gb, arg1_gb, _operate,
            nothing, false, nothing, nothing, nothing, nothing, 1)
    end
end

is_future_field(::CrossProduct) = true
operator_name(::CrossProduct) = "Cross"

# Register alias
register_alias!("cross", CrossProduct)

"""
    _operate_right_handed(op::CrossProduct, out, arg0_data, arg1_data)

Cross product for right-handed coordinate systems.
Uses Julia broadcasting (@.) instead of Python's numexpr.
"""
function _operate_right_handed(op::CrossProduct, out, arg0_data, arg1_data)
    # Component indices are 1-based
    d00 = selectdim(arg0_data, 1, 1)
    d01 = selectdim(arg0_data, 1, 2)
    d02 = selectdim(arg0_data, 1, 3)
    d10 = selectdim(arg1_data, 1, 1)
    d11 = selectdim(arg1_data, 1, 2)
    d12 = selectdim(arg1_data, 1, 3)
    out0 = selectdim(out.data, 1, 1)
    out1 = selectdim(out.data, 1, 2)
    out2 = selectdim(out.data, 1, 3)
    @. out0 = d01 * d12 - d02 * d11
    @. out1 = d02 * d10 - d00 * d12
    @. out2 = d00 * d11 - d01 * d10
end

"""
    _operate_left_handed(op::CrossProduct, out, arg0_data, arg1_data)

Cross product for left-handed coordinate systems.
"""
function _operate_left_handed(op::CrossProduct, out, arg0_data, arg1_data)
    d00 = selectdim(arg0_data, 1, 1)
    d01 = selectdim(arg0_data, 1, 2)
    d02 = selectdim(arg0_data, 1, 3)
    d10 = selectdim(arg1_data, 1, 1)
    d11 = selectdim(arg1_data, 1, 2)
    d12 = selectdim(arg1_data, 1, 3)
    out0 = selectdim(out.data, 1, 1)
    out1 = selectdim(out.data, 1, 2)
    out2 = selectdim(out.data, 1, 3)
    @. out0 = d02 * d11 - d01 * d12
    @. out1 = d00 * d12 - d02 * d10
    @. out2 = d01 * d10 - d00 * d11
end

"""
    operate(op::CrossProduct, out)

Perform the cross product operation.
"""
function operate(op::CrossProduct, out)
    arg0, arg1 = op.args[1], op.args[2]
    preset_layout(out, arg0.layout)
    arg0_data = ghost_cast(op.arg0_ghost_broadcaster, arg0)
    arg1_data = ghost_cast(op.arg1_ghost_broadcaster, arg1)
    op._operate_fn(op, out, arg0_data, arg1_data)
end

function new_operands(op::CrossProduct, arg0, arg1; kw...)
    if arg0 == 0 || arg1 == 0
        return 0
    end
    return CrossProduct(arg0, arg1; kw...)
end

function GammaCoord(op::CrossProduct, A_tensorsig, B_tensorsig, C_tensorsig)
    cs = A_tensorsig[1]
    G = zeros(Int, 3, 3, 3)
    # Levi-Civita symbol
    G[1, 2, 3] = 1; G[2, 3, 1] = 1; G[3, 1, 2] = 1
    G[1, 3, 2] = -1; G[3, 2, 1] = -1; G[2, 1, 3] = -1
    if !cs.right_handed
        G .*= -1
    end
    return G
end

# ============================================================================
# Multiply — Abstract multiplication operator
# ============================================================================

"""
    Multiply

Abstract type for multiplication operators. Concrete subtypes handle
field-field and number-field multiplication.
"""
abstract type Multiply <: Product end

operator_name(::Multiply) = "Mul"

function Base.show(io::IO, op::Multiply)
    function paren_str(arg)
        if arg isa Add
            return "(" * string(arg) * ")"
        else
            return string(arg)
        end
    end
    str_args = [paren_str(arg) for arg in op.args]
    print(io, join(str_args, "*"))
end

function new_operands(op::Multiply, arg0, arg1; kw...)
    return dedalus_multiply(arg0, arg1; kw...)
end

function GammaCoord(op::Multiply, A_tensorsig, B_tensorsig, C_tensorsig)
    A_dim = prod(cs_dim(cs) for cs in A_tensorsig; init=1)
    B_dim = prod(cs_dim(cs) for cs in B_tensorsig; init=1)
    C_dim = prod(cs_dim(cs) for cs in C_tensorsig; init=1)
    G = zeros(Int, A_dim, B_dim, C_dim)
    for (ia, a) in enum_indices(A_tensorsig)
        for (ib, b) in enum_indices(B_tensorsig)
            for (ic, c) in enum_indices(C_tensorsig)
                if (a..., b...) == c
                    G[ia, ib, ic] = 1
                end
            end
        end
    end
    return G
end

# ============================================================================
# MultiplyFields — Field-field multiplication
# ============================================================================

"""
    MultiplyFields

Multiplication operator for field-field products. Performs element-wise
multiplication with tensor broadcasting (outer product on tensor indices).
"""
mutable struct MultiplyFields <: Multiply
    args::Vector{Any}
    original_args::Vector{Any}
    out::Any
    dist::Any
    domain::Any
    tensorsig::Any
    dtype::DataType
    gamma_args::Vector{Any}
    arg0_ghost_broadcaster::Any
    arg1_ghost_broadcaster::Any
    arg0_exp_tshape::Tuple
    arg1_exp_tshape::Tuple
    ncc_method::String
    # NCC fields (lazily set)
    _ncc_vars::Any
    ncc_first::Bool
    operand::Any
    ncc::Any
    _ncc_data::Any
    _ncc_matrices::Any
    scales::Any

    function MultiplyFields(arg0, arg1; out=nothing, kw...)
        dist = unify_attributes((arg0, arg1), :dist)
        bases = product_build_bases(arg0, arg1; kw...)
        domain = Domain(dist, bases)
        tensorsig = (arg0.tensorsig..., arg1.tensorsig...)
        dtype = promote_type(arg0.dtype, arg1.dtype)
        # Setup ghost broadcasting
        broadcast_dims = collect(domain.nonconstant)
        arg0_gb = GhostBroadcaster(arg0.domain, dist.grid_layout, broadcast_dims)
        arg1_gb = GhostBroadcaster(arg1.domain, dist.grid_layout, broadcast_dims)
        # Compute expanded shapes for broadcasting data
        arg0_order = length(arg0.tensorsig)
        arg1_order = length(arg1.tensorsig)
        arg0_tshape = Tuple(cs_dim(cs) for cs in arg0.tensorsig)
        arg1_tshape = Tuple(cs_dim(cs) for cs in arg1.tensorsig)
        arg0_exp_tshape = (arg0_tshape..., ntuple(_ -> 1, arg1_order)...)
        arg1_exp_tshape = (ntuple(_ -> 1, arg0_order)..., arg1_tshape...)

        args_list = Any[arg0, arg1]
        new(args_list, copy(args_list), out, dist, domain, tensorsig, dtype,
            Any[], arg0_gb, arg1_gb, arg0_exp_tshape, arg1_exp_tshape,
            "tensor_product_ncc",
            nothing, false, nothing, nothing, nothing, nothing, 1)
    end
end

is_future_field(::MultiplyFields) = true

"""
    operate(op::MultiplyFields, out)

Perform field-field multiplication with tensor broadcasting.
"""
function operate(op::MultiplyFields, out)
    arg0, arg1 = op.args[1], op.args[2]
    # Set output layout
    preset_layout(out, arg0.layout)
    # Broadcast
    arg0_data = ghost_cast(op.arg0_ghost_broadcaster, arg0)
    arg1_data = ghost_cast(op.arg1_ghost_broadcaster, arg1)
    # Reshape arg data to broadcast properly for output tensorsig
    rank0 = length(arg0.tensorsig)
    rank1 = length(arg1.tensorsig)
    spatial_shape0 = size(arg0_data)[rank0+1:end]
    spatial_shape1 = size(arg1_data)[rank1+1:end]
    arg0_exp_data = reshape(arg0_data, op.arg0_exp_tshape..., spatial_shape0...)
    arg1_exp_data = reshape(arg1_data, op.arg1_exp_tshape..., spatial_shape1...)
    @. out.data = arg0_exp_data * arg1_exp_data
end

# ============================================================================
# GhostBroadcaster
# ============================================================================

"""
    GhostBroadcaster

Broadcasts field data over constant distributed dimensions for arithmetic
broadcasting. In serial mode, simply returns `field.data`. With MPI, uses
subcommunicator broadcasts to ensure all ranks have the necessary data.
"""
struct GhostBroadcaster
    domain::Any
    layout::Any
    broadcast_dims::Vector{Bool}
    deploy_dims::Vector{Bool}
    subcomm::Any  # MPI subcommunicator or nothing
    skip::Bool    # If true, just return field.data directly

    function GhostBroadcaster(domain, layout, broadcast_dims)
        broadcast_arr = collect(Bool, broadcast_dims)
        constant_arr = collect(Bool, domain.constant)
        # Determine deployment dimensions: broadcast AND constant
        deploy_dims_ext = broadcast_arr .& constant_arr
        # Filter to non-local (distributed) dimensions
        local_arr = collect(Bool, layout.local)
        deploy_dims = deploy_dims_ext[.!local_arr]
        # Build subcomm or skip casting
        if any(deploy_dims)
            # In MPI mode, create subcommunicator for broadcasting
            # For now, use domain.dist.comm_cart if available
            subcomm = nothing
            try
                remain_dims = Int.(deploy_dims)
                subcomm = domain.dist.comm_cart.Sub(remain_dims=remain_dims)
            catch
                # No MPI; fall through to skip
            end
            if subcomm !== nothing
                return new(domain, layout, broadcast_arr, deploy_dims, subcomm, false)
            end
        end
        # No deployment needed; skip casting
        return new(domain, layout, broadcast_arr, deploy_dims, nothing, true)
    end
end

"""
    ghost_cast(gb::GhostBroadcaster, field)

Return the field data, broadcasting over ghost dimensions if necessary.
In serial mode, simply returns `field.data`.
"""
function ghost_cast(gb::GhostBroadcaster, field)
    if gb.skip
        return field.data
    end
    # MPI broadcasting path
    shape = size(field.data)
    dtype = field.dtype
    # Get or create ghost data buffer
    if gb.subcomm.rank == 0
        ghost_data = field.data
    else
        ghost_data = zeros(dtype, shape...)
    end
    # Broadcast data from rank 0 of subcomm
    if length(ghost_data) > 0
        gb.subcomm.Bcast(ghost_data, root=0)
    end
    return ghost_data
end

# ============================================================================
# MultiplyNumberField — Number-field multiplication
# ============================================================================

"""
    MultiplyNumberField

Multiplication of a scalar number by a field. The number is always stored
as the first argument.
"""
mutable struct MultiplyNumberField <: Multiply
    args::Vector{Any}
    original_args::Vector{Any}
    out::Any
    dist::Any
    domain::Any
    tensorsig::Any
    dtype::DataType
    # NCC fields (lazily set)
    _ncc_vars::Any
    ncc_first::Bool
    operand::Any
    ncc::Any
    _ncc_data::Any
    _ncc_matrices::Any
    scales::Any

    function MultiplyNumberField(arg0, arg1; out=nothing, kw...)
        # Make number come first
        if arg1 isa Number
            arg0, arg1 = arg1, arg0
        end
        # arg0 is now the number, arg1 is the field
        domain = arg1.domain
        tensorsig = arg1.tensorsig
        dtype = promote_type(typeof(arg0), arg1.dtype)
        dist = arg1.dist
        args_list = Any[arg0, arg1]
        new(args_list, copy(args_list), out, dist, domain, tensorsig, dtype,
            nothing, false, nothing, nothing, nothing, nothing, 1)
    end
end

is_future_field(::MultiplyNumberField) = true

"""
    check_conditions(op::MultiplyNumberField)

Number-field multiplication works in any layout.
"""
check_conditions(op::MultiplyNumberField) = true

"""
    enforce_conditions(op::MultiplyNumberField)

No conditions to enforce for number-field multiplication.
"""
enforce_conditions(op::MultiplyNumberField) = nothing

"""
    operate(op::MultiplyNumberField, out)

Perform scalar multiplication.
"""
function operate(op::MultiplyNumberField, out)
    arg0, arg1 = op.args[1], op.args[2]
    # Set output layout
    preset_layout(out, arg1.layout)
    # Multiply argument data
    @. out.data = arg0 * arg1.data
end

"""
    matrix_dependence(op::MultiplyNumberField, vars...)

Matrix dependence for number-field multiply delegates to the field argument.
"""
function matrix_dependence(op::MultiplyNumberField, vars...)
    return matrix_dependence(op.args[2], vars...)
end

"""
    matrix_coupling(op::MultiplyNumberField, vars...)

Matrix coupling for number-field multiply delegates to the field argument.
"""
function matrix_coupling(op::MultiplyNumberField, vars...)
    return matrix_coupling(op.args[2], vars...)
end

"""
    expression_matrices(op::MultiplyNumberField, subproblem, vars; kw...)

Build expression matrices for number-field multiplication.
"""
function expression_matrices(op::MultiplyNumberField, subproblem, vars; kw...)
    # Intercept calls to compute matrices over expressions
    if op in vars
        size_val = field_size(subproblem, op)
        matrix = sparse(1.0I, size_val, size_val)
        return Dict(op => matrix)
    end
    arg0, arg1 = op.args[1], op.args[2]
    # Build field matrices
    arg1_mats = expression_matrices(arg1, subproblem, vars; kw...)
    # Multiply field matrices by scalar
    return Dict(var => arg0 * arg1_mats[var] for var in keys(arg1_mats))
end

"""
    build_ncc_matrices(op::MultiplyNumberField, separability, vars; kw...)

Precompute NCC matrices for number-field multiplication.
"""
function build_ncc_matrices(op::MultiplyNumberField, separability, vars; kw...)
    nccs, operand_idx = require_linearity(op, vars...)
    # Continue NCC matrix construction on the field operand
    build_ncc_matrices(op.args[2], separability, vars; kw...)
end

"""
    reinitialize(op::MultiplyNumberField; kw...)

Reinitialize: only reinitialize the field argument (arg1), not the number.
"""
function reinitialize(op::MultiplyNumberField; kw...)
    arg0 = op.args[1]
    arg1 = reinitialize(op.args[2]; kw...)
    return new_operands(op, arg0, arg1; kw...)
end

"""
    sym_diff(op::MultiplyNumberField, var)

Symbolically differentiate number * field: number * d(field)/d(var).
"""
function sym_diff(op::MultiplyNumberField, var)
    return op.args[1] * sym_diff(op.args[2], var)
end

# ============================================================================
# Factory functions (replacing Python MultiClass dispatch)
# ============================================================================

"""
    dedalus_add(args...; kw...)

Factory function for creating Add nodes. Replaces Python's MultiClass
dispatch on `Add`.

Preprocessing:
- Drops zero arguments
- Returns the single remaining argument if only one left
- Casts all arguments to Operands if any Operand is present
- Falls back to plain Julia summation if no Operands
"""
function dedalus_add(args...; kw...)
    # Drop zeros
    filtered = [arg for arg in args if arg != 0]
    # Return zero if all dropped
    if isempty(filtered)
        return 0
    end
    # Return single argument
    if length(filtered) == 1
        return filtered[1]
    end
    # Cast all args to Operands, if any present
    if any(arg isa AbstractOperand for arg in filtered)
        dist = unify_attributes(filtered, :dist; require=false)
        tensorsig = unify_attributes(filtered, :tensorsig; require=false)
        dtype = unify_attributes(filtered, :dtype; require=false)
        casted = [operand_cast(arg, dist, tensorsig, dtype) for arg in filtered]
        # Create AddFields (all should be field-like after casting)
        return AddFields(casted...; kw...)
    end
    # Use plain Julia summation
    return sum(filtered)
end

"""
    dedalus_multiply(args...; kw...)

Factory function for creating Multiply nodes. Replaces Python's MultiClass
dispatch on `Multiply`.

Preprocessing:
- Drops ones
- Returns zero for any zero argument
- Returns the single remaining argument if only one left
- Dispatches to MultiplyFields or MultiplyNumberField based on argument types
"""
function dedalus_multiply(args...; kw...)
    # Drop ones
    filtered = [arg for arg in args if arg != 1]
    # Return single argument
    if length(filtered) == 1
        return filtered[1]
    end
    # Return zero for any zero arguments
    if any(arg == 0 for arg in filtered)
        return 0
    end
    # No arguments left means all were ones
    if isempty(filtered)
        return 1
    end
    # Dispatch based on argument types
    arg0, arg1 = filtered[1], filtered[2]
    if arg0 isa Number && !(arg0 isa AbstractOperand)
        return MultiplyNumberField(arg0, arg1; kw...)
    elseif arg1 isa Number && !(arg1 isa AbstractOperand)
        return MultiplyNumberField(arg0, arg1; kw...)
    elseif (arg0 isa AbstractOperand) && (arg1 isa AbstractOperand)
        return MultiplyFields(arg0, arg1; kw...)
    else
        # Fall back to plain multiplication
        return prod(filtered)
    end
end

# ============================================================================
# Operand arithmetic operator overloads
# ============================================================================
#
# These methods define +, -, *, /, @  for Dedalus operands.
# They are meant to be imported by field.jl or the main module.
# We define them here on AbstractOperand so that the expression tree
# is built automatically.
#
# Note: These may conflict with existing definitions if field.jl also defines
# them. The pattern in Dedalus Python is that Operand.__add__ etc. call
# the arithmetic module's Add, Multiply, etc.
# ============================================================================

# operand_cast is defined in field.jl (loaded before this file)

# ============================================================================
# Exports
# ============================================================================

export Add,
       AddFields,
       Product,
       DotProduct,
       CrossProduct,
       Multiply,
       MultiplyFields,
       MultiplyNumberField,
       GhostBroadcaster,
       dedalus_add,
       dedalus_multiply,
       enum_indices,
       ARITHMETIC_ALIASES,
       register_alias!,
       convert_operand,
       operator_name,
       is_future_field,
       ghost_cast,
       cs_dim,
       operand_cast,
       cartesian_mode_matrix,
       build_cartesian_ncc_matrix
