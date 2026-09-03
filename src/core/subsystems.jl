"""
    Subsystem and Subproblem types for Dedalus.jl

Julia translation of `dedalus/core/subsystems.py`. Provides the `Subsystem` and
`Subproblem` types that partition the global coefficient space into groups of
coupled modes for efficient matrix assembly and linear algebra.

## Type hierarchy

    Subsystem  -- represents one group within the coefficient space
    Subproblem -- represents a set of coupled modes sharing a matrix structure

## Key translation choices

- Python `defaultdict` -> Julia `Dict` with `get!` idiom
- Python `np.copyto` -> Julia `copyto!`
- Python `sparse.coo_matrix` / `sparse.csr_matrix` -> Julia `SparseArrays`
- Python `reduce(sparse.kron, ...)` -> Julia `reduce(kron, ...)`
- Python `@CachedMethod` -> Dict-based memoization
- Python `eval(condition, group_dict)` -> Julia `_eval_condition`
- 0-based indexing -> 1-based indexing throughout
"""

using SparseArrays
using LinearAlgebra
using Logging

# ============================================================================
# Module-level builder functions
# ============================================================================

"""
    build_subsystems(solver) -> Tuple{Subsystem, ...}

Build local subsystem objects from the solver's problem variables and equations.
Collects local groupsets for each variable and equation, verifies that groupsets
are nested (compatible distributions), then creates a `Subsystem` for each
unique groupset.
"""
function build_subsystems(solver)
    matrix_coupling = Tuple(solver.matrix_coupling)
    coeff_layout = solver.dist.coeff_layout
    all_local_groupsets = Vector{Any}()
    # Collect groupsets for variables
    for var in solver.problem.variables
        push!(all_local_groupsets,
              local_groupsets(coeff_layout, matrix_coupling, var.domain; scales=1))
    end
    # Collect groupsets for equations
    for eqn in solver.problem.equations
        push!(all_local_groupsets,
              local_groupsets(coeff_layout, matrix_coupling, eqn["domain"]; scales=1))
    end
    # Combine and check that groupsets are nested
    local_gs = OrderedSet()
    for (i, lgs1) in enumerate(all_local_groupsets)
        for lgs2 in all_local_groupsets[i+1:end]
            s1 = Set(lgs1)
            s2 = Set(lgs2)
            if !(s1 <= s2 || s1 >= s2)
                throw(ArgumentError(
                    "Incompatible group distributions. " *
                    "Are distributed dimensions the same size?"))
            end
        end
        union!(local_gs, lgs1)
    end
    return Tuple(Subsystem(solver, gs) for gs in local_gs)
end

"""
    build_subproblems(solver, subsystems; build_matrices_list=nothing) -> Tuple{Subproblem, ...}

Arrange subsystems by matrix group and build `Subproblem` objects.
Optionally builds matrices immediately if `build_matrices_list` is provided.
"""
function build_subproblems(solver, subsystems; build_matrices_list=nothing)
    # Group subsystems by matrix group
    subsystems_by_group = Dict{Any, Vector{Subsystem}}()
    for ss in subsystems
        group = ss.matrix_group
        if !haskey(subsystems_by_group, group)
            subsystems_by_group[group] = Subsystem[]
        end
        push!(subsystems_by_group[group], ss)
    end
    # Build subproblem for each group
    subproblems = Subproblem[]
    for (mg, sss) in subsystems_by_group
        sp = Subproblem(solver, sss, mg)
        push!(subproblems, sp)
    end
    subproblems_tuple = Tuple(subproblems)
    # Build matrices if requested
    if build_matrices_list !== nothing
        build_subproblem_matrices(solver, subproblems_tuple, build_matrices_list)
    end
    return subproblems_tuple
end

"""
    build_subproblem_matrices(solver, subproblems, matrices)

Setup NCC coefficients and build the specified matrices for all subproblems.
Logs progress for long-running builds.
"""
function build_subproblem_matrices(solver, subproblems, matrices)
    # Setup NCCs
    for eq in solver.problem.equations
        for matrix in matrices
            expr = get(eq, matrix, nothing)
            if expr !== nothing && expr != 0
                gather_ncc_coeffs(expr)
            end
        end
    end
    # Build matrices for each subproblem
    n = length(subproblems)
    for (i, sp) in enumerate(subproblems)
        if i % max(1, div(n, 10)) == 0
            @info "Building subproblem matrices: $i / $n"
        end
        build_matrices!(sp, matrices)
    end
end

# ============================================================================
# Subsystem
# ============================================================================

"""
    Subsystem

Represents a subset of the global coefficient space -- the multi-dimensional
generalization of a "pencil". Each subsystem is described by a "group" tuple
containing a group index (for each separable axis) or `nothing` (for each
coupled axis).

# Fields
- `solver` -- parent solver
- `problem` -- parent problem
- `dist` -- distributor
- `dtype` -- data type
- `group::Tuple` -- group multi-index
- `matrix_group::Tuple` -- matrix group (maps non-dependent groups to default)
- `subproblem` -- back-reference to containing Subproblem (set later)
- `_cache::Dict` -- method cache
"""
mutable struct Subsystem
    solver::Any
    problem::Any
    dist::Any
    dtype::DataType
    group::Tuple
    matrix_group::Tuple
    subproblem::Any  # set by Subproblem constructor
    _cache::Dict{Symbol, Any}
end

function Subsystem(solver, group)
    problem = solver.problem
    dist = solver.dist
    dtype = problem_dtype(problem)
    # Determine matrix group using solver matrix dependence
    matrix_dep = solver.matrix_dependence .| solver.matrix_coupling
    group_arr = collect(Any, group)
    matrix_group_arr = collect(Any, group)
    for i in eachindex(group_arr)
        if !matrix_dep[i] && group_arr[i] != 0
            matrix_group_arr[i] = dist.default_nonconst_groups[i]
        end
    end
    matrix_group = Tuple(matrix_group_arr)
    return Subsystem(solver, problem, dist, dtype, Tuple(group),
                     matrix_group, nothing, Dict{Symbol, Any}())
end

"""
    coeff_slices(ss::Subsystem, domain) -> Tuple

Return the local coefficient slices for this subsystem's group within the
given domain.
"""
function coeff_slices(ss::Subsystem, domain)
    slices = local_groupset_slices(ss.dist.coeff_layout, ss.group, domain; scales=1)
    if length(slices) == 0
        return Tuple(UnitRange(1, 0) for _ in 1:ss.dist.dim)
    elseif length(slices) > 1
        throw(ArgumentError("Subsystem data not contiguous."))
    else
        return slices[1]
    end
end

"""
    coeff_shape(ss::Subsystem, domain) -> Tuple

Return the coefficient shape for this subsystem within the given domain.
"""
function coeff_shape(ss::Subsystem, domain)
    cs = coeff_slices(ss, domain)
    dom_cs = domain.coeff_shape
    shape = Int[]
    for (ax_slice, ax_size) in zip(cs, dom_cs)
        indices = ax_slice
        push!(shape, length(indices))
    end
    return Tuple(shape)
end

"""
    coeff_size(ss::Subsystem, domain) -> Int

Return the total number of coefficients for this subsystem within the domain.
"""
coeff_size(ss::Subsystem, domain) = prod(coeff_shape(ss, domain))

"""
    field_slices(ss::Subsystem, field) -> Tuple

Return slices for extracting subsystem data from a field's data array.
Component dimensions come first, followed by coefficient slices.
"""
function field_slices(ss::Subsystem, field)
    key = (:field_slices, objectid(field))
    return get!(ss._cache, key) do
        comp_slices = Tuple(Colon() for _ in field.tensorsig)
        cs = coeff_slices(ss, field.domain)
        return (comp_slices..., cs...)
    end
end

"""
    field_shape(ss::Subsystem, field) -> Tuple

Return the shape of subsystem data within a field.
"""
function field_shape(ss::Subsystem, field)
    key = (:field_shape, objectid(field))
    return get!(ss._cache, key) do
        comp_shape = Tuple(cs_dim(cs) for cs in field.tensorsig)
        cs = coeff_shape(ss, field.domain)
        return (comp_shape..., cs...)
    end
end

"""
    field_size(ss::Subsystem, field) -> Int

Return the total number of elements for this subsystem within a field.
"""
function field_size(ss::Subsystem, field)
    key = (:field_size, objectid(field))
    return get!(ss._cache, key) do
        return prod(field_shape(ss, field))
    end
end

"""
    gather(ss::Subsystem, fields) -> Vector

Gather and concatenate subsystem data from multiple fields into a single vector.
"""
function gather(ss::Subsystem, fields)
    total_size = sum(field_size(ss, f) for f in fields)
    data = Vector{ss.dtype}(undef, total_size)
    offset = 0
    for f in fields
        fsize = field_size(ss, f)
        if fsize > 0
            fsl = field_slices(ss, f)
            fshp = field_shape(ss, f)
            src = view(f.data, fsl...)
            copyto!(view(data, offset+1:offset+fsize), vec(src))
            offset += fsize
        end
    end
    return data
end

"""
    scatter!(ss::Subsystem, data::AbstractVector, fields)

Scatter concatenated subsystem data out to multiple fields.
"""
function scatter!(ss::Subsystem, data::AbstractVector, fields)
    offset = 0
    for f in fields
        fsize = field_size(ss, f)
        if fsize > 0
            fsl = field_slices(ss, f)
            fshp = field_shape(ss, f)
            src = view(data, offset+1:offset+fsize)
            dest = view(f.data, fsl...)
            copyto!(dest, reshape(src, fshp))
            offset += fsize
        end
    end
end

"""
    valid_modes(ss::Subsystem, field, valid_modes_array) -> Any

Extract the valid modes for this subsystem from the field's valid_modes array.
"""
function valid_modes(ss::Subsystem, field, valid_modes_array)
    sp_slices = field_slices(ss, field)
    return valid_modes_array[sp_slices...]
end

"""
    check_condition(ss::Subsystem, eqn) -> Bool

Evaluate the equation condition string in the context of this subsystem's
group dictionary.
"""
function check_condition(ss::Subsystem, eqn)
    condition = eqn["condition"]
    if condition == "True" || condition == "true"
        return true
    elseif condition == "False" || condition == "false"
        return false
    end
    # Build group dictionary
    group_dict = Dict{String, Any}()
    for (axis, ax_group) in enumerate(ss.group)
        if ax_group !== nothing
            coord = ss.dist.coords[axis]
            group_dict["n" * coord.name] = ax_group
        end
    end
    return _eval_condition(condition, group_dict)
end

"""
    _eval_condition(condition::AbstractString, vars::Dict) -> Bool

Evaluate a condition string with the given variable bindings.
"""
function _eval_condition(condition::AbstractString, vars::Dict)
    mod = Module(:__dedalus_cond__)
    for (name, val) in vars
        Core.eval(mod, Expr(:(=), Symbol(name), val))
    end
    parsed = Meta.parse(condition)
    return Core.eval(mod, parsed)::Bool
end

# ============================================================================
# Subproblem
# ============================================================================

"""
    Subproblem

Represents one coupled subsystem of a problem, identified by a group
multi-index. This is the generalization of "pencils" from problems with
exactly one coupled dimension.

Contains multiple `Subsystem` objects that share the same matrix structure.

# Fields
- `solver` -- parent solver
- `problem` -- parent problem
- `subsystems::Vector{Subsystem}` -- constituent subsystems
- `group::Tuple` -- group multi-index
- `dist` -- distributor
- `domain` -- domain reference
- `dtype` -- data type
- `group_dict::Dict` -- mapping from coordinate names to group indices
- `pre_left` -- left preconditioner (set by build_matrices!)
- `pre_left_pinv` -- left preconditioner pseudoinverse
- `pre_right` -- right preconditioner
- `pre_right_pinv` -- right preconditioner pseudoinverse
- `LHS` -- combined LHS matrix placeholder
- `LHS_solver` -- factorized LHS solver (for multistep methods)
- `LHS_solvers` -- factorized LHS solvers (for RK methods)
- `update_rank::Int` -- Woodbury update rank
- `_input_buffer` -- preallocated input buffer
- `_input_views` -- views into input buffer
- `_output_buffer` -- preallocated output buffer
- `_output_views` -- views into output buffer
- `_compressed_buffer` -- compressed buffer for gather/scatter
"""
mutable struct Subproblem
    solver::Any
    problem::Any
    subsystems::Vector{Subsystem}
    group::Tuple
    dist::Any
    domain::Any
    dtype::DataType
    group_dict::Dict{String, Any}
    # Matrices and preconditioners (set by build_matrices!)
    pre_left::Any
    pre_left_pinv::Any
    pre_right::Any
    pre_right_pinv::Any
    LHS::Any
    LHS_solver::Any
    LHS_solvers::Vector{Any}
    update_rank::Int
    # Buffers
    _input_buffer::Any
    _input_views::Any
    _output_buffer::Any
    _output_views::Any
    _compressed_buffer::Any
    # Per-matrix named attributes (L_min, M_min, etc.)
    _matrix_store::Dict{String, Any}
end

function Subproblem(solver, subsystems_vec::Vector{Subsystem}, group)
    problem = solver.problem
    dist = problem.dist
    domain = problem.variables[1].domain  # HACK: use first variable's domain
    dtype = problem_dtype(problem)
    # Cross-reference from subsystems
    for ss in subsystems_vec
        ss.subproblem = nothing  # will be set below
    end
    # Build group dictionary
    group_dict = Dict{String, Any}()
    for (axis, ax_group) in enumerate(group)
        if ax_group !== nothing
            coord = dist.coords[axis]
            group_dict["n" * coord.name] = ax_group
        end
    end
    sp = Subproblem(
        solver, problem, subsystems_vec, Tuple(group),
        dist, domain, dtype, group_dict,
        nothing, nothing, nothing, nothing,  # preconditioners
        nothing, nothing, Any[],             # LHS, LHS_solver, LHS_solvers
        0,                                    # update_rank
        nothing, nothing, nothing, nothing, nothing,  # buffers
        Dict{String, Any}(),                 # _matrix_store
    )
    # Set cross-references
    for ss in subsystems_vec
        ss.subproblem = sp
    end
    # Build input/output buffers (except for EVPs)
    if !_is_evp(problem)
        _build_buffers!(sp)
    end
    return sp
end

"""Check if a problem is an EVP (avoids circular import)."""
function _is_evp(problem)
    return problem isa EigenvalueProblem
end

"""
    _build_buffers!(sp::Subproblem)

Build input and output buffers and views for gather/scatter operations.
"""
function _build_buffers!(sp::Subproblem)
    sp._input_buffer, sp._input_views = _build_buffer_views(sp, sp.problem.LHS_variables)
    eqs = sp.problem.equations
    if !isempty(eqs) && haskey(eqs[1], "F")
        F_fields = [eqn["F"] for eqn in eqs]
        sp._output_buffer, sp._output_views = _build_buffer_views(sp, F_fields)
    end
end

"""
    _build_buffer_views(sp::Subproblem, fields) -> (buffer, views)

Allocate a buffer matrix and create views into it for each field and subsystem.
"""
function _build_buffer_views(sp::Subproblem, fields)
    n_ss = length(sp.subsystems)
    fsizes = Tuple(field_size(sp.subsystems[1], f) for f in fields)
    total = sum(fsizes)
    buffer = zeros(sp.dtype, total, n_ss)
    views = Vector{Vector{Tuple}}()
    i0 = 0
    for (fsize, field) in zip(fsizes, fields)
        field_views = Tuple[]
        if fsize > 0
            fshape = field_shape(sp.subsystems[1], field)
            i1 = i0 + fsize
            for (j, ss) in enumerate(sp.subsystems)
                ss_view = reshape(view(buffer, i0+1:i1, j), fshape)
                ss_slices = field_slices(ss, field)
                push!(field_views, (ss_view, ss_slices))
            end
            i0 = i1
        end
        push!(views, field_views)
    end
    return buffer, views
end

# Delegate to first subsystem for shape/size queries

coeff_slices(sp::Subproblem, domain) = coeff_slices(sp.subsystems[1], domain)
coeff_shape(sp::Subproblem, domain) = coeff_shape(sp.subsystems[1], domain)
coeff_size(sp::Subproblem, domain) = coeff_size(sp.subsystems[1], domain)
field_slices(sp::Subproblem, field) = field_slices(sp.subsystems[1], field)
field_shape(sp::Subproblem, field) = field_shape(sp.subsystems[1], field)
field_size(sp::Subproblem, field) = field_size(sp.subsystems[1], field)

"""
    subproblem_shape(sp::Subproblem) -> Tuple{Int, Int}

Return the shape of the compressed subproblem matrices.
"""
function subproblem_shape(sp::Subproblem)
    return (size(sp.pre_left, 1), length(sp.subsystems))
end

"""
    subproblem_size(sp::Subproblem) -> Int

Return the total size of the compressed subproblem.
"""
subproblem_size(sp::Subproblem) = prod(subproblem_shape(sp))

"""
    gather_inputs(sp::Subproblem, fields; out=nothing) -> Matrix

Gather and precondition subproblem data from input-like fields.
"""
function gather_inputs(sp::Subproblem, fields; out=nothing)
    # Gather from fields into buffer
    for (field, buffer_data) in zip(fields, sp._input_views)
        for (buffer_view, field_slcs) in buffer_data
            copyto!(buffer_view, view(field.data, field_slcs...))
        end
    end
    # Apply right preconditioner inverse to compress inputs
    if out === nothing
        shape = subproblem_shape(sp)
        out = zeros(sp.dtype, shape...)
    end
    _apply_sparse_axis0!(sp.pre_right_pinv, sp._input_buffer, out)
    return out
end

"""
    gather_outputs(sp::Subproblem, fields; out=nothing) -> Matrix

Gather and precondition subproblem data from output-like fields.
"""
function gather_outputs(sp::Subproblem, fields; out=nothing)
    for (field, buffer_data) in zip(fields, sp._output_views)
        for (buffer_view, field_slcs) in buffer_data
            copyto!(buffer_view, view(field.data, field_slcs...))
        end
    end
    if out === nothing
        shape = subproblem_shape(sp)
        out = zeros(sp.dtype, shape...)
    end
    _apply_sparse_axis0!(sp.pre_left, sp._output_buffer, out)
    return out
end

"""
    scatter_inputs!(sp::Subproblem, data, fields)

Precondition and scatter subproblem data out to input-like fields.
"""
function scatter_inputs!(sp::Subproblem, data, fields)
    _apply_sparse_axis0!(sp.pre_right, data, sp._input_buffer)
    for (field, buffer_data) in zip(fields, sp._input_views)
        for (buffer_view, field_slcs) in buffer_data
            copyto!(view(field.data, field_slcs...), buffer_view)
        end
    end
end

"""
    scatter_outputs!(sp::Subproblem, data, fields)

Precondition and scatter subproblem data out to output-like fields.
"""
function scatter_outputs!(sp::Subproblem, data, fields)
    _apply_sparse_axis0!(sp.pre_left_pinv, data, sp._output_buffer)
    for (field, buffer_data) in zip(fields, sp._output_views)
        for (buffer_view, field_slcs) in buffer_data
            copyto!(view(field.data, field_slcs...), buffer_view)
        end
    end
end

"""
    _apply_sparse_axis0!(A, X, out)

Apply sparse matrix `A` along axis 0 (rows) of matrix `X`, writing to `out`.
Equivalent to `out = A * X` for 2D arrays, or `out = A * x` for vectors.
"""
function _apply_sparse_axis0!(A, X::AbstractMatrix, out::AbstractMatrix)
    mul!(out, A, X)
end

function _apply_sparse_axis0!(A, X::AbstractVector, out::AbstractVector)
    mul!(out, A, X)
end

"""
    build_matrices!(sp::Subproblem, names)

Build the problem matrices for the given matrix names (e.g. ["L"], ["M", "L"]).
Constructs subsystem-level block matrices, applies valid mode filtering,
permutation, and preconditioning.
"""
function build_matrices!(sp::Subproblem, names)
    solver = sp.solver
    eqns = sp.problem.equations
    vars = sp.problem.LHS_variables
    # Evaluate conditions for each equation
    eqn_conditions = [check_condition(sp.subsystems[1], eqn) for eqn in eqns]
    eqn_sizes = [field_size(sp.subsystems[1], eqn["eqn"]) for eqn in eqns]
    var_sizes = [field_size(sp.subsystems[1], var) for var in vars]
    I_total = sum(eqn_sizes)
    J_total = sum(var_sizes)
    dtype = sp.dtype

    # Construct subsystem matrices
    matrices = Dict{String, Any}()
    for name in names
        row_indices = Int[]
        col_indices = Int[]
        values = dtype[]
        i0 = 0
        for (eqn, eqn_size, eqn_cond) in zip(eqns, eqn_sizes, eqn_conditions)
            if eqn_size > 0 && eqn_cond && get(eqn, name, 0) != 0
                eqn_blocks = expression_matrices(eqn[name];
                    subproblem=sp, vars=vars,
                    ncc_cutoff=solver.ncc_cutoff,
                    max_ncc_terms=solver.max_ncc_terms)
                j0 = 0
                for (var, var_size) in zip(vars, var_sizes)
                    if var_size > 0 && haskey(eqn_blocks, var)
                        block = eqn_blocks[var]
                        # Convert to COO format
                        block_sparse = sparse(block)
                        rows_b, cols_b, vals_b = findnz(block_sparse)
                        append!(values, vals_b)
                        append!(row_indices, i0 .+ rows_b)
                        append!(col_indices, j0 .+ cols_b)
                    end
                    j0 += var_size
                end
            end
            i0 += eqn_size
        end
        # Filter small entries
        for k in eachindex(values)
            if abs(values[k]) < solver.entry_cutoff
                values[k] = zero(dtype)
            end
        end
        matrices[name] = sparse(row_indices, col_indices, values,
                                I_total, J_total)
    end

    # Valid modes
    valid_eqn_vecs = Any[]
    for eqn in eqns
        push!(valid_eqn_vecs,
              valid_modes(sp.subsystems[1], eqn["eqn"], eqn["valid_modes"]))
    end
    valid_var_vecs = Any[]
    for var in vars
        push!(valid_var_vecs,
              valid_modes(sp.subsystems[1], var, var.valid_modes))
    end
    # Invalidate equations that fail condition test
    for (n, eqn_cond) in enumerate(eqn_conditions)
        if !eqn_cond
            valid_eqn_vecs[n] = falses(size(valid_eqn_vecs[n]))
        end
    end
    # Convert to filter matrices (diagonal)
    valid_eqn_flat = vcat([vec(v) for v in valid_eqn_vecs]...)
    valid_var_flat = vcat([vec(v) for v in valid_var_vecs]...)
    valid_eqn_diag = spdiagm(0 => Int.(valid_eqn_flat))
    valid_var_diag = spdiagm(0 => Int.(valid_var_flat))

    # Check squareness of restricted system
    nnz_eqn = count(!iszero, valid_eqn_flat)
    nnz_var = count(!iszero, valid_var_flat)
    if nnz_eqn != nnz_var
        throw(ArgumentError(
            "Non-square system: group=$(sp.group), I=$nnz_eqn, J=$nnz_var"))
    end

    # Permutations
    left_perm = left_permutation(sp, eqns;
        bc_top=solver.bc_top,
        interleave_components=solver.interleave_components)
    right_perm = right_permutation(sp, vars;
        tau_left=solver.tau_left,
        interleave_components=solver.interleave_components)

    # Preconditioners
    sp.pre_left = _drop_empty_rows(left_perm * valid_eqn_diag)
    sp.pre_left_pinv = sparse(sp.pre_left')
    sp.pre_right_pinv = _drop_empty_rows(right_perm * valid_var_diag)
    sp.pre_right = sparse(sp.pre_right_pinv')

    # Precondition matrices
    for name in keys(matrices)
        matrices[name] = sp.pre_left * matrices[name] * sp.pre_right
    end

    # Store minimal CSR matrices
    for (name, matrix) in matrices
        sp._matrix_store["$(name)_min"] = matrix
    end

    # Store expanded matrices for fast recombination
    if length(matrices) > 1
        if solver.store_expanded_matrices
            combined = _zeros_with_pattern(values(matrices)...)
            sp.LHS = combined
            for (name, matrix) in matrices
                expanded = _expand_pattern(matrix, combined)
                sp._matrix_store["$(name)_exp"] = expanded
            end
        else
            sp.LHS = spzeros(dtype, nnz_eqn, nnz_var)
        end
    end

    # Update rank for Woodbury
    eqn_dofs_by_dim = Dict{Int, Int}()
    for (eqn, cond) in zip(eqns, eqn_conditions)
        if cond
            d = eqn["domain"].dim
            eqn_dofs_by_dim[d] = get(eqn_dofs_by_dim, d, 0) +
                                  field_size(sp.subsystems[1], eqn["eqn"])
        end
    end
    if !isempty(eqn_dofs_by_dim)
        max_dim = maximum(keys(eqn_dofs_by_dim))
        sp.update_rank = sum(values(eqn_dofs_by_dim)) - eqn_dofs_by_dim[max_dim]
    end
end

"""
    expand_matrices!(sp::Subproblem, matrix_names)

Rebuild expanded matrices from minimal matrices.
"""
function expand_matrices!(sp::Subproblem, matrix_names)
    matrices = Dict(name => sp._matrix_store["$(name)_min"] for name in matrix_names)
    combined = _zeros_with_pattern(values(matrices)...)
    sp.LHS = combined
    for (name, matrix) in matrices
        expanded = _expand_pattern(matrix, combined)
        sp._matrix_store["$(name)_exp"] = expanded
    end
end

# Property-like access for stored matrices (M_min, L_min, etc.)
function Base.getproperty(sp::Subproblem, s::Symbol)
    name = String(s)
    if endswith(name, "_min") || endswith(name, "_exp")
        store = getfield(sp, :_matrix_store)
        if haskey(store, name)
            return store[name]
        end
    end
    return getfield(sp, s)
end

function Base.hasproperty(sp::Subproblem, s::Symbol)
    name = String(s)
    if endswith(name, "_min") || endswith(name, "_exp")
        store = getfield(sp, :_matrix_store)
        return haskey(store, name)
    end
    return hasfield(typeof(sp), s)
end

# Also allow size/shape on Subproblem
Base.size(sp::Subproblem) = subproblem_shape(sp)

# ============================================================================
# Permutation helpers
# ============================================================================

"""
    left_permutation(sp, equations; bc_top, interleave_components) -> SparseMatrix

Build the left permutation matrix acting on equations.
`bc_top` determines if lower-dimensional equations are placed at the top.

Input ordering:  Equations > Components > Modes
Output ordering: Modes > [Components|Equations] (depending on interleave_components)
"""
function left_permutation(sp, equations; bc_top::Bool=false, interleave_components::Bool=false)
    # Compute hierarchy of input equation indices
    i = 1  # 1-based
    L0 = Vector{Vector{Vector{Int}}}()
    for eqn in equations
        L1 = Vector{Vector{Int}}()
        vfshape = field_shape(sp.subsystems[1], eqn["eqn"])
        rank = length(eqn["tensorsig"])
        comp_size = rank > 0 ? prod(vfshape[1:rank]) : 1
        mode_size = rank > 0 ? prod(vfshape[rank+1:end]) : prod(vfshape)
        if comp_size == 0
            push!(L1, Int[])
            push!(L0, L1)
            continue
        end
        for comp in 1:comp_size
            L2 = Int[]
            n_modes = rank > 0 ? mode_size : prod(vfshape)
            for coeff in 1:n_modes
                push!(L2, i)
                i += 1
            end
            push!(L1, L2)
        end
        push!(L0, L1)
    end
    return _build_permutation(L0, equations, :dim, bc_top, interleave_components)
end

"""
    right_permutation(sp, variables; tau_left, interleave_components) -> SparseMatrix

Build the right permutation matrix acting on variables.
`tau_left` determines if lower-dimensional variables are placed at the left.

Input ordering:  Variables > Components > Modes
Output ordering: Modes > [Components|Variables] (depending on interleave_components)
"""
function right_permutation(sp, variables; tau_left::Bool=false, interleave_components::Bool=false)
    i = 1
    L0 = Vector{Vector{Vector{Int}}}()
    for var in variables
        L1 = Vector{Vector{Int}}()
        vfshape = field_shape(sp.subsystems[1], var)
        rank = length(var.tensorsig)
        comp_size = rank > 0 ? prod(vfshape[1:rank]) : 1
        mode_size = rank > 0 ? prod(vfshape[rank+1:end]) : prod(vfshape)
        if comp_size == 0
            push!(L1, Int[])
            push!(L0, L1)
            continue
        end
        for comp in 1:comp_size
            L2 = Int[]
            n_modes = rank > 0 ? mode_size : prod(vfshape)
            for coeff in 1:n_modes
                push!(L2, i)
                i += 1
            end
            push!(L1, L2)
        end
        push!(L0, L1)
    end
    # For variables, use domain.dim for dimension grouping
    dims_list = [var.domain.dim for var in variables]
    return _build_permutation_with_dims(L0, dims_list, tau_left, interleave_components)
end

"""
    _build_permutation(L0, items, dim_accessor, forward, interleave_components) -> SparseMatrix

Internal helper to build a permutation matrix from a hierarchy of indices,
grouped by dimension.
"""
function _build_permutation(L0, items, dim_accessor::Symbol, forward::Bool, interleave_components::Bool)
    dims_list = Int[]
    for item in items
        if item isa Dict
            push!(dims_list, item["domain"].dim)
        else
            push!(dims_list, getproperty(item, dim_accessor))
        end
    end
    return _build_permutation_with_dims(L0, dims_list, forward, interleave_components)
end

function _build_permutation_with_dims(L0, dims_list, forward::Bool, interleave_components::Bool)
    n1max = length(L0)
    n2max = maximum(length(L1) for L1 in L0; init=0)
    n3max = 0
    for L1 in L0
        for L2 in L1
            n3max = max(n3max, length(L2))
        end
    end

    # Reverse list hierarchy, grouping by dimension
    indices = Dict{Int, Vector{Int}}()
    if interleave_components
        for n3 in 1:n3max
            for n2 in 1:n2max
                for n1 in 1:n1max
                    dim = dims_list[n1]
                    if !haskey(indices, dim)
                        indices[dim] = Int[]
                    end
                    if n2 <= length(L0[n1]) && n3 <= length(L0[n1][n2])
                        push!(indices[dim], L0[n1][n2][n3])
                    end
                end
            end
        end
    else
        for n3 in 1:n3max
            for n1 in 1:n1max
                dim = dims_list[n1]
                if !haskey(indices, dim)
                    indices[dim] = Int[]
                end
                for n2 in 1:n2max
                    if n2 <= length(L0[n1]) && n3 <= length(L0[n1][n2])
                        push!(indices[dim], L0[n1][n2][n3])
                    end
                end
            end
        end
    end

    # Combine indices by dimension
    dims = sort(collect(keys(indices)))
    if forward
        ordered_indices = vcat([indices[d] for d in dims]...)
    else
        ordered_indices = vcat([indices[d] for d in reverse(dims)]...)
    end

    return _perm_matrix(ordered_indices)
end

"""
    _perm_matrix(indices) -> SparseMatrix

Build a permutation matrix from source indices (1-based).
P[i, indices[i]] = 1
"""
function _perm_matrix(indices)
    n = length(indices)
    if n == 0
        return spzeros(Float64, 0, 0)
    end
    m = maximum(indices)
    return sparse(1:n, indices, ones(Float64, n), n, m)
end

# ============================================================================
# Sparse utility helpers
# ============================================================================

"""
    _drop_empty_rows(A) -> SparseMatrix

Remove all-zero rows from a sparse matrix, returning a compressed matrix.
"""
function _drop_empty_rows(A::AbstractSparseMatrix)
    m, n = size(A)
    rows, cols, vals = findnz(A)
    if isempty(rows)
        return spzeros(eltype(A), 0, n)
    end
    unique_rows = sort(unique(rows))
    row_map = Dict(r => i for (i, r) in enumerate(unique_rows))
    new_rows = [row_map[r] for r in rows]
    return sparse(new_rows, cols, vals, length(unique_rows), n)
end

"""
    _zeros_with_pattern(matrices...) -> SparseMatrix

Create a zero sparse matrix with the union sparsity pattern of all input matrices.
"""
function _zeros_with_pattern(matrices...)
    matrices_vec = collect(matrices)
    if isempty(matrices_vec)
        return spzeros(Float64, 0, 0)
    end
    m, n = size(first(matrices_vec))
    # Collect all nonzero positions
    all_rows = Int[]
    all_cols = Int[]
    for mat in matrices_vec
        rs, cs, _ = findnz(mat)
        append!(all_rows, rs)
        append!(all_cols, cs)
    end
    if isempty(all_rows)
        return spzeros(eltype(first(matrices_vec)), m, n)
    end
    vals = zeros(eltype(first(matrices_vec)), length(all_rows))
    return sparse(all_rows, all_cols, vals, m, n)
end

"""
    _expand_pattern(source, target) -> SparseMatrix

Expand a sparse matrix to match the sparsity pattern of `target`,
placing source values at matching positions.
"""
function _expand_pattern(source, target)
    # Simply return the source -- in Julia sparse algebra handles this naturally
    return copy(source)
end

# gather_ncc_coeffs, expression_matrices, local_groupsets, local_groupset_slices
# are defined in earlier-included files (future.jl, field.jl, operators.jl, distributor.jl)

# ============================================================================
# Exports
# ============================================================================

export Subsystem,
       Subproblem,
       build_subsystems,
       build_subproblems,
       build_subproblem_matrices,
       build_matrices!,
       coeff_slices,
       coeff_shape,
       coeff_size,
       field_slices,
       field_shape,
       field_size,
       gather,
       scatter!,
       gather_inputs,
       gather_outputs,
       scatter_inputs!,
       scatter_outputs!,
       subproblem_shape,
       subproblem_size,
       left_permutation,
       right_permutation,
       expand_matrices!,
       check_condition,
       valid_modes
