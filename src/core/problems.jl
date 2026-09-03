"""
    Problem types for Dedalus.jl

Julia translation of `dedalus/core/problems.py`. Provides abstract and concrete
problem types that represent systems of equations to be solved.

## Type hierarchy

    ProblemBase (abstract)
    +-- LinearBoundaryValueProblem (LBVP)
    +-- NonlinearBoundaryValueProblem (NLBVP)
    +-- InitialValueProblem (IVP)
    +-- EigenvalueProblem (EVP)

## Key translation choices

- Python `eval(expr_str, namespace)` for equation parsing -> Julia `Meta.parse`
  + `Core.eval` within a dynamically-built `Module`.
- Python `ChainMap` for namespace -> Julia Dict merge with priority ordering.
- Python class attribute `solver_class` -> Julia function `solver_class(::Type)`.
- Python `isinstance` checks -> Julia multiple dispatch.
"""

using SparseArrays
using LinearAlgebra
using Logging

# ============================================================================
# Parseable namespace (operators + arithmetic aliases)
# ============================================================================

"""
    PARSEABLE_NAMESPACE :: Dict{String, Any}

Global namespace mapping operator/arithmetic names to their Julia constructors,
used for parsing equation strings. Populated during module initialization from
the operator and arithmetic modules' exported names.
"""
const PARSEABLE_NAMESPACE = Dict{String, Any}()

"""
    populate_parseables!(operator_names, arithmetic_names)

Populate the global parseable namespace with operator and arithmetic names.
Called during module initialization once operator and arithmetic modules are loaded.
"""
function populate_parseables!(operator_names::Dict, arithmetic_names::Dict)
    merge!(PARSEABLE_NAMESPACE, operator_names)
    merge!(PARSEABLE_NAMESPACE, arithmetic_names)
end

# ============================================================================
# ProblemBase -- abstract base
# ============================================================================

"""
    ProblemBase

Abstract base type for all Dedalus problem types. Stores the problem variables,
equations, and a namespace for equation parsing.

Concrete subtypes must implement:
- `_check_equation_conditions(problem, eqn)` -- validate equation structure
- `_build_matrix_expressions(problem, eqn)` -- extract matrix operator expressions
- `solver_class(::Type{<:ProblemBase})` -- return the corresponding solver constructor
"""
abstract type ProblemBase end

# ---------------------------------------------------------------------------
# Concrete problem data storage (shared by all problem types)
# ---------------------------------------------------------------------------

"""
    ProblemData

Mutable container holding the common state for all problem types:
variables, equations, namespace, and distributor reference.
"""
mutable struct ProblemData
    variables::Vector{Any}
    LHS_variables::Vector{Any}
    dist::Any
    equations::Vector{Dict{String, Any}}
    local_namespace::Dict{String, Any}
    namespace::Dict{String, Any}
end

"""
    _make_problem_data(variables; namespace=nothing) -> ProblemData

Construct the shared `ProblemData` for a problem, setting up the namespace
chain.  Priority: local_namespace > external namespace > parseables.
"""
function _make_problem_data(variables; namespace=nothing)
    dist = unify_attributes(variables, :dist)
    local_ns = Dict{String, Any}()
    for var in variables
        name = _get_name(var)
        if name !== nothing && name != ""
            local_ns[name] = var
        end
    end
    # Build merged namespace (local has highest priority, then external, then parseables)
    merged = copy(PARSEABLE_NAMESPACE)
    if namespace !== nothing
        merge!(merged, namespace)
    end
    merge!(merged, local_ns)
    return ProblemData(
        collect(Any, variables),
        collect(Any, variables),
        dist,
        Dict{String, Any}[],
        local_ns,
        merged,
    )
end

"""
    _get_name(obj) -> Union{String, Nothing}

Safely extract the `name` field from an object, returning `nothing` if
the field does not exist or is not set.
"""
function _get_name(obj)
    try
        return obj.name
    catch
        return nothing
    end
end

# ============================================================================
# Helper: evaluate expression string in the problem namespace
# ============================================================================

"""
    _eval_in_namespace(expr_str::AbstractString, namespace::Dict) -> Any

Parse and evaluate a Julia expression string within the given namespace
dictionary.  Uses a temporary module to avoid polluting the global scope.
"""
function _eval_in_namespace(expr_str::AbstractString, namespace::Dict)
    mod = Module(:__dedalus_parse__)
    # Inject namespace bindings into the temporary module
    for (name, val) in namespace
        Core.eval(mod, Expr(:(=), Symbol(name), val))
    end
    parsed = Meta.parse(expr_str)
    return Core.eval(mod, parsed)
end

# ============================================================================
# Common problem interface methods
# ============================================================================

"""
    get_equations(problem::ProblemBase) -> Vector{Dict{String, Any}}

Return the list of equation dictionaries for the problem.
"""
get_equations(p::ProblemBase) = _data(p).equations

"""
    get_variables(problem::ProblemBase) -> Vector{Any}

Return the problem variables.
"""
get_variables(p::ProblemBase) = _data(p).variables

"""
    get_LHS_variables(problem::ProblemBase) -> Vector{Any}

Return the LHS variables (may differ from variables for NLBVP).
"""
get_LHS_variables(p::ProblemBase) = _data(p).LHS_variables

"""
    get_dist(problem::ProblemBase) -> Any

Return the distributor associated with the problem.
"""
get_dist(p::ProblemBase) = _data(p).dist

"""
    matrix_dependence(problem::ProblemBase) -> BitVector

Return the logical OR of matrix dependence across all equations.
"""
function problem_matrix_dependence(p::ProblemBase)
    eqs = get_equations(p)
    deps = [eqn["matrix_dependence"] for eqn in eqs]
    return reduce((a, b) -> a .| b, deps)
end

"""
    matrix_coupling(problem::ProblemBase) -> BitVector

Return the logical OR of matrix coupling across all equations.
"""
function problem_matrix_coupling(p::ProblemBase)
    eqs = get_equations(p)
    couplings = [eqn["matrix_coupling"] for eqn in eqs]
    return reduce((a, b) -> a .| b, couplings)
end

"""
    problem_dtype(problem::ProblemBase) -> DataType

Return the promoted data type across all equations.
"""
function problem_dtype(p::ProblemBase)
    eqs = get_equations(p)
    dtypes = [eqn["dtype"] for eqn in eqs]
    return reduce(promote_type, dtypes)
end

"""
    add_equation!(problem::ProblemBase, equation; condition="True")

Add an equation to the problem. The equation can be:
- A string of the form "LHS = RHS"
- A tuple `(LHS, RHS)` of operand expressions

Returns the equation dictionary.
"""
function add_equation!(problem::ProblemBase, equation; condition::String="True")
    data = _data(problem)
    @debug "Adding equation $(length(data.equations))"
    # Split equation into LHS and RHS
    if equation isa AbstractString
        namespace = copy(data.namespace)
        lhs_str, rhs_str = split_equation(equation)
        LHS = _eval_in_namespace(lhs_str, namespace)
        RHS = _eval_in_namespace(rhs_str, namespace)
    else
        # Tuple/pair of expressions
        LHS, RHS = equation
    end
    @debug "  LHS: $LHS"
    @debug "  RHS: $RHS"
    @debug "  condition: $condition"
    # Build basic equation dictionary
    expr = LHS - RHS
    eqn = Dict{String, Any}(
        "eqn" => expr,
        "LHS" => LHS,
        "RHS" => RHS,
        "condition" => condition,
        "tensorsig" => expr.tensorsig,
        "dtype" => expr.dtype,
        "valid_modes" => copy(expr.valid_modes),
    )
    _check_equation_conditions(problem, eqn)
    _build_matrix_expressions(problem, eqn)
    push!(data.equations, eqn)
    return eqn
end

"""
    build_solver(problem::ProblemBase, args...; kw...)

Construct the solver class corresponding to this problem type.
"""
function build_solver(problem::ProblemBase, args...; kw...)
    sc = solver_class(problem)
    return sc(problem, args...; kw...)
end

"""
    _check_domain_containment(subexpr, supexpr, subname, supname)

Verify that the domain of `subexpr` is contained within the domain of `supexpr`.
Throws `UnsupportedEquationError` if the sub-expression has a larger domain.
"""
function _check_domain_containment(subexpr, supexpr, subname, supname)
    sub_nc = subexpr.domain.nonconstant
    sup_const = supexpr.domain.constant
    if any(sub_nc .& sup_const)
        throw(UnsupportedEquationError(
            "$subname domain cannot be larger than $supname domain."))
    end
end

# ============================================================================
# LinearBoundaryValueProblem (LBVP)
# ============================================================================

"""
    LinearBoundaryValueProblem <: ProblemBase

Linear boundary value problem of the form:

    L . X = F

where L is linear in the problem variables X, and F is independent of X.

# Fields
- `data::ProblemData` -- shared problem data

# Construction
```julia
prob = LinearBoundaryValueProblem(variables; namespace=nothing)
```
"""
mutable struct LinearBoundaryValueProblem <: ProblemBase
    data::ProblemData
end

function LinearBoundaryValueProblem(variables; namespace=nothing)
    data = _make_problem_data(variables; namespace=namespace)
    return LinearBoundaryValueProblem(data)
end

_data(p::LinearBoundaryValueProblem) = p.data

solver_class(::LinearBoundaryValueProblem) = LinearBoundaryValueSolver

function _check_equation_conditions(p::LinearBoundaryValueProblem, eqn)
    vars = get_variables(p)
    dist = get_dist(p)
    ts = eqn["tensorsig"]
    dt = eqn["dtype"]
    LHS = operand_cast(eqn["LHS"], dist, ts, dt)
    RHS = operand_cast(eqn["RHS"], dist, ts, dt)
    # LHS must be linear in variables (no affine allowed)
    require_linearity(LHS, vars...;
        allow_affine=false,
        self_name="LBVP LHS",
        vars_name="problem variables",
        error_type=UnsupportedEquationError)
    # RHS must be independent of variables
    require_independent(RHS, vars...;
        self_name="LBVP RHS",
        vars_name="problem variables",
        error_type=UnsupportedEquationError)
    _check_domain_containment(RHS, LHS, "RHS", "LHS")
end

function _build_matrix_expressions(p::LinearBoundaryValueProblem, eqn)
    vars = get_variables(p)
    dist = get_dist(p)
    ts = eqn["tensorsig"]
    dt = eqn["dtype"]
    # Extract matrix expressions
    L = eqn["LHS"]
    F = eqn["RHS"]
    # Reinitialize and prep NCCs
    L = reinitialize(L; ncc=true, ncc_vars=vars)
    prep_nccs(L; vars=vars)
    # Convert to same domain
    domain = (L - F).domain
    L = convert_operand(L, domain.bases)
    if F != 0
        F = operand_cast(F, dist, ts, dt)
        F = convert_operand(F, domain.bases)
    else
        F = _make_zero_field(dist, domain.bases, ts, dt)
    end
    # Save expressions and metadata
    eqn["L"] = L
    eqn["F"] = F
    eqn["domain"] = domain
    eqn["matrix_dependence"] = matrix_dependence(L, vars...)
    eqn["matrix_coupling"] = matrix_coupling(L, vars...)
    @debug "  L: $L"
    @debug "  F: $F"
end

# ============================================================================
# NonlinearBoundaryValueProblem (NLBVP)
# ============================================================================

"""
    NonlinearBoundaryValueProblem <: ProblemBase

Nonlinear boundary value problem of the form:

    G(X) = H(X)

Solved via Newton-Kantorovich iteration:

    dF(X_n) . dX = -F(X_n)
    X_{n+1} = X_n + dX

where F(X) = G(X) - H(X) and dF is the Frechet differential.

# Fields
- `data::ProblemData` -- shared problem data
- `perturbations::Vector{Any}` -- perturbation variables for Newton iteration
"""
mutable struct NonlinearBoundaryValueProblem <: ProblemBase
    data::ProblemData
    perturbations::Vector{Any}
end

function NonlinearBoundaryValueProblem(variables; namespace=nothing)
    data = _make_problem_data(variables; namespace=namespace)
    # Build perturbation variables
    perturbations = Any[]
    for var in variables
        pert = copy(var)
        preset_scales(pert, 1)
        pert["c"] = 0
        name = _get_name(var)
        if name !== nothing && name != ""
            pert.name = string("δ", name)  # delta prefix
        end
        push!(perturbations, pert)
    end
    data.LHS_variables = perturbations
    return NonlinearBoundaryValueProblem(data, perturbations)
end

_data(p::NonlinearBoundaryValueProblem) = p.data

solver_class(::NonlinearBoundaryValueProblem) = NonlinearBoundaryValueSolver

function _check_equation_conditions(::NonlinearBoundaryValueProblem, eqn)
    # No conditions for NLBVP
    nothing
end

function _build_matrix_expressions(p::NonlinearBoundaryValueProblem, eqn)
    vars = get_variables(p)
    perts = p.perturbations
    # F = LHS - RHS (the residual)
    F = eqn["LHS"] - eqn["RHS"]
    dF = frechet_differential(F, vars, perts)
    # Remove field locks
    dF = replace_op(dF, Lock, x -> x)
    for field in atoms(dF, LockedField)
        dF = replace_op(dF, field, unlock(field))
    end
    # Reinitialize and prep NCCs
    dF = reinitialize(dF; ncc=true, ncc_vars=perts)
    prep_nccs(dF; vars=perts)
    # Convert to same domain
    domain = (dF + F).domain
    F = convert_operand(F, domain.bases)
    dF = convert_operand(dF, domain.bases)
    # Save expressions and metadata
    eqn["F"] = F
    eqn["dF"] = dF
    eqn["domain"] = domain
    eqn["matrix_dependence"] = matrix_dependence(dF, perts...)
    eqn["matrix_coupling"] = matrix_coupling(dF, perts...)
    @debug "  F: $F"
    @debug "  dF: $dF"
end

# ============================================================================
# InitialValueProblem (IVP)
# ============================================================================

"""
    InitialValueProblem <: ProblemBase

Initial value problem of the form:

    M . dt(X) + L . X = F(X, t)

where:
- M and L are linear in variables X and time-independent
- The LHS is first-order in time derivatives
- The RHS contains no explicit time derivatives

# Fields
- `data::ProblemData` -- shared problem data
- `time::Any` -- time field
"""
mutable struct InitialValueProblem <: ProblemBase
    data::ProblemData
    time::Any
end

function InitialValueProblem(variables; time="t", namespace=nothing)
    data = _make_problem_data(variables; namespace=namespace)
    dist = data.dist
    if time isa AbstractString
        time_field = _make_scalar_field(dist; name=time, dtype=Float64)
    elseif _is_field(time)
        if any(time.domain.nonconstant)
            throw(ArgumentError("Time field cannot have any bases."))
        end
        time_field = time
    else
        throw(ArgumentError("Time must be a string or Field object."))
    end
    return InitialValueProblem(data, time_field)
end

_data(p::InitialValueProblem) = p.data

solver_class(::InitialValueProblem) = InitialValueSolver

function _check_equation_conditions(p::InitialValueProblem, eqn)
    vars = get_variables(p)
    dist = get_dist(p)
    ts = eqn["tensorsig"]
    dt_type = eqn["dtype"]
    LHS = operand_cast(eqn["LHS"], dist, ts, dt_type)
    RHS = operand_cast(eqn["RHS"], dist, ts, dt_type)
    # LHS must be linear in variables
    require_linearity(LHS, vars...;
        allow_affine=false,
        self_name="IVP LHS",
        vars_name="problem variables",
        error_type=UnsupportedEquationError)
    # LHS must be independent of time
    require_independent(LHS, p.time;
        self_name="IVP LHS",
        vars_name="time",
        error_type=UnsupportedEquationError)
    # LHS must be first order in time derivatives
    require_first_order(LHS, TimeDerivative;
        self_name="IVP LHS",
        ops_name="time derivatives",
        error_type=UnsupportedEquationError)
    # RHS must be independent of time derivatives
    require_independent(RHS, TimeDerivative;
        self_name="IVP RHS",
        vars_name="time derivatives",
        error_type=UnsupportedEquationError)
    _check_domain_containment(RHS, LHS, "RHS", "LHS")
end

function _build_matrix_expressions(p::InitialValueProblem, eqn)
    vars = get_variables(p)
    dist = get_dist(p)
    ts = eqn["tensorsig"]
    dt = eqn["dtype"]
    # Extract matrix expressions: split LHS into M (with dt) and L (without dt)
    M, L = split_operator(eqn["LHS"], TimeDerivative)
    F = eqn["RHS"]
    # Drop time derivatives from M
    if M !== nothing && M != 0
        M = replace_op(M, TimeDerivative, x -> x)
    end
    # Reinitialize and prep NCCs
    if M !== nothing && M != 0
        M = reinitialize(M; ncc=true, ncc_vars=vars)
        prep_nccs(M; vars=vars)
    end
    if L !== nothing && L != 0
        L = reinitialize(L; ncc=true, ncc_vars=vars)
        prep_nccs(L; vars=vars)
    end
    # Convert to same domain
    domain = _combined_domain(M, L, F)
    if M !== nothing && M != 0
        M = convert_operand(M, domain.bases)
    end
    if L !== nothing && L != 0
        L = convert_operand(L, domain.bases)
    end
    if F != 0
        F = operand_cast(F, dist, ts, dt)
        F = convert_operand(F, domain.bases)
    else
        F = _make_zero_field(dist, domain.bases, ts, dt)
    end
    # Save expressions and metadata
    eqn["M"] = M
    eqn["L"] = L
    eqn["F"] = F
    eqn["domain"] = domain
    ml_combined = _safe_add(M, L)
    eqn["matrix_dependence"] = matrix_dependence(ml_combined, vars...)
    eqn["matrix_coupling"] = matrix_coupling(ml_combined, vars...)
    @debug "  M: $M"
    @debug "  L: $L"
    @debug "  F: $F"
end

"""
    build_EVP(ivp::InitialValueProblem; eigenvalue=nothing, backgrounds=nothing,
              perturbations=nothing, kw...)

Create an eigenvalue problem from an initial value problem.

Converts time-independent IVP equations of the form:
    M.dt(X) + L.X = F(X)
to EVP equations as:
    lambda*M.X1 + L.X1 - F'(X0).X1 = 0

Parameters:
- `eigenvalue`: Field for eigenvalue (default: creates new scalar field named "lambda")
- `backgrounds`: Background fields for linearization (default: IVP variables)
- `perturbations`: Perturbation fields for EVP (default: copies of IVP variables)
"""
function build_EVP(ivp::InitialValueProblem;
                   eigenvalue=nothing,
                   backgrounds=nothing,
                   perturbations=nothing,
                   kw...)
    variables = get_variables(ivp)
    dist = get_dist(ivp)
    if eigenvalue === nothing
        eigenvalue = _make_scalar_field(dist; name="λ")  # lambda
    end
    if perturbations === nothing
        perturbations = [copy(var) for var in variables]
        for (pert, var) in zip(perturbations, variables)
            name = _get_name(var)
            if name !== nothing && name != ""
                pert.name = string("δ", name)
            end
        end
    end
    for (pert, var) in zip(perturbations, variables)
        pert.valid_modes .= var.valid_modes
    end
    evp = EigenvalueProblem(perturbations, eigenvalue; kw...)
    # Convert equations from IVP
    for eqn in get_equations(ivp)
        M, L = split_operator(eqn["LHS"], TimeDerivative)
        F = eqn["RHS"]
        # Convert M@dt(X) to lambda*M@Y
        if M !== nothing && M != 0
            M = replace_op(M, TimeDerivative, x -> eigenvalue * x)
            for (var, pert) in zip(variables, perturbations)
                M = replace_op(M, var, pert)
            end
        end
        # Convert L@X to L@Y
        if L !== nothing && L != 0
            for (var, pert) in zip(variables, perturbations)
                L = replace_op(L, var, pert)
            end
        end
        # Take Frechet differential of F(X)
        if F != 0
            if has(F, ivp.time)
                throw(UnsupportedEquationError(
                    "Cannot convert time-dependent IVP to EVP."))
            end
            dF = frechet_differential(F;
                variables=variables,
                perturbations=perturbations,
                backgrounds=backgrounds)
        else
            dF = 0
        end
        # Add linearized equation and copy valid modes
        evp_eqn = add_equation!(evp, (_safe_add(M, L) - dF, 0))
        evp_eqn["valid_modes"] .= eqn["valid_modes"]
    end
    # Add backgrounds to EVP namespace
    if backgrounds !== nothing
        for var in backgrounds
            name = _get_name(var)
            if name !== nothing && name != ""
                _data(evp).local_namespace[name] = var
            end
        end
    end
    return evp
end

# ============================================================================
# EigenvalueProblem (EVP)
# ============================================================================

"""
    EigenvalueProblem <: ProblemBase

Linear eigenvalue problem of the form:

    lambda * M . X + L . X = 0

The LHS must be linear in the problem variables and affine in the eigenvalue.
The RHS must be identically zero.

# Fields
- `data::ProblemData` -- shared problem data
- `eigenvalue::Any` -- the eigenvalue field
"""
mutable struct EigenvalueProblem <: ProblemBase
    data::ProblemData
    eigenvalue::Any
end

function EigenvalueProblem(variables, eigenvalue; namespace=nothing)
    data = _make_problem_data(variables; namespace=namespace)
    if any(eigenvalue.domain.nonconstant)
        throw(ArgumentError("Eigenvalue field cannot have any bases."))
    end
    return EigenvalueProblem(data, eigenvalue)
end

_data(p::EigenvalueProblem) = p.data

solver_class(::EigenvalueProblem) = EigenvalueSolver

function _check_equation_conditions(p::EigenvalueProblem, eqn)
    vars = get_variables(p)
    dist = get_dist(p)
    ts = eqn["tensorsig"]
    dt = eqn["dtype"]
    LHS = operand_cast(eqn["LHS"], dist, ts, dt)
    # LHS must be linear in variables
    require_linearity(LHS, vars...;
        allow_affine=false,
        self_name="EVP LHS",
        vars_name="problem variables",
        error_type=UnsupportedEquationError)
    # LHS must be affine in eigenvalue (linear + constant allowed)
    require_linearity(LHS, p.eigenvalue;
        allow_affine=true,
        self_name="EVP LHS",
        vars_name="the eigenvalue",
        error_type=UnsupportedEquationError)
    if eqn["RHS"] != 0
        throw(UnsupportedEquationError("EVP RHS must be identically zero."))
    end
end

function _build_matrix_expressions(p::EigenvalueProblem, eqn)
    vars = get_variables(p)
    # Extract matrix expressions: split LHS into M (with eigenvalue) and L (without)
    M, L = split_operator(eqn["LHS"], p.eigenvalue)
    # Drop eigenvalue from M
    if M !== nothing && M != 0
        M = replace_op(M, p.eigenvalue, 1)
    end
    # Reinitialize and prep NCCs
    if M !== nothing && M != 0
        M = reinitialize(M; ncc=true, ncc_vars=vars)
        prep_nccs(M; vars=vars)
    end
    if L !== nothing && L != 0
        L = reinitialize(L; ncc=true, ncc_vars=vars)
        prep_nccs(L; vars=vars)
    end
    # Convert to same domain
    domain = _safe_add(M, L).domain
    if M !== nothing && M != 0
        M = convert_operand(M, domain.bases)
    end
    if L !== nothing && L != 0
        L = convert_operand(L, domain.bases)
    end
    # Save expressions and metadata
    eqn["M"] = M
    eqn["L"] = L
    eqn["domain"] = domain
    ml_combined = _safe_add(M, L)
    eqn["matrix_dependence"] = matrix_dependence(ml_combined, vars...)
    eqn["matrix_coupling"] = matrix_coupling(ml_combined, vars...)
    @debug "  M: $M"
    @debug "  L: $L"
end

# ============================================================================
# Helper functions (stubs for forward references)
# ============================================================================

"""
    _safe_add(a, b)

Add two expressions, handling nothing/zero cases gracefully.
"""
function _safe_add(a, b)
    if (a === nothing || a == 0) && (b === nothing || b == 0)
        return 0
    elseif a === nothing || a == 0
        return b
    elseif b === nothing || b == 0
        return a
    else
        return a + b
    end
end

"""
    _combined_domain(args...)

Compute the combined domain from multiple expressions, ignoring nothing/zero values.
"""
function _combined_domain(args...)
    valid = [a for a in args if a !== nothing && a != 0]
    if isempty(valid)
        error("No valid expressions to compute combined domain.")
    end
    combined = valid[1]
    for i in 2:length(valid)
        combined = combined + valid[i]
    end
    return combined.domain
end

"""
    _make_zero_field(dist, bases, tensorsig, dtype) -> Any

Create a zero-valued field with the given specifications.
Forward reference to field construction infrastructure.
"""
function _make_zero_field(dist, bases, tensorsig, dtype)
    # Stub: creates a zero field. Actual implementation depends on Field infrastructure.
    f = Field(; dist=dist, bases=bases, tensorsig=tensorsig, dtype=dtype)
    f["c"] = 0
    return f
end

"""
    _make_scalar_field(dist; name="", dtype=Float64) -> Any

Create a scalar field with no spatial bases.
Forward reference to field construction infrastructure.
"""
function _make_scalar_field(dist; name="", dtype=Float64)
    return Field(; dist=dist, name=name, dtype=dtype)
end

"""
    _is_field(obj) -> Bool

Check if an object is a Field.
"""
function _is_field(obj)
    try
        return hasproperty(obj, :domain) && hasproperty(obj, :data)
    catch
        return false
    end
end

# ============================================================================
# Types and functions referenced here (Field, LockedField, TimeDerivative,
# solver types, expression tree methods) are defined in other core modules
# that are included before or after this file in Dedalus.jl.

# ============================================================================
# Aliases
# ============================================================================

"""Alias for `LinearBoundaryValueProblem`."""
const LBVP = LinearBoundaryValueProblem

"""Alias for `NonlinearBoundaryValueProblem`."""
const NLBVP = NonlinearBoundaryValueProblem

"""Alias for `InitialValueProblem`."""
const IVP = InitialValueProblem

"""Alias for `EigenvalueProblem`."""
const EVP = EigenvalueProblem

# ============================================================================
# Exports
# ============================================================================

export ProblemBase,
       LinearBoundaryValueProblem,
       NonlinearBoundaryValueProblem,
       InitialValueProblem,
       EigenvalueProblem,
       LBVP, NLBVP, IVP, EVP,
       add_equation!,
       build_solver,
       get_equations,
       get_variables,
       get_LHS_variables,
       get_dist,
       problem_matrix_dependence,
       problem_matrix_coupling,
       problem_dtype,
       build_EVP,
       PARSEABLE_NAMESPACE,
       populate_parseables!
