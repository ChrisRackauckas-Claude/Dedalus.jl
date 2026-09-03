"""
    Solver types for Dedalus.jl

Julia translation of `dedalus/core/solvers.py`. Provides solver classes that
build and solve the matrix systems arising from Dedalus problem types.

## Type hierarchy

    SolverBase (abstract)
    +-- EigenvalueSolver
    +-- LinearBoundaryValueSolver
    +-- NonlinearBoundaryValueSolver
    +-- InitialValueSolver

## Key translation choices

- Python `config` lookups -> Julia `get_config` / `get_config_bool` from config.jl
- Python `MPI.SUM` reductions -> Julia MPI or serial fallback
- Python `cProfile` -> Julia `@timed` or `Profile` module
- Python `scipy.linalg.eig` -> Julia `LinearAlgebra.eigen`
- Python `scipy.sparse.linalg.eigs` -> Julia Arnoldi iteration
- Python `h5py` for state loading -> Julia HDF5.jl
- Python class attributes `matsolver_default`, `matrices` -> Julia functions
"""

using SparseArrays
using LinearAlgebra
using Logging

# ============================================================================
# Configuration defaults
# ============================================================================

"""
    _solver_config(section, key; default=nothing) -> Any

Retrieve solver configuration with a fallback default.
"""
function _solver_config(section::String, key::String; default=nothing)
    try
        return get_config(section, key)
    catch
        return default
    end
end

function _solver_config_bool(section::String, key::String; default=false)
    try
        return get_config_bool(section, key)
    catch
        return default
    end
end

# ============================================================================
# SolverBase -- abstract base
# ============================================================================

"""
    SolverBase

Abstract base type for all PDE solvers.

Concrete subtypes must implement solver-specific initialization and solve/step
methods. The base type provides common infrastructure for:
- Matrix coupling and dependence analysis
- Subsystem and subproblem construction
- Matrix solver selection
- Evaluator setup

# Common keyword arguments
- `ncc_cutoff::Float64` -- mode amplitude cutoff for NCC expansions (default: 1e-6)
- `max_ncc_terms` -- max terms in NCC expansions (default: nothing)
- `entry_cutoff::Float64` -- matrix entry cutoff (default: 1e-12)
- `matrix_coupling` -- coupling override (default: nothing)
- `matsolver` -- matrix solver type or name (default: from config)
- `bc_top::Bool` -- boundary conditions at top of matrices
- `tau_left::Bool` -- tau columns at left of matrices
- `interleave_components::Bool` -- interleave components before variables
- `store_expanded_matrices::Bool` -- store right-preconditioned matrices
"""
abstract type SolverBase end

"""
    SolverData

Mutable container for solver state shared across all solver types.
"""
mutable struct SolverData
    problem::Any
    dist::Any
    dtype::DataType
    state::Vector{Any}
    ncc_cutoff::Float64
    max_ncc_terms::Any
    entry_cutoff::Float64
    matrix_coupling::BitVector
    matrix_dependence::BitVector
    matsolver::Any
    bc_top::Bool
    tau_left::Bool
    interleave_components::Bool
    store_expanded_matrices::Bool
    subsystems::Any
    subproblems::Any
    subproblems_by_group::Dict{Any, Any}
    evaluator::Any
end

"""
    _init_solver_base(problem; ncc_cutoff, max_ncc_terms, entry_cutoff,
                      matrix_coupling, matsolver, bc_top, tau_left,
                      interleave_components, store_expanded_matrices,
                      matsolver_default) -> SolverData

Initialize the common solver infrastructure. This is the Julia equivalent of
`SolverBase.__init__` in Python.
"""
function _init_solver_base(problem;
                           ncc_cutoff::Float64=1e-6,
                           max_ncc_terms=nothing,
                           entry_cutoff::Float64=1e-12,
                           matrix_coupling=nothing,
                           matsolver=nothing,
                           bc_top=nothing,
                           tau_left=nothing,
                           interleave_components=nothing,
                           store_expanded_matrices=nothing,
                           matsolver_default::String="MATRIX_FACTORIZER")
    dist = get_dist(problem)
    dtype = problem_dtype(problem)
    state = get_variables(problem)
    # Process matrix coupling
    prob_coupling = problem_matrix_coupling(problem)
    if matrix_coupling === nothing
        mc = BitVector(prob_coupling)
        # Couple fully separable problems along last axis by default
        if !any(mc)
            mc[end] = true
        end
    else
        mc = BitVector(matrix_coupling)
        if any(.~mc .& BitVector(prob_coupling))
            throw(ArgumentError(
                "Specified solver coupling is incompatible with problem coupling: $prob_coupling"))
        end
    end
    # Check that coupled dimensions are local
    cl = dist.coeff_layout
    coupled_nonlocal = mc .& .~BitVector(cl.local)
    if any(coupled_nonlocal)
        nonlocal_axes = findall(coupled_nonlocal)
        throw(ArgumentError(
            "Problem is coupled along distributed dimensions: $(Tuple(nonlocal_axes))"))
    end
    # Determine matrix dependence
    mat_dep = BitVector(problem_matrix_dependence(problem))
    for eq in get_equations(problem)
        for basis in eq["domain"].bases
            first_axis = get_basis_axis(dist, basis)
            d = get_dim(basis)
            sl = first_axis:first_axis+d-1
            mat_dep[sl] .= mat_dep[sl] .| BitVector(basis_matrix_dependence(basis, mc[sl]))
        end
    end
    # Process config options
    if matsolver === nothing
        matsolver_name = _solver_config("linear_algebra", matsolver_default;
                                        default="SuperLUColamdFactorizedTranspose")
        matsolver = get_solver(string(matsolver_name))
    elseif matsolver isa AbstractString
        matsolver = get_solver(matsolver)
    end
    if bc_top === nothing
        bc_top = _solver_config_bool("matrix_construction", "BC_TOP"; default=false)
    end
    if tau_left === nothing
        tau_left = _solver_config_bool("matrix_construction", "TAU_LEFT"; default=false)
    end
    if interleave_components === nothing
        interleave_components = _solver_config_bool("matrix_construction",
                                                    "INTERLEAVE_COMPONENTS"; default=false)
    end
    if store_expanded_matrices === nothing
        store_expanded_matrices = _solver_config_bool("matrix_construction",
                                                      "STORE_EXPANDED_MATRICES"; default=false)
    end
    # Build initial solver data (subsystems/subproblems set later)
    sd = SolverData(
        problem, dist, dtype, collect(Any, state),
        ncc_cutoff, max_ncc_terms, entry_cutoff,
        mc, mat_dep,
        matsolver, bc_top, tau_left, interleave_components, store_expanded_matrices,
        nothing, nothing, Dict{Any, Any}(),
        nothing,
    )
    return sd
end

"""
    _finalize_solver_base!(sd::SolverData, solver::SolverBase)

Complete solver initialization by building subsystems, subproblems, and evaluator.
"""
function _finalize_solver_base!(sd::SolverData, solver::SolverBase)
    sd.subsystems = build_subsystems(solver)
    sd.subproblems = build_subproblems(solver, sd.subsystems)
    sd.subproblems_by_group = Dict(sp.group => sp for sp in sd.subproblems)
    # Build evaluator
    namespace = Dict{String, Any}()
    sd.evaluator = Evaluator(sd.dist, namespace)
end

"""
    solver_build_matrices!(solver::SolverBase; subproblems=nothing, matrices=nothing)

Build matrices for selected subproblems.
"""
function solver_build_matrices!(solver::SolverBase; subproblems=nothing, matrices=nothing)
    sd = _solver_data(solver)
    if subproblems === nothing
        subproblems = sd.subproblems
    end
    if matrices === nothing
        matrices = solver_matrices(solver)
    end
    build_subproblem_matrices(solver, subproblems, matrices)
end

# Interface to be implemented by concrete solvers
function _solver_data end
function solver_matrices end

# ============================================================================
# Stub types for forward references
# ============================================================================

"""Stub: Evaluator constructor."""

"""Stub: basis_matrix_dependence."""
function basis_matrix_dependence end

# ============================================================================
# EigenvalueSolver
# ============================================================================

"""
    EigenvalueSolver <: SolverBase

Solver for linear eigenvalue problems of the form:

    lambda * M . X + L . X = 0

Supports dense and sparse eigenvalue computations.

# Attributes
- `state` -- problem variables containing solution (after set_state! is called)
- `eigenvalues` -- vector of eigenvalues
- `eigenvectors` -- array of eigenvectors (column j is eigenvector j)
- `eigenvalue_subproblem` -- subproblem for which EVP was last solved
"""
mutable struct EigenvalueSolver <: SolverBase
    sd::SolverData
    eigenvalues::Any
    eigenvectors::Any
    right_eigenvectors::Any
    left_eigenvectors::Any
    modified_left_eigenvectors::Any
    left_eigenvalues::Any
    eigenvalue_subproblem::Any
end

function EigenvalueSolver(problem; kw...)
    @debug "Beginning EVP instantiation"
    sd = _init_solver_base(problem;
                           matsolver_default="MATRIX_FACTORIZER", kw...)
    solver = EigenvalueSolver(sd,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    _finalize_solver_base!(sd, solver)
    @debug "Finished EVP instantiation"
    return solver
end

_solver_data(s::EigenvalueSolver) = s.sd
solver_matrices(::EigenvalueSolver) = ["M", "L"]

# Property accessors -- delegate SolverData fields transparently
function Base.getproperty(s::EigenvalueSolver, sym::Symbol)
    if sym in (:problem, :dist, :dtype, :state, :ncc_cutoff, :max_ncc_terms,
               :entry_cutoff, :matrix_coupling, :matrix_dependence, :matsolver,
               :bc_top, :tau_left, :interleave_components, :store_expanded_matrices,
               :subsystems, :subproblems, :subproblems_by_group, :evaluator)
        return getfield(getfield(s, :sd), sym)
    end
    return getfield(s, sym)
end

"""
    print_subproblem_ranks(solver::EigenvalueSolver; subproblems=nothing, target=0)

Print the matrix rank and condition number of each subproblem LHS.
"""
function print_subproblem_ranks(solver::EigenvalueSolver;
                                subproblems=nothing, target=0)
    if subproblems === nothing
        subproblems = solver.subproblems
    end
    for (i, sp) in enumerate(subproblems)
        if !hasproperty(sp, :L_min)
            continue
        end
        L = Matrix(sp.L_min)
        M = Matrix(sp.M_min)
        A = L .+ target .* M
        r = rank(A)
        c = cond(A)
        println("subproblem: $i, group: $(sp.group), " *
                "matrix rank: $r/$(size(A, 1)), cond: $(round(c, sigdigits=4))")
    end
end

"""
    solve_dense!(solver::EigenvalueSolver, subproblem;
                 rebuild_matrices=false, left=false, normalize_left=true, kw...)

Perform dense eigenvector search for the selected subproblem.
Finds all eigenvectors but is computationally expensive.

# Parameters
- `subproblem` -- subproblem for which to solve the EVP
- `rebuild_matrices` -- rebuild LHS matrices if coefficients changed (default: false)
- `left` -- also solve for left eigenvectors (default: false)
- `normalize_left` -- normalize left eigenvectors for biorthonormality (default: true)
"""
function solve_dense!(solver::EigenvalueSolver, subproblem;
                      rebuild_matrices::Bool=false,
                      left::Bool=false,
                      normalize_left::Bool=true,
                      kw...)
    solver.eigenvalue_subproblem = sp = subproblem
    # Build matrices if directed or not yet built
    if rebuild_matrices || !hasproperty(sp, :L_min)
        solver_build_matrices!(solver; subproblems=[sp], matrices=["M", "L"])
    end
    # Solve as dense general eigenvalue problem: A*v = lambda*B*v
    A = Matrix(sp.L_min)
    B = -Matrix(sp.M_min)
    if left
        # Right eigenvectors
        F_right = eigen(A, B)
        solver.eigenvalues = F_right.values
        pre_right_evecs = F_right.vectors
        # Left eigenvectors: solve A^H * w = conj(lambda) * B^H * w
        F_left = eigen(A', B')
        pre_left_evecs = F_left.vectors
        solver.right_eigenvectors = solver.eigenvectors = sp.pre_right * pre_right_evecs
        solver.left_eigenvectors = conj.(sp.pre_left') * pre_left_evecs
        solver.modified_left_eigenvectors = conj.((sp.M_min * sp.pre_right_pinv)') * pre_left_evecs
        if normalize_left
            norms = diag(pre_left_evecs' * sp.M_min * pre_right_evecs)
            solver.left_eigenvectors ./= conj.(norms)'
            solver.modified_left_eigenvectors ./= conj.(norms)'
        end
    else
        F = eigen(A, B)
        solver.eigenvalues = F.values
        pre_eigenvectors = F.vectors
        solver.eigenvectors = sp.pre_right * pre_eigenvectors
    end
end

"""
    solve_sparse!(solver::EigenvalueSolver, subproblem, N, target;
                  rebuild_matrices=false, left=false, normalize_left=true,
                  raise_on_mismatch=true, v0=nothing, kw...)

Perform targeted sparse eigenvector search for the selected subproblem.
Finds N eigenvectors near the specified target eigenvalue.
"""
function solve_sparse!(solver::EigenvalueSolver, subproblem, N::Int, target;
                       rebuild_matrices::Bool=false,
                       left::Bool=false,
                       normalize_left::Bool=true,
                       raise_on_mismatch::Bool=true,
                       v0=nothing,
                       kw...)
    solver.eigenvalue_subproblem = sp = subproblem
    if rebuild_matrices || !hasproperty(sp, :L_min)
        solver_build_matrices!(solver; subproblems=[sp], matrices=["M", "L"])
    end
    A = sp.L_min
    B = -sp.M_min
    # Precondition starting guess
    if v0 !== nothing
        v0 = sp.pre_right_pinv * v0
    end
    # Sparse eigenvalue solve via shift-invert Arnoldi
    eig_result = _sparse_eigs(A, B; N=N, target=target,
                              matsolver=solver.matsolver, v0=v0, left=left, kw...)
    if left
        solver.eigenvalues, pre_right_evecs, solver.left_eigenvalues, pre_left_evecs = eig_result
        solver.right_eigenvectors = solver.eigenvectors = sp.pre_right * pre_right_evecs
        solver.left_eigenvectors = conj.(sp.pre_left') * pre_left_evecs
        solver.modified_left_eigenvectors = conj.((sp.M_min * sp.pre_right_pinv)') * pre_left_evecs
        # Check eigenvalue match
        if !isapprox(solver.eigenvalues, conj.(solver.left_eigenvalues); atol=1e-10)
            if raise_on_mismatch
                error("Conjugate of left eigenvalues does not match right eigenvalues.")
            else
                @warn "Conjugate of left eigenvalues does not match right eigenvalues."
                if normalize_left
                    @warn "Cannot normalize left eigenvectors."
                    normalize_left = false
                end
            end
        end
        if normalize_left
            norms = diag(pre_left_evecs' * sp.M_min * pre_right_evecs)
            solver.left_eigenvectors ./= conj.(norms)'
            solver.modified_left_eigenvectors ./= conj.(norms)'
        end
    else
        solver.eigenvalues, pre_right_evecs = eig_result
        solver.right_eigenvectors = solver.eigenvectors = sp.pre_right * pre_right_evecs
    end
end

"""
    set_state!(solver::EigenvalueSolver, index; subsystem_idx=1)

Set the state vector to the specified eigenmode.

# Parameters
- `index` -- index of the desired eigenmode
- `subsystem_idx` -- index of subsystem within the eigenvalue subproblem (default: 1)
"""
function set_state!(solver::EigenvalueSolver, index::Int; subsystem_idx::Int=1)
    sp = solver.eigenvalue_subproblem
    ss = sp.subsystems[subsystem_idx]
    # Zero all state variables
    for var in solver.state
        var["c"] = 0
    end
    # Set eigenmode coefficients
    scatter!(ss, solver.eigenvectors[:, index], solver.state)
    # Set eigenvalue
    solver.problem.eigenvalue["g"] = solver.eigenvalues[index]
end

"""
    _sparse_eigs(A, B; N, target, matsolver, v0=nothing, left=false, kw...)

Sparse eigenvalue solve using shift-invert mode.
"""
function _sparse_eigs(A, B; N::Int, target, matsolver, v0=nothing, left::Bool=false, kw...)
    sigma = target
    C = A - sigma * B
    C_solver = matsolver(sparse(C))
    n = size(A, 1)
    function matvec(x)
        return solve(C_solver, B * x)
    end
    eigenvalues, eigenvectors = _arnoldi_eigs(matvec, n, N; v0=v0)
    # Transform back: lambda = sigma + 1/mu
    transformed_evals = sigma .+ 1.0 ./ eigenvalues
    if left
        function matvec_H(x)
            return B' * solve(C_solver, x)
        end
        left_evals, left_evecs = _arnoldi_eigs(matvec_H, n, N; v0=v0)
        left_transformed = sigma .+ 1.0 ./ left_evals
        return (transformed_evals, eigenvectors, conj.(left_transformed), left_evecs)
    end
    return (transformed_evals, eigenvectors)
end

"""
    _arnoldi_eigs(op, n, k; v0=nothing) -> (eigenvalues, eigenvectors)

Implicitly restarted Arnoldi iteration for computing k eigenvalues of the
linear operator `op` acting on vectors of length `n`.
"""
function _arnoldi_eigs(op, n::Int, k::Int; v0=nothing)
    m = min(k + 20, n)  # Krylov subspace dimension
    V = zeros(ComplexF64, n, m + 1)
    H = zeros(ComplexF64, m + 1, m)
    if v0 !== nothing
        V[:, 1] = v0 / norm(v0)
    else
        v = randn(ComplexF64, n)
        V[:, 1] = v / norm(v)
    end
    actual_m = m
    for j in 1:m
        w = op(V[:, j])
        for i in 1:j
            H[i, j] = dot(V[:, i], w)
            w .-= H[i, j] .* V[:, i]
        end
        H[j+1, j] = norm(w)
        if abs(H[j+1, j]) < 1e-14
            actual_m = j
            break
        end
        V[:, j+1] = w / H[j+1, j]
    end
    # Eigendecompose the Hessenberg matrix
    Hm = H[1:actual_m, 1:actual_m]
    F = eigen(Hm)
    # Select k eigenvalues with largest magnitude
    perm = sortperm(abs.(F.values); rev=true)
    k_actual = min(k, length(perm))
    selected = perm[1:k_actual]
    eigenvalues = F.values[selected]
    Vm = V[:, 1:actual_m]
    ritz_vectors = Vm * F.vectors[:, selected]
    return (eigenvalues, ritz_vectors)
end

# ============================================================================
# LinearBoundaryValueSolver
# ============================================================================

"""
    LinearBoundaryValueSolver <: SolverBase

Solver for linear boundary value problems of the form:

    L . X = F

# Attributes
- `state` -- problem variables containing solution (after solve! is called)
"""
mutable struct LinearBoundaryValueSolver <: SolverBase
    sd::SolverData
    subproblem_matsolvers::Dict{Any, Any}
    iteration::Int
    F::Vector{Any}
end

function LinearBoundaryValueSolver(problem; kw...)
    @debug "Beginning LBVP instantiation"
    sd = _init_solver_base(problem;
                           matsolver_default="MATRIX_FACTORIZER", kw...)
    solver = LinearBoundaryValueSolver(sd, Dict{Any, Any}(), 0, Any[])
    _finalize_solver_base!(sd, solver)
    # Create RHS handler
    F_handler = add_system_handler!(sd.evaluator; iter=1, group="F")
    for eq in get_equations(problem)
        add_task!(F_handler, eq["F"])
    end
    build_system!(F_handler)
    solver.F = F_handler.fields
    @debug "Finished LBVP instantiation"
    return solver
end

_solver_data(s::LinearBoundaryValueSolver) = s.sd
solver_matrices(::LinearBoundaryValueSolver) = ["L"]

function Base.getproperty(s::LinearBoundaryValueSolver, sym::Symbol)
    if sym in (:problem, :dist, :dtype, :state, :ncc_cutoff, :max_ncc_terms,
               :entry_cutoff, :matrix_coupling, :matrix_dependence, :matsolver,
               :bc_top, :tau_left, :interleave_components, :store_expanded_matrices,
               :subsystems, :subproblems, :subproblems_by_group, :evaluator)
        return getfield(getfield(s, :sd), sym)
    end
    return getfield(s, sym)
end

"""
    print_subproblem_ranks(solver::LinearBoundaryValueSolver; subproblems=nothing)

Print the matrix rank and condition number of each subproblem LHS.
"""
function print_subproblem_ranks(solver::LinearBoundaryValueSolver; subproblems=nothing)
    if subproblems === nothing
        subproblems = solver.subproblems
    end
    for (i, sp) in enumerate(subproblems)
        if !hasproperty(sp, :L_min)
            continue
        end
        L = Matrix(sp.L_min)
        r = rank(L)
        c = cond(L)
        println("subproblem: $i, group: $(sp.group), " *
                "matrix rank: $r/$(size(L, 1)), cond: $(round(c, sigdigits=4))")
    end
end

"""
    solve!(solver::LinearBoundaryValueSolver;
           subproblems=nothing, rebuild_matrices=false)

Solve the BVP over selected subproblems.

# Parameters
- `subproblems` -- subproblems to solve for (default: all)
- `rebuild_matrices` -- rebuild LHS matrices if coefficients changed (default: false)
"""
function solve!(solver::LinearBoundaryValueSolver;
                subproblems=nothing, rebuild_matrices::Bool=false)
    sd = solver.sd
    if subproblems === nothing
        subproblems = sd.subproblems
    end
    if subproblems isa Subproblem
        subproblems = [subproblems]
    end
    # Build matrices and matsolvers if directed or not yet built
    if rebuild_matrices
        sp_to_build = collect(subproblems)
    else
        sp_to_build = [sp for sp in subproblems if !haskey(solver.subproblem_matsolvers, sp)]
    end
    if !isempty(sp_to_build)
        solver_build_matrices!(solver; subproblems=sp_to_build, matrices=["L"])
        for sp in sp_to_build
            solver.subproblem_matsolvers[sp] = sd.matsolver(sp.L_min, solver)
        end
    end
    # Compute RHS
    evaluate_scheduled!(sd.evaluator; iteration=solver.iteration)
    # Ensure coeff space
    for field in solver.F
        change_layout!(field, "c")
    end
    for field in sd.state
        preset_layout!(field, "c")
    end
    # Solve system for each subproblem
    for sp in subproblems
        spF = gather_outputs(sp, solver.F)
        spX = solve(solver.subproblem_matsolvers[sp], spF)
        scatter_inputs!(sp, spX, sd.state)
    end
    solver.iteration += 1
end

"""
    evaluate_handlers!(solver::LinearBoundaryValueSolver; handlers=nothing)

Evaluate specified handlers (all by default).
"""
function evaluate_handlers!(solver::LinearBoundaryValueSolver; handlers=nothing)
    sd = solver.sd
    if handlers === nothing
        handlers = sd.evaluator.handlers
    end
    evaluate_handlers!(sd.evaluator, handlers; iteration=solver.iteration)
end

# ============================================================================
# NonlinearBoundaryValueSolver
# ============================================================================

"""
    NonlinearBoundaryValueSolver <: SolverBase

Solver for nonlinear boundary value problems using Newton-Kantorovich iteration.

Each iteration:
1. Evaluate F(X_n) (the residual)
2. Build the Jacobian dF(X_n)
3. Solve dF . dX = -F for the perturbation dX
4. Update X_{n+1} = X_n - damping * dX

# Attributes
- `state` -- problem variables containing solution
- `perturbations` -- perturbations from each Newton iteration
- `iteration` -- current iteration count
"""
mutable struct NonlinearBoundaryValueSolver <: SolverBase
    sd::SolverData
    perturbations::Vector{Any}
    iteration::Int
    F::Vector{Any}
end

function NonlinearBoundaryValueSolver(problem; kw...)
    @debug "Beginning NLBVP instantiation"
    sd = _init_solver_base(problem;
                           matsolver_default="MATRIX_SOLVER", kw...)
    perts = problem.perturbations
    # Copy valid modes from variables to perturbations
    for (pert, var) in zip(perts, get_variables(problem))
        pert.valid_modes .= var.valid_modes
    end
    solver = NonlinearBoundaryValueSolver(sd, collect(Any, perts), 0, Any[])
    _finalize_solver_base!(sd, solver)
    # Create RHS handler
    F_handler = add_system_handler!(sd.evaluator; iter=1, group="F")
    for eq in get_equations(problem)
        add_task!(F_handler, eq["F"])
    end
    build_system!(F_handler)
    solver.F = F_handler.fields
    @debug "Finished NLBVP instantiation"
    return solver
end

_solver_data(s::NonlinearBoundaryValueSolver) = s.sd
solver_matrices(::NonlinearBoundaryValueSolver) = ["dF"]

function Base.getproperty(s::NonlinearBoundaryValueSolver, sym::Symbol)
    if sym in (:problem, :dist, :dtype, :state, :ncc_cutoff, :max_ncc_terms,
               :entry_cutoff, :matrix_coupling, :matrix_dependence, :matsolver,
               :bc_top, :tau_left, :interleave_components, :store_expanded_matrices,
               :subsystems, :subproblems, :subproblems_by_group, :evaluator)
        return getfield(getfield(s, :sd), sym)
    end
    return getfield(s, sym)
end

"""
    print_subproblem_ranks(solver::NonlinearBoundaryValueSolver; subproblems=nothing)

Print the matrix rank and condition number of each subproblem Jacobian.
"""
function print_subproblem_ranks(solver::NonlinearBoundaryValueSolver; subproblems=nothing)
    if subproblems === nothing
        subproblems = solver.subproblems
    end
    for (i, sp) in enumerate(subproblems)
        if !hasproperty(sp, :dF_min)
            continue
        end
        dF = Matrix(sp.dF_min)
        r = rank(dF)
        c = cond(dF)
        println("subproblem: $i, group: $(sp.group), " *
                "matrix rank: $r/$(size(dF, 1)), cond: $(round(c, sigdigits=4))")
    end
end

"""
    newton_iteration!(solver::NonlinearBoundaryValueSolver; damping=1.0)

Perform one Newton iteration, updating the solution.
"""
function newton_iteration!(solver::NonlinearBoundaryValueSolver; damping::Float64=1.0)
    sd = solver.sd
    # Compute RHS (evaluate F(X_n))
    evaluate_scheduled!(sd.evaluator; iteration=solver.iteration)
    # Rebuild Jacobian
    solver_build_matrices!(solver; subproblems=sd.subproblems, matrices=["dF"])
    # Ensure coeff space
    for field in solver.F
        change_layout!(field, "c")
    end
    for field in solver.perturbations
        preset_layout!(field, "c")
    end
    # Solve system for each subproblem
    for sp in sd.subproblems
        spF = gather_outputs(sp, solver.F)
        sp_matsolver = sd.matsolver(sp.dF_min, solver)
        spX = solve(sp_matsolver, spF)
        scatter_inputs!(sp, spX, solver.perturbations)
    end
    # Update state: X_{n+1} = X_n - damping * dX
    for (var, pert) in zip(sd.state, solver.perturbations)
        var["c"] .-= damping .* pert["c"]
    end
    solver.iteration += 1
end

"""
    evaluate_handlers!(solver::NonlinearBoundaryValueSolver; handlers=nothing)

Evaluate specified handlers (all by default).
"""
function evaluate_handlers!(solver::NonlinearBoundaryValueSolver; handlers=nothing)
    sd = solver.sd
    if handlers === nothing
        handlers = sd.evaluator.handlers
    end
    evaluate_handlers!(sd.evaluator, handlers; iteration=solver.iteration)
end

# ============================================================================
# InitialValueSolver
# ============================================================================

"""
    InitialValueSolver <: SolverBase

Solver for initial value problems with IMEX timestepping.

    M . dt(X) + L . X = F(X, t)

# Attributes
- `state` -- problem variables containing solution
- `stop_sim_time` -- simulation stop time
- `stop_wall_time` -- wall-clock stop time (seconds from instantiation)
- `stop_iteration` -- stop iteration
- `sim_time` -- current simulation time
- `iteration` -- current iteration
- `dt` -- last timestep
"""
mutable struct InitialValueSolver <: SolverBase
    sd::SolverData
    timestepper::Any
    iteration::Int
    initial_iteration::Int
    sim_time::Float64
    initial_sim_time::Float64
    dt::Float64
    stop_sim_time::Float64
    stop_wall_time::Float64
    stop_iteration::Int
    enforce_real_cadence::Any
    warmup_iterations::Int
    total_modes::Int
    F::Vector{Any}
    init_time::Float64
    # Timing fields
    start_time_end::Float64
    warmup_time_start::Float64
    warmup_time_end::Float64
    run_time_start::Float64
    run_time_end::Float64
end

function InitialValueSolver(problem, timestepper_type;
                            enforce_real_cadence::Int=100,
                            warmup_iterations::Int=10,
                            kw...)
    @debug "Beginning IVP instantiation"
    sd = _init_solver_base(problem;
                           matsolver_default="MATRIX_FACTORIZER", kw...)
    init_time = time()
    solver = InitialValueSolver(
        sd,
        nothing,  # timestepper (set below)
        0, 0,     # iteration, initial_iteration
        0.0, 0.0, # sim_time, initial_sim_time
        0.0,      # dt
        Inf, Inf, typemax(Int),  # stop criteria
        nothing,  # enforce_real_cadence
        warmup_iterations,
        0,        # total_modes
        Any[],    # F
        init_time,
        0.0, 0.0, 0.0, 0.0, 0.0,  # timing fields
    )
    _finalize_solver_base!(sd, solver)
    # Build LHS matrices
    solver_build_matrices!(solver; subproblems=sd.subproblems, matrices=["M", "L"])
    # Compute total modes
    local_modes = sum(prod(subproblem_shape(sp)) for sp in sd.subproblems)
    solver.total_modes = local_modes  # Serial; MPI would allreduce
    # Create RHS handler
    F_handler = add_system_handler!(sd.evaluator; iter=1, group="F")
    for eq in get_equations(problem)
        add_task!(F_handler, eq["F"])
    end
    build_system!(F_handler)
    solver.F = F_handler.fields
    # Initialize timestepper
    if timestepper_type isa AbstractString
        timestepper_type = get_timestepper(timestepper_type)
    end
    solver.timestepper = timestepper_type(solver)
    # Set initial sim time from problem's time field
    solver.sim_time = _get_initial_sim_time(problem)
    solver.initial_sim_time = solver.sim_time
    # Set enforce_real_cadence
    if is_real_dtype(sd.dtype)
        solver.enforce_real_cadence = enforce_real_cadence
    else
        solver.enforce_real_cadence = nothing
    end
    @debug "Finished IVP instantiation"
    return solver
end

_solver_data(s::InitialValueSolver) = s.sd
solver_matrices(::InitialValueSolver) = ["M", "L"]

function Base.getproperty(s::InitialValueSolver, sym::Symbol)
    if sym in (:problem, :dist, :dtype, :state, :ncc_cutoff, :max_ncc_terms,
               :entry_cutoff, :matrix_coupling, :matrix_dependence, :matsolver,
               :bc_top, :tau_left, :interleave_components, :store_expanded_matrices,
               :subsystems, :subproblems, :subproblems_by_group, :evaluator)
        return getfield(getfield(s, :sd), sym)
    end
    return getfield(s, sym)
end

function _get_initial_sim_time(problem)
    try
        return Float64(problem.time["g"])
    catch
        return 0.0
    end
end

"""
    wall_time(solver::InitialValueSolver) -> Float64

Return seconds elapsed since solver instantiation.
"""
wall_time(solver::InitialValueSolver) = time() - solver.init_time

"""
    proceed(solver::InitialValueSolver) -> Bool

Check whether the solver should continue (stop conditions not yet met).
"""
function proceed(solver::InitialValueSolver)
    if solver.sim_time >= solver.stop_sim_time
        @info "Simulation stop time reached."
        return false
    elseif wall_time(solver) >= solver.stop_wall_time
        @info "Wall stop time reached."
        return false
    elseif solver.iteration >= solver.stop_iteration
        @info "Stop iteration reached."
        return false
    end
    return true
end

"""
    step!(solver::InitialValueSolver, dt::Float64)

Advance the system by one timestep of size `dt`.
"""
function step!(solver::InitialValueSolver, dt::Float64)
    if !isfinite(dt)
        throw(ArgumentError("Invalid timestep: $dt"))
    end
    wt = wall_time(solver)
    # Record timing checkpoints
    if solver.iteration == solver.initial_iteration
        solver.start_time_end = wt
        solver.warmup_time_start = wt
    end
    if solver.iteration == solver.initial_iteration + solver.warmup_iterations
        solver.warmup_time_end = wt
        solver.run_time_start = wt
    end
    # Advance using timestepper
    step!(solver.timestepper, dt, wt)
    # Enforce Hermitian symmetry for real variables
    if solver.enforce_real_cadence !== nothing
        if solver.iteration % solver.enforce_real_cadence < solver.timestepper.steps
            enforce_hermitian_symmetry!(solver, solver.state)
        end
    end
    # Update iteration
    solver.iteration += 1
    solver.dt = dt
end

"""
    enforce_hermitian_symmetry!(solver::InitialValueSolver, fields)

Transform fields to grid and back to enforce Hermitian symmetry.
"""
function enforce_hermitian_symmetry!(solver::InitialValueSolver, fields)
    sd = solver.sd
    for f in fields
        change_scales!(f, f.domain.dealias)
    end
    require_grid_space!(sd.evaluator, fields)
    require_coeff_space!(sd.evaluator, fields)
end

"""
    evolve!(solver::InitialValueSolver, timestep_function;
            log_cadence=100)

Advance the system until a stopping criterion is reached.

# Parameters
- `timestep_function` -- callable returning the next timestep
- `log_cadence` -- iteration cadence for info logging (default: 100)
"""
function evolve!(solver::InitialValueSolver, timestep_function;
                 log_cadence::Int=100)
    if isinf(solver.stop_sim_time) && isinf(solver.stop_wall_time) &&
       solver.stop_iteration == typemax(Int)
        throw(ArgumentError("No stopping criterion specified."))
    end
    try
        @info "Starting main loop"
        while proceed(solver)
            dt_val = timestep_function()
            step!(solver, dt_val)
            if (solver.iteration - 1) % log_cadence == 0
                @info "Iteration=$(solver.iteration), Time=$(solver.sim_time), Step=$dt_val"
            end
        end
    catch e
        @error "Exception raised, triggering end of main loop."
        rethrow()
    finally
        log_stats(solver)
    end
end

"""
    evaluate_handlers!(solver::InitialValueSolver; handlers=nothing, dt=0.0)

Evaluate specified handlers (all by default).
"""
function evaluate_handlers!(solver::InitialValueSolver; handlers=nothing, dt::Float64=0.0)
    sd = solver.sd
    if handlers === nothing
        handlers = sd.evaluator.handlers
    end
    evaluate_handlers!(sd.evaluator, handlers;
                       iteration=solver.iteration,
                       wall_time=wall_time(solver),
                       sim_time=solver.sim_time,
                       timestep=dt)
end

"""
    print_subproblem_ranks(solver::InitialValueSolver; subproblems=nothing, dt=1.0)

Print the matrix rank and condition number of each subproblem LHS.
"""
function print_subproblem_ranks(solver::InitialValueSolver;
                                subproblems=nothing, dt::Float64=1.0)
    if subproblems === nothing
        subproblems = solver.subproblems
    end
    for (i, sp) in enumerate(subproblems)
        M = sp.M_min
        L = sp.L_min
        A = Matrix(M + dt * L)
        r = rank(A)
        c = cond(A)
        println("subproblem: $i, group: $(sp.group), " *
                "matrix rank: $r/$(size(A, 1)), cond: $(round(c, sigdigits=4))")
    end
end

"""
    log_stats(solver::InitialValueSolver)

Log timing statistics.
"""
function log_stats(solver::InitialValueSolver)
    solver.run_time_end = wall_time(solver)
    start = solver.start_time_end
    @info "Final iteration: $(solver.iteration)"
    @info "Final sim time: $(solver.sim_time)"
    @info "Setup time (init - iter 0): $(round(start, sigdigits=4)) sec"
    if solver.iteration >= solver.initial_iteration + solver.warmup_iterations
        warmup = solver.warmup_time_end - solver.warmup_time_start
        run = solver.run_time_end - solver.run_time_start
        modes = solver.total_modes
        stages = (solver.iteration - solver.warmup_iterations - solver.initial_iteration) *
                 solver.timestepper.stages
        @info "Warmup time: $(round(warmup, sigdigits=4)) sec"
        @info "Run time: $(round(run, sigdigits=4)) sec"
        if run > 0
            @info "Speed: $(round(modes * stages / run, sigdigits=4)) mode-stages/sec"
        end
    else
        @info "Timings unavailable because warmup did not complete."
    end
end

# ============================================================================
# Forward-reference stubs for evaluator/field operations
# ============================================================================

"""Stub: change_layout! for fields."""

"""Stub: preset_layout! for fields."""

"""Stub: evaluate_scheduled! for evaluator."""

"""Stub: evaluate_handlers! for evaluator (with args)."""

"""Stub: add_system_handler! for evaluator."""

"""Stub: add_task! for handler."""

"""Stub: build_system! for handler."""



"""Stub: change_scales! for fields."""

"""Stub: get_timestepper by name."""
function get_timestepper end

# ============================================================================
# Exports
# ============================================================================

export SolverBase,
       SolverData,
       EigenvalueSolver,
       LinearBoundaryValueSolver,
       NonlinearBoundaryValueSolver,
       InitialValueSolver,
       solve_dense!,
       solve_sparse!,
       set_state!,
       solve!,
       newton_iteration!,
       step!,
       evolve!,
       proceed,
       wall_time,
       log_stats,
       print_subproblem_ranks,
       solver_build_matrices!,
       evaluate_handlers!
