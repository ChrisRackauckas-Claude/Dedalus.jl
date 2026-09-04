"""
    Timestepper types for Dedalus.jl

Julia translation of `dedalus/core/timesteppers.py`. Provides IMEX (implicit-
explicit) timestepping methods for initial value problems.

## Type hierarchy

    IMEXBase (abstract)
    +-- MultistepIMEX (abstract)
    |   +-- CNAB1  -- 1st-order Crank-Nicolson Adams-Bashforth
    |   +-- SBDF1  -- 1st-order semi-implicit BDF
    |   +-- CNAB2  -- 2nd-order Crank-Nicolson Adams-Bashforth
    |   +-- MCNAB2 -- 2nd-order modified Crank-Nicolson Adams-Bashforth
    |   +-- SBDF2  -- 2nd-order semi-implicit BDF
    |   +-- CNLF2  -- 2nd-order Crank-Nicolson leap-frog
    |   +-- SBDF3  -- 3rd-order semi-implicit BDF
    |   +-- SBDF4  -- 4th-order semi-implicit BDF
    +-- RungeKuttaIMEX (abstract)
        +-- RK111  -- 1st-order 1-stage DIRK+ERK
        +-- RK222  -- 2nd-order 2-stage DIRK+ERK
        +-- RK443  -- 3rd-order 4-stage DIRK+ERK
        +-- RKSMR  -- (3-epsilon)-order 3-stage DIRK+ERK
        +-- RKGFY  -- 2nd-order 2-stage scheme

## Key translation choices

- Python `deque` for timestep history -> Julia `Vector` with rotate helper
- Python `scipy.linalg.blas.axpy` -> Julia `LinearAlgebra.axpy!`
- Python `np.copyto` -> Julia `copyto!`
- Python `CoeffSystem` -> Julia `CoeffSystem` struct
- Python `@add_scheme` decorator -> Julia `register_scheme!` function
- Python `@classmethod compute_coefficients` -> Julia functions with type dispatch
"""

using LinearAlgebra
using SparseArrays
using Logging

# ============================================================================
# Scheme registry
# ============================================================================

"""
    SCHEME_REGISTRY :: Dict{String, Any}

Global registry of timestepper schemes, keyed by name.
"""
const SCHEME_REGISTRY = Dict{String, Any}()

"""
    register_scheme!(T::Type)

Register a timestepper type in the global scheme registry.
"""
function register_scheme!(T::Type)
    SCHEME_REGISTRY[string(nameof(T))] = T
    return T
end

"""
    get_timestepper(name::AbstractString) -> Type

Look up a registered timestepper by name.
"""
function get_timestepper(name::AbstractString)
    key = name
    if haskey(SCHEME_REGISTRY, key)
        return SCHEME_REGISTRY[key]
    end
    error("Unknown timestepper: $name. " *
          "Available: $(join(sort(collect(keys(SCHEME_REGISTRY))), ", "))")
end

# CoeffSystem is defined in system.jl (included before this file)

# ============================================================================
# IMEXBase -- abstract base for all IMEX timesteppers
# ============================================================================

"""
    IMEXBase

Abstract base type for all IMEX timestepper methods.
"""
abstract type IMEXBase end

# ============================================================================
# MultistepIMEX -- abstract base for multistep methods
# ============================================================================

"""
    MultistepIMEX <: IMEXBase

Abstract base type for implicit-explicit multistep methods.

These timesteppers discretize the system:
    M . dt(X) + L . X = F
into the general form:
    a_j M . X(n-j) + b_j L . X(n-j) = c_j F(n-j)
where j runs from {0, 0, 1} to {amax, bmax, cmax}.

The system is then solved as:
    (a_0 M + b_0 L) . X(n) = c_j F(n-j) - a_j M . X(n-j) - b_j L . X(n-j)
where j runs from {1, 1, 1} to {cmax, amax, bmax}.

References:
    D. Wang and S. J. Ruuth, Journal of Computational Mathematics 26, (2008).

    Our coefficients relate to Wang's as:
        amax = bmax = cmax = s
        a_j = alpha(s-j) / k(n+s-1)
        b_j = gamma(s-j)
        c_j = beta(s-j)
"""
abstract type MultistepIMEX <: IMEXBase end

"""
    MultistepIMEXData{T}

Mutable data container for multistep IMEX methods.
"""
mutable struct MultistepIMEXData{T}
    solver::Any
    RHS::CoeffSystem{T}
    dt_history::Vector{Float64}
    MX::Vector{CoeffSystem{T}}
    LX::Vector{CoeffSystem{T}}
    F::Vector{CoeffSystem{T}}
    _iteration::Int
    _LHS_params::Union{Nothing, Tuple{Float64, Float64}}
    _nonempty_subproblems::Vector{Any}
end

"""
    _init_multistep(solver, amax, bmax, cmax, steps; dtype) -> MultistepIMEXData

Initialize the multistep IMEX data structures.
"""
function _init_multistep(solver, amax::Int, bmax::Int, cmax::Int, steps::Int;
                         dtype::DataType=Float64)
    subproblems = solver.subproblems
    RHS = CoeffSystem(subproblems; dtype=dtype)
    dt_history = zeros(Float64, steps)
    MX = [CoeffSystem(subproblems; dtype=dtype) for _ in 1:amax]
    LX = [CoeffSystem(subproblems; dtype=dtype) for _ in 1:bmax]
    F_sys = [CoeffSystem(subproblems; dtype=dtype) for _ in 1:cmax]
    nonempty = Any[sp for sp in subproblems if subproblem_size(sp) > 0]
    return MultistepIMEXData{dtype}(solver, RHS, dt_history, MX, LX, F_sys, 0, nothing, nonempty)
end

"""
    _multistep_step!(data, stepper, dt, wall_time)

Advance the solver by one timestep using the multistep IMEX method.

This implements the core multistep algorithm:
1. Rotate timestep and coefficient histories
2. Compute IMEX coefficients for current timesteps
3. Evaluate M.X, L.X, and F(X)
4. Build RHS from history
5. Form and solve the LHS system
"""
function _multistep_step!(data::MultistepIMEXData, stepper::MultistepIMEX,
                          dt::Float64, wall_time::Float64)::Nothing
    solver = data.solver
    subproblems = data._nonempty_subproblems
    evaluator = solver.evaluator
    state_fields = solver.state
    F_fields = solver.F
    sim_time = solver.sim_time
    iteration = solver.iteration
    STORE_EXPANDED = solver.store_expanded_matrices

    MX = data.MX
    LX = data.LX
    F_sys = data.F
    RHS = data.RHS

    # Rotate timestep history
    _rotate_right!(data.dt_history)
    data.dt_history[1] = dt

    # Compute IMEX coefficients
    a, b, c = compute_coefficients(stepper, data.dt_history, data._iteration)
    data._iteration += 1

    # Rotate coefficient system histories
    _rotate_right!(MX)
    _rotate_right!(LX)
    _rotate_right!(F_sys)

    MX0 = MX[1]
    LX0 = LX[1]
    F0 = F_sys[1]
    a0 = a[1]
    b0 = b[1]

    # Check on updating LHS
    update_LHS = ((a0, b0) != data._LHS_params)
    data._LHS_params = (a0, b0)
    if update_LHS
        for sp in subproblems
            sp.LHS_solver = nothing
        end
    end

    # Evaluate M.X0 and L.X0
    require_coeff_space!(evaluator, state_fields)
    for sp in subproblems
        spX = gather_inputs(sp, state_fields)
        _apply_sparse_to_subdata!(sp.M_min, spX, get_subdata(MX0, sp))
        _apply_sparse_to_subdata!(sp.L_min, spX, get_subdata(LX0, sp))
    end

    # Evaluate F(X0)
    evaluate_scheduled!(evaluator; iteration=iteration, wall_time=wall_time,
                        sim_time=sim_time, timestep=dt)
    require_coeff_space!(evaluator, F_fields)
    for sp in subproblems
        gather_outputs(sp, F_fields; out=get_subdata(F0, sp))
    end

    # Build RHS
    if length(RHS.data) > 0
        # RHS = c[2] * F[1]
        RHS.data .= c[2] .* F_sys[1].data
        for j in 3:length(c)
            # RHS += c[j] * F[j-1]
            axpy!(c[j], F_sys[j-1].data, RHS.data)
        end
        for j in 2:length(a)
            # RHS -= a[j] * MX[j-1]
            axpy!(-a[j], MX[j-1].data, RHS.data)
        end
        for j in 2:length(b)
            # RHS -= b[j] * LX[j-1]
            axpy!(-b[j], LX[j-1].data, RHS.data)
        end
    end

    # Solve
    for field in state_fields
        preset_layout!(field, "c")
    end
    for sp in subproblems
        if update_LHS
            if STORE_EXPANDED
                sp.LHS.nzval .= a0 .* sp.M_exp.nzval .+ b0 .* sp.L_exp.nzval
            else
                sp.LHS = a0 * sp.M_min + b0 * sp.L_min
            end
            sp.LHS_solver = solver.matsolver(sp.LHS, solver)
        end
        spRHS = get_subdata(RHS, sp)
        spX = solve(sp.LHS_solver, spRHS)
        scatter_inputs!(sp, spX, state_fields)
    end

    # Update sim time
    solver.sim_time += dt
end

"""
    _rotate_right!(v::Vector)

Rotate vector elements to the right by one position (last element wraps to first).
Equivalent to Python's `deque.rotate()`.
"""
@inline function _rotate_right!(v::Vector)
    n = length(v)
    if n <= 1
        return v
    end
    last = v[n]
    for i in n:-1:2
        v[i] = v[i-1]
    end
    v[1] = last
    return v
end

"""
    _apply_sparse_to_subdata!(A, X, out)

Apply sparse matrix along the first axis of subdata arrays.
Handles both 1D and 2D data.
"""
@inline function _apply_sparse_to_subdata!(A, X, out)::Nothing
    if ndims(X) <= 1 && ndims(out) <= 1
        mul!(out, A, X)
    else
        out_flat = reshape(out, size(A, 1), :)
        X_flat = reshape(X, size(A, 2), :)
        mul!(out_flat, A, X_flat)
    end
    return nothing
end

# ============================================================================
# Multistep scheme: CNAB1
# ============================================================================

"""
    CNAB1 <: MultistepIMEX

1st-order Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.5.3].

Implicit: 2nd-order Crank-Nicolson
Explicit: 1st-order Adams-Bashforth (forward Euler)
"""
mutable struct CNAB1 <: MultistepIMEX
    data::MultistepIMEXData
    stages::Int
    steps::Int
end

const CNAB1_AMAX = 1
const CNAB1_BMAX = 1
const CNAB1_CMAX = 1

function CNAB1(solver)
    data = _init_multistep(solver, CNAB1_AMAX, CNAB1_BMAX, CNAB1_CMAX, 1;
                           dtype=solver.dtype)
    return CNAB1(data, 1, 1)
end

function compute_coefficients(::CNAB1, timesteps, iteration)
    a = zeros(CNAB1_AMAX + 1)
    b = zeros(CNAB1_BMAX + 1)
    c = zeros(CNAB1_CMAX + 1)
    k0 = timesteps[1]
    a[1] = 1 / k0;  a[2] = -1 / k0
    b[1] = 1 / 2;   b[2] = 1 / 2
    c[2] = 1
    return a, b, c
end

function step!(ts::CNAB1, dt::Float64, wt::Float64)
    _multistep_step!(ts.data, ts, dt, wt)
end

register_scheme!(CNAB1)

# ============================================================================
# Multistep scheme: SBDF1
# ============================================================================

"""
    SBDF1 <: MultistepIMEX

1st-order semi-implicit BDF scheme [Wang 2008 eqn 2.6].

Implicit: 1st-order BDF (backward Euler)
Explicit: 1st-order extrapolation (forward Euler)
"""
mutable struct SBDF1 <: MultistepIMEX
    data::MultistepIMEXData
    stages::Int
    steps::Int
end

const SBDF1_AMAX = 1
const SBDF1_BMAX = 1
const SBDF1_CMAX = 1

function SBDF1(solver)
    data = _init_multistep(solver, SBDF1_AMAX, SBDF1_BMAX, SBDF1_CMAX, 1;
                           dtype=solver.dtype)
    return SBDF1(data, 1, 1)
end

function compute_coefficients(::SBDF1, timesteps, iteration)
    a = zeros(SBDF1_AMAX + 1)
    b = zeros(SBDF1_BMAX + 1)
    c = zeros(SBDF1_CMAX + 1)
    k0 = timesteps[1]
    a[1] = 1 / k0;  a[2] = -1 / k0
    b[1] = 1
    c[2] = 1
    return a, b, c
end

function step!(ts::SBDF1, dt::Float64, wt::Float64)
    _multistep_step!(ts.data, ts, dt, wt)
end

register_scheme!(SBDF1)

# ============================================================================
# Multistep scheme: CNAB2
# ============================================================================

"""
    CNAB2 <: MultistepIMEX

2nd-order Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.9].

Implicit: 2nd-order Crank-Nicolson
Explicit: 2nd-order Adams-Bashforth
"""
mutable struct CNAB2 <: MultistepIMEX
    data::MultistepIMEXData
    stages::Int
    steps::Int
end

const CNAB2_AMAX = 2
const CNAB2_BMAX = 2
const CNAB2_CMAX = 2

function CNAB2(solver)
    data = _init_multistep(solver, CNAB2_AMAX, CNAB2_BMAX, CNAB2_CMAX, 2;
                           dtype=solver.dtype)
    return CNAB2(data, 1, 2)
end

function compute_coefficients(ts::CNAB2, timesteps, iteration)
    if iteration < 1
        return _cnab1_coefficients(timesteps)
    end
    a = zeros(CNAB2_AMAX + 1)
    b = zeros(CNAB2_BMAX + 1)
    c = zeros(CNAB2_CMAX + 1)
    k1 = timesteps[1]
    k0 = timesteps[2]
    w1 = k1 / k0
    a[1] = 1 / k1;    a[2] = -1 / k1
    b[1] = 1 / 2;     b[2] = 1 / 2
    c[2] = 1 + w1 / 2
    c[3] = -w1 / 2
    return a, b, c
end

function step!(ts::CNAB2, dt::Float64, wt::Float64)
    _multistep_step!(ts.data, ts, dt, wt)
end

register_scheme!(CNAB2)

# ============================================================================
# Multistep scheme: MCNAB2
# ============================================================================

"""
    MCNAB2 <: MultistepIMEX

2nd-order modified Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.10].

Implicit: 2nd-order modified Crank-Nicolson
Explicit: 2nd-order Adams-Bashforth
"""
mutable struct MCNAB2 <: MultistepIMEX
    data::MultistepIMEXData
    stages::Int
    steps::Int
end

const MCNAB2_AMAX = 2
const MCNAB2_BMAX = 2
const MCNAB2_CMAX = 2

function MCNAB2(solver)
    data = _init_multistep(solver, MCNAB2_AMAX, MCNAB2_BMAX, MCNAB2_CMAX, 2;
                           dtype=solver.dtype)
    return MCNAB2(data, 1, 2)
end

function compute_coefficients(ts::MCNAB2, timesteps, iteration)
    if iteration < 1
        return _cnab1_coefficients(timesteps)
    end
    a = zeros(MCNAB2_AMAX + 1)
    b = zeros(MCNAB2_BMAX + 1)
    c = zeros(MCNAB2_CMAX + 1)
    k1 = timesteps[1]
    k0 = timesteps[2]
    w1 = k1 / k0
    a[1] = 1 / k1;    a[2] = -1 / k1
    b[1] = (8 + 1 / w1) / 16
    b[2] = (7 - 1 / w1) / 16
    b[3] = 1 / 16
    c[2] = 1 + w1 / 2
    c[3] = -w1 / 2
    return a, b, c
end

function step!(ts::MCNAB2, dt::Float64, wt::Float64)
    _multistep_step!(ts.data, ts, dt, wt)
end

register_scheme!(MCNAB2)

# ============================================================================
# Multistep scheme: SBDF2
# ============================================================================

"""
    SBDF2 <: MultistepIMEX

2nd-order semi-implicit BDF scheme [Wang 2008 eqn 2.8].

Implicit: 2nd-order BDF
Explicit: 2nd-order extrapolation
"""
mutable struct SBDF2 <: MultistepIMEX
    data::MultistepIMEXData
    stages::Int
    steps::Int
end

const SBDF2_AMAX = 2
const SBDF2_BMAX = 2
const SBDF2_CMAX = 2

function SBDF2(solver)
    data = _init_multistep(solver, SBDF2_AMAX, SBDF2_BMAX, SBDF2_CMAX, 2;
                           dtype=solver.dtype)
    return SBDF2(data, 1, 2)
end

function compute_coefficients(ts::SBDF2, timesteps, iteration)
    if iteration < 1
        return _sbdf1_coefficients(timesteps)
    end
    a = zeros(SBDF2_AMAX + 1)
    b = zeros(SBDF2_BMAX + 1)
    c = zeros(SBDF2_CMAX + 1)
    k1 = timesteps[1]
    k0 = timesteps[2]
    w1 = k1 / k0
    a[1] = (1 + 2 * w1) / (1 + w1) / k1
    a[2] = -(1 + w1) / k1
    a[3] = w1^2 / (1 + w1) / k1
    b[1] = 1
    c[2] = 1 + w1
    c[3] = -w1
    return a, b, c
end

function step!(ts::SBDF2, dt::Float64, wt::Float64)
    _multistep_step!(ts.data, ts, dt, wt)
end

register_scheme!(SBDF2)

# ============================================================================
# Multistep scheme: CNLF2
# ============================================================================

"""
    CNLF2 <: MultistepIMEX

2nd-order Crank-Nicolson leap-frog scheme [Wang 2008 eqn 2.11].

Implicit: wide Crank-Nicolson
Explicit: 2nd-order leap-frog
"""
mutable struct CNLF2 <: MultistepIMEX
    data::MultistepIMEXData
    stages::Int
    steps::Int
end

const CNLF2_AMAX = 2
const CNLF2_BMAX = 2
const CNLF2_CMAX = 2

function CNLF2(solver)
    data = _init_multistep(solver, CNLF2_AMAX, CNLF2_BMAX, CNLF2_CMAX, 2;
                           dtype=solver.dtype)
    return CNLF2(data, 1, 2)
end

function compute_coefficients(ts::CNLF2, timesteps, iteration)
    if iteration < 1
        return _cnab1_coefficients(timesteps)
    end
    a = zeros(CNLF2_AMAX + 1)
    b = zeros(CNLF2_BMAX + 1)
    c = zeros(CNLF2_CMAX + 1)
    k1 = timesteps[1]
    k0 = timesteps[2]
    w1 = k1 / k0
    a[1] = 1 / (1 + w1) / k1
    a[2] = (w1 - 1) / k1
    a[3] = -w1^2 / (1 + w1) / k1
    b[1] = 1 / w1 / 2
    b[2] = (1 - 1 / w1) / 2
    b[3] = 1 / 2
    c[2] = 1
    return a, b, c
end

function step!(ts::CNLF2, dt::Float64, wt::Float64)
    _multistep_step!(ts.data, ts, dt, wt)
end

register_scheme!(CNLF2)

# ============================================================================
# Multistep scheme: SBDF3
# ============================================================================

"""
    SBDF3 <: MultistepIMEX

3rd-order semi-implicit BDF scheme [Wang 2008 eqn 2.14].

Implicit: 3rd-order BDF
Explicit: 3rd-order extrapolation
"""
mutable struct SBDF3 <: MultistepIMEX
    data::MultistepIMEXData
    stages::Int
    steps::Int
end

const SBDF3_AMAX = 3
const SBDF3_BMAX = 3
const SBDF3_CMAX = 3

function SBDF3(solver)
    data = _init_multistep(solver, SBDF3_AMAX, SBDF3_BMAX, SBDF3_CMAX, 3;
                           dtype=solver.dtype)
    return SBDF3(data, 1, 3)
end

function compute_coefficients(ts::SBDF3, timesteps, iteration)
    if iteration < 2
        return _sbdf2_coefficients(timesteps, iteration)
    end
    a = zeros(SBDF3_AMAX + 1)
    b = zeros(SBDF3_BMAX + 1)
    c = zeros(SBDF3_CMAX + 1)
    k2 = timesteps[1]
    k1 = timesteps[2]
    k0 = timesteps[3]
    w2 = k2 / k1
    w1 = k1 / k0
    a[1] = (1 + w2 / (1 + w2) + w1 * w2 / (1 + w1 * (1 + w2))) / k2
    a[2] = (-1 - w2 - w1 * w2 * (1 + w2) / (1 + w1)) / k2
    a[3] = w2^2 * (w1 + 1 / (1 + w2)) / k2
    a[4] = -w1^3 * w2^2 * (1 + w2) / (1 + w1) / (1 + w1 + w1 * w2) / k2
    b[1] = 1
    c[2] = (1 + w2) * (1 + w1 * (1 + w2)) / (1 + w1)
    c[3] = -w2 * (1 + w1 * (1 + w2))
    c[4] = w1^2 * w2 * (1 + w2) / (1 + w1)
    return a, b, c
end

function step!(ts::SBDF3, dt::Float64, wt::Float64)
    _multistep_step!(ts.data, ts, dt, wt)
end

register_scheme!(SBDF3)

# ============================================================================
# Multistep scheme: SBDF4
# ============================================================================

"""
    SBDF4 <: MultistepIMEX

4th-order semi-implicit BDF scheme [Wang 2008 eqn 2.15].

Implicit: 4th-order BDF
Explicit: 4th-order extrapolation
"""
mutable struct SBDF4 <: MultistepIMEX
    data::MultistepIMEXData
    stages::Int
    steps::Int
end

const SBDF4_AMAX = 4
const SBDF4_BMAX = 4
const SBDF4_CMAX = 4

function SBDF4(solver)
    data = _init_multistep(solver, SBDF4_AMAX, SBDF4_BMAX, SBDF4_CMAX, 4;
                           dtype=solver.dtype)
    return SBDF4(data, 1, 4)
end

function compute_coefficients(ts::SBDF4, timesteps, iteration)
    if iteration < 3
        return _sbdf3_coefficients(timesteps, iteration)
    end
    a = zeros(SBDF4_AMAX + 1)
    b = zeros(SBDF4_BMAX + 1)
    c = zeros(SBDF4_CMAX + 1)
    k3 = timesteps[1]
    k2 = timesteps[2]
    k1 = timesteps[3]
    k0 = timesteps[4]
    w3 = k3 / k2
    w2 = k2 / k1
    w1 = k1 / k0
    A1 = 1 + w1 * (1 + w2)
    A2 = 1 + w2 * (1 + w3)
    A3 = 1 + w1 * A2
    a[1] = (1 + w3 / (1 + w3) + w2 * w3 / A2 + w1 * w2 * w3 / A3) / k3
    a[2] = (-1 - w3 * (1 + w2 * (1 + w3) / (1 + w2) * (1 + w1 * A2 / A1))) / k3
    a[3] = w3 * (w3 / (1 + w3) + w2 * w3 * (A3 + w1) / (1 + w1)) / k3
    a[4] = -w2^3 * w3^2 * (1 + w3) / (1 + w2) * A3 / A2 / k3
    a[5] = (1 + w3) / (1 + w1) * A2 / A1 * w1^4 * w2^3 * w3^2 / A3 / k3
    b[1] = 1
    c[2] = w2 * (1 + w3) / (1 + w2) * ((1 + w3) * (A3 + w1) + (1 + w1) / w2) / A1
    c[3] = -A2 * A3 * w3 / (1 + w1)
    c[4] = w2^2 * w3 * (1 + w3) / (1 + w2) * A3
    c[5] = -w1^3 * w2^2 * w3 * (1 + w3) / (1 + w1) * A2 / A1
    return a, b, c
end

function step!(ts::SBDF4, dt::Float64, wt::Float64)
    _multistep_step!(ts.data, ts, dt, wt)
end

register_scheme!(SBDF4)

# ============================================================================
# Coefficient helper functions (for fallback in higher-order schemes)
# ============================================================================

"""Compute CNAB1 coefficients (used as fallback for early iterations)."""
function _cnab1_coefficients(timesteps)
    a = zeros(2)
    b = zeros(2)
    c = zeros(2)
    k0 = timesteps[1]
    a[1] = 1 / k0;  a[2] = -1 / k0
    b[1] = 1 / 2;   b[2] = 1 / 2
    c[2] = 1
    return a, b, c
end

"""Compute SBDF1 coefficients (used as fallback for early iterations)."""
function _sbdf1_coefficients(timesteps)
    a = zeros(2)
    b = zeros(2)
    c = zeros(2)
    k0 = timesteps[1]
    a[1] = 1 / k0;  a[2] = -1 / k0
    b[1] = 1
    c[2] = 1
    return a, b, c
end

"""Compute SBDF2 coefficients with fallback (used by SBDF3)."""
function _sbdf2_coefficients(timesteps, iteration)
    if iteration < 1
        return _sbdf1_coefficients(timesteps)
    end
    a = zeros(3)
    b = zeros(3)
    c = zeros(3)
    k1 = timesteps[1]
    k0 = timesteps[2]
    w1 = k1 / k0
    a[1] = (1 + 2 * w1) / (1 + w1) / k1
    a[2] = -(1 + w1) / k1
    a[3] = w1^2 / (1 + w1) / k1
    b[1] = 1
    c[2] = 1 + w1
    c[3] = -w1
    return a, b, c
end

"""Compute SBDF3 coefficients with fallback (used by SBDF4)."""
function _sbdf3_coefficients(timesteps, iteration)
    if iteration < 2
        return _sbdf2_coefficients(timesteps, iteration)
    end
    a = zeros(4)
    b = zeros(4)
    c = zeros(4)
    k2 = timesteps[1]
    k1 = timesteps[2]
    k0 = timesteps[3]
    w2 = k2 / k1
    w1 = k1 / k0
    a[1] = (1 + w2 / (1 + w2) + w1 * w2 / (1 + w1 * (1 + w2))) / k2
    a[2] = (-1 - w2 - w1 * w2 * (1 + w2) / (1 + w1)) / k2
    a[3] = w2^2 * (w1 + 1 / (1 + w2)) / k2
    a[4] = -w1^3 * w2^2 * (1 + w2) / (1 + w1) / (1 + w1 + w1 * w2) / k2
    b[1] = 1
    c[2] = (1 + w2) * (1 + w1 * (1 + w2)) / (1 + w1)
    c[3] = -w2 * (1 + w1 * (1 + w2))
    c[4] = w1^2 * w2 * (1 + w2) / (1 + w1)
    return a, b, c
end

# ============================================================================
# RungeKuttaIMEX -- abstract base for IMEX Runge-Kutta methods
# ============================================================================

"""
    RungeKuttaIMEX <: IMEXBase

Abstract base type for implicit-explicit Runge-Kutta methods.

These timesteppers discretize the system:
    M . dt(X) + L . X = F
by constructing s stages:
    M . X(n,i) - M . X(n,0) + k * H_{ij} * L . X(n,j) = k * A_{ij} * F(n,j)
where j runs from {0, 0} to {i, i-1}, and F(n,i) is evaluated at time
    t(n,i) = t(n,0) + k * c_i

The final stage is the advanced solution:
    X(n+1) = X(n,s)

The Butcher tableaux must satisfy:
    b_im = H[s, :]
    b_ex = A[s, :]
    c[s] = 1

References:
    U. M. Ascher, S. J. Ruuth, and R. J. Spiteri,
    Applied Numerical Mathematics (1997).
"""
abstract type RungeKuttaIMEX <: IMEXBase end

"""
    RungeKuttaIMEXData{T}

Mutable data container for Runge-Kutta IMEX methods.
"""
mutable struct RungeKuttaIMEXData{T}
    solver::Any
    RHS::CoeffSystem{T}
    MX0::CoeffSystem{T}
    LX::Vector{CoeffSystem{T}}
    F::Vector{CoeffSystem{T}}
    _LHS_params::Union{Nothing, Float64}
    _nonempty_subproblems::Vector{Any}
end

"""
    _init_rk(solver, num_stages; dtype) -> RungeKuttaIMEXData

Initialize the Runge-Kutta IMEX data structures.
"""
function _init_rk(solver, num_stages::Int; dtype::DataType=Float64)
    subproblems = solver.subproblems
    RHS = CoeffSystem(subproblems; dtype=dtype)
    MX0 = CoeffSystem(subproblems; dtype=dtype)
    LX = [CoeffSystem(subproblems; dtype=dtype) for _ in 1:num_stages]
    F_sys = [CoeffSystem(subproblems; dtype=dtype) for _ in 1:num_stages]
    nonempty = Any[sp for sp in subproblems if subproblem_size(sp) > 0]
    return RungeKuttaIMEXData{dtype}(solver, RHS, MX0, LX, F_sys, nothing, nonempty)
end

"""
    _rk_step!(data, stepper, dt, wall_time)

Advance the solver by one timestep using the IMEX Runge-Kutta method.

For each stage i = 1, ..., s:
    (M + k*H_{ii}*L) . X(n,i) = M . X(n,0)
        + k * sum_{j<i} A_{ij} * F(n,j)
        - k * sum_{j<i} H_{ij} * L . X(n,j)
"""
function _rk_step!(data::RungeKuttaIMEXData, stepper::RungeKuttaIMEX,
                   dt::Float64, wall_time::Float64)::Nothing
    solver = data.solver
    subproblems = data._nonempty_subproblems
    evaluator = solver.evaluator
    state_fields = solver.state
    F_fields = solver.F
    sim_time_0 = solver.sim_time
    iteration = solver.iteration
    STORE_EXPANDED = solver.store_expanded_matrices

    RHS = data.RHS
    MX0 = data.MX0
    LX = data.LX
    LX0 = LX[1]
    F_sys = data.F
    A_tab = stepper.A
    H_tab = stepper.H
    c_tab = stepper.c
    k = dt
    num_stages = stepper.stages

    # Check on updating LHS
    update_LHS = (k != data._LHS_params)
    data._LHS_params = k
    if update_LHS
        required_len = num_stages + 1
        for sp in subproblems
            if length(sp.LHS_solvers) != required_len
                resize!(sp.LHS_solvers, required_len)
            end
            fill!(sp.LHS_solvers, nothing)
        end
    end

    # Compute M.X(n,0) and L.X(n,0)
    require_coeff_space!(evaluator, state_fields)
    for sp in subproblems
        spX = gather_inputs(sp, state_fields)
        _apply_sparse_to_subdata!(sp.M_min, spX, get_subdata(MX0, sp))
        _apply_sparse_to_subdata!(sp.L_min, spX, get_subdata(LX0, sp))
    end

    # Compute stages (1-based: stages run from i=2 to i=num_stages+1)
    for i in 2:(num_stages + 1)
        # Compute L.X(n,i-1), already done for i=2 (stage 0)
        if i > 2
            LXi = LX[i - 1]
            require_coeff_space!(evaluator, state_fields)
            for sp in subproblems
                spX = gather_inputs(sp, state_fields)
                _apply_sparse_to_subdata!(sp.L_min, spX, get_subdata(LXi, sp))
            end
        end

        # Compute F(n,i-1)
        if i == 2
            evaluate_scheduled!(evaluator; iteration=iteration,
                                wall_time=wall_time,
                                sim_time=solver.sim_time, timestep=dt)
        else
            evaluate_group!(evaluator, "F")
        end
        Fi = F_sys[i - 1]
        for sp in subproblems
            gather_outputs(sp, F_fields; out=get_subdata(Fi, sp))
        end

        # Construct RHS(n,i)
        if length(RHS.data) > 0
            copyto!(RHS.data, MX0.data)
            for j in 1:(i - 1)
                axpy!(k * A_tab[i, j], F_sys[j].data, RHS.data)
                axpy!(-k * H_tab[i, j], LX[j].data, RHS.data)
            end
        end

        # Solve for stage
        k_Hii = k * H_tab[i, i]
        for field in state_fields
            preset_layout!(field, "c")
        end
        for sp in subproblems
            if update_LHS
                if STORE_EXPANDED
                    copyto!(sp.LHS.nzval, sp.M_exp.nzval)
                    axpy!(k_Hii, sp.L_exp.nzval, sp.LHS.nzval)
                else
                    sp.LHS = sp.M_min + k_Hii * sp.L_min
                end
                sp.LHS_solvers[i] = solver.matsolver(sp.LHS, solver)
            end
            spRHS = get_subdata(RHS, sp)
            spX = solve(sp.LHS_solvers[i], spRHS)
            scatter_inputs!(sp, spX, state_fields)
        end
        solver.sim_time = sim_time_0 + k * c_tab[i]
    end
end

# ============================================================================
# RK111 -- 1st-order 1-stage DIRK+ERK
# ============================================================================

"""
    RK111 <: RungeKuttaIMEX

1st-order 1-stage DIRK+ERK scheme [Ascher 1997 sec 2.1].
"""
mutable struct RK111 <: RungeKuttaIMEX
    data::RungeKuttaIMEXData
    stages::Int
    steps::Int
    c::Vector{Float64}
    A::Matrix{Float64}
    H::Matrix{Float64}
end

function RK111(solver)
    data = _init_rk(solver, 1; dtype=solver.dtype)
    c = [0.0, 1.0]
    A = [0.0 0.0;
         1.0 0.0]
    H = [0.0 0.0;
         0.0 1.0]
    return RK111(data, 1, 1, c, A, H)
end

function step!(ts::RK111, dt::Float64, wt::Float64)
    _rk_step!(ts.data, ts, dt, wt)
end

register_scheme!(RK111)

# ============================================================================
# RK222 -- 2nd-order 2-stage DIRK+ERK
# ============================================================================

"""
    RK222 <: RungeKuttaIMEX

2nd-order 2-stage DIRK+ERK scheme [Ascher 1997 sec 2.6].
"""
mutable struct RK222 <: RungeKuttaIMEX
    data::RungeKuttaIMEXData
    stages::Int
    steps::Int
    c::Vector{Float64}
    A::Matrix{Float64}
    H::Matrix{Float64}
end

function RK222(solver)
    data = _init_rk(solver, 2; dtype=solver.dtype)
    gamma = (2 - sqrt(2)) / 2
    delta = 1 - 1 / gamma / 2
    c = [0.0, gamma, 1.0]
    A = [0.0   0.0     0.0;
         gamma 0.0     0.0;
         delta 1-delta 0.0]
    H = [0.0     0.0     0.0;
         0.0     gamma   0.0;
         0.0     1-gamma gamma]
    return RK222(data, 2, 1, c, A, H)
end

function step!(ts::RK222, dt::Float64, wt::Float64)
    _rk_step!(ts.data, ts, dt, wt)
end

register_scheme!(RK222)

# ============================================================================
# RK443 -- 3rd-order 4-stage DIRK+ERK
# ============================================================================

"""
    RK443 <: RungeKuttaIMEX

3rd-order 4-stage DIRK+ERK scheme [Ascher 1997 sec 2.8].
"""
mutable struct RK443 <: RungeKuttaIMEX
    data::RungeKuttaIMEXData
    stages::Int
    steps::Int
    c::Vector{Float64}
    A::Matrix{Float64}
    H::Matrix{Float64}
end

function RK443(solver)
    data = _init_rk(solver, 4; dtype=solver.dtype)
    c = [0.0, 1/2, 2/3, 1/2, 1.0]
    A = [  0.0    0.0    0.0    0.0  0.0;
           1/2    0.0    0.0    0.0  0.0;
          11/18   1/18   0.0    0.0  0.0;
           5/6   -5/6    1/2    0.0  0.0;
           1/4    7/4    3/4   -7/4  0.0]
    H = [0.0    0.0    0.0    0.0   0.0;
         0.0    1/2    0.0    0.0   0.0;
         0.0    1/6    1/2    0.0   0.0;
         0.0   -1/2    1/2    1/2   0.0;
         0.0    3/2   -3/2    1/2   1/2]
    return RK443(data, 4, 1, c, A, H)
end

function step!(ts::RK443, dt::Float64, wt::Float64)
    _rk_step!(ts.data, ts, dt, wt)
end

register_scheme!(RK443)

# ============================================================================
# RKSMR -- (3-epsilon)-order 3-stage DIRK+ERK
# ============================================================================

"""
    RKSMR <: RungeKuttaIMEX

(3-epsilon)-order 3-stage DIRK+ERK scheme [Spalart 1991 Appendix].
"""
mutable struct RKSMR <: RungeKuttaIMEX
    data::RungeKuttaIMEXData
    stages::Int
    steps::Int
    c::Vector{Float64}
    A::Matrix{Float64}
    H::Matrix{Float64}
end

function RKSMR(solver)
    data = _init_rk(solver, 3; dtype=solver.dtype)
    alpha1, alpha2, alpha3 = (29/96, -3/40, 1/6)
    beta1, beta2, beta3 = (37/160, 5/24, 1/6)
    gamma1, gamma2, gamma3 = (8/15, 5/12, 3/4)
    zeta2, zeta3 = (-17/60, -5/12)
    c = [0.0, 8/15, 2/3, 1.0]
    A = [0.0           0.0           0.0     0.0;
         gamma1        0.0           0.0     0.0;
         gamma1+zeta2  gamma2        0.0     0.0;
         gamma1+zeta2  gamma2+zeta3  gamma3  0.0]
    H = [0.0     0.0            0.0            0.0;
         alpha1  beta1          0.0            0.0;
         alpha1  beta1+alpha2   beta2          0.0;
         alpha1  beta1+alpha2   beta2+alpha3   beta3]
    return RKSMR(data, 3, 1, c, A, H)
end

function step!(ts::RKSMR, dt::Float64, wt::Float64)
    _rk_step!(ts.data, ts, dt, wt)
end

register_scheme!(RKSMR)

# ============================================================================
# RKGFY -- 2nd-order 2-stage scheme
# ============================================================================

"""
    RKGFY <: RungeKuttaIMEX

2nd-order 2-stage scheme from Hollerbach and Marti.
"""
mutable struct RKGFY <: RungeKuttaIMEX
    data::RungeKuttaIMEXData
    stages::Int
    steps::Int
    c::Vector{Float64}
    A::Matrix{Float64}
    H::Matrix{Float64}
end

function RKGFY(solver)
    data = _init_rk(solver, 2; dtype=solver.dtype)
    c = [0.0, 1.0, 1.0]
    A = [0.0  0.0  0.0;
         1.0  0.0  0.0;
         0.5  0.5  0.0]
    H = [0.0   0.0  0.0;
         0.5   0.5  0.0;
         0.5   0.0  0.5]
    return RKGFY(data, 2, 1, c, A, H)
end

function step!(ts::RKGFY, dt::Float64, wt::Float64)
    _rk_step!(ts.data, ts, dt, wt)
end

register_scheme!(RKGFY)

# ============================================================================
# Forward-reference stubs
# ============================================================================

"""Stub: require_coeff_space! for evaluator."""

"""Stub: evaluate_scheduled! for evaluator."""

"""Stub: evaluate_group! for evaluator."""

"""Stub: preset_layout! for fields."""

"""Stub: gather_inputs from subproblem."""

"""Stub: gather_outputs from subproblem."""

"""Stub: scatter_inputs! to subproblem."""

"""Stub: subproblem_shape query."""

"""Stub: subproblem_size query."""

"""Stub: solve from matsolvers."""

# ============================================================================
# Exports
# ============================================================================

export IMEXBase,
       MultistepIMEX,
       RungeKuttaIMEX,
       CoeffSystem,
       get_subdata,
       compute_coefficients,
       CNAB1, CNAB2, MCNAB2,
       SBDF1, SBDF2, SBDF3, SBDF4,
       CNLF2,
       RK111, RK222, RK443,
       RKSMR, RKGFY,
       SCHEME_REGISTRY,
       get_timestepper
