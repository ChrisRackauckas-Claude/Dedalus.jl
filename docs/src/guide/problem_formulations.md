# [Problem Formulations](@id problem_formulations)

Dedalus.jl supports four problem types, each corresponding to a different class of
partial differential equation. All four share the same spectral discretization
machinery but differ in how the resulting algebraic system is assembled and solved.

## Initial Value Problems (IVP)

An [`IVP`](@ref) evolves a system forward in time. The equations take the form

```math
\mathbf{M} \, \partial_t \mathbf{X} + \mathbf{L} \, \mathbf{X} = \mathbf{F}(\mathbf{X}, t)
```

where ``\mathbf{X}`` is the state vector, ``\mathbf{M}`` is the mass matrix,
``\mathbf{L}`` is a linear operator, and ``\mathbf{F}`` collects all nonlinear
and explicit forcing terms.

### IMEX Splitting

Dedalus.jl uses implicit-explicit (IMEX) time-stepping. The left-hand side (LHS)
containing ``\mathbf{M}`` and ``\mathbf{L}`` is treated **implicitly**, while the
right-hand side (RHS) ``\mathbf{F}`` is treated **explicitly**. This splitting is
essential for stiff systems where diffusive terms would otherwise impose severe
time-step restrictions.

Available IMEX schemes include:

| Family | Schemes |
|:-------|:--------|
| Multistep | `CNAB1`, `CNAB2`, `MCNAB2`, `CNLF2`, `SBDF1`–`SBDF4` |
| Runge-Kutta | `RK111`, `RK222`, `RK443`, `RKSMR`, `RKGFY` |

### Constructing an IVP

```julia
problem = IVP(variables; time="t", namespace=@locals)
add_equation!(problem, "M*dt(u) + L*u = F(u)")
solver = build_solver(problem, SBDF2)
```

The `time` keyword sets the name of the time variable (default `"t"`). The
[`@locals`](@ref) macro (Julia's `Base.@locals`) captures the current scope so
that variables referenced in equation strings are resolved automatically.

## Eigenvalue Problems (EVP)

An [`EVP`](@ref) finds the eigenvalues ``\sigma`` and eigenvectors ``\mathbf{X}``
of a generalized eigenvalue problem:

```math
\sigma \, \mathbf{M} \, \mathbf{X} = \mathbf{L} \, \mathbf{X}
```

The LHS must be linear in both ``\mathbf{X}`` and the eigenvalue ``\sigma``, and
the RHS must be zero. Dedalus.jl assembles the pencil matrices and delegates to
a sparse generalized eigenvalue solver.

### Constructing an EVP

```julia
problem = EVP(variables; eigenvalue="sigma", namespace=@locals)
add_equation!(problem, "sigma*M*X + L*X = 0")
solver = build_solver(problem)
```

### Converting an IVP to an EVP

A common workflow is to linearize an IVP around a background state. The helper
[`build_EVP`](@ref) automates this:

```julia
evp = build_EVP(ivp; eigenvalue="sigma",
                backgrounds=background_fields,
                perturbations=perturbation_fields)
```

This replaces ``\partial_t`` with the eigenvalue parameter and substitutes the
background state into the nonlinear terms.

## Linear Boundary Value Problems (LBVP)

An [`LBVP`](@ref) solves a linear system with no time dependence:

```math
\mathbf{L} \, \mathbf{X} = \mathbf{F}
```

where ``\mathbf{L}`` is linear in the unknowns and ``\mathbf{F}`` is independent
of them. The system is assembled once and solved with a sparse direct solver.

### Constructing an LBVP

```julia
problem = LBVP(variables; namespace=@locals)
add_equation!(problem, "L*u = f")
solver = build_solver(problem)
solver.solve!()
```

Typical uses include Poisson problems, Helmholtz equations, and computing initial
conditions that satisfy constraints.

## Nonlinear Boundary Value Problems (NLBVP)

An [`NLBVP`](@ref) solves a nonlinear system without time dependence:

```math
\mathbf{G}(\mathbf{X}) = \mathbf{H}(\mathbf{X})
```

Internally, Dedalus.jl applies Newton-Kantorovich iteration. At each Newton step
it forms the Frechet differential of the residual and solves a linear correction
problem for perturbation variables ``\delta \mathbf{X}``:

```math
\mathbf{J}(\mathbf{X}_n) \, \delta\mathbf{X} = -\mathbf{R}(\mathbf{X}_n)
```

where ``\mathbf{J}`` is the Jacobian (Frechet derivative) and ``\mathbf{R}`` is
the residual. The state is updated as
``\mathbf{X}_{n+1} = \mathbf{X}_n + \delta\mathbf{X}`` until convergence.

### Constructing an NLBVP

```julia
problem = NLBVP(variables; namespace=@locals)
add_equation!(problem, "G(u) = H(u)")
solver = build_solver(problem)

for iteration in 1:max_iterations
    solver.newton_iteration!()
    if solver.perturbation_norm < tolerance
        break
    end
end
```

Perturbation variables are created automatically with a `delta_` prefix.

## Boundary Conditions and Tau Terms

All four problem types support boundary conditions added as additional equations.
In a spectral method, boundary conditions cannot simply replace rows of the
interior equations without destroying the spectral accuracy of the solution. Instead,
Dedalus.jl uses the **tau method**: each boundary condition is paired with a
**tau term** — a polynomial correction in the highest modes that absorbs the
boundary-condition constraint.

A tau term is added to an equation using the [`Lift`](@ref) operator:

```julia
tau_u = Field(dist; name="tau_u")  # scalar tau field
lift_basis = derivative_basis(zbasis, 1)
lift = A -> Lift(A, lift_basis, -1)

# Interior equation with tau correction
add_equation!(problem, "dx(dx(u)) + lift(tau_u) = f")
# Boundary condition
add_equation!(problem, "u(z=0) = 0")
```

The number of tau terms per equation must equal the number of boundary conditions
imposed on that variable. See [The Tau Method](@ref tau_method) for a detailed
treatment.
