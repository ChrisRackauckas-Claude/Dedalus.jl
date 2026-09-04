# [Tutorial 3: Problems and Solvers](@id tutorial_problems_solvers)

This tutorial covers formulating PDE problems and solving them with Dedalus.jl.
We demonstrate all four problem types: LBVP, EVP, IVP, and NLBVP.

## Prerequisites

This tutorial assumes you have read [Tutorial 1](@ref tutorial_coords_bases) and
[Tutorial 2](@ref tutorial_fields_operators).

## The Dedalus.jl Workflow

Every Dedalus.jl simulation follows the same pattern:

1. Define coordinates, distributor, and bases.
2. Create fields (unknowns, tau variables, parameters).
3. Create a problem object, passing the unknown fields.
4. Add equations and boundary conditions with `add_equation!`.
5. Build a solver with `build_solver`.
6. Run the solver (`solve!` for LBVP/EVP/NLBVP, or `step!` in a time loop for
   IVP).

## Linear Boundary Value Problems (LBVP)

An LBVP solves a linear system ``L\mathbf{X} = \mathbf{F}`` with boundary
conditions.  This is the simplest problem type and a good starting point.

### Example: 2D Poisson equation

We solve the Poisson equation on a periodic-by-bounded domain:

```math
\nabla^2 u = f, \quad u(y=0) = g, \quad \partial_y u(y=L_y) = h.
```

```julia
using Dedalus

# Parameters
Lx, Ly = 2*pi, pi
Nx, Ny = 256, 128

# Domain
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; dtype=Float64)
xbasis = RealFourier(coords["x"], Nx; bounds=(0, Lx))
ybasis = ChebyshevT(coords["y"], Ny; bounds=(0, Ly))

# Unknown field and tau variables
u = Field(dist; name="u", bases=(xbasis, ybasis))
tau_1 = Field(dist; name="tau_1", bases=(xbasis,))
tau_2 = Field(dist; name="tau_2", bases=(xbasis,))

# Forcing and boundary data
f = Field(dist; bases=(xbasis, ybasis))
g = Field(dist; bases=(xbasis,))
h = Field(dist; bases=(xbasis,))
x, y = local_grids(dist, xbasis, ybasis)
fill_random!(f, "g"; seed=40)
low_pass_filter!(f; shape=(64, 32))
g["g"] = sin.(8 .* x) .* 0.025
h["g"] .= 0

# Substitutions for equation strings
dy = A -> Differentiate(A, coords["y"])
lift_basis = derivative_basis(ybasis, 2)
lift = (A, n) -> Lift(A, lift_basis, n)

# Problem
problem = LBVP([u, tau_1, tau_2]; namespace=@locals)
add_equation!(problem, "lap(u) + lift(tau_1,-1) + lift(tau_2,-2) = f")
add_equation!(problem, "u(y=0) = g")
add_equation!(problem, "dy(u)(y=Ly) = h")

# Solve
solver = build_solver(problem)
solve!(solver)
```

### Key points

**Variables list**: The first argument to `LBVP` (and all problem types) is a
vector of the unknown fields, including tau variables.  Every field that appears
on the left-hand side of an equation must be in this list.

**Namespace**: The `namespace` keyword provides a dictionary of names that the
equation parser can reference.  The `@locals` macro captures all local
variables into a `Dict`, making names like `f`, `g`, `h`, `dy`, `lift`, `Lx`,
and `Ly` available in equation strings.

**Equation strings**: Equations are written as strings of the form
`"LHS = RHS"`.  The parser evaluates both sides in the provided namespace.
Linear terms involving unknowns go on the left; known forcing goes on the
right.

**Tau terms**: The `lift(tau_1, -1) + lift(tau_2, -2)` terms are the tau
corrections needed because a second-order equation on a Chebyshev basis
requires two boundary conditions (and therefore two tau terms).

**Boundary conditions**: Expressions like `"u(y=0) = g"` are syntactic sugar
for interpolation at the boundary.

## Eigenvalue Problems (EVP)

An EVP solves a generalized eigenvalue problem
``\sigma M \mathbf{X} = L \mathbf{X}`` for the eigenvalue ``\sigma`` and
eigenvector ``\mathbf{X}``.

### Example: Rayleigh-Benard stability

This example computes the growth rates of Rayleigh-Benard convection modes:

```julia
using Dedalus

Lz = 1.0
Nz = 64
Rayleigh = 1e4
Prandtl = 1.0

# Use a single Fourier mode at a prescribed wavenumber
kx = 3.14
Nx = 2
Lx = 2*pi / kx

# Domain (complex-valued for EVP)
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=ComplexF64)
xbasis = ComplexFourier(coords["x"], Nx; bounds=(0, Lx))
zbasis = ChebyshevT(coords["z"], Nz; bounds=(0, Lz))

# Fields
omega = Field(dist; name="omega")              # Eigenvalue
p = Field(dist; name="p", bases=(xbasis, zbasis))
b = Field(dist; name="b", bases=(xbasis, zbasis))
u = VectorField(dist, coords; name="u", bases=(xbasis, zbasis))
tau_p = Field(dist; name="tau_p")
tau_b1 = Field(dist; name="tau_b1", bases=(xbasis,))
tau_b2 = Field(dist; name="tau_b2", bases=(xbasis,))
tau_u1 = VectorField(dist, coords; name="tau_u1", bases=(xbasis,))
tau_u2 = VectorField(dist, coords; name="tau_u2", bases=(xbasis,))

# Substitutions
kappa = (Rayleigh * Prandtl)^(-1/2)
nu = (Rayleigh / Prandtl)^(-1/2)
x, z = local_grids(dist, xbasis, zbasis)
ex, ez = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
lift = A -> Lift(A, lift_basis, -1)
grad_u = gradient(u) + ez * lift(tau_u1)
grad_b = gradient(b) + ez * lift(tau_b1)
dt = A -> -1im * omega * A  # Temporal eigenvalue substitution

# Problem
problem = EVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2];
              eigenvalue=omega, namespace=@locals)
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "dt(b) - kappa*div(grad_b) + lift(tau_b2) - ez@u = 0")
add_equation!(problem, "dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = 0")
add_equation!(problem, "b(z=0) = 0")
add_equation!(problem, "u(z=0) = 0")
add_equation!(problem, "b(z=Lz) = 0")
add_equation!(problem, "u(z=Lz) = 0")
add_equation!(problem, "integ(p) = 0")

# Solve
solver = build_solver(problem)
solve!(solver)
```

### Key points

**Eigenvalue field**: The `eigenvalue` keyword tells the EVP which field
represents the eigenvalue ``\sigma``.  The eigenvalue field should be a
zero-dimensional `Field` (no bases).

**Temporal substitution**: The `dt` closure maps `dt(A)` to `-i\omega A`,
converting the time-derivative notation into the eigenvalue form.  This
makes the equation strings look identical to the IVP version.

**Complex data type**: EVPs typically use `ComplexF64` since eigenvalues and
eigenvectors are generally complex.

**After solving**: The eigenvalues are accessible through `solver.eigenvalues`,
and the eigenvectors can be loaded into the problem fields using
`solver.set_state(index)`.

## Initial Value Problems (IVP)

An IVP integrates a time-dependent system forward in time using IMEX
time-stepping.

### Example: 2D Rayleigh-Benard convection

```julia
using Dedalus
using Logging

# Parameters
Lx, Lz = 4.0, 1.0
Nx, Nz = 256, 64
Rayleigh = 2e6
Prandtl = 1.0
dealias = 3/2
stop_sim_time = 50.0
max_timestep = 0.125

# Domain
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64)
xbasis = RealFourier(coords["x"], Nx; bounds=(0, Lx), dealias=dealias)
zbasis = ChebyshevT(coords["z"], Nz; bounds=(0, Lz), dealias=dealias)

# Fields
p = Field(dist; name="p", bases=(xbasis, zbasis))
b = Field(dist; name="b", bases=(xbasis, zbasis))
u = VectorField(dist, coords; name="u", bases=(xbasis, zbasis))
tau_p = Field(dist; name="tau_p")
tau_b1 = Field(dist; name="tau_b1", bases=(xbasis,))
tau_b2 = Field(dist; name="tau_b2", bases=(xbasis,))
tau_u1 = VectorField(dist, coords; name="tau_u1", bases=(xbasis,))
tau_u2 = VectorField(dist, coords; name="tau_u2", bases=(xbasis,))

# Substitutions
kappa = (Rayleigh * Prandtl)^(-1/2)
nu = (Rayleigh / Prandtl)^(-1/2)
x, z = local_grids(dist, xbasis, zbasis)
ex, ez = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
lift = A -> Lift(A, lift_basis, -1)
grad_u = gradient(u) + ez * lift(tau_u1)
grad_b = gradient(b) + ez * lift(tau_b1)

# Problem
problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2];
              namespace=@locals)
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "dt(b) - kappa*div(grad_b) + lift(tau_b2) = -u@grad(b)")
add_equation!(problem, "dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = -u@grad(u)")
add_equation!(problem, "b(z=0) = Lz")
add_equation!(problem, "u(z=0) = 0")
add_equation!(problem, "b(z=Lz) = 0")
add_equation!(problem, "u(z=Lz) = 0")
add_equation!(problem, "integ(p) = 0")

# Solver
solver = build_solver(problem, RK222)
solver.stop_sim_time = stop_sim_time

# Initial conditions
fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3)
b["g"] .*= z .* (Lz .- z)   # Damp noise at walls
b["g"] .+= Lz .- z           # Add linear background

# Main loop
while solver.proceed
    step!(solver, max_timestep)
    if solver.iteration % 100 == 0
        @info "Iteration=$(solver.iteration), Time=$(solver.sim_time)"
    end
end
```

### Key points

**Timestepper**: The second argument to `build_solver` for an IVP is the
timestepper type.  Common choices are `RK222` (second-order Runge-Kutta,
self-starting) and `SBDF2` (second-order multistep, more efficient but requires
startup).

**LHS vs RHS**: In IVP equations, linear terms go on the left-hand side and
nonlinear terms on the right.  The parser separates these for IMEX treatment:
left-hand side terms are treated implicitly and right-hand side terms
explicitly.

**`dt()` notation**: The `dt()` function marks time-derivative terms.  This is
built in and does not need to be defined in the namespace (unlike for EVP,
where `dt` must be redefined to map to the eigenvalue).

**Stopping conditions**: Set `solver.stop_sim_time`, `solver.stop_wall_time`,
or `solver.stop_iteration` to control when the loop ends.  The `solver.proceed`
property checks all stopping conditions.

**Time stepping**: Call `step!(solver, dt)` to advance by one timestep of size
`dt`.  For adaptive time-stepping, use the `CFL` object (see
[Tutorial 4](@ref tutorial_analysis)).

## Nonlinear Boundary Value Problems (NLBVP)

An NLBVP solves a nonlinear system ``H(\mathbf{X}) = 0`` using Newton
iteration.  Each Newton step internally solves an LBVP for the correction.

### Example: Lane-Emden equation

The Lane-Emden equation describes polytropic stellar structure:

```math
\frac{1}{r^2} \frac{d}{dr}\left(r^2 \frac{df}{dr}\right) + f^n = 0,
\quad f(0) = 1, \quad f'(0) = 0.
```

```julia
using Dedalus

# Domain
coords = SphericalCoordinates("phi", "theta", "r")
dist = Distributor(coords; dtype=Float64)
ball = BallBasis(coords, (1, 1, 64), Float64; radius=1.0)
# ... (see the nlbvp_ball_lane_emden example for the full setup)
```

For NLBVPs, the key difference is that **all terms go on the left-hand side**
and the right-hand side is zero:

```julia
problem = NLBVP([f, tau]; namespace=@locals)
add_equation!(problem, "lap(f) + f^n + lift(tau, -1) = 0")
add_equation!(problem, "f(r=0) = 1")
```

The solver iterates until convergence:

```julia
solver = build_solver(problem)

# Set initial guess
f["g"] = 1.0  # Uniform initial guess

# Newton iteration
for i in 1:20
    solve!(solver)
    # Check convergence (the solver updates fields in-place)
end
```

## Equation Syntax Reference

### Left-hand side vs right-hand side

| Problem type | LHS contains                     | RHS contains              |
|:-------------|:---------------------------------|:--------------------------|
| `LBVP`       | Linear terms in unknowns         | Known forcing             |
| `EVP`        | Linear terms (including `dt()`)  | Zero                      |
| `IVP`        | Linear terms (including `dt()`)  | Nonlinear terms, forcing  |
| `NLBVP`      | All terms (linear + nonlinear)   | Zero                      |

### Common equation patterns

```julia
# Laplacian with tau terms (second-order formulation)
"lap(u) + lift(tau_1, -1) + lift(tau_2, -2) = f"

# First-order form with pre-defined grad_u
"trace(grad_u) + tau_p = 0"                           # Divergence-free
"dt(b) - kappa*div(grad_b) + lift(tau_b2) = -u@grad(b)"  # Advection-diffusion

# Boundary conditions
"u(z=0) = 0"             # Dirichlet
"dy(u)(y=Ly) = h"        # Neumann
"integ(p) = 0"           # Integral (gauge) condition
```

### Passing expressions directly

Instead of equation strings, you can also pass expression tuples:

```julia
# These are equivalent:
add_equation!(problem, "lap(u) = f")
add_equation!(problem, (laplacian(u, coords), f))
```

The string form is more concise; the tuple form avoids parsing and gives you
full Julia expressiveness.

## Summary

- All problem types follow the same pattern: create problem, add equations,
  build solver, run.
- `LBVP` and `NLBVP` use `solve!`; `IVP` uses `step!` in a loop.
- `EVP` uses `solve!` and the eigenvalues are in `solver.eigenvalues`.
- Equation strings are parsed in the provided `namespace`.
- The LHS/RHS split matters: it determines what is treated implicitly (IMEX)
  or what defines the linear operator (LBVP/EVP).

Next, learn how to extract diagnostics and save output in
[Tutorial 4: Analysis](@ref tutorial_analysis).
