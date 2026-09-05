# [Gauge Conditions](@id gauge_conditions)

Some PDE systems admit families of solutions that differ by an arbitrary constant or
function. A **gauge condition** pins this freedom so the system has a unique solution.
This page explains when gauge conditions are needed in Dedalus.jl and how to apply
them.

## The Pressure Gauge in Incompressible Flow

The incompressible Navier-Stokes equations are the most common case requiring a gauge:

```math
\partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u}
  = -\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}
```
```math
\nabla \cdot \mathbf{u} = 0
```

The pressure ``p`` appears only through its gradient ``\nabla p``. Consequently,
if ``(p, \mathbf{u})`` is a solution then so is ``(p + C, \mathbf{u})`` for any
constant ``C``. The continuity equation ``\nabla \cdot \mathbf{u} = 0`` constrains
the velocity but not the absolute level of the pressure.

Without a gauge condition, the linear system for the implicit (LHS) solve is
**singular**: the matrix has a null space corresponding to the constant-pressure
mode. This causes the sparse direct solver to fail or return garbage.

## Applying a Pressure Gauge

The standard fix is to impose that the spatial average of the pressure vanishes:

```math
\int_\Omega p \, dV = 0
```

In Dedalus.jl this is a single additional equation:

```julia
add_equation!(problem, "integ(p) = 0")
```

This removes the constant mode from the pressure, making the system
non-singular. The `integ` operator computes the integral over the full domain in
all spatial coordinates.

### Tau Term for the Gauge

The gauge equation is a constraint on ``p`` analogous to a boundary condition. It
requires a corresponding **tau term** in the continuity equation:

```julia
tau_p = Field(dist; name="tau_p")
lift_basis = derivative_basis(zbasis, 1)
lift = A -> Lift(A, lift_basis, -1)

# Continuity with tau term for the gauge
add_equation!(problem, "div(u) + lift(tau_p) = 0")
# Pressure gauge
add_equation!(problem, "integ(p) = 0")
```

The `tau_p` variable absorbs the gauge constraint just as boundary-condition tau
terms absorb boundary data. See [The Tau Method](@ref tau_method) for background.

## When Are Gauge Conditions Needed?

Gauge conditions are needed whenever:

1. **A variable appears only through derivatives.** If the equation system
   involves only ``\nabla p`` and never ``p`` itself, the constant mode of ``p``
   is undetermined.

2. **Periodic directions with no boundary conditions.** In a fully periodic
   domain, variables satisfying Laplace-type equations have an undetermined mean.

3. **Divergence-free constraints.** The continuity equation constrains velocity
   derivatives but not pressure, leaving the pressure mean free.

!!! tip "Quick test"
    If your solve fails with a singular-matrix error and you have a variable that
    only appears differentiated, try adding `integ(variable) = 0`.

## Other Gauge Examples

### Stream-function formulation

In a 2D vorticity-streamfunction formulation:

```math
\nabla^2 \psi = -\omega
```

the stream function ``\psi`` is defined up to a constant. The gauge is:

```julia
add_equation!(problem, "integ(psi) = 0")
```

### Magnetic vector potential

In MHD, the magnetic vector potential ``\mathbf{A}`` (where
``\mathbf{B} = \nabla \times \mathbf{A}``) has gauge freedom. A common choice
is the Coulomb gauge:

```math
\nabla \cdot \mathbf{A} = 0
```

implemented similarly to the velocity divergence constraint with an appropriate
tau term and integral condition on a potential.

## Multi-Dimensional Domains

For problems with mixed periodic and bounded directions, the gauge integral only
needs to cover the directions where the variable lacks boundary conditions. In a
channel flow with periodic ``x`` and bounded ``z``:

```julia
# p has boundary conditions in z but is periodic in x
# The gauge fixes the x-averaged constant mode
add_equation!(problem, "integ(p) = 0")
```

Dedalus.jl's `integ` operator integrates over all coordinates of the field by
default. For partial integration, specify the coordinate:

```julia
add_equation!(problem, "integ(p, 'x') = 0")
```
