# [The Tau Method](@id tau_method)

The tau method is the mechanism by which Dedalus.jl incorporates boundary conditions
into spectral discretizations without sacrificing spectral accuracy. This page
explains why tau terms are needed, how they work, and how to use them correctly.

## Why Tau Terms?

Consider a second-order ODE on ``[-1, 1]``:

```math
\frac{d^2 u}{d z^2} = f(z), \qquad u(-1) = a, \quad u(1) = b
```

A Galerkin approach would restrict the trial space to functions satisfying the
boundary conditions, but constructing such bases for general coupled systems is
impractical. The **collocation** approach evaluates equations at grid points and
replaces two rows with boundary conditions, but this destroys the banded structure
of spectral derivative matrices and couples all modes.

The **tau method** instead solves a *modified* problem:

```math
\frac{d^2 u}{d z^2} + \tau_0 P_0(z) + \tau_1 P_1(z) = f(z)
```

where ``P_0`` and ``P_1`` are polynomials (typically from a related basis) and
``\tau_0, \tau_1`` are unknown constants determined by the boundary conditions.
The tau terms live in the highest spectral modes and are algebraically fixed by
the boundary equations.

## The Lift Operator

In Dedalus.jl, tau terms are added using the [`Lift`](@ref) operator. `Lift`
places a lower-dimensional field into specific modes of a higher-dimensional basis:

```julia
Lift(tau_field, output_basis, n)
```

- `tau_field`: the unknown tau variable (a `Field` on the boundary or a scalar).
- `output_basis`: the basis into which the tau term is "lifted."
- `n`: a **negative** integer selecting which mode to place the tau data into.
  `n = -1` is the last mode, `n = -2` the second-to-last, etc.

A typical setup for a second-order equation:

```julia
tau_u1 = Field(dist; name="tau_u1")
tau_u2 = Field(dist; name="tau_u2")

lift_basis = derivative_basis(zbasis, 1)
lift = A -> Lift(A, lift_basis, -1)

add_equation!(problem, "dz(dz(u)) + lift(tau_u1) + lift(tau_u2) = f")
add_equation!(problem, "u(z=-1) = a")
add_equation!(problem, "u(z= 1) = b")
```

Two tau variables absorb the two boundary conditions.

## Derivative Bases

The function [`derivative_basis`](@ref) returns the "natural" basis for the
derivative of a given basis. For Jacobi polynomials ``P_n^{(a,b)}``, the
derivative maps into ``P_n^{(a+1, b+1)}``:

```math
\frac{d}{dz} P_n^{(a,b)}(z) \propto P_{n-1}^{(a+1,\, b+1)}(z)
```

In code:

```julia
derivative_basis(b::JacobiBasis; order=1)
# Returns a new JacobiBasis with a -> a + order, b -> b + order
```

For a Chebyshev basis (``a = b = -1/2``), the first derivative basis is
``P_n^{(1/2, 1/2)}`` (Legendre-like), and the second derivative basis is
``P_n^{(3/2, 3/2)}``.

The derivative basis is the natural choice for `Lift` because it preserves the
banded structure of the combined differentiation-plus-tau-correction system. Using
the wrong basis for lifting can destroy sparsity and slow the solve.

## Counting Tau Terms

The rule is straightforward:

!!! tip "Tau Counting Rule"
    Each variable needs **one tau term per boundary condition** imposed on it,
    and those tau terms should appear in the **highest-order equation** for that
    variable.

For example, in a fourth-order beam equation with four boundary conditions on
``u``, you need four tau variables:

```julia
tau_u1 = Field(dist; name="tau_u1")
tau_u2 = Field(dist; name="tau_u2")
tau_u3 = Field(dist; name="tau_u3")
tau_u4 = Field(dist; name="tau_u4")

lift_basis = derivative_basis(zbasis, 1)
lift = A -> Lift(A, lift_basis, -1)

add_equation!(problem, "dz(dz(dz(dz(u)))) + lift(tau_u1) + lift(tau_u2) + lift(tau_u3) + lift(tau_u4) = f")
add_equation!(problem, "u(z=-1) = 0")
add_equation!(problem, "dz(u)(z=-1) = 0")
add_equation!(problem, "u(z=1) = 0")
add_equation!(problem, "dz(u)(z=1) = 0")
```

## First-Order Reductions

In practice, high-order equations are often reduced to first-order systems. A
second-order diffusion equation can be split as:

```math
\partial_t u - \partial_z w = 0, \qquad w - \partial_z u = 0
```

with boundary conditions on ``u``. In this formulation:

- The equation for ``u`` gets **one** tau term (the ``\partial_z w`` equation has
  first-order derivatives and one boundary condition on ``u``).
- The auxiliary equation for ``w`` also gets **one** tau term.

```julia
tau_u = Field(dist; name="tau_u")
tau_w = Field(dist; name="tau_w")

lift_basis = derivative_basis(zbasis, 1)
lift = A -> Lift(A, lift_basis, -1)

add_equation!(problem, "dt(u) - dz(w) + lift(tau_u) = 0")
add_equation!(problem, "w - dz(u) + lift(tau_w) = 0")
add_equation!(problem, "u(z=-1) = 0")
add_equation!(problem, "u(z= 1) = 0")
```

## Common Mistakes

!!! warning "Mismatched tau count"
    If the number of tau terms does not equal the number of boundary conditions
    for a variable, the system will be either over- or under-determined. Dedalus.jl
    will raise a dimension mismatch error during matrix construction.

!!! warning "Wrong lift basis"
    Using the original basis instead of `derivative_basis` for the lift will
    produce a dense (rather than banded) system. The solver will still work but
    will be much slower and may suffer from numerical instability.

!!! warning "Tau terms on the RHS"
    Tau variables must appear on the **LHS** of the equation so they enter the
    implicitly solved system. Placing them on the RHS will not incorporate them
    into the matrix pencil.
