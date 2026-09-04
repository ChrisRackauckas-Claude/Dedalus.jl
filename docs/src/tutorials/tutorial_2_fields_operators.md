# [Tutorial 2: Fields and Operators](@id tutorial_fields_operators)

This tutorial covers creating fields, populating them with data, and applying
the differential operators that form the building blocks of PDE equations in
Dedalus.jl.

## Prerequisites

This tutorial assumes you have read [Tutorial 1](@ref tutorial_coords_bases)
and know how to set up coordinates, a distributor, and bases.

## Creating Fields

### Scalar fields

A scalar [`Field`](@ref) (also aliased as `ScalarField`) is the fundamental
data container in Dedalus.jl.  Create one by specifying the distributor and
the bases it lives on:

```julia
using Dedalus

# Setup
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64)
xbasis = RealFourier(coords["x"], 128; bounds=(0, 4.0), dealias=3/2)
zbasis = ChebyshevT(coords["z"], 64; bounds=(0, 1.0), dealias=3/2)

# A 2D scalar field
u = Field(dist; name="u", bases=(xbasis, zbasis))

# A scalar field with no spatial dependence (a constant)
tau = Field(dist; name="tau")

# A field depending on only one dimension (e.g., a boundary tau variable)
tau_bc = Field(dist; name="tau_bc", bases=(xbasis,))
```

The `name` keyword is optional but recommended -- it is used when parsing
equation strings.

### Vector fields

A [`VectorField`](@ref) represents a vector-valued field with components
defined by a coordinate system:

```julia
# A 2D velocity vector field with components (u_x, u_z)
u = VectorField(dist, coords; name="u", bases=(xbasis, zbasis))

# A boundary tau vector
tau_u = VectorField(dist, coords; name="tau_u", bases=(xbasis,))
```

### Tensor fields

A [`TensorField`](@ref) represents a higher-order tensor.  The `order`
keyword (default 2) controls the tensor rank:

```julia
# A rank-2 tensor field (e.g., a stress tensor)
sigma = TensorField(dist, coords; name="sigma", order=2, bases=(xbasis, zbasis))
```

## Accessing Field Data

Every field stores data in two representations:

- **Grid space** (`"g"`) -- the field evaluated at physical grid points.
- **Coefficient space** (`"c"`) -- the spectral expansion coefficients.

### Reading and writing grid-space data

Use the `["g"]` accessor to get or set grid-space values:

```julia
x, z = local_grids(dist, xbasis, zbasis)

# Set the entire grid-space array
u["g"] = @. sin(2*pi*x/4.0) * z * (1 - z)

# Modify in place
u["g"] .*= 2.0

# Read the data
data = u["g"]
```

### Reading coefficient-space data

Use the `["c"]` accessor for spectral coefficients:

```julia
coeffs = u["c"]
```

Accessing `"c"` after writing to `"g"` triggers the forward transform
automatically.  Likewise, accessing `"g"` after modifying `"c"` triggers the
backward transform.

### Vector field components

For vector fields, the `["g"]` accessor returns an array with an extra leading
dimension for the components:

```julia
u = VectorField(dist, coords; name="u", bases=(xbasis, zbasis))
x, z = local_grids(dist, xbasis, zbasis)

# Set x-component and z-component separately
u["g"][1, :, :] .= sin.(2*pi*x/4.0)   # u_x
u["g"][2, :, :] .= 0.0                 # u_z
```

## Utility Functions

### Filling with random data

The `fill_random!` function populates a field with random data in grid space,
with an optional seed for reproducibility:

```julia
fill_random!(u, "g"; seed=42)

# With a specific distribution
fill_random!(u, "g"; seed=42, distribution="normal", scale=1e-3)
```

### Low-pass filtering

After filling with random data, you often want to filter out high-frequency
modes:

```julia
fill_random!(u, "g"; seed=42)
low_pass_filter!(u; shape=(64, 32))
```

### Gathering global data

For post-processing or visualization on the root process, use `allgather_data`
to collect the full field from all MPI ranks:

```julia
u_global = allgather_data(u, "g")
```

This returns `nothing` on non-root ranks (by default) and the full array on
rank 0.

## Unit Vector Fields

When building equations in vector form, you often need unit vectors:

```julia
ex, ez = unit_vector_fields(coords, dist)
```

These are constant vector fields pointing along each coordinate direction,
useful for projecting forces or constructing source terms:

```julia
# Buoyancy force: b * ez
buoyancy_force = b * ez
```

## Differential Operators

Dedalus.jl provides a rich set of differential operators that act on fields and
return operator expression trees (lazy representations that are evaluated when
needed).

### Differentiation

The `differentiate` function (or its capitalized constructor `Differentiate`)
computes partial derivatives along a coordinate:

```julia
# Partial derivative of u with respect to z
du_dz = differentiate(u, coords["z"])

# You can also define shorthand closures
dz = A -> Differentiate(A, coords["z"])
du_dz = dz(u)
```

### Gradient

The `gradient` function computes the gradient vector of a scalar field:

```julia
grad_u = gradient(u, coords)
```

For vector fields, `gradient` produces a rank-2 tensor (the velocity gradient
tensor, for instance).

### Divergence

The `divergence` function computes the divergence of a vector field:

```julia
div_u = divergence(u)
```

This is commonly used for the incompressibility constraint ``\nabla \cdot \mathbf{u} = 0``.

### Curl

The `curl` function computes the curl of a vector field:

```julia
omega = curl(u)
```

In 2D, the curl of a 2D vector field produces a scalar (the vorticity).

### Laplacian

The `laplacian` function computes the Laplacian of a scalar field:

```julia
lap_u = laplacian(u, coords)
```

Note that in many Dedalus.jl examples, the Laplacian is computed indirectly
using a first-order formulation (see below).

### Trace and transpose

For tensor fields, `trace_op` computes the trace and `transpose_components`
transposes tensor indices:

```julia
trace_grad_u = trace_op(gradient(u, coords))  # equivalent to divergence
```

### Dot product

The dot product of two vector fields uses the `@` operator in equation strings,
or `DotProduct` in Julia expressions:

```julia
# In equation strings: "u@grad(b)" means u . grad(b)
# In Julia expressions:
u_dot_grad_b = DotProduct(u, gradient(b, coords))
```

## Integration and Interpolation

### Integration

The `integrate` function computes definite integrals over one or all dimensions:

```julia
# Integrate over z
u_bar = integrate(u, coords["z"])

# Integrate over all dimensions (total integral)
total = integrate(u)
```

### Interpolation

The `interpolate` function evaluates a field at specific coordinate values:

```julia
# Evaluate u at z = 0 (returns a field on the remaining bases)
u_boundary = interpolate(u, coords["z"], 0.0)
```

This is used internally by the equation parser when you write boundary
conditions like `"u(z=0) = g"`.

## The Lift Operator

The `lift` function is essential for the tau method (see
[Methodology](@ref methodology)).  It "lifts" a lower-dimensional field into
the full function space by multiplying it by a basis function:

```julia
# Get the basis for lifting (typically the derivative basis)
lift_basis = derivative_basis(zbasis, 2)

# Lift a tau variable into the equation's function space
tau_1 = Field(dist; name="tau_1", bases=(xbasis,))
lift_term = Lift(tau_1, lift_basis, -1)
```

The integer argument (`-1`, `-2`, etc.) selects which basis function is used for
the lift.  Negative indices count from the end of the basis, so `-1` is the
highest-degree polynomial and `-2` is the next highest.

## First-Order Reductions

Many Dedalus.jl examples use a **first-order formulation** for second-order
PDEs.  Instead of working directly with ``\nabla^2 u``, an auxiliary gradient
variable is introduced:

```julia
# Standard second-order form
# add_equation!(problem, "lap(u) + lift(tau_1,-1) + lift(tau_2,-2) = f")

# First-order form
lift_basis = derivative_basis(zbasis, 1)
lift = A -> Lift(A, lift_basis, -1)
grad_u = gradient(u, coords) + ez * lift(tau_u1)  # First-order reduction
# Then: div(grad_u) replaces lap(u), and trace(grad_u) gives div(u)
```

This approach reduces the number of tau terms per equation and can improve
the conditioning of the resulting matrix systems.  It is the recommended
approach for most problems involving second-order operators with multiple
boundary conditions.

## Putting It Together

Here is a complete example that creates a field, applies operators, and
inspects the result:

```julia
using Dedalus

# Domain setup
Lx, Lz = 2*pi, 1.0
Nx, Nz = 128, 64
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64)
xbasis = RealFourier(coords["x"], Nx; bounds=(0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"], Nz; bounds=(0, Lz), dealias=3/2)

# Create and populate a field
u = Field(dist; name="u", bases=(xbasis, zbasis))
x, z = local_grids(dist, xbasis, zbasis)
u["g"] = @. sin(2*pi*x/Lx) * z * (1 - z)

# Apply operators
du_dx = differentiate(u, coords["x"])
du_dz = differentiate(u, coords["z"])
lap_u = laplacian(u, coords)

# Integrate over the domain
total_integral = integrate(u)
```

## Summary

Key points from this tutorial:

- Fields are created with `Field`, `VectorField`, or `TensorField`, specifying
  the distributor, bases, and optionally a name.
- Grid-space data is accessed via `["g"]` and coefficient-space via `["c"]`.
- Differential operators (`differentiate`, `gradient`, `divergence`, `curl`,
  `laplacian`) return lazy expression trees.
- The `lift` operator and `derivative_basis` are essential for the tau method.
- First-order formulations are recommended for most second-order problems.

Next, learn how to assemble these operators into PDE problems in
[Tutorial 3: Problems and Solvers](@ref tutorial_problems_solvers).
