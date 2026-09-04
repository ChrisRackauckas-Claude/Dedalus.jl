# [Differences from Python Dedalus](@id python_differences)

This page is a reference for users migrating from [Python Dedalus v3](https://dedalus-project.org/)
to Dedalus.jl. It highlights the key syntactic and structural differences between
the two implementations.

## Constructor Syntax

Python Dedalus uses method-based construction from distributor and basis objects.
Dedalus.jl uses standalone constructors with keyword arguments.

### Fields

**Python:**
```python
f = dist.Field(bases=xbasis, name='f')
u = dist.VectorField(coords, bases=xbasis, name='u')
```

**Julia:**
```julia
f = Field(dist; bases=xbasis, name="f")
u = VectorField(dist, coords; bases=xbasis, name="u")
```

Note that `dist` is the first positional argument in Julia, not the object that
owns the constructor. Keyword arguments follow a semicolon.

### Coordinates and Bases

**Python:**
```python
coords = d3.CartesianCoordinates('x', 'z')
xbasis = d3.RealFourier(coords['x'], size=64, bounds=(0, Lx), dealias=3/2)
zbasis = d3.ChebyshevT(coords['z'], size=32, bounds=(-1, 1), dealias=3/2)
```

**Julia:**
```julia
coords = CartesianCoordinates("x", "z")
xbasis = RealFourier(coords["x"], 64, (0, Lx); dealias=3/2)
zbasis = ChebyshevT(coords["z"], 32, (-1, 1); dealias=3/2)
```

In Julia, `size` and `bounds` are positional arguments rather than keyword
arguments.

## Mutating Functions (The `!` Convention)

Julia signals functions that modify their arguments with a trailing `!`. In
Python Dedalus, mutation is implicit:

| Operation | Python | Julia |
|:----------|:-------|:------|
| Add equation | `problem.add_equation("...")` | `add_equation!(problem, "...")` |
| Fill random | `f.fill_random('g')` | `fill_random!(f; layout="g")` |
| Build solver | `solver = problem.build_solver(ts)` | `solver = build_solver(problem, ts)` |
| Solver step | `solver.step(dt)` | `solver.step!(dt)` |
| Newton step | `solver.newton_iteration()` | `solver.newton_iteration!()` |

## Problem Construction

**Python:**
```python
problem = d3.IVP([u, p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)")
problem.add_equation("div(u) = 0")
```

**Julia:**
```julia
problem = IVP([u, p]; namespace=@locals)
add_equation!(problem, "dt(u) + grad(p) - nu*lap(u) = -dot(u, grad(u))")
add_equation!(problem, "div(u) = 0")
```

Key differences:
- `@locals` (Julia's `Base.@locals`) replaces `locals()`.
- `add_equation!` is a free function, not a method on the problem.
- Keyword arguments use `;` instead of `,` after positional arguments.

## The `@locals` Macro

Python's `locals()` returns a dictionary of local variables. Julia's equivalent
is `Base.@locals`, which returns a `Dict{Symbol, Any}` of all bindings in the
current scope:

**Python:**
```python
nu = 1e-3
Lx = 2 * np.pi
problem = d3.IVP([u], namespace=locals())
```

**Julia:**
```julia
nu = 1e-3
Lx = 2 * pi
problem = IVP([u]; namespace=@locals)
```

!!! warning
    `@locals` captures bindings at the point where it is called. Variables
    defined *after* the `@locals` call will not be in the namespace. Define all
    parameters before constructing the problem.

## Indexing: 1-Based

Julia uses 1-based indexing everywhere. This affects:

- Array slicing: `data[1, :]` instead of `data[0, :]`.
- Coordinate indexing: `coords["x"]` returns the first coordinate (though
  string-based access is usually preferred over integer indexing).
- Mode numbers and component indices.

## Dispatch Instead of Class Hierarchies

Python Dedalus uses metaclasses (`MultiClass`) to dispatch constructors based on
argument types. Dedalus.jl uses Julia's multiple dispatch via a registration
system:

```julia
# Internal: registering a basis subtype
@register_dispatch BasisType
register_subtype!(BasisType, MyNewBasis)
```

As a user, this is transparent — you call `ChebyshevT(...)` and dispatch handles
the rest. But if you are extending Dedalus.jl with custom bases or operators, you
use `register_subtype!` instead of subclassing.

## Data Access

**Python:**
```python
f['g']         # grid-space data
f['c']         # coefficient-space data
f.fill_random('g')
```

**Julia:**
```julia
f["g"]         # grid-space data
f["c"]         # coefficient-space data
fill_random!(f; layout="g")
```

Double quotes for strings (Julia does not use single quotes for strings — single
quotes denote characters).

## Operator Names

Most operator names are the same, but some differ:

| Concept | Python | Julia |
|:--------|:-------|:------|
| Gradient | `d3.grad(f)` or `grad(f)` | `gradient(f)` or `grad(f)` |
| Divergence | `d3.div(u)` | `divergence(u)` or `div(u)` |
| Curl | `d3.curl(u)` | `curl(u)` |
| Laplacian | `d3.lap(f)` | `laplacian(f)` or `lap(f)` |
| Dot product | `u @ v` | `dot(u, v)` |
| Lift | `d3.Lift(tau, basis, -1)` | `Lift(tau, basis, -1)` |
| Integrate | `d3.integ(f)` | `integrate(f)` or `integ(f)` |
| Average | `d3.Average(f, coord)` | `average(f, coord)` |
| Interpolate | `d3.Interpolate(f, coord, val)` | `interpolate(f, coord, val)` |
| Differentiate | `d3.Differentiate(f, coord)` | `differentiate(f, coord)` |

In equation strings, both long and short names are available (e.g., `"div(u)"` and
`"divergence(u)"` both work).

## Numerical Comparisons

**Python:**
```python
np.allclose(a, b, atol=1e-10)
```

**Julia:**
```julia
isapprox(a, b; atol=1e-10)
```

Or using the infix form:

```julia
a ≈ b  # uses default tolerances
```

## Field Locking

Dedalus.jl introduces `LockedField`, which restricts a field to specific data
layouts for performance. This has no direct Python equivalent:

```julia
lock_to_layouts!(field, ["g"])      # only grid-space access allowed
lock_axis_to_grid!(field, axis)     # lock one axis to grid layout
unlock(field)                       # remove restrictions
```

## Package Namespacing

**Python:**
```python
import dedalus.public as d3
d3.IVP(...)
d3.RealFourier(...)
```

**Julia:**
```julia
using Dedalus
IVP(...)
RealFourier(...)
```

Julia's `using` brings exported names into scope directly. There is no need for
a namespace prefix.

## Summary Table

| Feature | Python Dedalus | Dedalus.jl |
|:--------|:---------------|:-----------|
| Constructor style | `dist.Field(...)` | `Field(dist; ...)` |
| Mutation | implicit | `!` suffix |
| Local namespace | `locals()` | `@locals` |
| String quotes | `'single'` or `"double"` | `"double"` only |
| Indexing | 0-based | 1-based |
| Dot product | `@` operator | `dot()` function |
| Approximate equality | `np.allclose()` | `isapprox()` / `≈` |
| Extension mechanism | Subclassing | `register_subtype!` |
| Import style | `import ... as d3` | `using Dedalus` |
