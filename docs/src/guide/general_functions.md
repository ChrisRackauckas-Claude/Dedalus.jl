# [General Functions and NCCs](@id general_functions)

Many PDEs have coefficients that vary in space — viscosity profiles, background
states, coordinate-dependent forcing. Dedalus.jl handles these through
**non-constant coefficient (NCC) expansion**, converting spatially varying
functions into spectral representations that can be folded into the operator
matrices.

## What Is an NCC?

A non-constant coefficient is any function of the spatial coordinates that
multiplies an unknown variable in the equations. For example, in

```math
\partial_t u = \nu(z) \, \partial_z^2 u
```

the viscosity ``\nu(z)`` is an NCC. Dedalus.jl cannot simply store ``\nu`` as a
scalar; it must represent the *product* ``\nu(z) \cdot u(z)`` in the spectral
basis.

## NCC Expansion via Clenshaw Recurrence

When Dedalus.jl encounters a product of an NCC with an operand, it expands the
NCC in the spectral basis of the last (non-trivial) axis and builds a
**product matrix** that represents pointwise multiplication in coefficient space.
The expansion uses **Clenshaw recurrence** (`matrix_clenshaw`), which evaluates
the product matrix from the NCC's spectral coefficients without explicitly forming
the full polynomial sum.

For a Chebyshev-represented NCC ``f(z) = \sum_n \hat{f}_n T_n(z)`` acting on an
operand in the same basis, the product ``f \cdot g`` is represented as a banded
matrix multiplication in coefficient space:

```math
(\hat{f \cdot g})_m = \sum_n C_{mn} \, \hat{g}_n
```

where ``C_{mn}`` is the NCC product matrix computed via Clenshaw recurrence from
the coefficients ``\hat{f}_n``.

## Using NCCs in Practice

To use a spatially varying coefficient, create a `Field`, set its data, and
reference it in the equation string:

```julia
# Create a viscosity profile
nu = Field(dist; bases=zbasis, name="nu")
nu.fill_mode = "grid"
nu["g"] .= 1.0 .+ 0.5 * sin.(z)

# Reference in equation
problem = IVP([u]; namespace=@locals)
add_equation!(problem, "dt(u) - nu*dz(dz(u)) = 0")
```

Dedalus.jl automatically detects that `nu` is an NCC (it depends on spatial
coordinates but is not an unknown variable) and builds the appropriate product
matrix during system assembly.

## NCC Truncation

Not all spectral coefficients of the NCC are significant. Dedalus.jl truncates the
NCC expansion by dropping coefficients below a threshold, controlled by the
configuration key `ncc_cutoff` (default `1e-6`). This keeps the product matrices
sparse.

The maximum number of retained NCC terms can also be limited with `max_ncc_terms`.
Both settings live in the `dedalus.toml` configuration file under the
`[linear_algebra]` section. See [Configuration](@ref configuration) for details.

!!! warning "Insufficient NCC resolution"
    If the NCC has fine spatial structure, the default cutoff may discard important
    modes. Symptoms include inaccurate solutions or poor convergence. Decrease
    `ncc_cutoff` or increase the resolution of the NCC field to resolve the
    structure.

## Coordinate-Dependent Coefficients

Some problems have coefficients that are known analytically but vary with
coordinates — for example, ``r`` or ``1/r`` in cylindrical or spherical geometry.
These are handled the same way: create a field, fill it with the coordinate values
on the grid, and pass it through the equation namespace:

```julia
r_field = Field(dist; bases=rbasis, name="r")
r_field["g"] .= r_grid
```

Dedalus.jl's basis implementations for disk, annulus, sphere, and ball geometries
pre-build common coordinate NCCs internally so that operators like `div`, `grad`,
and `curl` in curvilinear coordinates work without user intervention.

## Tips for Efficient NCC Usage

1. **Smooth NCCs are cheap.** A smooth profile has rapidly decaying spectral
   coefficients, leading to narrow-banded product matrices and fast solves.

2. **Discontinuous NCCs are expensive.** Step functions or sharp interfaces
   require many spectral modes and produce dense product matrices. Consider
   smoothing with a tanh profile.

3. **Constant NCCs are free.** If the NCC is spatially uniform, Dedalus.jl
   recognizes it as a constant and avoids building a product matrix entirely.

4. **Evaluate NCCs in grid space.** Fill NCC fields in grid layout
   (`field["g"]`) and let Dedalus.jl transform to coefficient space. Avoid
   manually setting coefficient data unless you are certain of the spectral
   representation.

5. **Check NCC bandwidth.** After building the solver, inspect the NCC product
   matrices to verify they are sufficiently sparse. Dense NCC matrices dominate
   the cost of implicit solves.
