# [Tutorial 1: Coordinates and Bases](@id tutorial_coords_bases)

Every Dedalus.jl simulation starts by defining the coordinate system and
spectral bases for the computational domain.  This tutorial covers the
foundational objects: coordinate systems, distributors, and bases.

## Cartesian Coordinates

The most common starting point is a Cartesian coordinate system.  Create one by
listing the coordinate names:

```julia
using Dedalus

# 1D coordinate
xcoord = CartesianCoordinates("x")

# 2D coordinates
coords = CartesianCoordinates("x", "z")

# 3D coordinates
coords3d = CartesianCoordinates("x", "y", "z")
```

Individual coordinates can be extracted by name using bracket indexing:

```julia
coords = CartesianCoordinates("x", "z")
x = coords["x"]  # Coordinate object for x
z = coords["z"]  # Coordinate object for z
```

## The Distributor

The [`Distributor`](@ref) manages data distribution across MPI processes and
coordinates the spectral transforms between coefficient space and grid space.
Every field, basis, and transform goes through the distributor.

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64)
```

The `dtype` keyword specifies the default data type for fields on this domain.
Use `Float64` for real-valued problems and `ComplexF64` when the solution is
complex (common in eigenvalue problems).

### MPI mesh

For parallel runs, the `mesh` keyword controls how the domain is decomposed
across MPI ranks:

```julia
# 1D decomposition across 4 processes
dist = Distributor(coords; dtype=Float64, mesh=(4,))

# 2D pencil decomposition across a 2x2 grid of processes
dist = Distributor(coords; dtype=Float64, mesh=(2, 2))
```

When `mesh` is omitted, the distributor uses a default one-dimensional
decomposition.

## Fourier Bases

Fourier bases are used for periodic dimensions.  Specify the coordinate, number
of modes, and the domain bounds:

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64)

# Real Fourier basis (for real-valued fields)
xbasis = RealFourier(coords["x"], 128; bounds=(0, 2*pi))

# Complex Fourier basis (for complex-valued fields or eigenvalue problems)
xbasis_c = ComplexFourier(coords["x"], 128; bounds=(0, 2*pi))
```

### RealFourier vs ComplexFourier

- **`RealFourier`** stores paired cosine and sine modes, taking advantage of
  the Hermitian symmetry of real-valued transforms.  Use this for most
  physical problems where the solution is real.
- **`ComplexFourier`** stores independent complex exponential modes.  Use this
  when the solution is intrinsically complex or when you need to isolate a
  single Fourier mode (e.g., for eigenvalue problems with a prescribed
  wavenumber).

### Dealiasing

Nonlinear terms generate high-frequency content that can alias back into the
resolved modes.  The `dealias` keyword controls the dealiasing factor -- the
ratio of grid points to coefficient modes:

```julia
# 3/2 dealiasing (standard for quadratic nonlinearities)
xbasis = RealFourier(coords["x"], 128; bounds=(0, 2*pi), dealias=3/2)
```

A dealias factor of `3/2` (the default for the 2/3 dealiasing rule) provides
``3N/2`` grid points for ``N`` modes, which is sufficient for quadratic
nonlinearities.  The default `dealias=1` uses no extra grid points.

## Chebyshev Bases

Chebyshev bases are used for non-periodic (bounded) dimensions:

```julia
# Chebyshev-T basis on [0, 1] with 64 modes
zbasis = ChebyshevT(coords["z"], 64; bounds=(0, 1))

# With dealiasing
zbasis = ChebyshevT(coords["z"], 64; bounds=(0, 1), dealias=3/2)
```

The `ChebyshevT` basis uses Chebyshev polynomials of the first kind,
``T_n(x)``.  This is the most common choice for bounded dimensions because the
associated transforms (based on the DCT) are fast and numerically stable.

### Other polynomial bases

For advanced usage, you can also use other Jacobi-family bases:

```julia
# Chebyshev-U (second kind)
zbasis = ChebyshevU(coords["z"], 64; bounds=(0, 1))

# Legendre polynomials
zbasis = Legendre(coords["z"], 64; bounds=(0, 1))

# General Jacobi polynomials with parameters (a, b)
zbasis = Jacobi(coords["z"], 64, (0, 1), 1.0, 1.0)
```

These are mainly needed when working with specific tau-method formulations or
when the differential operator structure benefits from a particular polynomial
family.

## Combining Bases: A 2D Domain

A typical 2D Cartesian domain combines a Fourier basis (periodic direction)
with a Chebyshev basis (bounded direction):

```julia
using Dedalus

# Parameters
Lx = 4.0       # Periodic domain length
Lz = 1.0       # Bounded domain height
Nx = 256       # Fourier modes
Nz = 64        # Chebyshev modes

# Coordinates and distributor
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64)

# Bases
xbasis = RealFourier(coords["x"], Nx; bounds=(0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"], Nz; bounds=(0, Lz), dealias=3/2)
```

This sets up a channel-like domain: periodic in ``x`` and bounded in ``z``.

## Retrieving Grid Points

Once you have a distributor and bases, you can retrieve the physical-space grid
points using `local_grids`:

```julia
x, z = local_grids(dist, xbasis, zbasis)
```

Each returned array is shaped for broadcasting -- `x` varies along the first
dimension and `z` varies along the second.  This makes it easy to initialize
fields:

```julia
f = Field(dist; bases=(xbasis, zbasis))
f["g"] = @. sin(2*pi*x/Lx) * cos(pi*z/Lz)
```

!!! note "Local vs Global Grids"
    `local_grids` returns only the grid points owned by the current MPI rank.
    For post-processing or visualization on the root process, use `global_grid`
    to retrieve the full grid on a single basis:
    ```julia
    x_global = global_grid(xbasis, dist; scale=1)
    ```

## Curvilinear Coordinate Systems

Dedalus.jl supports several curvilinear coordinate systems with specialized
basis types.

### Polar coordinates (disk and annulus)

```julia
# Polar coordinates: azimuth (phi) and radius (r)
coords = PolarCoordinates("phi", "r")
dist = Distributor(coords; dtype=Float64)

# Full disk (r in [0, radius])
dbasis = DiskBasis(coords, (128, 64), Float64; radius=1.0)

# Annular domain (r in [r_inner, r_outer])
abasis = AnnulusBasis(coords, (128, 64), Float64; radii=(0.5, 1.0))
```

### Spherical surface

```julia
# S2 coordinates for the surface of a sphere
coords = S2Coordinates("phi", "theta")
dist = Distributor(coords; dtype=Float64)

sbasis = SphereBasis(coords, (128, 64), Float64; radius=1.0)
```

### Full spherical (shell and ball)

```julia
# Spherical coordinates: azimuth, colatitude, radius
coords = SphericalCoordinates("phi", "theta", "r")
dist = Distributor(coords; dtype=Float64)

# Spherical shell (r in [r_inner, r_outer])
shell = ShellBasis(coords, (128, 64, 48), Float64; radii=(0.5, 1.0))

# Full ball (r in [0, radius])
ball = BallBasis(coords, (128, 64, 48), Float64; radius=1.0)
```

### Grid points on curvilinear domains

Grid retrieval works the same way for curvilinear bases:

```julia
coords = PolarCoordinates("phi", "r")
dist = Distributor(coords; dtype=Float64)
dbasis = DiskBasis(coords, (128, 64), Float64; radius=1.0)

phi, r = local_grids(dist, dbasis)
```

## Quick Domain Helpers

For prototyping and testing, Dedalus.jl provides convenience functions that
create a coordinate system, distributor, and basis in a single call:

```julia
# 1D Fourier domain
xcoord, dist, xbasis = quick_fourier(128; dealias=3/2)

# 1D Chebyshev domain
xcoord, dist, xbasis = quick_chebyshev(64; dealias=3/2)

# 2D periodic domain
coords, dist, bases = quick_fourier_2d(128; dealias=3/2)

# 2D channel (Fourier x Chebyshev)
coords, dist, bases = quick_channel_2d(128; dealias=3/2)

# 3D channel (Fourier x Fourier x Chebyshev)
coords, dist, bases = quick_channel_3d(64; dealias=3/2)
```

These are handy for quick experiments but should be replaced with explicit
coordinate and basis definitions for production simulations.

## Summary

The basic setup workflow is:

1. Create a coordinate system (`CartesianCoordinates`, `PolarCoordinates`,
   `SphericalCoordinates`, etc.).
2. Create a `Distributor` with the desired data type and (optionally) MPI mesh.
3. Create spectral bases for each dimension (`RealFourier`, `ChebyshevT`, or a
   composite basis like `DiskBasis`).
4. Retrieve grid points with `local_grids`.

With these pieces in place, you are ready to create fields and operators, which
is the topic of the [next tutorial](@ref tutorial_fields_operators).
