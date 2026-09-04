# [Half Dimensions](@id half_dimensions)

Dedalus.jl's Fourier bases exploit the symmetry structure of real-valued fields to
reduce memory and computation. This page explains the "half dimension" concept and
when to choose each Fourier basis type.

## Real vs Complex Fourier Bases

Dedalus.jl provides two Fourier basis types:

| Basis | Modes | Group shape | Typical use |
|:------|:------|:------------|:------------|
| [`ComplexFourier`](@ref) | ``e^{i k x}`` | `(1,)` | Complex-valued fields, rotating frames |
| [`RealFourier`](@ref) | ``\cos(kx),\; \sin(kx)`` | `(2,)` | Real-valued fields (most physical problems) |

### ComplexFourier

The `ComplexFourier` basis uses complex exponentials:

```math
u(x) = \sum_{k} \hat{u}_k \, e^{i k x}
```

Each wavenumber ``k`` is an independent degree of freedom. The group shape is
`(1,)` — one coefficient per mode.

```julia
xcoord = Coordinate("x")
xbasis = ComplexFourier(xcoord, N, (0, 2pi); dealias=3/2)
```

### RealFourier

The `RealFourier` basis uses cosine-sine pairs:

```math
u(x) = \sum_{k \ge 0} \bigl[ a_k \cos(kx) + b_k \sin(kx) \bigr]
```

For real-valued data, the negative-frequency Fourier coefficients are conjugates of
the positive-frequency ones (``\hat{u}_{-k} = \hat{u}_k^*``), so storing both is
redundant. `RealFourier` stores only the non-redundant half, pairing the cosine
and sine coefficients for each wavenumber into a single group.

The group shape is `(2,)` — two coefficients (cosine and sine) per wavenumber
group. This is the origin of the term **"half dimension"**: the spectral
representation uses roughly half the storage of the equivalent `ComplexFourier`
basis.

```julia
xcoord = Coordinate("x")
xbasis = RealFourier(xcoord, N, (0, Lx); dealias=3/2)
```

## How Half Dimensions Affect Layout

In Dedalus.jl, data arrays are organized by **groups** rather than individual
modes. For a `RealFourier` dimension with ``N`` grid points:

- **Grid space**: ``N`` real values.
- **Coefficient space**: ``\lceil N/2 \rceil`` groups, each containing a cosine
  and sine coefficient (except the zero mode, which has zero sine component).

The internal mode ordering is:

```math
[\cos(0x),\; -\!\sin(0x),\; \cos(1 \cdot x),\; -\!\sin(1 \cdot x),\; \cos(2 \cdot x),\; -\!\sin(2 \cdot x),\; \ldots]
```

The negative signs on the sine components are a convention that simplifies the
transform implementation.

For `ComplexFourier`, each mode is its own group of size 1, so there is no
pairing.

## When to Use Which

!!! tip "Default choice"
    Use `RealFourier` unless you have a specific reason to use `ComplexFourier`.
    Most physical PDEs have real-valued solutions.

**Use `RealFourier` when:**
- All fields are real-valued.
- The problem has no preferred complex phase (e.g., thermal convection, channel
  flow, wave equations with real coefficients).
- You want the memory and speed benefits of storing only the non-redundant half.

**Use `ComplexFourier` when:**
- Fields are genuinely complex-valued.
- The problem has asymmetric dispersion (e.g., rotating flows where positive and
  negative wavenumbers behave differently).
- You are performing a stability analysis where eigenmodes at ``+k`` and ``-k``
  are independent.

## Interaction with Other Bases

Half dimensions only apply to Fourier (periodic) directions. Bounded directions
using Chebyshev or other Jacobi bases do not have this pairing. In a 2D problem
with a periodic ``x`` and bounded ``z``:

```julia
xbasis = RealFourier(xcoord, Nx, (0, Lx); dealias=3/2)
zbasis = ChebyshevT(zcoord, Nz, (-1, 1); dealias=3/2)
```

the ``x`` direction uses half-dimension storage while the ``z`` direction stores
all ``N_z`` Chebyshev coefficients.

## Dealias Factors

Both Fourier types accept a `dealias` keyword (default `1`, i.e., no dealiasing).
A common choice is `3/2` (the "3/2 rule"), which pads the grid by 50% to
eliminate aliasing errors from quadratic nonlinearities:

```julia
xbasis = RealFourier(xcoord, N, (0, Lx); dealias=3/2)
```

The dealias factor multiplies the number of grid points used for transforms but
does not change the number of retained spectral coefficients.
