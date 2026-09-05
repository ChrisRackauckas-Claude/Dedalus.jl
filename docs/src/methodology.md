# [Methodology](@id methodology)

This page provides an overview of the mathematical and computational methods
underlying Dedalus.jl.  For a rigorous treatment, see the
[Dedalus paper](https://doi.org/10.1103/PhysRevResearch.2.023068) (Burns et al.,
2020).

## Spectral Methods for PDEs

Dedalus.jl solves partial differential equations by expanding unknown fields in
truncated series of basis functions.  Given a set of basis functions
``\phi_n(x)``, a field ``u(x)`` is approximated as

```math
u(x) = \sum_{n=0}^{N-1} \hat{u}_n \, \phi_n(x),
```

where ``\hat{u}_n`` are the spectral coefficients.  The PDE is then recast as a
system of algebraic equations for these coefficients.

Spectral methods offer **exponential (spectral) convergence** for smooth
solutions: the error decreases faster than any polynomial in ``1/N``.  This
makes them extremely efficient for problems with smooth solutions, requiring far
fewer degrees of freedom than finite-difference or finite-element methods of
comparable accuracy.

## Galerkin Discretization

Dedalus.jl uses a **Galerkin** (or more precisely, Petrov-Galerkin) approach.
The PDE residual is projected onto a set of test functions ``\psi_m(x)``,
requiring that

```math
\langle \psi_m, \, \mathcal{L}[u] - f \rangle = 0, \quad m = 0, 1, \ldots, N-1,
```

where ``\mathcal{L}`` is the differential operator and ``f`` is the forcing.
The inner product ``\langle \cdot, \cdot \rangle`` depends on the choice of
basis.  For orthogonal polynomial bases on ``[-1, 1]`` with weight function
``w(x)``:

```math
\langle f, g \rangle = \int_{-1}^{1} f(x) \, g(x) \, w(x) \, dx.
```

By selecting test and trial bases carefully, Dedalus.jl ensures that the
resulting matrix systems are **sparse** (typically banded), enabling efficient
direct solves.

## The Tau Method

Boundary conditions cannot be imposed directly in a Galerkin framework because
the basis functions generally do not satisfy the boundary conditions
individually.  Dedalus.jl uses the **tau method**, which replaces the
highest-order Galerkin equations with boundary condition equations and adds
compensating **tau terms** to the PDE.

For a second-order ODE such as ``u'' = f`` with two boundary conditions, the
tau-corrected equation becomes

```math
u''(x) = f(x) + \tau_1 \, P_{N-1}(x) + \tau_2 \, P_{N-2}(x),
```

where ``\tau_1`` and ``\tau_2`` are additional unknowns (the "tau variables")
and ``P_k`` are polynomials from an appropriate basis.  The tau variables are
determined jointly with the solution coefficients by the full system (PDE
equations + boundary conditions).

In Dedalus.jl, tau terms are added using the [`Lift`](@ref) operator, which
lifts a lower-dimensional field (the tau variable) into the equation's function
space:

```julia
lift_basis = derivative_basis(zbasis, 2)
lift = (A, n) -> Lift(A, lift_basis, n)
add_equation!(problem, "lap(u) + lift(tau_1, -1) + lift(tau_2, -2) = f")
```

## Supported Bases

### Fourier Bases

For periodic dimensions, Dedalus.jl provides:

- **`ComplexFourier`** -- complex exponential basis
  ``\phi_n(x) = e^{i k_n x}`` where ``k_n = 2\pi n / L``.  Used when the
  solution is complex-valued or when both positive and negative wavenumbers are
  needed.
- **`RealFourier`** -- paired cosine/sine basis for real-valued fields.
  Stores cosine and sine modes together, halving the storage compared to a full
  complex representation.

Fourier bases are naturally periodic and require no boundary conditions or tau
terms.  Differentiation in Fourier space is a simple multiplication by ``ik_n``.

### Chebyshev and Jacobi Bases

For non-periodic (bounded) dimensions, Dedalus.jl supports:

- **`ChebyshevT`** -- Chebyshev polynomials of the first kind, ``T_n(x)``.
  The most commonly used non-periodic basis, combining spectral accuracy with
  well-conditioned transforms via the DCT.
- **`ChebyshevU`** -- Chebyshev polynomials of the second kind, ``U_n(x)``.
- **`Legendre`** -- Legendre polynomials ``P_n(x)``, a special case of Jacobi
  polynomials with ``\alpha = \beta = 0``.
- **`Ultraspherical`** -- Gegenbauer (ultraspherical) polynomials
  ``C_n^{(\alpha)}(x)``, a one-parameter family useful for sparse
  differentiation matrices.
- **`Jacobi`** -- General Jacobi polynomials ``P_n^{(\alpha, \beta)}(x)``,
  the most general family.  All of the above are special cases.

The Jacobi polynomial ``P_n^{(a,b)}(x)`` satisfies the orthogonality relation

```math
\int_{-1}^{1} P_m^{(a,b)}(x) \, P_n^{(a,b)}(x) \, (1-x)^a (1+x)^b \, dx
= h_n^{(a,b)} \, \delta_{mn},
```

where ``h_n^{(a,b)}`` is the normalization constant.  A key property exploited
by Dedalus.jl is that differentiation maps Jacobi polynomials to Jacobi
polynomials with shifted parameters:

```math
\frac{d}{dx} P_n^{(a,b)}(x) = \frac{n + a + b + 1}{2} \, P_{n-1}^{(a+1, b+1)}(x).
```

This means the differentiation matrix, expressed as a conversion between Jacobi
bases with different parameters, is **sparse** (in fact, banded), which is
essential for efficient solves.

### Composite Bases for Curvilinear Geometries

For curvilinear coordinate systems, Dedalus.jl builds composite bases from
products of 1D bases with appropriate symmetry handling:

| Geometry   | Type             | Angular basis      | Radial basis          |
|:-----------|:-----------------|:-------------------|:----------------------|
| Disk       | `DiskBasis`      | Fourier (azimuth)  | Jacobi (radius)       |
| Annulus    | `AnnulusBasis`   | Fourier (azimuth)  | Jacobi (radius)       |
| Sphere     | `SphereBasis`    | Fourier (azimuth)  | Spin-weighted (colat) |
| Shell      | `ShellBasis`     | Sphere (angles)    | Jacobi (radius)       |
| Ball       | `BallBasis`      | Sphere (angles)    | Jacobi (radius)       |

These composite bases automatically handle regularity conditions at coordinate
singularities (e.g., the origin of a disk or the poles of a sphere).

## Problem Types

Dedalus.jl supports four classes of problems:

### Initial Value Problems (IVP)

Time-dependent PDEs of the form

```math
M \frac{\partial \mathbf{X}}{\partial t} = L \mathbf{X} + F(\mathbf{X}, t),
```

where ``M`` is the mass matrix, ``L`` collects linear terms, and ``F`` collects
nonlinear and forcing terms.  These are advanced in time using IMEX schemes.

### Eigenvalue Problems (EVP)

Generalized eigenvalue problems of the form

```math
\sigma M \mathbf{X} = L \mathbf{X},
```

where ``\sigma`` is the eigenvalue.  These arise from linearizing IVPs and
seeking solutions proportional to ``e^{\sigma t}``.

### Linear Boundary Value Problems (LBVP)

Linear systems of the form

```math
L \mathbf{X} = F,
```

where ``L`` is a linear operator and ``F`` is a known forcing.  Solved by
direct sparse matrix factorization.

### Nonlinear Boundary Value Problems (NLBVP)

Nonlinear systems of the form

```math
H(\mathbf{X}) = 0,
```

solved iteratively using Newton's method.  Each Newton step involves solving an
LBVP for the correction.

## IMEX Time-Stepping

Initial value problems are integrated using **implicit-explicit (IMEX)**
time-stepping schemes.  The idea is to split the right-hand side into a stiff
linear part ``L\mathbf{X}`` (treated implicitly) and a nonlinear part
``F(\mathbf{X})`` (treated explicitly):

```math
M \frac{\partial \mathbf{X}}{\partial t} = \underbrace{L \mathbf{X}}_{\text{implicit}}
+ \underbrace{F(\mathbf{X})}_{\text{explicit}}.
```

This avoids the severe timestep restriction that would result from treating
diffusive terms explicitly, while avoiding the cost of solving nonlinear
implicit systems at each step.

### Available Timesteppers

**Multistep schemes** (use information from previous timesteps):

| Scheme   | Order | Description                           |
|:---------|:------|:--------------------------------------|
| `CNAB1`  | 1     | Crank-Nicolson / Adams-Bashforth, 1st |
| `CNAB2`  | 2     | Crank-Nicolson / Adams-Bashforth, 2nd |
| `MCNAB2` | 2     | Modified CNAB2                        |
| `CNLF2`  | 2     | Crank-Nicolson / Leap-Frog, 2nd       |
| `SBDF1`  | 1     | Semi-implicit BDF, 1st                |
| `SBDF2`  | 2     | Semi-implicit BDF, 2nd                |
| `SBDF3`  | 3     | Semi-implicit BDF, 3rd                |
| `SBDF4`  | 4     | Semi-implicit BDF, 4th                |

**Runge-Kutta IMEX schemes** (single-step, multi-stage):

| Scheme  | Stages | Order | Description                |
|:--------|:-------|:------|:---------------------------|
| `RK111` | 1      | 1     | Forward-Backward Euler     |
| `RK222` | 2      | 2     | Ascher, Ruuth, Spiteri     |
| `RK443` | 4      | 3     | Ascher, Ruuth, Spiteri     |
| `RKSMR` | 4      | 3     | Spalart, Moser, Rogers     |
| `RKGFY` | 4      | 3     | Gear, Fukunaga, Yim        |

Runge-Kutta schemes are self-starting and do not require special initialization,
making them simpler to use.  Multistep schemes can be more efficient at higher
order but require a startup procedure (handled automatically by the solver).

## Sparse Linear Algebra

A key design principle of Dedalus.jl is that all matrix systems -- arising from
the implicit part of IMEX stepping, eigenvalue problems, or boundary value
problems -- are **sparse**.  This is achieved through:

1. Choosing bases where differentiation operators are banded (Jacobi family).
2. Using the tau method, which preserves sparsity by adding low-rank
   corrections.
3. Pencil decomposition, which reduces multi-dimensional problems to
   collections of independent 1D matrix problems along the "last" (non-periodic)
   axis.

The resulting sparse systems are solved using direct factorization (typically
sparse LU), giving robust and efficient solves even for high-resolution
problems.

## Parallelism

Dedalus.jl distributes computation across MPI ranks using a **pencil
decomposition**.  The data array for each field is distributed across processors
along one or more dimensions, and transforms are performed by transposing
between "layouts" that align data along the axis being transformed.

The key abstraction is the [`Distributor`](@ref), which manages:

- **Layouts** -- different data distributions optimized for coefficient-space
  or grid-space operations.
- **Transforms** -- spectral transforms along each axis (FFT, DCT, etc.).
- **Transposes** -- MPI all-to-all communications that redistribute data
  between layouts.

This approach scales efficiently to thousands of processors for
three-dimensional problems.
