# Dedalus.jl

**Dedalus.jl** is a Julia translation of the Python
[Dedalus v3](https://dedalus-project.org/) framework for solving partial
differential equations using spectral methods.  It provides a flexible,
high-performance environment for formulating and solving PDEs in a wide range of
geometries, with built-in support for MPI parallelism and implicit-explicit
(IMEX) time-stepping.

## Key Features

- **Spectral methods** -- Galerkin discretization with sparse tau formulations
  for high-accuracy PDE solutions.
- **Multiple coordinate systems** -- Cartesian, polar, cylindrical, spherical,
  disk, annulus, sphere, shell, and ball geometries.
- **Rich basis library** -- `RealFourier`, `ComplexFourier`, `ChebyshevT`,
  `ChebyshevU`, `Jacobi`, `Legendre`, and `Ultraspherical` bases for 1D
  intervals; composite bases (`DiskBasis`, `AnnulusBasis`, `SphereBasis`,
  `ShellBasis`, `BallBasis`) for curvilinear domains.
- **Flexible problem types** -- Initial value problems ([`IVP`](@ref)),
  eigenvalue problems ([`EVP`](@ref)), linear boundary value problems
  ([`LBVP`](@ref)), and nonlinear boundary value problems ([`NLBVP`](@ref)).
- **IMEX time-stepping** -- Multistep schemes (CNAB, SBDF) and implicit-explicit
  Runge-Kutta schemes (RK222, RK443, and others).
- **MPI parallelism** -- Distributed-memory parallelism via MPI.jl with
  configurable processor meshes and pencil decompositions.
- **Analysis framework** -- Built-in evaluator and file handlers for HDF5
  output, CFL computation, and global flow property tracking.

## Getting Started

If you are new to Dedalus.jl, start with the [Installation](@ref installation)
page, then read the [Methodology](@ref methodology) overview to understand the
mathematical framework.  The [Tutorials](@ref tutorials_overview) walk through
the core API step by step:

1. [Coordinates and Bases](@ref tutorial_coords_bases) -- setting up coordinate
   systems and spectral bases.
2. [Fields and Operators](@ref tutorial_fields_operators) -- creating fields,
   populating data, and applying differential operators.
3. [Problems and Solvers](@ref tutorial_problems_solvers) -- formulating and
   solving LBVP, EVP, and IVP problems.
4. [Analysis](@ref tutorial_analysis) -- output handling, CFL conditions, and
   flow diagnostics.

For worked physics examples, see the [Examples](@ref) section.  For a detailed
function-by-function listing, consult the [API Reference](@ref).

## Comparison with Python Dedalus

Dedalus.jl mirrors the Python Dedalus v3 API as closely as possible.
Constructor signatures, equation syntax, and overall workflow are intentionally
similar to ease migration.  Julia-specific differences -- such as 1-based
indexing, `snake_case!` mutating conventions, and multiple dispatch -- are
documented in the [User Guide](@ref).

## Citing

If you use Dedalus.jl in your research, please cite the original Dedalus paper:

> Burns, K. J., Vasil, G. M., Oishi, J. S., Lecoanet, D., & Brown, B. P.
> (2020). *Dedalus: A flexible framework for numerical simulations with spectral
> methods.* Physical Review Research, 2(2), 023068.

## Contents

```@contents
Pages = [
    "installation.md",
    "methodology.md",
    "tutorials/index.md",
    "tutorials/tutorial_1_coords_bases.md",
    "tutorials/tutorial_2_fields_operators.md",
    "tutorials/tutorial_3_problems_solvers.md",
    "tutorials/tutorial_4_analysis.md",
]
Depth = 2
```
