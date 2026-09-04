# API Reference

This section provides detailed documentation for every public type and function in **Dedalus.jl**. The API is organized into logical groups that mirror the package's layered architecture.

## Module structure

Dedalus.jl is a single module (`Dedalus`) whose source is split across several layers:

| Layer | Sub-page | Description |
|:------|:---------|:------------|
| Coordinates | [Coordinates](@ref) | Coordinate systems (`Coordinate`, `CartesianCoordinates`, `S2Coordinates`, etc.) |
| Fields | [Fields](@ref) | Data containers and lazy evaluation (`Field`, `ScalarField`, `VectorField`, `FutureField`, `Domain`) |
| Bases | [Bases](@ref) | Spectral basis sets and transforms (`ChebyshevT`, `Fourier`, `DiskBasis`, `BallBasis`, etc.) |
| Operators | [Operators](@ref) | Differential and algebraic operators (`differentiate`, `gradient`, `divergence`, `curl`, etc.) |
| Problems | [Problems](@ref) | Problem formulations (`IVP`, `EVP`, `LBVP`, `NLBVP`) and subsystem management |
| Solvers | [Solvers](@ref) | Solver pipelines, time integrators, evaluation, and MPI distribution |
| Extras | [Extras](@ref) | Convenience utilities: CFL conditions, flow properties, plotting helpers, quick domain constructors, and matrix solvers |

## Conventions

- All types and functions live in the `Dedalus` module and are accessed after `using Dedalus`.
- Source file paths shown in `@autodocs` blocks are relative to `src/` in the package tree.
- Docstrings follow Julia conventions: the first line is a one-sentence summary, followed by an extended description with arguments, examples, and cross-references.
