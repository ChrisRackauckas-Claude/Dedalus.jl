# Dedalus.jl

A Julia translation of the [Python Dedalus](https://dedalus-project.org/) spectral PDE solver (v3.0.5). Dedalus.jl solves differential equations using spectral methods and provides a flexible framework for specifying and solving a wide variety of partial differential equations.

## Features

- **Spectral methods**: Fourier, Chebyshev, Jacobi, and spherical harmonic bases
- **Multiple geometries**: Cartesian, cylindrical, spherical (ball, shell, disk, annulus)
- **Problem types**: Linear and nonlinear boundary value problems (LBVP/NLBVP), eigenvalue problems (EVP), and initial value problems (IVP)
- **IMEX time-stepping**: Implicit-explicit Runge-Kutta schemes for stiff systems
- **MPI parallelism**: Distributed-memory parallelism via MPI.jl
- **HDF5 output**: Snapshot and time-series output in HDF5 format

## Installation

```julia
using Pkg

# Install from repository URL
Pkg.add(url="https://github.com/DedalusProject/Dedalus.jl")

# Or for local development
Pkg.develop(path=".")
```

## Quick Example

Solve a 1D Poisson equation with Dirichlet boundary conditions:

```julia
using Dedalus

# Build a 1D Chebyshev domain
coord = Dedalus.Coordinate("x")
dist = Dedalus.Distributor(coord)
basis = Dedalus.Chebyshev(coord, size=32, bounds=(0, 1))

# Set up fields
u = dist.Field(bases=basis)
tau = dist.Field()  # lifting parameter for boundary condition

# Define the problem: d2(u)/dx2 = -1, u(0) = 0, u(1) = 0
problem = Dedalus.LBVP([u, tau])
problem.add_equation((Dedalus.d(u, 2) + tau, -1))
problem.add_equation((u(x=0), 0))

# Solve
solver = problem.build_solver()
solver.solve()
```

## Building Documentation

```bash
cd docs
julia --project=. -e 'using Pkg; Pkg.develop(PackageSpec(path="..")); Pkg.instantiate()'
OMP_NUM_THREADS=1 julia --project=. make.jl
```

## Running Tests

Run the unit test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run with example validation (runs the full example scripts and checks results against reference values):

```bash
DEDALUS_TEST_EXAMPLES=true julia --project=. -e 'using Pkg; Pkg.test()'
```

## Reference Values

The `test/reference_values/` directory contains TOML files with known analytical and published reference values used for validating the example scripts:

- `evp_mathieu.toml` -- Mathieu equation eigenvalues (analytical at q=0)
- `lbvp_poisson.toml` -- 2D Poisson residual check parameters
- `ivp_kdv_burgers.toml` -- KdV-Burgers initial condition and boundedness checks
- `nlbvp_lane_emden.toml` -- Lane-Emden first zeros from Boyd (2011)

## License

GPL-3.0. This project is a derivative work of the [Python Dedalus project](https://dedalus-project.org/).
