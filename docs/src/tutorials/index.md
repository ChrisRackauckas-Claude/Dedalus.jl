# [Tutorials](@id tutorials_overview)

These tutorials introduce the core components of Dedalus.jl step by step.  Each
tutorial builds on the previous one, progressing from setting up coordinate
systems through to running full simulations with analysis output.

## Prerequisites

All tutorials assume you have Dedalus.jl installed and working.  See the
[Installation](@ref installation) page if you have not set up the package yet.
A basic familiarity with Julia (arrays, functions, broadcasting) is assumed, but
no prior experience with spectral methods is required.

## Tutorial List

### 1. [Coordinates and Bases](@ref tutorial_coords_bases)

Learn how to create coordinate systems and spectral bases -- the foundation of
every Dedalus.jl simulation.  Covers:
- Defining Cartesian, polar, and spherical coordinate systems
- Creating Fourier and Chebyshev bases on those coordinates
- Setting up a `Distributor` for data management and parallelism
- Retrieving grid points with `local_grids`

### 2. [Fields and Operators](@ref tutorial_fields_operators)

Learn how to create scalar, vector, and tensor fields, populate them with data,
and apply differential operators.  Covers:
- Creating `Field`, `VectorField`, and `TensorField` objects
- Accessing grid-space (`"g"`) and coefficient-space (`"c"`) data
- Using `differentiate`, `gradient`, `divergence`, `curl`, and `laplacian`
- Interpolation, integration, and `lift` operations

### 3. [Problems and Solvers](@ref tutorial_problems_solvers)

Learn how to formulate PDEs as Dedalus problem objects and solve them.  Covers:
- Setting up a linear boundary value problem (`LBVP`)
- Adding equations and boundary conditions with `add_equation!`
- Building and running solvers with `build_solver` and `solve!`
- Brief examples of eigenvalue problems (`EVP`) and initial value problems
  (`IVP`)

### 4. [Analysis](@ref tutorial_analysis)

Learn how to extract and save simulation data during time-stepping.  Covers:
- The evaluator and handler system
- Writing field snapshots to HDF5 with `add_file_handler` and `add_task!`
- Computing adaptive timesteps with `CFL`
- Tracking global flow diagnostics with `GlobalFlowProperty`

## After the Tutorials

Once you have completed the tutorials, explore the [Examples](@ref) section for
full physics simulations spanning eigenvalue problems, boundary value problems,
and time-dependent simulations in Cartesian, disk, annulus, sphere, shell, and
ball geometries.  The [User Guide](@ref) covers advanced topics including the
tau method, gauge conditions, performance tuning, and configuration.
