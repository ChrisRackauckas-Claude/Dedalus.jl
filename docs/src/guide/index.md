# User Guide

Welcome to the Dedalus.jl guide. These pages cover the core concepts, techniques, and
practical considerations for solving PDEs with Dedalus.jl.

Dedalus.jl is a Julia translation of the Python
[Dedalus v3](https://dedalus-project.org/) spectral PDE solver. It solves partial
differential equations using spectral methods with a tau formulation, sparse matrix
solvers, and IMEX time-stepping schemes.

## How Dedalus.jl Works

At a high level, the workflow for solving a PDE with Dedalus.jl is:

1. **Define the domain** — Choose coordinates, bases (Fourier, Chebyshev, etc.),
   and create a `Distributor` to manage data layout.
2. **Create fields** — Allocate `Field` objects for unknowns and parameters.
3. **Formulate the problem** — Construct one of the four problem types
   ([`IVP`](@ref), [`EVP`](@ref), [`LBVP`](@ref), or [`NLBVP`](@ref)) and add
   equations as strings.
4. **Build and run the solver** — Call `build_solver` with an appropriate
   time-stepper (for IVPs) or directly solve (for boundary value and eigenvalue
   problems).
5. **Analyze results** — Extract data from fields in grid or coefficient space,
   write outputs with file handlers.

Equations are specified as strings (e.g., `"dt(u) - nu*lap(u) = f"`) that
Dedalus.jl parses, symbolically differentiates (for Jacobians), and discretizes
into sparse matrix pencils. The implicit (left-hand side) terms are solved with
sparse direct solvers, while explicit (right-hand side) terms are evaluated
pseudo-spectrally on the grid.

## Where to Start

If you are **new to spectral methods**, begin with
[Problem Formulations](@ref problem_formulations) and
[The Tau Method](@ref tau_method) — these explain the mathematical framework
underlying the solver.

If you are **migrating from Python Dedalus**, jump to
[Differences from Python Dedalus](@ref python_differences) for a quick
translation guide, then consult [Configuration](@ref configuration) and
[Performance Tips](@ref performance_tips) for Julia-specific setup.

If you are **debugging a simulation**, see
[Troubleshooting](@ref troubleshooting) for solutions to common issues.

## Contents

### Core Concepts

- **[Problem Formulations](@ref problem_formulations)** — The four problem types
  (IVP, EVP, LBVP, NLBVP) and their mathematical formulations. Start here to
  understand how Dedalus.jl frames PDEs as algebraic systems.

- **[The Tau Method](@ref tau_method)** — Why tau terms are needed in spectral
  discretizations, how `Lift` adds them, and choosing the correct number of tau
  terms per equation.

- **[Gauge Conditions](@ref gauge_conditions)** — When and why gauge conditions
  (such as `integ(p) = 0`) are required, with a focus on incompressible flow
  problems.

### Spectral Bases

- **[Half Dimensions](@ref half_dimensions)** — How `RealFourier` bases create
  "half" dimensions by pairing cosine and sine modes, and when to choose
  `RealFourier` vs `ComplexFourier`.

- **[General Functions and NCCs](@ref general_functions)** — Using arbitrary
  functions in equations via non-constant coefficient (NCC) expansion and
  the `GeneralFunction` wrapper.

### Practical Usage

- **[Performance Tips](@ref performance_tips)** — Julia-specific advice for
  getting the best performance: threading, MPI scaling, FFTW planning, memory
  management, and precompilation.

- **[Configuration](@ref configuration)** — Controlling Dedalus.jl behavior
  through `dedalus.toml` files: available sections, option hierarchy, and
  key settings.

- **[Troubleshooting](@ref troubleshooting)** — Solutions to common issues
  including MPI errors, FFTW planning failures, memory pressure, NLBVP
  convergence, and eigenvalue sorting.

### Migration

- **[Differences from Python Dedalus](@ref python_differences)** — A reference
  for users migrating from Python Dedalus v3: constructor syntax, mutating
  conventions, dispatch patterns, and idiomatic Julia equivalents.
