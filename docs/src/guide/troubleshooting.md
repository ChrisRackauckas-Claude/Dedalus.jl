# [Troubleshooting](@id troubleshooting)

This page covers common issues encountered when using Dedalus.jl and their
solutions.

## MPI Errors

### "MPI not initialized"

Dedalus.jl requires explicit MPI initialization before creating distributed
objects:

```julia
using Dedalus
init_mpi!()  # Must be called before creating Distributor, Field, etc.
```

If you forget this call, operations that require communication will fail with
MPI-related errors.

### Rank count mismatch

The number of MPI ranks must divide the number of modes in the distributed
dimension. If you run with 8 ranks but have only 6 Fourier modes, the
distributor will raise an error.

**Fix**: Choose a mode count that is divisible by your intended rank count, or
reduce the number of ranks.

### Deadlocks during transpose

If the simulation hangs during spectral transforms, possible causes include:

- **Mismatched operations across ranks**: All ranks must call the same sequence
  of solver operations. Conditional logic that causes some ranks to skip a
  `step!` call will deadlock.
- **Insufficient buffer memory**: Large transposes may exceed available memory.
  Reduce the problem size or the number of ranks.

## FFTW Planning Issues

### Slow startup

FFTW planning with `"measure"` or `"patient"` rigor can take seconds to minutes.
For quick debugging runs, reduce the rigor:

```toml
# ./dedalus.toml
[transforms-fftw]
PLANNING_RIGOR = "estimate"
```

### FFTW wisdom not reused

FFTW plans are recomputed each time the program starts. Unlike the Python version,
Dedalus.jl does not currently persist FFTW wisdom to disk. For long production
campaigns with identical grid sizes, this means repeated planning overhead at each
restart.

## Memory Issues

### Out-of-memory on a single node

Spectral PDE solvers can be memory-intensive, especially with:

- High resolution in multiple dimensions.
- Large dealias factors (e.g., `dealias=2`).
- `STORE_OUTPUTS = true` (the default), which caches intermediate operator
  results.

**Mitigations**:

1. Set `STORE_OUTPUTS = false` in `dedalus.toml` to reduce caching.
2. Reduce dealias factors if aliasing errors are acceptable.
3. Distribute across MPI ranks to split the memory footprint.
4. Use lower resolution during development and scale up for production.

### Growing memory over time

If memory usage grows steadily during a simulation, check for:

- **Accumulated analysis outputs** stored in memory. Ensure file handlers are
  flushing to disk.
- **Julia's GC not keeping up**: Call `GC.gc()` periodically in long loops
  (though this adds latency).

## NLBVP Convergence Failures

### Newton iteration does not converge

The Newton-Kantorovich iteration in [`NLBVP`](@ref) solvers can fail to converge
for several reasons:

1. **Poor initial guess**: Newton's method needs a starting point in the basin of
   attraction. Try a continuation strategy: solve at a simpler parameter value
   and gradually increase it.

2. **Insufficient resolution**: If the solution has fine structure that the
   spectral basis cannot resolve, the Newton update will be inaccurate.

3. **Incorrect Jacobian**: Verify that the equations are correctly formulated.
   The Frechet differential is computed symbolically from the equation strings,
   so typos in the equations can silently produce wrong Jacobians.

**Diagnostic**: Print the perturbation norm at each iteration:

```julia
for iter in 1:max_iterations
    newton_iteration!(solver)
    @info "Iteration $iter: perturbation = $(solver.perturbation_norm)"
    if solver.perturbation_norm < tolerance
        break
    end
end
```

Quadratic convergence (the norm decreases as ``\|\delta X\|^2``) indicates a
healthy Newton solve. Linear or no decrease suggests problems.

## Eigenvalue Problems

### Spurious eigenvalues

Spectral discretizations of eigenvalue problems produce spurious eigenvalues
alongside physical ones. These typically have:

- Very large magnitude (near machine precision limits).
- Eigenvectors dominated by the highest spectral modes.

**Filtering strategies**:

1. **Magnitude cutoff**: Discard eigenvalues with ``|\sigma| > \sigma_{\max}``
   for a physically motivated threshold.
2. **Resolution test**: Re-solve at higher resolution. Physical eigenvalues
   converge; spurious ones move.
3. **Eigenvector smoothness**: Physical eigenvectors are smooth; spurious ones
   oscillate at the grid scale.

### Wrong eigenvalue count

If the EVP returns fewer eigenvalues than expected, check that:

- The system is not singular (see [Gauge Conditions](@ref gauge_conditions)).
- The boundary conditions are consistent and sufficient.
- The eigenvalue parameter appears correctly in the equations.

## Equation Parsing

### "Variable not found" errors

Equation strings are parsed against the problem's namespace. If a variable or
parameter is not found:

```julia
# Wrong: 'nu' is not in scope
problem = IVP([u]; namespace=@locals)

# Right: capture 'nu' in the namespace
nu = 1e-3
problem = IVP([u]; namespace=@locals)
```

The [`@locals`](@ref) macro captures all local bindings at the point where it is
called. Ensure all referenced names are defined *before* the `@locals` call.

### Operator precedence surprises

Equation strings follow Python-like parsing rules. Multiplication binds tighter
than addition, but be explicit with parentheses for clarity:

```julia
# Potentially ambiguous
add_equation!(problem, "dt(u) - nu*dx(dx(u)) + lift(tau) = f")
# Clear
add_equation!(problem, "dt(u) - (nu * dx(dx(u))) + lift(tau) = f")
```

## Thread Oversubscription Warning

If you see the warning about `OMP_NUM_THREADS` at startup:

```bash
export OMP_NUM_THREADS=1
```

See [Performance Tips](@ref performance_tips) for details on why this matters.
