# [Performance Tips](@id performance_tips)

This page collects Julia-specific advice for getting the best performance out of
Dedalus.jl. Many of these tips also apply to the Python version, but some are
unique to the Julia implementation.

## Threading: Set `OMP_NUM_THREADS=1`

Dedalus.jl uses FFTW and sparse linear algebra libraries that spawn their own
threads via OpenMP. When Julia's task-based threading interacts with OpenMP
threading, the result is **thread oversubscription** — more threads than physical
cores — which causes severe slowdowns due to context switching.

Dedalus.jl warns at startup if `OMP_NUM_THREADS` is not set to `1`:

```bash
export OMP_NUM_THREADS=1
julia --threads=1 my_simulation.jl
```

!!! tip
    For single-node runs, use Julia with `--threads=1` and `OMP_NUM_THREADS=1`.
    Parallelism should come from MPI, not from threading.

## MPI Parallelism

Dedalus.jl supports distributed-memory parallelism via MPI. Initialize MPI before
creating any Dedalus objects:

```julia
using Dedalus
init_mpi!()
```

Run with:

```bash
export OMP_NUM_THREADS=1
mpiexecjl -n 4 julia my_simulation.jl
```

### Scaling Guidelines

- **1D decomposition**: Dedalus.jl distributes the first (fastest-varying)
  spectral dimension across MPI ranks. The number of ranks must divide the
  number of modes in that dimension.
- **Start small**: Profile with 1 rank first, then scale up. MPI communication
  costs (transposes) can dominate for small problems.
- **Minimize output frequency**: File I/O serializes across ranks. Write
  analysis outputs less frequently than every time step.

## FFTW Planning

Dedalus.jl uses FFTW for spectral transforms. The planning rigor controls how
much time FFTW spends optimizing the transform plan at startup:

| Rigor | Startup cost | Runtime speed | When to use |
|:------|:-------------|:--------------|:------------|
| `"estimate"` | Milliseconds | Good | Short tests, debugging |
| `"measure"` | Seconds | Better | Default; good for most runs |
| `"patient"` | Minutes | Best | Long production runs |
| `"exhaustive"` | Hours | Marginally better | Almost never worthwhile |

Set in `dedalus.toml`:

```toml
[transforms-fftw]
PLANNING_RIGOR = "patient"
```

For short debugging runs, drop to `"estimate"` to skip the planning overhead.

## Dealias Factors

Dealiasing pads the grid to prevent aliasing errors from nonlinear terms. The
standard `3/2` rule adds 50% more grid points, which increases memory and
transform cost:

```julia
xbasis = RealFourier(xcoord, N, (0, Lx); dealias=3/2)
```

- Use `dealias=3/2` (default for most examples) for quadratic nonlinearities.
- Use `dealias=2` for cubic nonlinearities.
- Use `dealias=1` (no padding) only for purely linear problems.

Reducing `dealias` saves memory and compute time but risks aliasing-driven
instabilities.

## Time-Stepper Selection

The choice of IMEX time-stepper affects both accuracy and computational cost per
step:

**Multistep schemes** (`SBDF1`–`SBDF4`, `CNAB1`, `CNAB2`, `MCNAB2`, `CNLF2`):
- One implicit solve per time step.
- Higher-order schemes (e.g., `SBDF4`) need more history and take longer to start
  but allow larger stable time steps.
- `SBDF2` is a good default for production IVPs.

**Runge-Kutta IMEX schemes** (`RK111`, `RK222`, `RK443`, `RKSMR`, `RKGFY`):
- Multiple implicit solves per time step (one per stage).
- Self-starting — no need for lower-order initialization.
- `RK443` is third-order with four stages: accurate and stable, but costs four
  solves per step.

!!! tip "Rule of thumb"
    Use `SBDF2` for long production runs (fewer solves per step). Use `RK222` or
    `RK443` for startup or when you need self-starting behavior.

## Avoiding Allocations in the Time Loop

Julia's garbage collector can introduce latency spikes. In the main time-stepping
loop:

1. **Pre-allocate scratch arrays** outside the loop.
2. **Use in-place operations** (functions ending in `!`) wherever possible.
3. **Avoid string interpolation** in hot paths — format log messages only when
   actually writing output.
4. **Avoid creating closures** that capture mutable state inside the loop.

```julia
# Good: pre-allocated outside the loop
flow = GlobalArrayReducer(dist)

while solver.proceed
    solver.step!(dt)
    if solver.iteration % 10 == 0
        max_u = reduce_scalar(flow, abs.(u["g"]), maximum)
    end
end
```

## Precompilation

Dedalus.jl benefits from Julia's precompilation. The first `using Dedalus` in a
fresh session compiles the package, which can take a minute or more. Subsequent
loads in the same depot are fast.

For MPI runs, ensure all ranks use the same compiled cache by pre-warming the
depot on a single process before launching the parallel job:

```bash
julia -e 'using Dedalus'         # compile once
mpiexecjl -n 16 julia sim.jl    # all ranks reuse the cache
```

If compile times are a pain during development, consider using
[Revise.jl](https://github.com/timholy/Revise.jl) for interactive iteration.

## Matrix Solver Selection

The implicit solve at each time step (or Runge-Kutta stage) factors and solves a
sparse linear system. Dedalus.jl defaults to `SuperLUColamdSpsolve`. For some
problems, other solvers may be faster:

Configure in `dedalus.toml`:

```toml
[linear_algebra]
MATRIX_SOLVER = "SuperLUColamdSpsolve"
```

## Profiling Checklist

When performance is unsatisfactory, check in this order:

1. `OMP_NUM_THREADS=1`? Thread oversubscription is the most common issue.
2. Is the problem resolution-limited or time-step-limited?
3. FFTW planning rigor appropriate for the run length?
4. Are NCC product matrices sparse? Dense NCCs dominate implicit-solve cost.
5. Is file I/O the bottleneck? Reduce output frequency.
6. MPI scaling: do transposes dominate? Try fewer ranks or 1D decomposition.
