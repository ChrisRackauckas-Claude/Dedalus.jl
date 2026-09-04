# [Installation](@id installation)

## Requirements

- **Julia 1.10 or later**.  Dedalus.jl relies on language features and standard
  library improvements introduced in Julia 1.10.
- **MPI** (optional but recommended for parallel runs).  The MPI.jl package
  handles installation automatically on most systems.
- **FFTW** and **HDF5** are provided through Julia's artifact system and do not
  need to be installed manually.

## Installing Dedalus.jl

### From the Julia package registry

Open a Julia REPL and run:

```julia
using Pkg
Pkg.add("Dedalus")
```

This installs the latest registered release along with all dependencies.

### Development install from Git

To work with the latest source (or to contribute), clone the repository and
install in development mode:

```julia
using Pkg
Pkg.develop(url="https://github.com/DedalusProject/Dedalus.jl.git")
```

Or, if you have already cloned the repository locally:

```julia
using Pkg
Pkg.develop(path="/path/to/Dedalus.jl")
```

## Dependencies

Dedalus.jl depends on the following Julia packages, all of which are installed
automatically:

| Package           | Purpose                                       |
|:------------------|:----------------------------------------------|
| `MPI.jl`          | Distributed-memory parallelism                |
| `FFTW.jl`         | Fast Fourier transforms                       |
| `HDF5.jl`         | Reading and writing HDF5 analysis output       |
| `SparseArrays`    | Sparse matrix assembly and storage            |
| `LinearAlgebra`   | Dense linear algebra (eigenvalue solves, etc.) |
| `SpecialFunctions`| Jacobi polynomial evaluation                  |
| `OrderedCollections`| Ordered dictionaries for deterministic output |

## Verifying the installation

After installing, verify that the package loads correctly:

```julia
using Dedalus
println(pkgversion(Dedalus))
```

For a quick functional test, try building a minimal domain:

```julia
using Dedalus

coords = CartesianCoordinates("x")
dist = Distributor(coords; dtype=Float64)
xbasis = ChebyshevT(coords["x"], 32; bounds=(0, 1))
f = Field(dist; bases=(xbasis,))
x = local_grids(dist, xbasis)
f["g"] = @. sin(pi * x[1])
println("Field created successfully with $(length(f["g"])) grid points.")
```

## MPI configuration

Dedalus.jl uses MPI.jl for parallel execution.  To run a script on multiple
processes:

```bash
mpiexec -n 4 julia my_simulation.jl
```

When running with MPI, you can specify a processor mesh to control the domain
decomposition:

```julia
dist = Distributor(coords; dtype=Float64, mesh=(2, 2))
```

The `mesh` tuple defines how the distributor splits the domain across MPI
ranks.  For a 2D problem on 4 processes, `mesh=(4,)` distributes one dimension
while `mesh=(2, 2)` distributes two.  If `mesh` is not specified, Dedalus.jl
uses a one-dimensional decomposition.

## Environment variables

### `OMP_NUM_THREADS`

When using MPI parallelism, it is strongly recommended to set:

```bash
export OMP_NUM_THREADS=1
```

This prevents FFTW and BLAS from spawning additional threads within each MPI
rank, which typically causes oversubscription and hurts performance on
multi-core nodes.

### `JULIA_NUM_THREADS`

Similarly, unless you are intentionally using Julia's multithreading in
combination with MPI, set:

```bash
export JULIA_NUM_THREADS=1
```

### FFTW planning

FFTW plan creation can be controlled through the FFTW.jl package.  By default,
Dedalus.jl uses `FFTW.MEASURE` for transform planning.  For faster startup at
the cost of slightly slower transforms, you can set:

```julia
import FFTW
FFTW.set_num_threads(1)
```

## Platform notes

### Linux

No special steps required.  MPI.jl downloads a compatible MPI implementation
automatically.  To use a system MPI instead (e.g., for optimized interconnect
support on a cluster), configure MPI.jl before installing Dedalus:

```julia
using MPIPreferences
MPIPreferences.use_system_binary()
```

### macOS

Works out of the box with the bundled MPI.  For Apple Silicon (ARM64), ensure
you are using a native ARM Julia build for optimal performance.

### HPC clusters

On HPC systems, you typically want to use the system MPI for best performance
with the cluster interconnect (InfiniBand, etc.).  See the
[MPI.jl documentation](https://juliaparallel.org/MPI.jl/stable/configuration/)
for detailed configuration instructions.

## Building the documentation locally

To build this documentation locally:

```bash
cd docs/
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. make.jl
```

The generated HTML will appear in `docs/build/`.
