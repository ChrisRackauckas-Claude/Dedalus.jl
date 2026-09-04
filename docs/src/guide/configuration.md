# [Configuration](@id configuration)

Dedalus.jl reads configuration from TOML files at three levels, merged with
increasing precedence. This page documents the hierarchy, available sections, and
key options.

## Configuration Hierarchy

Configuration files are named `dedalus.toml` and are loaded from three locations:

| Priority | Location | Scope |
|:---------|:---------|:------|
| 1 (lowest) | Package defaults (`src/dedalus.toml`) | Built-in defaults |
| 2 | `~/.dedalus/dedalus.toml` | Per-user settings |
| 3 (highest) | `./dedalus.toml` (working directory) | Per-project settings |

Files are **deep-merged**: a key set at a higher priority overrides the same key
from a lower priority, but unset keys fall through to the default. You only need
to include the keys you want to change.

### Example: Project-Local Override

To use patient FFTW planning for a production run, create `./dedalus.toml` in
your simulation directory:

```toml
[transforms-fftw]
PLANNING_RIGOR = "patient"

[linear_algebra]
MATRIX_SOLVER = "SuperLUColamdSpsolve"
```

All other settings inherit from the user and package defaults.

## Accessing Configuration in Code

```julia
using Dedalus: get_config, get_config_bool

solver_name = get_config("linear_algebra", "MATRIX_SOLVER")
store = get_config_bool("memory", "STORE_OUTPUTS")
```

These functions read the merged configuration. They are typically used internally,
but you can call them to inspect the active settings or to build configuration-aware
scripts.

## Configuration Sections

### `[logging]`

Controls log output levels and destinations.

| Key | Default | Description |
|:----|:--------|:------------|
| `nonroot_level` | `"WARNING"` | Log level for non-root MPI ranks |
| `stdout_level` | `"INFO"` | Log level for console output |
| `file_level` | `"DEBUG"` | Log level for log file output |
| `filename` | `""` | Log file path (empty = no file logging) |

### `[transforms]`

Settings for spectral transforms.

| Key | Default | Description |
|:----|:--------|:------------|
| `DEFAULT_LIBRARY` | `"fftw"` | Transform library to use |
| `GROUP_TRANSFORMS` | `false` | Batch transforms across components |
| `DEALIAS_BEFORE_CONVERTING` | `true` | Apply dealiasing before basis conversion |

### `[transforms-fftw]`

FFTW-specific transform settings.

| Key | Default | Description |
|:----|:--------|:------------|
| `PLANNING_RIGOR` | `"measure"` | FFTW plan optimization level: `"estimate"`, `"measure"`, `"patient"`, `"exhaustive"` |

### `[parallelism]`

MPI and transpose settings.

| Key | Default | Description |
|:----|:--------|:------------|
| `TRANSPOSE_LIBRARY` | `"fftw"` | Library for MPI transposes |
| `GROUP_TRANSPOSES` | `true` | Batch transposes across field components |

### `[matrix_construction]`

Controls how the discretized matrix system is assembled.

| Key | Default | Description |
|:----|:--------|:------------|
| `BC_TOP` | `false` | Place boundary condition rows at top of matrix |
| `TAU_LEFT` | `false` | Place tau columns on the left side |
| `INTERLEAVE_COMPONENTS` | `false` | Interleave vector components in the system matrix |

These settings affect the matrix structure and can influence solver performance.
The defaults match the Python Dedalus conventions.

### `[linear_algebra]`

Sparse linear algebra settings.

| Key | Default | Description |
|:----|:--------|:------------|
| `MATRIX_SOLVER` | `"SuperLUColamdSpsolve"` | Sparse direct solver for implicit systems |
| `MATRIX_FACTORIZER` | (solver-dependent) | Factorization backend |

### `[memory]`

Memory management settings.

| Key | Default | Description |
|:----|:--------|:------------|
| `STORE_OUTPUTS` | `true` | Cache operator output fields for reuse |

Setting `STORE_OUTPUTS = false` reduces memory usage at the cost of recomputing
outputs each time they are needed. Useful for memory-constrained runs.

### `[analysis]`

Settings for file output and analysis handlers.

| Key | Default | Description |
|:----|:--------|:------------|
| `FILEHANDLER_MODE_DEFAULT` | `"overwrite"` | Default file mode: `"overwrite"` or `"append"` |
| `FILEHANDLER_PARALLEL_DEFAULT` | `"virtual"` | Parallel I/O strategy: `"virtual"` or `"gather"` |

## Full Default Configuration

For reference, the complete default configuration shipped with the package is in
`src/dedalus.toml`. You can view it with:

```julia
using Dedalus
println(read(joinpath(pkgdir(Dedalus), "src", "dedalus.toml"), String))
```

## Tips

!!! tip "Keep overrides minimal"
    Only set the keys you need to change. This makes it easy to see what differs
    from defaults and avoids staleness if defaults change between versions.

!!! tip "Per-user vs per-project"
    Use `~/.dedalus/dedalus.toml` for machine-wide preferences (e.g., FFTW
    planning rigor appropriate for your hardware). Use `./dedalus.toml` for
    problem-specific settings (e.g., matrix solver choice for a particular
    simulation).
