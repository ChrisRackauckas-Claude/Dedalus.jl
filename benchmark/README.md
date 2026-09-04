# Dedalus.jl Performance Benchmarks

## Overview

This directory contains profiling and type-stability audit scripts for
measuring and improving Dedalus.jl performance. The optimizations target
three pillars: type stability, preallocation, and non-allocating inner loops.

## Scripts

- **`profile_ivp.jl`** — Runs a small 2D Rayleigh-Benard IVP (Nx=32, Nz=16)
  for 100 timesteps, collecting wall time, allocation counts, profile samples,
  and per-step timing distributions.
- **`typestability_audit.jl`** — Static and runtime analysis of type stability
  across core modules. Reports `::Any` annotations, unstable return types,
  and prioritized recommendations.

## Performance Annotation Summary

### Before (Milestone 3 baseline)

| Annotation     | Count |
|---------------|-------|
| `@inbounds`   | 20    |
| `@simd`       | 2     |
| `@inline`     | 0     |
| `@.` (fused)  | 20    |
| `mul!`        | 6     |
| `ldiv!`       | 1     |
| `::Any` fields| ~639  |
| `::Nothing` return annotations | 0 |

### After (Milestone 4 optimizations)

| Annotation     | Count | Change  |
|---------------|-------|---------|
| `@inbounds`   | 48    | +28     |
| `@simd`       | 10    | +8      |
| `@inline`     | 46    | +46     |
| `@.` (fused)  | 20    | (already optimal) |
| `mul!`        | 6     | (verified optimal) |
| `ldiv!`       | 1     | (verified optimal) |
| `::Any` fields| ~630  | -9 (struct fields tightened) |
| `::Nothing` return annotations | 66 | +66 |

## Key Optimizations Applied

### Phase 2: Type Stability
- **Container type tightening**: Replaced `::Any` with concrete types in
  `SolverData`, `FieldSystem`, `Subsystem`, `Subproblem`, `Evaluator`,
  `Distributor`, `MultistepIMEXData`, `RungeKuttaIMEXData` structs.
- **Return type annotations**: Added `::Nothing`, `::Bool`, `::String`,
  `::Layout` annotations to 66+ hot-path functions including `operate`,
  `evaluate_future`, `get_layout_object`, `preset_layout!`, `change_layout!`,
  `step!`, `evaluate_scheduled`, etc.
- **Field accessor stabilization**: `get_layout_object` returns `::Layout`
  instead of `Any`, eliminating type instability on every `f["g"]`/`f["c"]`
  access.

### Phase 3: Preallocation
- **Timestepper workspace**: Cached filtered subproblems list in data structs
  (`_nonempty_subproblems`), eliminating per-step array allocation. Reuse
  `LHS_solvers` vector via `resize!`+`fill!` instead of reallocating.
- **Evaluator buffers**: Pre-allocated `_scheduled_buf`, `_tasks_buf`,
  `_unfinished_buf` on the Evaluator struct, reused via `empty!`/`push!`
  pattern to avoid per-evaluation list comprehensions.
- **FutureField output reuse**: Verified that output fields are already cached
  via `_cache` Dict pattern (matching Python's `@CachedAttribute`).
- **Transform plan caching**: Verified FFTW plans and Jacobi matrices are
  lazily computed once via `CachedAttribute` and reused across evaluations.
- **Solver factorization reuse**: Verified LHS factorizations are cached and
  only recomputed when the timestep changes (`update_LHS` flag).

### Phase 4: Non-Allocating Inner Loops
- **Transform kernels**: Added `@inbounds` to `resize_coeffs_complex!`,
  `unpack_rescale_real!`, `repack_rescale_real!` loops. Added `@simd` to
  CSR matrix inner loops in `linalg.jl`.
- **Arithmetic operators**: Verified all `operate` methods already use `@.`
  fused broadcasting. Added `@inbounds` to tensor contraction loop in
  `_einsum_contract!`.
- **Operator apply methods**: Added `@inbounds` to 19 loops across
  `PolarMOperator`, `SphericalTrace`, `PolarTrace`, `DirectProductTrace`,
  `SphericalEllOperator`, `SphericalCurl`, and other operator `operate`
  methods.
- **Utility inlining**: Added `@inline` to 46 small functions across
  `distributor.jl`, `coords.jl`, `cache.jl`, `array.jl`, `timesteppers.jl`,
  `system.jl`.

## Remaining Performance Opportunities

1. **Expression tree polymorphism**: The `evaluate_future` recursive walk
   dispatches on heterogeneous `AbstractFuture` subtypes. A `FunctionWrappers.jl`
   devirtualization strategy could further reduce dispatch overhead.
2. **Sparse solver workspace**: Pre-allocating solve output buffers per
   subproblem would eliminate one allocation per subproblem per step.
3. **MPI transpose buffers**: When MPI mode is enabled, transpose communication
   buffers should be allocated once per layout pair.
4. **Parametric expression tree nodes**: Parameterizing expression tree nodes
   on operand types (currently `Any`) would allow full compile-time
   specialization, at the cost of increased compilation time for deep trees.
