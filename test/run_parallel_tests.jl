"""
MPI parallel test runner for Dedalus.jl.

Launches parallel test files using mpiexecjl with the appropriate number of
processes. Each test file is run as a separate MPI job.

Usage:
    julia run_parallel_tests.jl                  # Run all parallel tests
    julia run_parallel_tests.jl output            # Run only output tests
    julia run_parallel_tests.jl spherical         # Run only spherical tests

Prerequisites:
    - MPI.jl must be installed
    - mpiexecjl must be available (install via MPI.install_mpiexecjl())
    - At least 4 MPI processes are required for these tests
"""

using Pkg

# Ensure MPI.jl is available and mpiexecjl is installed
try
    @eval using MPI
catch
    @error "MPI.jl is not installed. Please add it with: Pkg.add(\"MPI\")"
    exit(1)
end

# Find mpiexec
mpiexec_path = nothing
try
    mpiexec_path = MPI.mpiexec_path
catch
    # Fallback: look for mpiexecjl or mpiexec in PATH
    mpiexec_path = Sys.which("mpiexecjl")
    if mpiexec_path === nothing
        mpiexec_path = Sys.which("mpiexec")
    end
end

if mpiexec_path === nothing
    @error """
    Could not find mpiexec or mpiexecjl.
    Install mpiexecjl with:
        using MPI
        MPI.install_mpiexecjl()
    Or ensure mpiexec is in your PATH.
    """
    exit(1)
end

# Configuration
const NPROCS = 4  # Minimum 4 processes for mesh=(2,2)
const TEST_DIR = @__DIR__

# Define parallel test files and their required process counts
const PARALLEL_TESTS = [
    (file="test_output_parallel.jl",                nprocs=NPROCS, label="Output Parallel"),
    (file="test_spherical3d_arithmetic_parallel.jl", nprocs=NPROCS, label="Spherical 3D Arithmetic Parallel"),
]

# Parse command-line filter
filter_arg = length(ARGS) >= 1 ? lowercase(ARGS[1]) : ""

function should_run(test_entry)
    isempty(filter_arg) && return true
    return occursin(filter_arg, lowercase(test_entry.label)) ||
           occursin(filter_arg, lowercase(test_entry.file))
end

# Build the wrapper script that each MPI worker will execute.
# This initialises MPI before loading Dedalus and running the test file.
function make_wrapper_script(test_file::String)
    return """
    using MPI
    MPI.Init()

    # Initialise Dedalus MPI support
    using Dedalus
    Dedalus.init_mpi!()

    using Test

    include("$(escape_string(test_file))")

    MPI.Finalize()
    """
end

# Run tests
println("=" ^ 70)
println("Dedalus.jl Parallel Test Runner")
println("=" ^ 70)
println("  mpiexec:   $(mpiexec_path)")
println("  processes: $(NPROCS)")
println("  test dir:  $(TEST_DIR)")
println("=" ^ 70)

results = Tuple{String, Bool, Float64}[]  # (label, passed, elapsed)

for test in PARALLEL_TESTS
    should_run(test) || continue

    test_path = joinpath(TEST_DIR, test.file)
    if !isfile(test_path)
        @warn "Test file not found: $(test_path)"
        push!(results, (test.label, false, 0.0))
        continue
    end

    println("\n--- Running: $(test.label) ($(test.nprocs) processes) ---")

    # Write a temporary wrapper script
    wrapper_path = joinpath(TEST_DIR, "_parallel_wrapper_$(test.file)")
    wrapper_code = make_wrapper_script(test_path)
    open(wrapper_path, "w") do io
        write(io, wrapper_code)
    end

    t0 = time()
    success = false
    try
        # Use mpiexec to launch the test
        cmd = `$(mpiexec_path) -n $(test.nprocs) $(Base.julia_cmd()) --project=$(dirname(TEST_DIR)) $(wrapper_path)`
        run(cmd)
        success = true
    catch e
        @error "Test failed" test=test.label exception=e
        success = false
    finally
        elapsed = time() - t0
        push!(results, (test.label, success, elapsed))
        # Clean up wrapper
        rm(wrapper_path; force=true)
    end
end

# Summary
println("\n" * "=" ^ 70)
println("RESULTS SUMMARY")
println("=" ^ 70)
all_passed = true
for (label, passed, elapsed) in results
    status = passed ? "PASS" : "FAIL"
    all_passed = all_passed && passed
    println("  [$(status)] $(label)  ($(round(elapsed, digits=1))s)")
end
println("=" ^ 70)
n_pass = count(r -> r[2], results)
n_total = length(results)
println("  $(n_pass)/$(n_total) test suites passed.")

if !all_passed
    exit(1)
end
