"""
Profiling benchmark for Dedalus.jl IVP solver.

Sets up a small 2D Rayleigh-Benard convection problem (Nx=32, Nz=16) and
runs it for 100 timesteps, collecting:
  1. Total wall time via @time
  2. Allocation count and total bytes via @allocated
  3. Profile.@profile flamegraph data (top-20 time-consuming functions)
  4. Per-step timing breakdown
  5. Per-step component breakdown (RHS evaluation, LHS solve, transform,
     transpose/MPI) estimated from Profile sample categorization

Based on examples/ivp_2d_rayleigh_benard/rayleigh_benard.jl but with reduced
resolution and fixed timestep for reproducible profiling.

Usage:
    julia benchmark/profile_ivp.jl
"""

using Profile
using Printf
using Dates

# ============================================================================
# Header
# ============================================================================

println("=" ^ 72)
println("Dedalus.jl IVP Profiling Benchmark")
println("=" ^ 72)
println()
println("Date:     ", Dates.now())
println("Julia:    ", VERSION)
println("Threads:  ", Threads.nthreads())
println("Machine:  ", Sys.MACHINE)
println("CPU:      ", Sys.CPU_NAME)
println()

# ============================================================================
# Load Dedalus
# ============================================================================

println("Loading Dedalus.jl ...")
load_t0 = time()

try
    using Dedalus
catch e
    # If the package is not installed, try activating the project first
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using Dedalus
end

load_time = time() - load_t0
@printf("Dedalus loaded in %.2f seconds\n\n", load_time)

# ============================================================================
# Problem parameters (small for profiling)
# ============================================================================

const PROFILE_Lx = 4.0
const PROFILE_Lz = 1.0
const PROFILE_Nx = 32
const PROFILE_Nz = 16
const PROFILE_Rayleigh = 1e4
const PROFILE_Prandtl = 1.0
const PROFILE_dealias = 3 / 2
const PROFILE_dt = 0.01
const PROFILE_dtype = Float64

const WARMUP_STEPS = 5
const BENCHMARK_STEPS = 100

println("Problem parameters:")
println("  Resolution:  Nx=$(PROFILE_Nx), Nz=$(PROFILE_Nz)")
println("  Domain:      Lx=$(PROFILE_Lx), Lz=$(PROFILE_Lz)")
println("  Rayleigh:    $(PROFILE_Rayleigh)")
println("  Prandtl:     $(PROFILE_Prandtl)")
println("  Dealias:     $(PROFILE_dealias)")
println("  Fixed dt:    $(PROFILE_dt)")
println("  Warmup:      $(WARMUP_STEPS) steps")
println("  Benchmark:   $(BENCHMARK_STEPS) steps")
println()

# ============================================================================
# Problem setup
# ============================================================================

function setup_rayleigh_benard()
    println("Setting up Rayleigh-Benard problem ...")
    setup_t0 = time()

    Lx = PROFILE_Lx
    Lz = PROFILE_Lz
    Nx = PROFILE_Nx
    Nz = PROFILE_Nz
    Rayleigh = PROFILE_Rayleigh
    Prandtl = PROFILE_Prandtl
    dealias = PROFILE_dealias
    dtype = PROFILE_dtype

    # Bases
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=dtype)
    xbasis = RealFourier(coords["x"], Nx; bounds=(0, Lx), dealias=dealias)
    zbasis = ChebyshevT(coords["z"], Nz; bounds=(0, Lz), dealias=dealias)

    # Fields
    p = Field(dist; name="p", bases=(xbasis, zbasis))
    b = Field(dist; name="b", bases=(xbasis, zbasis))
    u = VectorField(dist, coords; name="u", bases=(xbasis, zbasis))
    tau_p = Field(dist; name="tau_p")
    tau_b1 = Field(dist; name="tau_b1", bases=(xbasis,))
    tau_b2 = Field(dist; name="tau_b2", bases=(xbasis,))
    tau_u1 = VectorField(dist, coords; name="tau_u1", bases=(xbasis,))
    tau_u2 = VectorField(dist, coords; name="tau_u2", bases=(xbasis,))

    # Substitutions
    kappa = (Rayleigh * Prandtl)^(-1 / 2)
    nu = (Rayleigh / Prandtl)^(-1 / 2)
    x, z = local_grids(dist, xbasis, zbasis)
    ex, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    lift = A -> Lift(A, lift_basis, -1)
    grad_u = gradient(u) + ez * lift(tau_u1)
    grad_b = gradient(b) + ez * lift(tau_b1)

    # Build namespace dictionary for equation parsing
    ns = Dict{String, Any}(
        "p" => p, "b" => b, "u" => u,
        "tau_p" => tau_p, "tau_b1" => tau_b1, "tau_b2" => tau_b2,
        "tau_u1" => tau_u1, "tau_u2" => tau_u2,
        "kappa" => kappa, "nu" => nu,
        "x" => x, "z" => z,
        "ex" => ex, "ez" => ez,
        "lift" => lift,
        "grad_u" => grad_u, "grad_b" => grad_b,
        "Lx" => Lx, "Lz" => Lz,
    )

    # Problem
    problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2]; namespace=ns)
    add_equation!(problem, "trace(grad_u) + tau_p = 0")
    add_equation!(problem, "dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
    add_equation!(problem, "dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
    add_equation!(problem, "b(z=0) = Lz")
    add_equation!(problem, "u(z=0) = 0")
    add_equation!(problem, "b(z=Lz) = 0")
    add_equation!(problem, "u(z=Lz) = 0")
    add_equation!(problem, "integ(p) = 0")

    # Solver
    solver = build_solver(problem, RK222)

    # Initial conditions
    fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3)
    b["g"] .*= z .* (Lz .- z)
    b["g"] .+= Lz .- z

    setup_time = time() - setup_t0
    @printf("Setup completed in %.2f seconds\n\n", setup_time)

    return solver
end

# ============================================================================
# Stepping helpers
# ============================================================================

"""Run N timesteps with fixed dt. Returns a vector of per-step wall times."""
function run_steps!(solver, n_steps::Int, dt::Float64)
    step_times = Vector{Float64}(undef, n_steps)
    for i in 1:n_steps
        t0 = time_ns()
        step!(solver, dt)
        t1 = time_ns()
        step_times[i] = (t1 - t0) / 1e9
    end
    return step_times
end

# ============================================================================
# Main benchmark
# ============================================================================

function run_benchmark()
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    local solver
    try
        solver = setup_rayleigh_benard()
    catch e
        println("ERROR during problem setup:")
        println("  ", e)
        println()
        println("Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        println()
        println("The IVP solver infrastructure may not be fully functional yet.")
        println("Benchmark cannot proceed.")
        exit(1)
    end

    # ------------------------------------------------------------------
    # Phase 1: Warmup (compile and JIT)
    # ------------------------------------------------------------------
    println("-" ^ 72)
    println("Phase 1: Warmup ($(WARMUP_STEPS) steps)")
    println("-" ^ 72)
    try
        warmup_times = run_steps!(solver, WARMUP_STEPS, PROFILE_dt)
        @printf("  Warmup completed. Mean step: %.4f s, Total: %.4f s\n",
                sum(warmup_times) / WARMUP_STEPS, sum(warmup_times))
    catch e
        println("  ERROR during warmup:")
        println("  ", e)
        println()
        println("  The step! function may not be fully functional yet.")
        println("  Attempting to continue with remaining benchmarks ...")
        println()
    end
    println()

    # ------------------------------------------------------------------
    # Phase 2: Timed run with @time and @allocated
    # ------------------------------------------------------------------
    println("-" ^ 72)
    println("Phase 2: Timed benchmark ($(BENCHMARK_STEPS) steps)")
    println("-" ^ 72)

    local step_times
    local total_alloc_bytes
    local total_time_s

    try
        # Measure allocations
        total_alloc_bytes = @allocated begin
            step_times = run_steps!(solver, BENCHMARK_STEPS, PROFILE_dt)
        end
        total_time_s = sum(step_times)

        println()
        println("  @time summary:")
        @printf("    Total wall time:    %.4f s\n", total_time_s)
        @printf("    Mean step time:     %.6f s\n", total_time_s / BENCHMARK_STEPS)
        @printf("    Min step time:      %.6f s\n", minimum(step_times))
        @printf("    Max step time:      %.6f s\n", maximum(step_times))
        @printf("    Median step time:   %.6f s\n", sort(step_times)[div(BENCHMARK_STEPS, 2)])
        println()
        @printf("  @allocated summary:\n")
        @printf("    Total bytes:        %d (%.2f MiB)\n",
                total_alloc_bytes, total_alloc_bytes / 1024^2)
        @printf("    Per step:           %d bytes (%.2f KiB)\n",
                div(total_alloc_bytes, BENCHMARK_STEPS),
                total_alloc_bytes / BENCHMARK_STEPS / 1024)
    catch e
        println("  ERROR during timed run:")
        println("  ", e)
        step_times = Float64[]
        total_alloc_bytes = 0
        total_time_s = 0.0
    end
    println()

    # ------------------------------------------------------------------
    # Phase 3: Profiling run
    # ------------------------------------------------------------------
    println("-" ^ 72)
    println("Phase 3: Profile run ($(BENCHMARK_STEPS) steps)")
    println("-" ^ 72)

    try
        Profile.clear()
        Profile.@profile begin
            run_steps!(solver, BENCHMARK_STEPS, PROFILE_dt)
        end

        println()
        println("  Top 20 functions by inclusive time:")
        println("  " * "-" ^ 68)
        Profile.print(maxdepth=20, noisefloor=2.0, mincount=1, sortedby=:count)
        println()
    catch e
        println("  ERROR during profiling:")
        println("  ", e)
    end
    println()

    # ------------------------------------------------------------------
    # Phase 4: Per-step timing breakdown
    # ------------------------------------------------------------------
    if !isempty(step_times)
        println("-" ^ 72)
        println("Phase 4: Per-step timing breakdown")
        println("-" ^ 72)

        sorted_times = sort(step_times)
        n = length(sorted_times)

        println()
        println("  Step time distribution (seconds):")
        @printf("    P5:    %.6f\n", sorted_times[max(1, ceil(Int, 0.05 * n))])
        @printf("    P25:   %.6f\n", sorted_times[max(1, ceil(Int, 0.25 * n))])
        @printf("    P50:   %.6f\n", sorted_times[max(1, ceil(Int, 0.50 * n))])
        @printf("    P75:   %.6f\n", sorted_times[max(1, ceil(Int, 0.75 * n))])
        @printf("    P95:   %.6f\n", sorted_times[max(1, ceil(Int, 0.95 * n))])
        @printf("    P99:   %.6f\n", sorted_times[max(1, ceil(Int, 0.99 * n))])
        println()

        # Simple histogram (10 bins)
        lo, hi = minimum(step_times), maximum(step_times)
        if hi > lo
            n_bins = 10
            bin_width = (hi - lo) / n_bins
            bins = zeros(Int, n_bins)
            for t in step_times
                idx = min(n_bins, 1 + floor(Int, (t - lo) / bin_width))
                bins[idx] += 1
            end
            println("  Histogram:")
            max_count = maximum(bins)
            bar_scale = max_count > 40 ? 40.0 / max_count : 1.0
            for i in 1:n_bins
                bin_lo = lo + (i - 1) * bin_width
                bin_hi = lo + i * bin_width
                bar_len = round(Int, bins[i] * bar_scale)
                @printf("    [%8.5f, %8.5f) %4d  %s\n",
                        bin_lo, bin_hi, bins[i], "#" ^ bar_len)
            end
        end
        println()
    end

    # ------------------------------------------------------------------
    # Phase 5: Per-step component breakdown from Profile data
    # ------------------------------------------------------------------
    println("-" ^ 72)
    println("Phase 5: Per-step component breakdown (estimated from Profile samples)")
    println("-" ^ 72)
    println()

    try
        # Profile.retrieve() -> (ips::Vector{UInt64}, lidict::Dict{UInt64, Vector{StackFrame}})
        # ips is a flat stream of instruction pointers separated by 0 (one backtrace per sample).
        # lidict maps each IP to its resolved stack frames.
        ips_raw, lidict = Profile.retrieve()
        if isempty(ips_raw)
            println("  No profile data collected; skipping component breakdown.")
        else
            # Category regex patterns (applied to lowercase function name)
            rhs_patterns   = [r"evaluate_scheduled", r"evaluate_group",
                              r"evaluate_handlers", r"(?<![a-z])operate(?![a-z])", r"evaluate_future"]
            lhs_patterns   = [r"(?<![a-z])solve(?![a-z])", r"ldiv!", r"lu!", r"factorize"]
            trans_patterns = [r"forward!", r"backward!", r"(?<![a-z])fft", r"rfft", r"plan_"]
            mpi_patterns   = [r"transpose", r"alltoallv", r"gather", r"scatter"]

            function categorize_frame(func_name::AbstractString)
                fn = lowercase(func_name)
                for pat in rhs_patterns
                    occursin(pat, fn) && return :RHS
                end
                for pat in lhs_patterns
                    occursin(pat, fn) && return :LHS
                end
                for pat in trans_patterns
                    occursin(pat, fn) && return :Transform
                end
                for pat in mpi_patterns
                    occursin(pat, fn) && return :TransposeMPI
                end
                return :Other
            end

            # Count samples per category by walking every instruction pointer
            # in every backtrace.  Each backtrace represents one sample; we
            # attribute the sample to the leaf-most recognized category to
            # avoid double-counting nested calls.
            category_counts = Dict{Symbol, Int}(
                :RHS          => 0,
                :LHS          => 0,
                :Transform    => 0,
                :TransposeMPI => 0,
                :Other        => 0,
            )
            total_samples = 0

            # Helper: classify a single backtrace (vector of IPs) using lidict.
            function classify_backtrace(ips, lidict)
                # Walk from leaf (index 1) outward; first recognized category wins.
                for ip in ips
                    frames = get(lidict, ip, nothing)
                    frames === nothing && continue
                    for sf in frames
                        cat = categorize_frame(string(sf.func))
                        if cat !== :Other
                            return cat
                        end
                    end
                end
                return :Other
            end

            # Parse the flat IP vector: backtraces are separated by 0
            ips_buf = UInt64[]
            for ip in ips_raw
                if ip == 0
                    if !isempty(ips_buf)
                        cat = classify_backtrace(ips_buf, lidict)
                        category_counts[cat] += 1
                        total_samples += 1
                        empty!(ips_buf)
                    end
                else
                    push!(ips_buf, ip)
                end
            end
            if !isempty(ips_buf)
                cat = classify_backtrace(ips_buf, lidict)
                category_counts[cat] += 1
                total_samples += 1
            end

            println("  Component breakdown (from $total_samples profile samples):")
            println()
            @printf("    %-20s  %8s  %6s\n", "Component", "Samples", "  %")
            println("    " * "-" ^ 40)

            labels = [
                (:RHS,          "RHS evaluation"),
                (:LHS,          "LHS solve"),
                (:Transform,    "Transform"),
                (:TransposeMPI, "Transpose / MPI"),
                (:Other,        "Other"),
            ]
            for (key, label) in labels
                cnt = category_counts[key]
                pct = total_samples > 0 ? 100.0 * cnt / total_samples : 0.0
                @printf("    %-20s  %8d  %5.1f%%\n", label, cnt, pct)
            end
            println()

            recognized = total_samples - category_counts[:Other]
            recog_pct = total_samples > 0 ? 100.0 * recognized / total_samples : 0.0
            @printf("  Recognized solver work: %d / %d samples (%.1f%%)\n",
                    recognized, total_samples, recog_pct)
            println()
            println("  NOTE: Percentages are estimates from statistical profiling.")
            println("        'Other' includes runtime, GC, I/O, and minor solver bookkeeping.")
        end
    catch e
        println("  ERROR during component breakdown analysis:")
        println("  ", e)
    end
    println()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    println("=" ^ 72)
    println("BENCHMARK SUMMARY")
    println("=" ^ 72)
    println()
    @printf("  Resolution:           %d x %d\n", PROFILE_Nx, PROFILE_Nz)
    @printf("  Fixed dt:             %.4f\n", PROFILE_dt)
    @printf("  Benchmark steps:      %d\n", BENCHMARK_STEPS)
    if !isempty(step_times)
        @printf("  Total wall time:      %.4f s\n", total_time_s)
        @printf("  Mean step time:       %.6f s\n", total_time_s / BENCHMARK_STEPS)
        @printf("  Throughput:           %.1f steps/s\n", BENCHMARK_STEPS / total_time_s)
        @printf("  Total allocations:    %.2f MiB\n", total_alloc_bytes / 1024^2)
        @printf("  Alloc per step:       %.2f KiB\n",
                total_alloc_bytes / BENCHMARK_STEPS / 1024)
    else
        println("  (timing data unavailable due to errors)")
    end
    println()
    println("Solver state at end:")
    try
        @printf("  Iteration:  %d\n", solver.iteration)
        @printf("  Sim time:   %.6f\n", solver.sim_time)
    catch
        println("  (solver state unavailable)")
    end
    println()
    println("=" ^ 72)
    println("Benchmark complete.")
    println("=" ^ 72)
end

# Run the benchmark
run_benchmark()
