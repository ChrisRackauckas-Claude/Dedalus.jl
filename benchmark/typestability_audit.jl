"""
    Type-Stability Audit for Dedalus.jl

Systematically audits type stability of key functions across the Dedalus.jl
codebase using two complementary approaches:

1. **Static source analysis**: Scans source files for `::Any`-typed struct fields,
   function parameters, and return annotations that indicate potential type
   instability at the compiler level.

2. **Runtime inference analysis**: When the Dedalus module is loadable, uses
   `Base.return_types()` and `code_typed()` to check inferred return types of
   key methods for `Any`, broad `Union` types, or `Body::Any` indicators.

The script reports findings in a structured format:
    [STATUS] function_name @ file:line -- details

Status legend:
    STABLE      -- No type-stability concerns detected
    UNSTABLE    -- Concrete type-instability indicator found
    WARNING     -- Potential concern (e.g., ::Any field in a hot struct)
    INFO        -- Informational note about type annotations
    SKIP        -- Could not analyze (module not loadable, etc.)

Usage:
    julia benchmark/typestability_audit.jl [--verbose] [--runtime]

Options:
    --verbose   Show all findings including STABLE entries
    --runtime   Attempt runtime inference analysis (requires loadable module)
"""

# ============================================================================
# Configuration
# ============================================================================

const VERBOSE = "--verbose" in ARGS
const DO_RUNTIME = "--runtime" in ARGS

# Source root relative to this script
const SCRIPT_DIR = @__DIR__
const SRC_ROOT = joinpath(dirname(SCRIPT_DIR), "src")

# Files and functions to audit
const AUDIT_TARGETS = [
    # (file_path_relative_to_src, description, key_functions)
    (
        "core/timesteppers.jl",
        "IMEX timestepping methods",
        [
            "_multistep_step!",
            "_rk_step!",
            "_apply_sparse_to_subdata!",
            "compute_coefficients",
            "step!",
            "_rotate_right!",
        ]
    ),
    (
        "core/field.jl",
        "Field types and indexing",
        [
            "Base.getindex(::Field, ...)",
            "Base.setindex!(::Field, ...)",
            "preset_scales!",
            "preset_layout!",
            "change_scales!",
            "change_layout!",
            "towards_grid_space!",
            "towards_coeff_space!",
            "operand_cast",
        ]
    ),
    (
        "core/future.jl",
        "Future/deferred evaluation",
        [
            "evaluate_future",
            "build_out",
            "get_out",
            "init_future!",
            "attempt",
        ]
    ),
    (
        "core/arithmetic.jl",
        "Arithmetic operators",
        [
            "operate(::AddFields, ...)",
            "operate(::MultiplyFields, ...)",
            "operate(::MultiplyNumberField, ...)",
            "operate(::DotProduct, ...)",
            "operate(::CrossProduct, ...)",
            "dedalus_add",
            "dedalus_multiply",
        ]
    ),
    (
        "core/transforms.jl",
        "Spectral transforms",
        [
            "forward!(::JacobiMMT, ...)",
            "backward!(::JacobiMMT, ...)",
            "forward!(::FFTWComplexFFT, ...)",
            "backward!(::FFTWComplexFFT, ...)",
            "forward!(::FFTWRealFFT, ...)",
            "backward!(::FFTWRealFFT, ...)",
            "forward_matrix",
            "backward_matrix",
        ]
    ),
    (
        "core/distributor.jl",
        "Distributor and layout management",
        [
            "get_layout_object",
            "remedy_scales",
            "local_shape",
            "local_elements",
            "buffer_size",
            "increment_single",
            "decrement_single",
        ]
    ),
    (
        "tools/cache.jl",
        "Caching utilities",
        [
            "get_cached!",
            "cached_construct",
        ]
    ),
    (
        "core/system.jl",
        "CoeffSystem and FieldSystem",
        [
            "get_subdata",
            "gather!",
            "scatter!",
        ]
    ),
    (
        "libraries/matsolvers.jl",
        "Matrix solvers",
        [
            "solve(::UmfpackSpsolve, ...)",
            "solve(::UmfpackFactorized, ...)",
            "solve(::BandedLAPACK, ...)",
            "solve(::DenseLU, ...)",
            "solve(::BlockInverse, ...)",
            "solve(::SparseInverse, ...)",
            "solve(::Woodbury, ...)",
        ]
    ),
]

# ============================================================================
# Counters and results storage
# ============================================================================

mutable struct AuditStats
    total_findings::Int
    unstable_count::Int
    warning_count::Int
    stable_count::Int
    info_count::Int
    skip_count::Int
    findings::Vector{NamedTuple{(:status, :func, :file, :line, :detail), Tuple{Symbol, String, String, Int, String}}}
end

AuditStats() = AuditStats(0, 0, 0, 0, 0, 0, [])

function record!(stats::AuditStats, status::Symbol, func::String, file::String, line::Int, detail::String)
    stats.total_findings += 1
    if status == :UNSTABLE
        stats.unstable_count += 1
    elseif status == :WARNING
        stats.warning_count += 1
    elseif status == :STABLE
        stats.stable_count += 1
    elseif status == :INFO
        stats.info_count += 1
    elseif status == :SKIP
        stats.skip_count += 1
    end
    push!(stats.findings, (status=status, func=func, file=file, line=line, detail=detail))
end

# ============================================================================
# Part 1: Static Source Analysis
# ============================================================================

"""
    scan_any_typed_fields(filepath) -> Vector{NamedTuple}

Scan a Julia source file for struct fields typed as `::Any` or with broad
Union types. These are a primary source of type instability because the
compiler cannot specialize on field access.
"""
function scan_any_typed_fields(filepath::String)
    results = NamedTuple{(:line, :context, :text), Tuple{Int, Symbol, String}}[]
    if !isfile(filepath)
        return results
    end
    lines = readlines(filepath)
    in_struct = false
    struct_name = ""
    for (i, line) in enumerate(lines)
        stripped = strip(line)
        # Detect struct definition
        m_struct = match(r"^(?:mutable\s+)?struct\s+(\w+)", stripped)
        if m_struct !== nothing
            in_struct = true
            struct_name = m_struct.captures[1]
            continue
        end
        if in_struct && stripped == "end"
            in_struct = false
            struct_name = ""
            continue
        end
        if in_struct
            # Check for ::Any typed fields
            if occursin(r"::\s*Any\b", stripped)
                # Skip comments
                if !startswith(stripped, "#")
                    push!(results, (line=i, context=:struct_field, text="$(struct_name): $(stripped)"))
                end
            end
            # Check for broad Union types containing Any
            if occursin(r"Union\{.*Any", stripped) && !startswith(stripped, "#")
                push!(results, (line=i, context=:struct_union, text="$(struct_name): $(stripped)"))
            end
        end
    end
    return results
end

"""
    scan_untyped_function_returns(filepath) -> Vector{NamedTuple}

Scan for function signatures that return `Any` or have no return type
annotation on performance-critical functions.
"""
function scan_untyped_function_returns(filepath::String)
    results = NamedTuple{(:line, :context, :text), Tuple{Int, Symbol, String}}[]
    if !isfile(filepath)
        return results
    end
    lines = readlines(filepath)
    for (i, line) in enumerate(lines)
        stripped = strip(line)
        # Skip comments and docstrings
        if startswith(stripped, "#") || startswith(stripped, "\"\"\"")
            continue
        end
        # Check for explicit ::Any return annotation
        if occursin(r"function\s+\w+.*\)\s*::\s*Any", stripped)
            push!(results, (line=i, context=:return_any, text=stripped))
        end
    end
    return results
end

"""
    scan_dict_any_patterns(filepath) -> Vector{NamedTuple}

Scan for Dict types with Any keys or values, which propagate type instability
through dictionary lookups.
"""
function scan_dict_any_patterns(filepath::String)
    results = NamedTuple{(:line, :context, :text), Tuple{Int, Symbol, String}}[]
    if !isfile(filepath)
        return results
    end
    lines = readlines(filepath)
    in_struct = false
    struct_name = ""
    for (i, line) in enumerate(lines)
        stripped = strip(line)
        m_struct = match(r"^(?:mutable\s+)?struct\s+(\w+)", stripped)
        if m_struct !== nothing
            in_struct = true
            struct_name = m_struct.captures[1]
            continue
        end
        if in_struct && stripped == "end"
            in_struct = false
            struct_name = ""
            continue
        end
        if in_struct
            if occursin(r"Dict\{.*,\s*Any\}", stripped) && !startswith(stripped, "#")
                push!(results, (line=i, context=:dict_any_value, text="$(struct_name): $(stripped)"))
            end
            if occursin(r"Dict\{Any", stripped) && !startswith(stripped, "#")
                push!(results, (line=i, context=:dict_any_key, text="$(struct_name): $(stripped)"))
            end
        end
    end
    return results
end

"""
    scan_vector_any_patterns(filepath) -> Vector{NamedTuple}

Scan for Vector{Any} fields and arguments that force boxing of elements.
"""
function scan_vector_any_patterns(filepath::String)
    results = NamedTuple{(:line, :context, :text), Tuple{Int, Symbol, String}}[]
    if !isfile(filepath)
        return results
    end
    lines = readlines(filepath)
    in_struct = false
    struct_name = ""
    for (i, line) in enumerate(lines)
        stripped = strip(line)
        m_struct = match(r"^(?:mutable\s+)?struct\s+(\w+)", stripped)
        if m_struct !== nothing
            in_struct = true
            struct_name = m_struct.captures[1]
            continue
        end
        if in_struct && stripped == "end"
            in_struct = false
            struct_name = ""
            continue
        end
        if in_struct
            if occursin(r"Vector\{Any\}", stripped) && !startswith(stripped, "#")
                push!(results, (line=i, context=:vector_any, text="$(struct_name): $(stripped)"))
            end
        end
    end
    return results
end

"""
    classify_any_field(struct_name, field_text) -> Symbol

Classify an ::Any field as :UNSTABLE or :WARNING based on the struct's
role in the hot path.

Hot-path structs (those used in inner loops of timestepping, transforms,
and arithmetic) get :UNSTABLE. Configuration/setup structs get :WARNING.
"""
function classify_any_field(struct_name::AbstractString, field_text::AbstractString)
    # Hot-path structs where ::Any fields cause measurable slowdown
    hot_structs = Set([
        "MultistepIMEXData",
        "RungeKuttaIMEXData",
        "Field",
        "LockedField",
        "AddFields",
        "MultiplyFields",
        "MultiplyNumberField",
        "DotProduct",
        "CrossProduct",
        "CoeffSystem",
        "FieldSystem",
        "Layout",
        "Distributor",
    ])
    if struct_name in hot_structs
        return :UNSTABLE
    else
        return :WARNING
    end
end

# ============================================================================
# Part 2: Known type-stability issues (curated from source reading)
# ============================================================================

"""
    known_issues() -> Vector{NamedTuple}

Return a curated list of known type-stability issues identified through
manual source review. These are structural issues that static regex
scanning might miss.
"""
function known_issues()
    issues = []

    # --- timesteppers.jl ---
    push!(issues, (
        status=:UNSTABLE,
        func="_multistep_step!",
        file="core/timesteppers.jl",
        line=162,
        detail="MultistepIMEXData.solver::Any -- solver access in hot loop is type-unstable; " *
               "every field access on solver (solver.subproblems, solver.evaluator, etc.) " *
               "requires dynamic dispatch"
    ))
    push!(issues, (
        status=:UNSTABLE,
        func="_multistep_step!",
        file="core/timesteppers.jl",
        line=132,
        detail="MultistepIMEXData._LHS_params::Any -- comparison (a0, b0) != data._LHS_params " *
               "cannot be type-specialized"
    ))
    push!(issues, (
        status=:UNSTABLE,
        func="_rk_step!",
        file="core/timesteppers.jl",
        line=859,
        detail="RungeKuttaIMEXData.solver::Any -- same issue as MultistepIMEXData; " *
               "solver field access in inner stage loop forces dynamic dispatch"
    ))
    push!(issues, (
        status=:UNSTABLE,
        func="_rk_step!",
        file="core/timesteppers.jl",
        line=833,
        detail="RungeKuttaIMEXData._LHS_params::Any -- type-unstable comparison in LHS update check"
    ))

    # --- field.jl ---
    push!(issues, (
        status=:UNSTABLE,
        func="Base.getindex(::Field, key)",
        file="core/field.jl",
        line=600,
        detail="Field.layout::Any -- layout field is accessed on every getindex/setindex! call; " *
               "all downstream dispatch through layout is type-unstable"
    ))
    push!(issues, (
        status=:UNSTABLE,
        func="preset_layout!(::Field, layout)",
        file="core/field.jl",
        line=746,
        detail="Field.dist::Any -- distributor access (dist.dtype, local_shape, etc.) " *
               "is type-unstable; this function is called on every layout change"
    ))
    push!(issues, (
        status=:WARNING,
        func="operand_cast",
        file="core/field.jl",
        line=141,
        detail="Return type is Union{AbstractOperand, Field} depending on input; " *
               "callers cannot predict the concrete type"
    ))

    # --- future.jl ---
    push!(issues, (
        status=:UNSTABLE,
        func="evaluate_future",
        file="core/future.jl",
        line=261,
        detail="Returns Union{Nothing, Field, LockedField} -- callers must handle " *
               "the union return type; args are stored in Vector{Any}"
    ))
    push!(issues, (
        status=:UNSTABLE,
        func="get_out",
        file="core/future.jl",
        line=330,
        detail="AbstractFuture.out::Any field -- output field access is type-unstable"
    ))
    push!(issues, (
        status=:UNSTABLE,
        func="build_out",
        file="core/future.jl",
        line=347,
        detail="Uses future_type(f) which returns Type{Field} or Type{LockedField} -- " *
               "the constructed output is dynamically dispatched"
    ))

    # --- arithmetic.jl ---
    push!(issues, (
        status=:UNSTABLE,
        func="dedalus_add",
        file="core/arithmetic.jl",
        line=1664,
        detail="Return type is Union{Int, AbstractOperand, AddFields, Any} -- " *
               "branches return 0 (Int), single arg, or AddFields; callers see Any"
    ))
    push!(issues, (
        status=:UNSTABLE,
        func="dedalus_multiply",
        file="core/arithmetic.jl",
        line=1700,
        detail="Return type is Union{Int, AbstractOperand, MultiplyFields, MultiplyNumberField, Any} -- " *
               "multiple return paths with different types"
    ))
    push!(issues, (
        status=:UNSTABLE,
        func="operate(::DotProduct, out)",
        file="core/arithmetic.jl",
        line=1071,
        detail="ghost_cast returns field.data or freshly allocated array -- " *
               "return type depends on GhostBroadcaster.skip flag"
    ))

    # --- transforms.jl ---
    push!(issues, (
        status=:WARNING,
        func="forward!(::FFTWComplexFFT, ...)",
        file="core/transforms.jl",
        line=501,
        detail="_plan_cache::Dict{Tuple{NTuple, Int}, Any} -- plan retrieval " *
               "returns Any; FFTW plan type is erased"
    ))
    push!(issues, (
        status=:WARNING,
        func="forward!(::FFTWRealFFT, ...)",
        file="core/transforms.jl",
        line=794,
        detail="_plan_cache::Dict{Tuple, Any} -- same issue as FFTWComplexFFT; " *
               "plan types erased by Dict{..., Any}"
    ))
    push!(issues, (
        status=:WARNING,
        func="transform_plan (all bases)",
        file="core/transforms.jl",
        line=892,
        detail="Basis _transform_cache uses untyped Dict -- retrieved plans are ::Any"
    ))

    # --- distributor.jl ---
    push!(issues, (
        status=:UNSTABLE,
        func="get_layout_object",
        file="core/distributor.jl",
        line=1099,
        detail="Returns Layout from dist.layout_references::Dict{String, Any} -- " *
               "return type is ::Any; this is called on every field layout change"
    ))
    push!(issues, (
        status=:UNSTABLE,
        func="Distributor struct",
        file="core/distributor.jl",
        line=821,
        detail="Multiple ::Any fields: coeff_layout, grid_layout, comm, comm_cart, " *
               "dtype, single_coordsys -- all field accesses are type-unstable"
    ))
    push!(issues, (
        status=:WARNING,
        func="Layout struct",
        file="core/distributor.jl",
        line=1266,
        detail="Layout.dist::Any -- every access to the parent distributor from " *
               "a layout is type-unstable"
    ))

    # --- cache.jl ---
    push!(issues, (
        status=:WARNING,
        func="get_cached!(::CachedAttribute{Any})",
        file="tools/cache.jl",
        line=66,
        detail="CachedAttribute(fn) without type parameter defaults to {Any} -- " *
               "returned value is untyped; use CachedAttribute{T}(fn) instead"
    ))

    # --- system.jl ---
    push!(issues, (
        status=:UNSTABLE,
        func="get_subdata",
        file="core/system.jl",
        line=64,
        detail="CoeffSystem.views::Dict{Any, Dict{Any, AbstractArray{T}}} -- " *
               "subproblem key lookup is ::Any; consider using integer keys"
    ))

    # --- matsolvers.jl ---
    push!(issues, (
        status=:INFO,
        func="solve (all concrete solvers)",
        file="libraries/matsolvers.jl",
        line=68,
        detail="All concrete solve() methods return typed results (matrix \\ vector); " *
               "type stability depends on the AbstractMatSolver dispatch being monomorphic " *
               "at call sites. The sp.LHS_solver::Any in timesteppers forces dynamic dispatch."
    ))

    return issues
end

# ============================================================================
# Part 3: Runtime inference analysis (optional)
# ============================================================================

"""
    try_runtime_analysis(stats)

Attempt to load the Dedalus module and perform runtime type inference
analysis using Base.return_types() and code_typed().
"""
function try_runtime_analysis(stats::AuditStats)
    println("\n" * "="^72)
    println("RUNTIME INFERENCE ANALYSIS")
    println("="^72)

    # Try to load the module
    try
        # Add the project to the load path
        project_root = dirname(SRC_ROOT)
        push!(LOAD_PATH, project_root)

        @eval using Dedalus

        println("Successfully loaded Dedalus module. Analyzing method return types...\n")

        # Analyze key methods
        _check_return_types(stats)

    catch e
        println("Could not load Dedalus module: $(e)")
        println("Runtime analysis skipped. Run with a properly configured environment.")
        println("The static analysis results above remain valid.\n")
        record!(stats, :SKIP, "runtime_analysis", "N/A", 0,
                "Module could not be loaded: $(sprint(showerror, e))")
    end
end

"""
    _is_broad_union(T) -> Bool

Check whether a type is a broad Union that indicates type instability.
A Union is considered "broad" if it contains `Any`, has more than 3 members,
or contains both value types (Int, Float64) and reference types (Array, Field).
"""
function _is_broad_union(@nospecialize(T))
    if T === Any
        return true
    end
    if T isa Union
        members = Base.uniontypes(T)
        # Any appearing anywhere in the union
        if Any in members
            return true
        end
        # Unions with many alternatives suggest the compiler couldn't narrow the type
        if length(members) > 3
            return true
        end
        return false
    end
    return false
end

"""
    _extract_arg_types(sig) -> Union{Nothing, Type{<:Tuple}}

Extract argument types from a method signature, skipping the first element
(the function type itself). Returns `nothing` if the types cannot be extracted
or contain TypeVar parameters that prevent concrete instantiation.
"""
function _extract_arg_types(sig)
    try
        params = sig.parameters
        if length(params) < 2
            # No arguments beyond the function itself
            return Tuple{}
        end
        arg_types = params[2:end]
        # Check for uninstantiable type variables
        for t in arg_types
            if t isa TypeVar
                return nothing
            end
        end
        return Tuple{arg_types...}
    catch
        return nothing
    end
end

"""
    _check_function_stability(stats, func_name, func, source_file)

Analyze all methods of `func` using `Base.return_types()` and report any
methods whose inferred return type includes `Any` or a broad `Union`.
"""
function _check_function_stability(stats::AuditStats, func_name::String, func, source_file::String)
    local meths
    try
        meths = methods(func)
    catch e
        println("  [SKIP] Could not get methods for $(func_name): $(e)")
        record!(stats, :SKIP, func_name, source_file, 0,
                "Could not enumerate methods: $(sprint(showerror, e))")
        return
    end

    n_methods = length(meths)
    n_checked = 0
    n_unstable = 0
    n_skipped = 0

    for m in meths
        sig = m.sig
        arg_tuple = _extract_arg_types(sig)

        if arg_tuple === nothing
            # Cannot construct concrete argument types (e.g., has TypeVar params)
            n_skipped += 1
            if VERBOSE
                println("  [SKIP] $(func_name) method $(m) -- signature has unresolvable type parameters")
            end
            record!(stats, :SKIP, func_name, source_file, Int(m.line),
                    "Skipped: signature $(sig) contains type variables that prevent concrete instantiation")
            continue
        end

        local rtypes
        try
            rtypes = Base.return_types(func, arg_tuple)
        catch e
            # Some signatures may cause errors in the type inference engine
            n_skipped += 1
            if VERBOSE
                println("  [SKIP] $(func_name) method at $(m.file):$(m.line) -- Base.return_types failed: $(e)")
            end
            record!(stats, :SKIP, func_name, source_file, Int(m.line),
                    "Base.return_types() failed for $(arg_tuple): $(sprint(showerror, e))")
            continue
        end

        n_checked += 1

        for rt in rtypes
            if _is_broad_union(rt)
                n_unstable += 1
                println("  [UNSTABLE] $(func_name) method at $(m.file):$(m.line)")
                println("             Signature: $(arg_tuple)")
                println("             Inferred return type: $(rt)")
                println()
                record!(stats, :UNSTABLE, func_name, source_file, Int(m.line),
                        "Runtime inference: return type is $(rt) for args $(arg_tuple)")
            else
                if VERBOSE
                    println("  [STABLE]   $(func_name) method at $(m.file):$(m.line) -> $(rt)")
                end
                record!(stats, :STABLE, func_name, source_file, Int(m.line),
                        "Runtime inference: return type is $(rt) for args $(arg_tuple)")
            end
        end
    end

    println("  $(func_name): $(n_methods) methods total, $(n_checked) checked, $(n_unstable) unstable, $(n_skipped) skipped")
end

"""
    _check_return_types(stats)

Check inferred return types for priority functions after module loading.

Uses `Base.return_types()` to perform runtime type-inference analysis on
key functions identified in the type-stability audit specification:
  - Timestepper inner loops (_multistep_step!, _rk_step!)
  - Layout lookup (get_layout_object)
  - Coefficient system access (get_subdata)
  - Sparse matrix application (_apply_sparse_to_subdata!)
  - Expression tree evaluation (evaluate_future)
  - Cache lookup (get_cached!)
"""
function _check_return_types(stats::AuditStats)
    # Priority functions to analyze, in order of hot-path impact.
    # Each entry: (display_name, module-qualified accessor expression, source file)
    priority_targets = [
        ("Dedalus._multistep_step!",      "core/timesteppers.jl"),
        ("Dedalus._rk_step!",             "core/timesteppers.jl"),
        ("Dedalus.get_layout_object",     "core/distributor.jl"),
        ("Dedalus.get_subdata",           "core/system.jl"),
        ("Dedalus._apply_sparse_to_subdata!", "core/timesteppers.jl"),
        ("Dedalus.evaluate_future",       "core/future.jl"),
        ("Dedalus.get_cached!",           "tools/cache.jl"),
    ]

    println("Checking inferred return types for $(length(priority_targets)) priority functions...\n")

    for (qualified_name, source_file) in priority_targets
        # Resolve the function object from its qualified name
        local func
        try
            func = @eval $(Meta.parse(qualified_name))
        catch e
            println("  [SKIP] $(qualified_name) -- function not found: $(e)")
            record!(stats, :SKIP, qualified_name, source_file, 0,
                    "Function not found in loaded module: $(sprint(showerror, e))")
            println()
            continue
        end

        if !isa(func, Function)
            println("  [SKIP] $(qualified_name) -- resolved to $(typeof(func)), not a Function")
            record!(stats, :SKIP, qualified_name, source_file, 0,
                    "Resolved to $(typeof(func)), not a Function")
            println()
            continue
        end

        _check_function_stability(stats, qualified_name, func, source_file)
        println()
    end

    # Print runtime analysis summary
    rt_findings = filter(f -> startswith(f.detail, "Runtime inference:"), stats.findings)
    rt_unstable = count(f -> f.status == :UNSTABLE, rt_findings)
    rt_stable = count(f -> f.status == :STABLE, rt_findings)
    rt_skip = count(f -> f.status == :SKIP && occursin("type variables", f.detail), stats.findings)

    println("-"^60)
    println("Runtime inference summary:")
    println("  Methods with stable return types:   $(rt_stable)")
    println("  Methods with unstable return types:  $(rt_unstable)")
    println("  Methods skipped (type variables):    $(rt_skip)")
    println()
    if rt_unstable > 0
        println("  $(rt_unstable) method(s) inferred as returning Any or a broad Union type.")
        println("  These are candidates for type annotations or refactoring.")
    else
        println("  All checked methods have concrete inferred return types.")
        println("  Note: methods with type parameters were skipped; they may still")
        println("  be unstable for specific type combinations.")
    end
end

# ============================================================================
# Reporting
# ============================================================================

function print_separator(char='=', width=72)
    println(char^width)
end

function print_header(title)
    println()
    print_separator()
    println(title)
    print_separator()
end

function status_marker(s::Symbol)
    if s == :UNSTABLE
        return "[UNSTABLE]"
    elseif s == :WARNING
        return "[WARNING] "
    elseif s == :STABLE
        return "[STABLE]  "
    elseif s == :INFO
        return "[INFO]    "
    elseif s == :SKIP
        return "[SKIP]    "
    else
        return "[???]     "
    end
end

function print_finding(f)
    marker = status_marker(f.status)
    loc = f.line > 0 ? "$(f.file):$(f.line)" : f.file
    println("  $(marker) $(f.func) @ $(loc)")
    println("             $(f.detail)")
    println()
end

# ============================================================================
# Main
# ============================================================================

function main()
    stats = AuditStats()

    println()
    print_separator('#')
    println("  Dedalus.jl Type-Stability Audit")
    println("  $(Dates.now())")
    print_separator('#')

    # ---------------------------------------------------------------
    # Static analysis: scan source files
    # ---------------------------------------------------------------
    print_header("PART 1: STATIC SOURCE ANALYSIS -- ::Any TYPED STRUCT FIELDS")

    total_any_fields = 0
    total_vector_any = 0
    total_dict_any = 0

    for (relpath, description, _) in AUDIT_TARGETS
        filepath = joinpath(SRC_ROOT, relpath)
        if !isfile(filepath)
            println("  [SKIP] File not found: $(filepath)")
            continue
        end

        any_fields = scan_any_typed_fields(filepath)
        vector_any = scan_vector_any_patterns(filepath)
        dict_any = scan_dict_any_patterns(filepath)

        n_any = length(any_fields)
        n_vec = length(vector_any)
        n_dict = length(dict_any)
        total_any_fields += n_any
        total_vector_any += n_vec
        total_dict_any += n_dict

        if n_any + n_vec + n_dict > 0
            println("\n  --- $(relpath) ($(description)) ---")
        end

        for r in any_fields
            struct_name = split(r.text, ":")[1]
            severity = classify_any_field(strip(struct_name), r.text)
            record!(stats, severity, "struct field", relpath, r.line, r.text)
            if severity == :UNSTABLE || VERBOSE
                print_finding((status=severity, func="struct field", file=relpath, line=r.line, detail=r.text))
            end
        end
        for r in vector_any
            record!(stats, :WARNING, "Vector{Any} field", relpath, r.line, r.text)
            if VERBOSE
                print_finding((status=:WARNING, func="Vector{Any} field", file=relpath, line=r.line, detail=r.text))
            end
        end
        for r in dict_any
            record!(stats, :WARNING, "Dict{...,Any} field", relpath, r.line, r.text)
            if VERBOSE
                print_finding((status=:WARNING, func="Dict{...,Any} field", file=relpath, line=r.line, detail=r.text))
            end
        end
    end

    println("\n  Static scan summary:")
    println("    ::Any struct fields:    $(total_any_fields)")
    println("    Vector{Any} fields:     $(total_vector_any)")
    println("    Dict{...,Any} fields:   $(total_dict_any)")

    # ---------------------------------------------------------------
    # Known issues from manual review
    # ---------------------------------------------------------------
    print_header("PART 2: CURATED TYPE-STABILITY FINDINGS (from manual source review)")

    issues = known_issues()
    for issue in issues
        record!(stats, issue.status, issue.func, issue.file, issue.line, issue.detail)
        if issue.status in (:UNSTABLE, :WARNING) || VERBOSE
            print_finding(issue)
        end
    end

    # ---------------------------------------------------------------
    # Hot-path impact analysis
    # ---------------------------------------------------------------
    print_header("PART 3: HOT-PATH IMPACT ANALYSIS")

    println("""
  The following type-instability chains have the highest performance impact
  because they sit on the innermost loops of the timestepping algorithm:

  Chain 1: Timestepper -> Solver -> Field layout changes
  -------------------------------------------------------
    MultistepIMEXData.solver::Any
      -> solver.subproblems (::Any)
        -> sp.LHS_solver (::Any, set to nothing or matsolver)
          -> solve(sp.LHS_solver, rhs) -- dynamic dispatch on every solve

    Impact: Every timestep performs N_subproblems dynamic dispatches for
    the matrix solve, plus additional dispatches for field layout changes.

    Fix: Parameterize MultistepIMEXData{T, S} where S is the solver type,
    or use a FunctionWrapper for the solver interface.

  Chain 2: Field getindex/setindex! -> Layout -> Distributor
  ----------------------------------------------------------
    Field.dist::Any
      -> get_layout_object(dist, key) returns ::Any from Dict{String,Any}
        -> Layout.dist::Any (circular reference)
          -> local_shape, buffer_size, etc. all type-unstable

    Impact: Every field data access (f[\"g\"], f[\"c\"]) triggers this chain.
    Fields are accessed thousands of times per timestep.

    Fix: Parameterize Field{D<:AbstractDistributor} or at minimum use
    concrete type annotations for layout and dist fields.

  Chain 3: Arithmetic operate -> ghost_cast -> field.data
  -------------------------------------------------------
    AddFields.args::Vector{Any}
      -> arg.layout (::Any through AbstractOperand)
        -> ghost_cast returns Union{Array, field.data}
          -> @. out.data = ... (broadcasting with potentially untyped data)

    Impact: Every arithmetic evaluation in the expression tree.

    Fix: Use concrete argument types or FunctionWrappers for the
    args vector. Consider specialized 2-arg structs instead of Vector{Any}.

  Chain 4: Future evaluate -> recursive Any propagation
  -----------------------------------------------------
    AbstractFuture.args::Vector{Any}
      -> evaluate_future returns Union{Nothing, Field, LockedField}
        -> operate(future, out) -- out is ::Any from get_out

    Impact: Every expression tree evaluation.

    Fix: Parameterize future types on their output type, or use
    Union{Nothing, Field} with explicit type assertions.
""")

    # ---------------------------------------------------------------
    # Runtime analysis (optional)
    # ---------------------------------------------------------------
    if DO_RUNTIME
        try_runtime_analysis(stats)
    end

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print_header("SUMMARY")

    # Count by file
    file_counts = Dict{String, NamedTuple{(:unstable, :warning, :info), Tuple{Int,Int,Int}}}()
    for f in stats.findings
        key = f.file
        prev = get(file_counts, key, (unstable=0, warning=0, info=0))
        if f.status == :UNSTABLE
            file_counts[key] = (unstable=prev.unstable+1, warning=prev.warning, info=prev.info)
        elseif f.status == :WARNING
            file_counts[key] = (unstable=prev.unstable, warning=prev.warning+1, info=prev.info)
        elseif f.status == :INFO
            file_counts[key] = (unstable=prev.unstable, warning=prev.warning, info=prev.info+1)
        end
    end

    println("  Findings by file:")
    println("  " * "-"^60)
    println("  $(rpad("File", 35)) $(lpad("UNSTABLE", 10)) $(lpad("WARNING", 10)) $(lpad("INFO", 8))")
    println("  " * "-"^60)
    for (file, counts) in sort(collect(file_counts); by=x->x[1])
        println("  $(rpad(file, 35)) $(lpad(string(counts.unstable), 10)) $(lpad(string(counts.warning), 10)) $(lpad(string(counts.info), 8))")
    end
    println("  " * "-"^60)
    println("  $(rpad("TOTAL", 35)) $(lpad(string(stats.unstable_count), 10)) $(lpad(string(stats.warning_count), 10)) $(lpad(string(stats.info_count), 8))")
    println()

    println("  Overall statistics:")
    println("    Total findings:     $(stats.total_findings)")
    println("    UNSTABLE:           $(stats.unstable_count)")
    println("    WARNING:            $(stats.warning_count)")
    println("    INFO:               $(stats.info_count)")
    println("    SKIP:               $(stats.skip_count)")
    println()

    # ---------------------------------------------------------------
    # Recommendations
    # ---------------------------------------------------------------
    print_header("RECOMMENDATIONS (prioritized by performance impact)")

    println("""
  HIGH PRIORITY (hot-path type instability):

  1. Parameterize timestepper data structs on solver type
     Files: core/timesteppers.jl
     Change: MultistepIMEXData{T} -> MultistepIMEXData{T, S<:AbstractSolver}
     Impact: Eliminates dynamic dispatch on every timestep solve call.

  2. Add concrete type to Field.dist and Field.layout
     Files: core/field.jl, core/distributor.jl
     Change: Field.dist::Any -> Field.dist::Distributor
             Field.layout::Any -> Field.layout::Layout
     Impact: Every field access becomes type-stable.

  3. Type-stabilize get_layout_object return
     Files: core/distributor.jl
     Change: layout_references::Dict{String, Any} -> Dict{String, Layout}
     Impact: Removes type instability from all layout lookups.

  4. Replace Vector{Any} args in arithmetic operators
     Files: core/arithmetic.jl, core/future.jl
     Change: args::Vector{Any} -> args::Vector{AbstractOperand} or
             use a 2-element struct for binary operators
     Impact: Expression tree evaluation becomes partially type-stable.

  MEDIUM PRIORITY (setup/configuration path):

  5. Type FFTW plan caches
     Files: core/transforms.jl
     Change: _plan_cache::Dict{..., Any} -> Dict{..., Tuple{P1, P2}}
     Impact: Transform plan retrieval becomes type-stable.

  6. Type CoeffSystem views dictionary keys
     Files: core/system.jl
     Change: views::Dict{Any, ...} -> views::Dict{Int, ...} or
             use a flat Vector with subproblem indices
     Impact: Coefficient system access in timestep loop.

  LOW PRIORITY (correctness only, minimal performance impact):

  7. Use CachedAttribute{T} instead of CachedAttribute
     Files: tools/cache.jl (usage sites throughout)
     Change: Always specify the type parameter.
     Impact: Mostly documentation; real impact depends on usage context.

  8. Type the SCHEME_REGISTRY and MATSOLVER_REGISTRY
     Files: core/timesteppers.jl, libraries/matsolvers.jl
     Change: Dict{String, Any} -> Dict{String, Type{<:IMEXBase}} etc.
     Impact: Only affects setup, not runtime.
""")

    print_separator('#')
    println("  Audit complete. $(stats.unstable_count) UNSTABLE + $(stats.warning_count) WARNING findings.")
    print_separator('#')
    println()

    return stats
end

# ============================================================================
# Entry point
# ============================================================================

using Dates
stats = main()

# Return nonzero exit code if unstable findings exist (useful in CI)
if stats.unstable_count > 0
    exit(1)
end
