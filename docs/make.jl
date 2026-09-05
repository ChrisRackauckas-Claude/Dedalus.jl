using Documenter
using Literate
using Dedalus

# ---------------------------------------------------------------------------
# Literate.jl preprocessing: convert leading triple-quoted docstrings in
# example scripts into Literate-compatible `# ` markdown comment lines.
#
# Edge cases handled:
#   - LaTeX `$` must not be treated as Julia string interpolation
#   - Escaped `\$` characters
#   - `#` symbols inside docstrings (passed through as-is)
#   - Scripts without a leading docstring (returned unchanged)
#   - Multi-paragraph docstrings with blank lines
# ---------------------------------------------------------------------------

function preprocess_example(content::AbstractString)
    # Only transform if the file starts with a triple-quoted docstring
    startswith(lstrip(content), "\"\"\"") || return content

    # Find the leading whitespace + opening """
    m_open = match(r"^(\s*)\"\"\"", content)
    m_open === nothing && return content

    open_end   = m_open.offset + ncodeunits(m_open.match)  # position right after opening """

    # Locate the closing """ that ends the docstring
    close_idx = findnext("\"\"\"", content, open_end)
    close_idx === nothing && return content

    close_start = first(close_idx)
    close_end   = last(close_idx)

    # Extract the docstring body (between the two """)
    body = content[open_end:close_start - 1]

    # Strip a single leading newline if present (triple-quote convention)
    if startswith(body, "\n")
        body = body[2:end]
    end

    # Convert each line of the docstring body to a Literate `# ` comment.
    # Blank lines become just `#` (for Markdown paragraph breaks).
    doc_lines = split(body, "\n")
    comment_lines = String[]
    for line in doc_lines
        stripped = rstrip(line)
        if isempty(stripped)
            push!(comment_lines, "#")
        else
            push!(comment_lines, "# " * stripped)
        end
    end

    # Remainder of the file after the closing """ (skip trailing newline if any)
    rest = content[close_end + 1:end]
    if startswith(rest, "\n")
        rest = rest[2:end]
    end

    return join(comment_lines, "\n") * "\n" * rest
end

# ---------------------------------------------------------------------------
# Process all 16 example scripts with Literate.markdown
# ---------------------------------------------------------------------------

const EXAMPLES_SRC = joinpath(@__DIR__, "..", "examples")
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
mkpath(EXAMPLES_OUT)

# (directory_name, script_file) pairs for every example
const EXAMPLE_SCRIPTS = [
    ("evp_1d_mathieu",                         "mathieu_evp.jl"),
    ("evp_1d_rayleigh_benard",                 "rayleigh_benard_evp.jl"),
    ("evp_1d_waves_on_a_string",               "waves_on_a_string.jl"),
    ("evp_disk_pipe_flow",                      "pipe_flow.jl"),
    ("evp_shell_rotating_convection",           "rotating_convection.jl"),
    ("ivp_1d_kdv_burgers",                      "kdv_burgers.jl"),
    ("ivp_2d_rayleigh_benard",                  "rayleigh_benard.jl"),
    ("ivp_2d_shear_flow",                       "shear_flow.jl"),
    ("ivp_2d_ensemble_rbc",                     "ensemble_rbc.jl"),
    ("ivp_disk_libration",                      "libration.jl"),
    ("ivp_annulus_centrifugal_convection",       "centrifugal_convection.jl"),
    ("ivp_sphere_shallow_water",                "shallow_water.jl"),
    ("ivp_shell_convection",                    "shell_convection.jl"),
    ("ivp_ball_internally_heated_convection",   "internally_heated_convection.jl"),
    ("lbvp_2d_poisson",                         "poisson.jl"),
    ("nlbvp_ball_lane_emden",                   "lane_emden.jl"),
]

example_pages = Pair{String,String}[]

for (dirname, scriptfile) in EXAMPLE_SCRIPTS
    script = joinpath(EXAMPLES_SRC, dirname, scriptfile)
    if !isfile(script)
        @warn "Example script not found, skipping" script
        continue
    end
    @info "Processing example" dirname scriptfile
    Literate.markdown(
        script,
        EXAMPLES_OUT;
        name       = dirname,
        preprocess = preprocess_example,
        documenter = false,
        execute    = false,
    )
    push!(example_pages, dirname => "examples/$(dirname).md")
end

@info "Finished processing $(length(example_pages)) example scripts"

# ---------------------------------------------------------------------------
# Build documentation with Documenter.jl
# ---------------------------------------------------------------------------

makedocs(;
    sitename = "Dedalus.jl",
    format   = Documenter.HTML(;
        mathengine = MathJax3(),
        prettyurls = false,
        size_threshold = 500_000,
        size_threshold_warn = 200_000,
    ),
    remotes  = nothing,
    modules  = [Dedalus],
    pages    = [
        "Home"        => "index.md",
        "Installation" => "installation.md",
        "Methodology"  => "methodology.md",
        "Tutorials"   => [
            "Overview" => "tutorials/index.md",
            "Coordinates & Bases" => "tutorials/tutorial_1_coords_bases.md",
            "Fields & Operators" => "tutorials/tutorial_2_fields_operators.md",
            "Problems & Solvers" => "tutorials/tutorial_3_problems_solvers.md",
            "Analysis" => "tutorials/tutorial_4_analysis.md",
        ],
        "User Guide"  => [
            "Overview" => "guide/index.md",
            "Problem Formulations" => "guide/problem_formulations.md",
            "Tau Method" => "guide/tau_method.md",
            "Gauge Conditions" => "guide/gauge_conditions.md",
            "Half Dimensions" => "guide/half_dimensions.md",
            "General Functions" => "guide/general_functions.md",
            "Performance Tips" => "guide/performance_tips.md",
            "Configuration" => "guide/configuration.md",
            "Troubleshooting" => "guide/troubleshooting.md",
            "Python Differences" => "guide/python_differences.md",
        ],
        "Examples"    => [
            "Overview" => "examples/index.md",
            example_pages...,
        ],
        "API Reference" => [
            "Overview" => "api/index.md",
            "Coordinates" => "api/coordinates.md",
            "Fields" => "api/fields.md",
            "Bases" => "api/bases.md",
            "Operators" => "api/operators.md",
            "Problems" => "api/problems.md",
            "Solvers" => "api/solvers.md",
            "Extras" => "api/extras.md",
        ],
    ],
    doctest  = false,
    warnonly = [:missing_docs, :parse_error],
)

@info "Documentation build complete"
