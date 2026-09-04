using Test

function preprocess_example(content::AbstractString)
    startswith(lstrip(content), "\"\"\"") || return content

    m_open = match(r"^(\s*)\"\"\"", content)
    m_open === nothing && return content

    open_end   = m_open.offset + ncodeunits(m_open.match)

    close_idx = findnext("\"\"\"", content, open_end)
    close_idx === nothing && return content

    close_start = first(close_idx)
    close_end   = last(close_idx)

    body = content[open_end:close_start - 1]

    if startswith(body, "\n")
        body = body[2:end]
    end

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

    rest = content[close_end + 1:end]
    if startswith(rest, "\n")
        rest = rest[2:end]
    end

    return join(comment_lines, "\n") * "\n" * rest
end

@testset "preprocess_example" begin
    @testset "simple docstring" begin
        input = "\"\"\"\nHello world\n\"\"\"\ncode_here()"
        result = preprocess_example(input)
        @test startswith(result, "# Hello world")
        @test occursin("code_here()", result)
    end

    @testset "LaTeX dollar signs" begin
        input = "\"\"\"\nThe equation \$x^2\$ is important.\n\"\"\"\nx = 1"
        result = preprocess_example(input)
        @test occursin("# The equation \$x^2\$ is important.", result)
    end

    @testset "escaped dollar signs" begin
        input = "\"\"\"\nCost is \\\$5.\n\"\"\"\ny = 2"
        result = preprocess_example(input)
        @test occursin("\\\$5", result)
    end

    @testset "hash symbols in docstring" begin
        input = "\"\"\"\n# Section heading\nSome text with # inside.\n\"\"\"\nz = 3"
        result = preprocess_example(input)
        @test occursin("# # Section heading", result)
        @test occursin("# Some text with # inside.", result)
    end

    @testset "no leading docstring" begin
        input = "using LinearAlgebra\nx = 1"
        result = preprocess_example(input)
        @test result == input
    end

    @testset "multi-paragraph with blank lines" begin
        input = "\"\"\"\nFirst paragraph.\n\nSecond paragraph.\n\"\"\"\ncode()"
        result = preprocess_example(input)
        @test occursin("# First paragraph.\n#\n# Second paragraph.", result)
    end
end
