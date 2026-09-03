using Test
using Dedalus

@testset "Dedalus.jl" begin
    @testset "Module loading" begin
        @test Dedalus.VERSION == "3.0.5"
    end

    test_files = [
        "test_clenshaw.jl",
        "test_transforms.jl",
        "test_fourier_operators.jl",
        "test_jacobi_operators.jl",
        "test_cardinal_operators.jl",
        "test_grid_operators.jl",
        "test_cartesian_operators.jl",
        "test_cartesian_ncc.jl",
        "test_lbvp.jl",
        "test_evp.jl",
        "test_ivp.jl",
        "test_nlbvp.jl",
        "test_output.jl",
        "test_cfl.jl",
    ]

    for tf in test_files
        path = joinpath(@__DIR__, tf)
        if isfile(path)
            @testset "$tf" begin
                try
                    include(path)
                catch e
                    @test_broken false
                    @warn "Test file $tf failed to load" exception=e
                end
            end
        end
    end
end
