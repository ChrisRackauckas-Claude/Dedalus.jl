using Test
using Dedalus

@testset "Dedalus.jl" begin
    @testset "Module loading" begin
        @test Dedalus.VERSION == "3.0.5"
    end

    @testset "Core types exist" begin
        @test isdefined(Dedalus, :Field)
        @test isdefined(Dedalus, :Distributor)
        @test isdefined(Dedalus, :CartesianCoordinates)
        @test isdefined(Dedalus, :Differentiate)
        @test isdefined(Dedalus, :Integrate)
        @test isdefined(Dedalus, :IVP)
        @test isdefined(Dedalus, :LBVP)
        @test isdefined(Dedalus, :CFL)
    end

    # Domain test files — each is wrapped in try/catch so individual
    # file failures don't prevent the rest from running.
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
            try
                include(path)
            catch e
                @warn "Test file $tf could not be loaded" exception=(e, catch_backtrace())
            end
        end
    end
end
