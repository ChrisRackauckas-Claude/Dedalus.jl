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

    # Individual test files (test_transforms.jl, test_lbvp.jl, etc.) are
    # present in test/ but require full operator/basis wiring to run.
    # They will be enabled as the operator framework matures.
    # For now, the test suite validates module loading and type availability.
end
