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

    @testset "Milestone 2 types exist" begin
        @test isdefined(Dedalus, :MultidimensionalBasis)
        @test isdefined(Dedalus, :SpinBasis)
        @test isdefined(Dedalus, :PolarBasis)
        @test isdefined(Dedalus, :DiskBasis)
        @test isdefined(Dedalus, :AnnulusBasis)
        @test isdefined(Dedalus, :SphereBasis)
        @test isdefined(Dedalus, :PolarGradient)
        @test isdefined(Dedalus, :PolarDivergence)
        @test isdefined(Dedalus, :PolarLaplacian)
        @test isdefined(Dedalus, :MulCosine)
        @test isdefined(Dedalus, :SpinSkew)
        @test isdefined(Dedalus, :PolarTrace)
        @test isdefined(Dedalus, :RadialComponent)
        @test isdefined(Dedalus, :AngularComponent)
        @test isdefined(Dedalus, :AzimuthalComponent)
        @test isdefined(Dedalus, :SphereEllProduct)
        @test isdefined(Dedalus, :SWSHColatitudeTransform)
        @test isdefined(Dedalus, :DiskRadialTransform)
        @test isdefined(Dedalus, :SphereWrapper)
        @test isdefined(Dedalus, :Intertwiner)
    end

    include("test_dedalus_sphere.jl")
    include("test_polar_calculus.jl")
    include("test_polar_ncc.jl")
    include("test_sphere_calculus.jl")
    include("test_cylinder_calculus.jl")
end
