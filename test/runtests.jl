using Test

@testset "Dedalus.jl" begin
    @testset "Module loading" begin
        @test begin
            @eval using Dedalus
            true
        end
    end
end
