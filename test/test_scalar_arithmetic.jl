using Test
using Dedalus

@testset "Scalar field arithmetic" begin
    c = CartesianCoordinates("x")
    d = Distributor(c, Float64)
    f = Field(d, dtype = Float64)
    @test (f + f).domain.bases == ()
end
