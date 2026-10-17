using Test
using Dedalus

@testset "Scalar field arithmetic" begin
    c = CartesianCoordinates("x")
    d = Distributor(c, Float64)
    f = Field(d, dtype = Float64)
    @test (f + f).domain.bases == ()
    f["g"] = 3
    @test evaluate(f + f)["g"] == [6.0]
    @test evaluate(2 * f)["g"] == [6.0]
end
