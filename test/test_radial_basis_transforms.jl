using Test
using Dedalus

@testset "Azimuth regularity transform on radial basis data" begin
    c = SphericalCoordinates("phi", "theta", "r")
    d = Distributor(c, Float64)
    b_r = radial_basis(BallBasis(c, (8, 10, 8), Float64; radius = 1.5))
    g = Field(d, bases = (b_r,), dtype = Float64)
    gdata = reshape(collect(1.0:8.0), 1, 1, 8)
    cdata = fill(NaN, 2, 1, 8)
    Dedalus.forward_transform_azimuth_regularity(b_r, g, 1, gdata, cdata)
    @test cdata[1:1, :, :] == gdata
    @test all(iszero, cdata[2, :, :])
    gout = zeros(1, 1, 8)
    Dedalus.backward_transform_azimuth_regularity(b_r, g, 1, cdata, gout)
    @test gout == gdata
end
