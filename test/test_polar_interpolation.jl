using Test
using Dedalus

@testset "Axisymmetric polar interpolation" begin
    @testset "$ctor T=$T Nphi=$Nphi scale=$scale" for
        ctor in (DiskBasis, AnnulusBasis), T in (Float64, ComplexF64),
            Nphi in (1, 2), scale in (1.0, 1.5)
        c = PolarCoordinates("phi", "r")
        d = Distributor(c, T)
        b = ctor === DiskBasis ?
            ctor(c, (Nphi, 8), T; radius = 1.0, dealias = (scale, scale)) :
            ctor(c, (Nphi, 8), T; radii = (0.5, 1.0), dealias = (scale, scale))
        _, r = local_grids(d, b; scales = scale)
        f = Field(d; bases = (b,), dtype = T)
        preset_scales!(f, scale)
        f["g"] = r .^ 2
        a = evaluate(interpolate(f; phi = 0.3))
        @test a["g"] ≈ r .^ 2 atol = 1.0e-12
        @test f["g"] ≈ r .^ 2 atol = 1.0e-12
        f["g"] = 2 .* r .^ 2
        @test evaluate(interpolate(f; phi = 1.7))["g"] ≈ 2 .* r .^ 2 atol = 1.0e-12
        @test a["g"] ≈ r .^ 2 atol = 1.0e-12
        @test only(evaluate(interpolate(f; phi = 0.3, r = 0.75))["g"]) ≈ 2 * 0.75^2 atol = 1.0e-12
    end
end
