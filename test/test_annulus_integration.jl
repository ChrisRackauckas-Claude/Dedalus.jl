using Test
using Dedalus
@testset "Thin annulus constant integral" begin
    c = PolarCoordinates("phi", "r")
    d = Distributor(c, Float64)
    b = AnnulusBasis(c, (8, 10), Float64; radii = (1.0, 1.00001), k = 0)
    f = Field(d; bases = (b,), dtype = Float64)
    f["g"] = 1.0
    area = pi * (b.radii[2] - b.radii[1]) * (b.radii[2] + b.radii[1])
    @test only(evaluate(integrate(f, c))["g"]) ≈ area rtol = 1.0e-12
    @test only(evaluate(average(f, c))["g"]) ≈ 1.0 rtol = 1.0e-12
end

@testset "Annulus analytic moments and repeated evaluation" begin
    @testset "radii=$radii T=$T scale=$scale" for
        radii in ((1.0, 1.00001), (1.0, 1.000000001), (1.0e8, 1.0e8 + 0.001), (0.01, 1.0)),
            T in (Float64, ComplexF64), scale in (1.0, 1.5)
        c = PolarCoordinates("phi", "r")
        d = Distributor(c, T)
        b = AnnulusBasis(c, (8, 10), T; radii = radii, k = 0, dealias = (scale, scale))
        _, r = local_grids(d, b; scales = scale)
        r0, r1 = radii
        area = pi * (r1 - r0) * (r1 + r0)
        f = Field(d; bases = (b,), dtype = T)
        preset_scales!(f, scale)
        f["g"] = 1.0
        integral = evaluate(integrate(f, c))
        @test only(integral["g"]) ≈ area rtol = 1.0e-12
        @test only(evaluate(average(f, c))["g"]) ≈ 1.0 rtol = 1.0e-12
        @test f["g"] ≈ ones(size(f["g"])) rtol = 1.0e-12
        expected = repeat(r .^ 2, size(f["g"], 1), 1)
        f["g"] = expected
        mean_r2 = (r0^2 + r1^2) / 2
        @test only(evaluate(integrate(f, c))["g"]) ≈ area * mean_r2 rtol = 1.0e-12
        @test only(evaluate(average(f, c))["g"]) ≈ mean_r2 rtol = 1.0e-12
        @test only(integral["g"]) ≈ area rtol = 1.0e-12
        @test f["g"] ≈ expected rtol = 1.0e-12
    end
end
