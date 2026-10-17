"""
Radial quadrature weights against analytic moments.

Shell and annulus weights integrate with unit measure, so `sum(w .* r .^ p)`
equals `∫ r^p dr` over the radii. Ball weights carry the Zernike measure
`r^2 dr` on the unit ball, so `sum(w .* r .^ p)` equals `1 / (p + 3)` for even
`p`. Both rules are exact for polynomials below the radial resolution.
"""

using Test
using Dedalus

shell_moment(p, (r0, r1)) = (r1^(p + 1) - r0^(p + 1)) / (p + 1)

@testset "Radial quadrature weights" begin
    @testset "shell Nr=$Nr alpha=$alpha" for Nr in (8, 9), alpha in ((0, 0), (-0.5, -0.5), (0.5, 1.5))
        radii = (0.5, 2.0)
        c = SphericalCoordinates("phi", "theta", "r")
        d = Distributor(c, Float64)
        b = ShellBasis(c, (8, 8, Nr), Float64; radii = radii, alpha = alpha)
        rb = b.radial_basis
        w = vec(Dedalus.local_weights(rb, d))
        r = vec(Dedalus.local_grid(rb, d, 1))
        @test length(w) == Nr
        @test vec(Dedalus.global_weights(rb, d)) == w
        for p in 0:(Nr - 1)
            @test sum(w .* r .^ p) ≈ shell_moment(p, radii) rtol = 1.0e-12
        end
    end

    @testset "annulus alpha=$alpha" for alpha in ((0, 0), (-0.5, -0.5))
        radii = (0.5, 2.0)
        c = PolarCoordinates("phi", "r")
        b = AnnulusBasis(c, (8, 10), Float64; radii = radii, alpha = alpha)
        w = Dedalus._radius_weights(b, 1)
        r = Dedalus._radius_grid(b, 1)
        @test w isa AbstractVector
        @test length(w) == 10
        for p in 0:9
            @test sum(w .* r .^ p) ≈ shell_moment(p, radii) rtol = 1.0e-12
        end
    end

    @testset "ball Nr=$Nr" for Nr in (8, 9)
        c = SphericalCoordinates("phi", "theta", "r")
        d = Distributor(c, Float64)
        b = BallBasis(c, (8, 8, Nr), Float64; radius = 1.0)
        rb = b.radial_basis
        w = vec(Dedalus.local_weights(rb, d))
        r = vec(Dedalus.local_grid(rb, d, 1))
        @test length(w) == Nr
        @test vec(Dedalus.global_weights(rb, d)) == w
        for p in 0:2:(2Nr - 2)
            @test sum(w .* r .^ p) ≈ 1 / (p + 3) rtol = 1.0e-12
        end
    end
end
