using Test
using Dedalus

# ∫₀¹ r^(2p) (1 - r²)^α r dr = B(p + 1, α + 1) / 2 = p! / (2 ∏_{j=1}^{p+1} (α + j))
disk_moment(p, α) = factorial(p) / (2 * prod(α + j for j in 1:(p + 1)))

@testset "Disk Zernike quadrature" begin
    @testset "zernike_quadrature alpha=$α" for α in (0, 0.0, 0.5, 1.5, 2)
        n = 8
        z, w = Dedalus.zernike_quadrature(2, n; k = α)
        r = sqrt.((z .+ 1) ./ 2)
        @test length(z) == n
        @test all(0 .< r .< 1)
        @test sum(w) ≈ Dedalus.zernike_mass(2; k = α)
        @test sum(w) ≈ disk_moment(0, α)
        # Gauss quadrature with n nodes is exact through degree 2n - 1 in z = 2r² - 1.
        for p in 1:(2n - 1)
            @test sum(w .* r .^ (2p)) ≈ disk_moment(p, α) rtol = 1.0e-12
        end
    end

    @testset "DiskBasis radial grid alpha=$α" for α in (0.0, 0.5)
        c = PolarCoordinates("phi", "r")
        d = Distributor(c, Float64)
        Nr = 8
        radius = 1.5
        b = DiskBasis(c, (16, Nr), Float64; radius = radius, alpha = α)
        phi, r = local_grids(d, b; scales = 1)
        z, _ = Dedalus.zernike_quadrature(2, Nr; k = α)
        @test vec(r) ≈ radius .* sqrt.((z .+ 1) ./ 2)
        @test all(0 .< vec(r) .< radius)
    end
end
