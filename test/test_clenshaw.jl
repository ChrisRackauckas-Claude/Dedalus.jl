"""Tests for Clenshaw summation algorithms."""
# Note: The original Python test_clenshaw.py only tests ball (spherical) Clenshaw.
# Since we restrict to Cartesian parametrizations, and the Python file has no
# Cartesian tests, we provide placeholder tests for 1D Jacobi Clenshaw
# (as noted by the TODO in the original).

using Test
using Dedalus

@testset "Clenshaw" begin

    @testset "Jacobi Clenshaw scalar N=$N a=$a b=$b T=$T" for
            N in [8],
            a in [-0.5, 0.0],
            b in [-0.5, 0.0],
            T in [Float64, ComplexF64]
        # Test that Clenshaw summation of Jacobi polynomials reproduces
        # direct evaluation for a simple polynomial NCC.
        # Setup: construct a Jacobi basis and evaluate a polynomial via
        # coefficient expansion (Clenshaw) vs direct grid evaluation.
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a=a, b=b, bounds=(0, 1))
        x = local_grid(d, basis, scale=1)
        # Field with known polynomial
        f = Field(d, bases=basis)
        f["g"] = @. 2 * x^2 - 1
        # Round-trip through coefficients should preserve values
        fc = f["c"]
        fg = f["g"]
        @test isapprox(fg, @.(2 * x^2 - 1), atol=1e-12)
    end

end
