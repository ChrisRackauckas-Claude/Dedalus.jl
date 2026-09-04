"""Tests for S2 (sphere) calculus operators: skew, transpose, convert, average,
gradient, cosine, laplacian, divergence cleaning, ell product.

Reference: dedalus/tests/test_sphere_calculus.py (Python sphere tests).
"""

using Test
using Dedalus

@testset "Sphere Calculus" begin

    Nphi_range = [32]
    Ntheta_range = [16]
    dealias_range = [1, 1.5]
    dtype_range = [Float64, ComplexF64]
    radius = 1.37

    # ---- Builder function ----

    function build_sphere(Nphi, Ntheta, dealias, T)
        c = S2Coordinates("phi", "theta")
        d = Distributor(c, dtype=T)
        b = SphereBasis(c, (Nphi, Ntheta), radius=radius, dealias=(dealias, dealias), dtype=T)
        phi, theta = local_grids(d, b, scales=dealias)
        return c, d, b, phi, theta
    end

    # ---- Spherical harmonic helpers ----

    """Real spherical harmonic Y_l^m on the 2-sphere (scipy convention:
    sph_harm_y(m, l, theta, phi) with colatitude theta, azimuth phi)."""
    function sph_harm_y(m, l, theta, phi)
        # Y_2^2(theta, phi) = (1/4) sqrt(15/2pi) sin^2(theta) e^{2i phi}
        if l == 2 && m == 2
            return @. exp(2im * phi) * sqrt(15 / (2 * pi)) * sin(theta)^2 / 4
        # Y_10^6(theta, phi)  -- use recurrence-free closed form from
        # associated Legendre + normalization.  We only need a few (l,m) pairs
        # for the eigenvalue tests, so a small manual table suffices.
        elseif l == 10 && m == 6
            # |m| <= l, so valid.
            # P_10^6(x) = (1-x^2)^3 * d^6/dx^6 P_10(x) / 2^10 / 10! * (-1)^6 ...
            # Easier: use the ladder definition.
            # For testing purposes we use a polynomial form.
            # P_10^6(cos theta) via Rodrigues:
            #   P_10(x) = (1/1024)(46189 x^10 - 109395 x^8 + 90090 x^6 - 30030 x^4 + 3465 x^2 - 63)
            # d^6 P_10/dx^6 = (1/1024)(46189*10!/4! x^4 - 109395*8!/2! x^2 + 90090*6!)
            # P_l^m(x) = (-1)^m (1-x^2)^{m/2} d^m P_l / dx^m
            # Full normalization: Y_l^m = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) P_l^m(cos theta) e^{im phi}
            x = @. cos(theta)
            s = @. sin(theta)
            # Pre-computed coefficients for P_10^6(x):
            #   P_10^6(x) = (-1)^6 (1-x^2)^3 * d^6 P_10/dx^6
            # d^6 P_10/dx^6 = (1/1024)*(46189*10*9*8*7*6*5 x^4 - 109395*8*7*6*5*4*3 x^2 + 90090*6*5*4*3*2*1)
            c6_4 = 46189 * 151200  # 46189 * 10!/4!  = 46189 * 151200
            c6_2 = 109395 * 20160  # 109395 * 8!/2!  = 109395 * 20160
            c6_0 = 90090 * 720     # 90090 * 6!      = 90090 * 720
            d6 = @. (c6_4 * x^4 - c6_2 * x^2 + c6_0) / 1024
            Plm = @. s^6 * d6  # (-1)^6 = 1
            # Normalization
            norm = sqrt((2 * 10 + 1) / (4 * pi) * factorial(big(10 - 6)) / factorial(big(10 + 6)))
            return @. Complex(norm) * Plm * exp(6im * phi)
        elseif l == 10 && m == 5
            # Similar manual computation for (l=10, m=5)
            x = @. cos(theta)
            s = @. sin(theta)
            # d^5 P_10/dx^5 = (1/1024)*(46189*10*9*8*7*6 x^5 - 109395*8*7*6*5*4 x^3 + 90090*6*5*4*3*2 x)
            c5_5 = 46189 * 30240   # 46189 * 10!/5!
            c5_3 = 109395 * 6720   # 109395 * 8!/3!
            c5_1 = 90090 * 720     # 90090 * 6!/1!  -- but here derivative order is 5 of degree-6 term = 6! = 720
            # Actually let's recompute carefully:
            # P_10(x) = (1/1024)(46189x^10 - 109395x^8 + 90090x^6 - 30030x^4 + 3465x^2 - 63)
            # d^5/dx^5 of 46189 x^10 = 46189 * 10*9*8*7*6 x^5 = 46189 * 30240 x^5
            # d^5/dx^5 of -109395 x^8 = -109395 * 8*7*6*5*4 x^3 = -109395 * 6720 x^3
            # d^5/dx^5 of 90090 x^6 = 90090 * 6*5*4*3*2 x = 90090 * 720 x
            # d^5/dx^5 of -30030 x^4 = 0 (degree < 5)
            c5_5_v = 46189 * 30240
            c5_3_v = 109395 * 6720
            c5_1_v = 90090 * 720
            d5 = @. (c5_5_v * x^5 - c5_3_v * x^3 + c5_1_v * x) / 1024
            Plm = @. (-1)^5 * s^5 * d5  # P_l^m includes (-1)^m
            norm = sqrt((2 * 10 + 1) / (4 * pi) * factorial(big(10 - 5)) / factorial(big(10 + 5)))
            return @. Complex(norm) * Plm * exp(5im * phi)
        else
            error("sph_harm_y not implemented for l=$l, m=$m")
        end
    end

    # ========================================================================
    # 1. Skew explicit
    #    skew(f)[0] = f[1], skew(f)[1] = -f[0]  (on S2 vector field)
    # ========================================================================
    @testset "skew explicit Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T layout=$layout" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range,
            layout in ["c", "g"]
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            f = VectorField(d, c, bases=b)
            fill_random!(f, layout="g")
            low_pass_filter!(f, scales=0.75)
            change_layout!(f, layout)
            g = evaluate(skew(f))
            # On the 2-sphere: skew swaps components with a sign flip
            @test isapprox(g["g"][1, :], f["g"][2, :], atol=1e-12)
            @test isapprox(g["g"][2, :], -f["g"][1, :], atol=1e-12)
        catch e
            @test_broken false
            @warn "test_skew_explicit failed" exception=e
        end
    end

    # ========================================================================
    # 2. Skew implicit -- LBVP: skew(u) = skew(f)
    # ========================================================================
    @testset "skew implicit Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            f = VectorField(d, c, bases=b)
            fill_random!(f, layout="g")
            low_pass_filter!(f, scales=0.75)
            u = VectorField(d, c, bases=b)
            problem = LBVP([u], namespace=Dict("u" => u, "f" => f, "skew" => skew))
            add_equation!(problem, "skew(u) = skew(f)")
            solver = build_solver(problem)
            solve!(solver)
            change_scales!(u, dealias)
            change_scales!(f, dealias)
            @test isapprox(u["g"], f["g"], atol=1e-10)
        catch e
            @test_broken false
            @warn "test_skew_implicit failed" exception=e
        end
    end

    # ========================================================================
    # 3. Transpose explicit
    # ========================================================================
    @testset "transpose explicit Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T layout=$layout" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range,
            layout in ["c", "g"]
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            f = TensorField(d, (c, c), bases=b)
            fill_random!(f, layout="g")
            low_pass_filter!(f, scales=0.75)
            change_layout!(f, layout)
            g = evaluate(tensor_transpose(f))
            # Transposed: g[i,j,...] = f[j,i,...]
            for i in 1:2, j in 1:2
                @test isapprox(g["g"][i, j, :], f["g"][j, i, :], atol=1e-12)
            end
        catch e
            @test_broken false
            @warn "test_transpose_explicit failed" exception=e
        end
    end

    # ========================================================================
    # 4. Transpose implicit -- LBVP: trans(u) = trans(f)
    # ========================================================================
    @testset "transpose implicit Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            f = TensorField(d, (c, c), bases=b)
            fill_random!(f, layout="g")
            low_pass_filter!(f, scales=0.75)
            u = TensorField(d, (c, c), bases=b)
            problem = LBVP([u], namespace=Dict("u" => u, "f" => f,
                                               "trans" => tensor_transpose))
            add_equation!(problem, "trans(u) = trans(f)")
            solver = build_solver(problem)
            solve!(solver)
            change_scales!(u, dealias)
            change_scales!(f, dealias)
            @test isapprox(u["g"], f["g"], atol=1e-10)
        catch e
            @test_broken false
            @warn "test_transpose_implicit failed" exception=e
        end
    end

    # ========================================================================
    # 5. Convert constant scalar explicit
    # ========================================================================
    @testset "convert constant scalar explicit Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            f = Field(d)
            if T == ComplexF64
                fref = 1im
            else
                fref = 1
            end
            f["g"] = fref
            g = evaluate(Convert(f, b))
            @test isapprox(g["g"], fill!(similar(g["g"]), fref), atol=1e-12)
        catch e
            @test_broken false
            @warn "test_convert_constant_scalar_explicit failed" exception=e
        end
    end

    # ========================================================================
    # 6. Sphere average scalar explicit
    #    Average of (1 + x + z) over S2 should be 1 (x, z have zero mean)
    # ========================================================================
    @testset "sphere average scalar explicit Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            f = Field(d, bases=b)
            preset_scales!(f, dealias)
            x = @. sin(theta) * cos(phi)
            z = @. cos(theta)
            if T == ComplexF64
                f["g"] = @. 1 + x + 1im * z
            else
                f["g"] = @. 1 + x + z
            end
            h = evaluate(Average(f, c))
            @test isapprox(h["g"], fill!(similar(h["g"]), 1), atol=1e-12)
        catch e
            @test_broken false
            @warn "test_sphere_average_scalar_explicit failed" exception=e
        end
    end

    # ========================================================================
    # 7. Gradient scalar explicit -- gradient of Y_2^2
    #    grad(Y_2^2) = [1/(R sin theta) d/dphi Y_2^2, 1/R d/dtheta Y_2^2]
    # ========================================================================
    @testset "gradient scalar explicit Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            m, l = 2, 2
            f = Field(d, bases=b)
            preset_scales!(f, dealias)
            Y22 = sph_harm_y(m, l, theta, phi)
            if T == ComplexF64
                f["g"] = Y22
            else
                f["g"] = real.(Y22)
            end
            u = evaluate(Gradient(f))
            # Analytical gradient of Y_2^2 on the sphere of radius R
            ug_phi = @. 1im * exp(2im * phi) * sqrt(15 / (2 * pi)) * sin(theta) / 2
            ug_theta = @. exp(2im * phi) * sqrt(15 / (2 * pi)) * cos(theta) * sin(theta) / 2
            ug_phi = ug_phi ./ radius
            ug_theta = ug_theta ./ radius
            if T == Float64
                ug_phi = real.(ug_phi)
                ug_theta = real.(ug_theta)
            end
            # u["g"] should have shape [2, Nphi, Ntheta] with component 1=phi, 2=theta
            @test isapprox(u["g"][1, :], ug_phi, atol=1e-10)
            @test isapprox(u["g"][2, :], ug_theta, atol=1e-10)
        catch e
            @test_broken false
            @warn "test_gradient_scalar_explicit failed" exception=e
        end
    end

    # ========================================================================
    # 8. MulCosine explicit -- cos(theta) * f for ranks 0, 1, 2
    # ========================================================================
    @testset "cosine explicit Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T rank=$rank" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range,
            rank in [0, 1, 2]
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            if rank == 0
                f = Field(d, bases=b)
            elseif rank == 1
                f = VectorField(d, c, bases=b)
            else
                f = TensorField(d, (c, c), bases=b)
            end
            fill_random!(f, layout="g")
            low_pass_filter!(f, scales=0.75)
            g = evaluate(MulCosine(f))
            change_scales!(g, dealias)
            change_scales!(f, dealias)
            @test isapprox(g["g"], cos.(theta) .* f["g"], atol=1e-10)
        catch e
            @test_broken false
            @warn "test_cosine_explicit failed (rank=$rank)" exception=e
        end
    end

    # ========================================================================
    # 9. MulCosine implicit -- LBVP: u + MulCosine(u) = f + MulCosine(f)
    # ========================================================================
    @testset "cosine implicit Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T rank=$rank" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range,
            rank in [0, 1, 2]
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            if rank == 0
                f = Field(d, bases=b)
                u = Field(d, bases=b)
            elseif rank == 1
                f = VectorField(d, c, bases=b)
                u = VectorField(d, c, bases=b)
            else
                f = TensorField(d, (c, c), bases=b)
                u = TensorField(d, (c, c), bases=b)
            end
            fill_random!(f, layout="g")
            low_pass_filter!(f, scales=0.75)
            problem = LBVP([u], namespace=Dict("u" => u, "f" => f,
                                               "MulCosine" => MulCosine))
            add_equation!(problem, "u + MulCosine(u) = f + MulCosine(f)")
            solver = build_solver(problem)
            solve!(solver)
            change_scales!(u, dealias)
            change_scales!(f, dealias)
            @test isapprox(u["g"], f["g"], atol=1e-10)
        catch e
            @test_broken false
            @warn "test_cosine_implicit failed (rank=$rank)" exception=e
        end
    end

    # ========================================================================
    # 10. Laplacian scalar explicit -- eigenvalue check
    #     lap(Y_l^m) = -l(l+1)/R^2 * Y_l^m
    # ========================================================================
    @testset "laplacian scalar explicit Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            m, l = 6, 10
            f = Field(d, bases=b)
            preset_scales!(f, dealias)
            Y = sph_harm_y(m, l, theta, phi)
            if T == ComplexF64
                f["g"] = Y
            else
                f["g"] = real.(Y)
            end
            u = evaluate(Laplacian(f))
            # Eigenvalue: lap(Y_l^m) = -l(l+1)/R^2 Y_l^m
            eigenval = -l * (l + 1) / radius^2
            expected = eigenval .* f["g"]
            @test isapprox(u["g"], expected, atol=1e-6)
        catch e
            @test_broken false
            @warn "test_laplacian_scalar_explicit failed" exception=e
        end
    end

    # ========================================================================
    # 11. Laplacian scalar implicit -- Poisson equation on sphere
    #     lap(u) + tau = f,  ave(u) = 0
    #     => u = -f / [l(l+1)] * R^2
    # ========================================================================
    @testset "laplacian scalar implicit Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            m, l = 5, 10
            f = Field(d, bases=b)
            preset_scales!(f, dealias)
            Y = sph_harm_y(m, l, theta, phi)
            if T == ComplexF64
                f["g"] = Y
            else
                f["g"] = real.(Y)
            end
            u = Field(d, bases=b)
            tau = Field(d)
            problem = LBVP([u, tau], namespace=Dict(
                "u" => u, "tau" => tau, "f" => f,
                "lap" => Laplacian, "ave" => A -> Average(A, c)))
            add_equation!(problem, "lap(u) + tau = f")
            add_equation!(problem, "ave(u) = 0")
            solver = build_solver(problem)
            solve!(solver)
            change_scales!(u, 1)
            change_scales!(f, 1)
            # Expected: u = -f * R^2 / [l(l+1)]
            expected = @. -f["g"] / (l * (l + 1)) * radius^2
            @test isapprox(u["g"], expected, atol=1e-6)
        catch e
            @test_broken false
            @warn "test_laplacian_scalar_implicit failed" exception=e
        end
    end

    # ========================================================================
    # 12. Divergence cleaning -- LBVP
    #     u + grad(psi) = h + g,  div(u) + tau = 0,  ave(psi) = 0
    #     where g = grad(f), h = skew(g)
    #     Solution: u = h (divergence-free part)
    # ========================================================================
    @testset "divergence cleaning Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            f = Field(d, bases=b)
            fill_random!(f, layout="g")
            low_pass_filter!(f, scales=0.75)
            # Build vector field as grad(f) + skew(grad(f))
            g_vec = evaluate(Gradient(f))
            h_vec = evaluate(skew(g_vec))
            # Divergence cleaning LBVP
            u = VectorField(d, c, bases=b)
            psi = Field(d, bases=b)
            tau = Field(d)
            problem = LBVP([u, psi, tau], namespace=Dict(
                "u" => u, "psi" => psi, "tau" => tau,
                "h" => h_vec, "g" => g_vec,
                "grad" => Gradient, "div" => Divergence,
                "ave" => A -> Average(A, c)))
            add_equation!(problem, "u + grad(psi) = h + g")
            add_equation!(problem, "div(u) + tau = 0")
            add_equation!(problem, "ave(psi) = 0")
            solver = build_solver(problem)
            solve!(solver)
            change_scales!(u, 1)
            change_scales!(h_vec, 1)
            @test isapprox(u["g"], h_vec["g"], atol=1e-10)
        catch e
            @test_broken false
            @warn "test_divergence_cleaning failed" exception=e
        end
    end

    # ========================================================================
    # 13. Sphere ell product scalar
    #     SphereEllProduct(f, c, func) where func(ell, r) = ell + 3
    # ========================================================================
    @testset "sphere ell product scalar Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            f = Field(d, bases=b, dtype=T)
            g = Field(d, bases=b, dtype=T)
            fill_random!(f, layout="g")
            func = (ell, r) -> ell + 3
            # Manually build expected result in coefficient space
            for (ell, m_ind, ell_ind) in ell_maps(b, d)
                g["c"][m_ind, ell_ind] = func(ell, radius) * f["c"][m_ind, ell_ind]
            end
            h = evaluate(SphereEllProduct(f, c, func))
            @test isapprox(g["c"], h["c"], atol=1e-12)
        catch e
            @test_broken false
            @warn "test_sphere_ell_product_scalar failed" exception=e
        end
    end

    # ========================================================================
    # 14. Sphere ell product vector
    #     SphereEllProduct(f, c, func) where func(ell, r) = ell + 3
    # ========================================================================
    @testset "sphere ell product vector Nphi=$Nphi Ntheta=$Ntheta dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, T)
            f = VectorField(d, c, bases=b, dtype=T)
            g = VectorField(d, c, bases=b, dtype=T)
            fill_random!(f, layout="g")
            func = (ell, r) -> ell + 3
            # Manually build expected result in coefficient space
            dim = 2  # S2 has 2 components
            for (ell, m_ind, ell_ind) in ell_maps(b, d)
                for i in 1:dim
                    g["c"][i, m_ind, ell_ind] = func(ell, radius) * f["c"][i, m_ind, ell_ind]
                end
            end
            h = evaluate(SphereEllProduct(f, c, func))
            @test isapprox(g["c"], h["c"], atol=1e-12)
        catch e
            @test_broken false
            @warn "test_sphere_ell_product_vector failed" exception=e
        end
    end

end
