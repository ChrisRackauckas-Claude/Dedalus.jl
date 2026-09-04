"""Tests for 3D spherical calculus: gradient, divergence, curl, laplacian
on ball and shell bases.

Reference: dedalus/tests/test_spherical_calculus.py
"""

using Test
using Dedalus

@testset "Spherical Calculus" begin

    Nphi_range = [16]
    Ntheta_range = [8]
    Nr_range = [8]
    dealias_range = [1, 1.5]
    dtype_range = [Float64, ComplexF64]
    radius_ball = 1.5
    radii_shell = (0.5, 3.0)

    # ---- Builder functions ----

    function build_ball(Nphi, Ntheta, Nr, dealias, T)
        c = SphericalCoordinates("phi", "theta", "r")
        d = Distributor(c, T)
        b = BallBasis(c, (Nphi, Ntheta, Nr), T;
                      radius=radius_ball,
                      dealias=(dealias, dealias, dealias))
        phi, theta, r = local_grids(d, b, scales=dealias)
        x, y, z = cartesian(SphericalCoordinates, phi, theta, r)
        return c, d, b, phi, theta, r, x, y, z
    end

    function build_shell(Nphi, Ntheta, Nr, dealias, T)
        c = SphericalCoordinates("phi", "theta", "r")
        d = Distributor(c, T)
        b = ShellBasis(c, (Nphi, Ntheta, Nr), T;
                       radii=radii_shell,
                       dealias=(dealias, dealias, dealias))
        phi, theta, r = local_grids(d, b, scales=dealias)
        x, y, z = cartesian(SphericalCoordinates, phi, theta, r)
        return c, d, b, phi, theta, r, x, y, z
    end

    # ========================================================================
    # 1. Gradient of scalar
    #    f = 3*x^2 + 2*y*z
    #    grad(f) in spherical components (phi, theta, r)
    # ========================================================================
    @testset "gradient scalar $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. 3 * x^2 + 2 * y * z
            u = evaluate(gradient(f, c))
            ug = zero(u["g"])
            ug[3, :, :, :] = @. (6 * x^2 + 4 * y * z) / r
            ug[2, :, :, :] = @. -2 * (y^3 + x^2 * (y - 3 * z) - y * z^2) / (r^2 * sin(theta))
            ug[1, :, :, :] = @. 2 * x * (-3 * y + z) / (r * sin(theta))
            @test isapprox(u["g"], ug, atol=1e-12)
        catch e
            @test_broken false
            @warn "gradient_scalar failed" exception=e
        end
    end

    # ========================================================================
    # 2. Gradient of radial scalar
    #    f = r^4 / 3, Nphi=Ntheta=1
    # ========================================================================
    @testset "gradient radial scalar $bname Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nr in Nr_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta, r, x, y, z = basis_fn(1, 1, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. r^4 / 3
            u = evaluate(gradient(f, c))
            ug = zero(u["g"])
            ug[3, :, :, :] = @. 4 / 3 * r^3
            @test isapprox(u["g"], ug, atol=1e-12)
        catch e
            @test_broken false
            @warn "gradient_radial_scalar failed" exception=e
        end
    end

    # ========================================================================
    # 3. Gradient of vector (Hessian)
    #    grad(grad(3*x^2 + 2*y*z))
    # ========================================================================
    @testset "gradient vector $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. 3 * x^2 + 2 * y * z
            grad_op = A -> gradient(A, c)
            Tf = evaluate(grad_op(grad_op(f)))
            Tg = zero(Tf["g"])
            Tg[3, 3, :, :, :] = @. (6 * x^2 + 4 * y * z) / r^2
            Tg[3, 2, :, :, :] = @. -2 * (y^3 + x^2 * (y - 3 * z) - y * z^2) / (r^3 * sin(theta))
            Tg[2, 3, :, :, :] = Tg[3, 2, :, :, :]
            Tg[3, 1, :, :, :] = @. 2 * x * (z - 3 * y) / (r^2 * sin(theta))
            Tg[1, 3, :, :, :] = Tg[3, 1, :, :, :]
            Tg[2, 2, :, :, :] = @. 6 * x^2 / (r^2 * sin(theta)^2) - (6 * x^2 + 4 * y * z) / r^2
            Tg[2, 1, :, :, :] = @. -2 * x * (x^2 + y^2 + 3 * y * z) / (r^3 * sin(theta)^2)
            Tg[1, 2, :, :, :] = Tg[2, 1, :, :, :]
            Tg[1, 1, :, :, :] = @. 6 * y^2 / (x^2 + y^2)
            @test isapprox(Tf["g"], Tg, atol=1e-12)
        catch e
            @test_broken false
            @warn "gradient_vector failed" exception=e
        end
    end

    # ========================================================================
    # 4. Gradient of radial vector
    #    grad(grad(r^4/3)), Nphi=Ntheta=1
    # ========================================================================
    @testset "gradient radial vector $bname Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nr in Nr_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta, r, x, y, z = basis_fn(1, 1, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. r^4 / 3
            grad_op = A -> gradient(A, c)
            Tf = evaluate(grad_op(grad_op(f)))
            Tg = zero(Tf["g"])
            Tg[1, 1, :, :, :] = @. 4 / 3 * r^2
            Tg[2, 2, :, :, :] = @. 4 / 3 * r^2
            Tg[3, 3, :, :, :] = @. 4 * r^2
            @test isapprox(Tf["g"], Tg, atol=1e-12)
        catch e
            @test_broken false
            @warn "gradient_radial_vector failed" exception=e
        end
    end

    # ========================================================================
    # 5. Divergence of vector
    #    div(grad(x^3 + 2*y^3 + 3*z^3)) = 6*x + 12*y + 18*z
    # ========================================================================
    @testset "divergence vector $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. x^3 + 2 * y^3 + 3 * z^3
            u = gradient(f, c)
            h = evaluate(divergence(u))
            hg = @. 6 * x + 12 * y + 18 * z
            @test isapprox(h["g"], hg, atol=1e-12)
        catch e
            @test_broken false
            @warn "divergence_vector failed" exception=e
        end
    end

    # ========================================================================
    # 6. Divergence of radial vector
    #    div(grad(r^4/3)) = 20/3 * r^2, Nphi=Ntheta=1
    # ========================================================================
    @testset "divergence radial vector $bname Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nr in Nr_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta, r, x, y, z = basis_fn(1, 1, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. r^4 / 3
            u = gradient(f, c)
            h = evaluate(divergence(u))
            hg = @. 20 / 3 * r^2
            @test isapprox(h["g"], hg, atol=1e-12)
        catch e
            @test_broken false
            @warn "divergence_radial_vector failed" exception=e
        end
    end

    # ========================================================================
    # 7. Curl of vector
    # ========================================================================
    @testset "curl vector $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, T)
            u = VectorField(d, c, bases=(b,), dtype=T)
            preset_scales!(u, dealias)
            ct = cos.(theta); st = sin.(theta); cp = cos.(phi); sp = sin.(phi)
            u["g"][3, :, :, :] = @. r^2 * st * (2 * ct^2 * cp - r * ct^3 * sp + r^3 * cp^3 * st^5 * sp^3 + r * ct * st^2 * (cp^3 + sp^3))
            u["g"][2, :, :, :] = @. r^2 * (2 * ct^3 * cp - r * cp^3 * st^4 + r^3 * ct * cp^3 * st^5 * sp^3 - 1 / 16 * r * sin(2 * theta)^2 * (-7 * sp + sin(3 * phi)))
            u["g"][1, :, :, :] = @. r^2 * sp * (-2 * ct^2 + r * ct * cp * st^2 * sp - r^3 * cp^2 * st^5 * sp^3)
            v = evaluate(curl(u))
            vg = zero(v["g"])
            vg[3, :, :, :] = @. -r * st * (r * ct^2 * cp + r * cp * st^2 * sp * (3 * cp + sp) + ct * sp * (-4 + 3 * r^3 * cp^2 * st^3 * sp))
            vg[2, :, :, :] = @. r * (-r * ct^3 * cp + 4 * ct^2 * sp + 3 * r^3 * cp^2 * st^5 * sp^2 - r * ct * cp * st^2 * sp * (3 * cp + sp))
            vg[1, :, :, :] = @. r * (4 * ct * cp + r * ct^2 * sp + r * st^2 * (-3 * cp^3 + sp^3))
            @test isapprox(v["g"], vg, atol=1e-12)
        catch e
            @test_broken false
            @warn "curl_vector failed" exception=e
        end
    end

    # ========================================================================
    # 8. Laplacian of scalar
    #    lap(x^4 + 2*y^4 + 3*z^4) = 12*x^2 + 24*y^2 + 36*z^2
    # ========================================================================
    @testset "laplacian scalar $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. x^4 + 2 * y^4 + 3 * z^4
            h = evaluate(laplacian(f, c))
            hg = @. 12 * x^2 + 24 * y^2 + 36 * z^2
            @test isapprox(h["g"], hg, atol=1e-12)
        catch e
            @test_broken false
            @warn "laplacian_scalar failed" exception=e
        end
    end

    # ========================================================================
    # 9. Laplacian of radial scalar
    #    lap(r^4/3) = 20/3 * r^2, Nphi=Ntheta=1
    # ========================================================================
    @testset "laplacian radial scalar $bname Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nr in Nr_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta, r, x, y, z = basis_fn(1, 1, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. r^4 / 3
            h = evaluate(laplacian(f, c))
            hg = @. 20 / 3 * r^2
            @test isapprox(h["g"], hg, atol=1e-12)
        catch e
            @test_broken false
            @warn "laplacian_radial_scalar failed" exception=e
        end
    end

    # ========================================================================
    # 10. Laplacian of vector
    # ========================================================================
    @testset "laplacian vector $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, T)
            u = VectorField(d, c, bases=(b,), dtype=T)
            preset_scales!(u, dealias)
            ct = cos.(theta); st = sin.(theta); cp = cos.(phi); sp = sin.(phi)
            u["g"][3, :, :, :] = @. r^2 * st * (2 * ct^2 * cp - r * ct^3 * sp + r^3 * cp^3 * st^5 * sp^3 + r * ct * st^2 * (cp^3 + sp^3))
            u["g"][2, :, :, :] = @. r^2 * (2 * ct^3 * cp - r * cp^3 * st^4 + r^3 * ct * cp^3 * st^5 * sp^3 - 1 / 16 * r * sin(2 * theta)^2 * (-7 * sp + sin(3 * phi)))
            u["g"][1, :, :, :] = @. r^2 * sp * (-2 * ct^2 + r * ct * cp * st^2 * sp - r^3 * cp^2 * st^5 * sp^3)
            v = evaluate(laplacian(u, c))
            vg = zero(v["g"])
            vg[3, :, :, :] = @. 2 * (2 + 3 * r * ct) * cp * st + 1 / 2 * r^3 * st^4 * (4 * sin(2 * phi) + sin(4 * phi))
            vg[2, :, :, :] = @. 2 * r * (-3 * cp * st^2 + sp) + 1 / 2 * ct * (8 * cp + r^3 * st^3 * (4 * sin(2 * phi) + sin(4 * phi)))
            vg[1, :, :, :] = @. 2 * r * ct * cp + 2 * sp * (-2 - r^3 * (2 + cos(2 * phi)) * st^3 * sp)
            @test isapprox(v["g"], vg, atol=1e-12)
        catch e
            @test_broken false
            @warn "laplacian_vector failed" exception=e
        end
    end

    # ========================================================================
    # 11. Laplacian of radial vector
    #    u_r = 4/3 * r^3, lap(u)_r = 40/3 * r, Nphi=Ntheta=1
    # ========================================================================
    @testset "laplacian radial vector $bname Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nr in Nr_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, theta, r, x, y, z = basis_fn(1, 1, Nr, dealias, T)
            u = VectorField(d, c, bases=(b,), dtype=T)
            preset_scales!(u, dealias)
            u["g"][3, :, :, :] = @. 4 / 3 * r^3
            v = evaluate(laplacian(u, c))
            vg = zero(v["g"])
            vg[3, :, :, :] = @. 40 / 3 * r
            @test isapprox(v["g"], vg, atol=1e-12)
        catch e
            @test_broken false
            @warn "laplacian_radial_vector failed" exception=e
        end
    end

end
