"""Tests for 3D spherical arithmetic: S2-radial products, cross product,
dot product, and outer-product multiplication at various tensor ranks.

Reference: dedalus/tests/test_spherical_arithmetic.py
"""

using Test
using Dedalus

@testset "Spherical Arithmetic" begin

    Nphi_range = [8]
    Ntheta_range = [10]
    Nr_range = [6]
    dealias_range = [1, 1.5]

    radius_ball = 1.5
    radii_shell = (0.5, 3.0)

    # ---- Builder functions ----

    function build_ball(Nphi, Ntheta, Nr, dealias, T)
        c = SphericalCoordinates("phi", "theta", "r")
        d = Distributor(c, T)
        b = BallBasis(c, (Nphi, Ntheta, Nr), T;
                      radius=radius_ball,
                      dealias=(dealias, dealias, dealias))
        phi, theta, r = local_grids(d, b)
        x, y, z = cartesian(SphericalCoordinates, phi, theta, r)
        return c, d, b, phi, theta, r, x, y, z
    end

    function build_shell(Nphi, Ntheta, Nr, dealias, T)
        c = SphericalCoordinates("phi", "theta", "r")
        d = Distributor(c, T)
        b = ShellBasis(c, (Nphi, Ntheta, Nr), T;
                       radii=radii_shell,
                       dealias=(dealias, dealias, dealias))
        phi, theta, r = local_grids(d, b)
        x, y, z = cartesian(SphericalCoordinates, phi, theta, r)
        return c, d, b, phi, theta, r, x, y, z
    end

    # ========================================================================
    # 1. S2-radial scalar * scalar multiplication
    # ========================================================================
    @testset "S2 radial scalar scalar multiplication Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = build_shell(Nphi, Ntheta, Nr, dealias, ComplexF64)
        f0 = Field(d, bases=(b,), dtype=ComplexF64)
        preset_scales!(f0, dealias)
        phi2, theta2, r2 = local_grids(d, b, scales=dealias)
        f0["g"] = @. (r2^2 - 0.5 * r2^3) * (5 * cos(theta2)^2 - 1) * sin(theta2) * exp(1im * phi2)

        b_S2 = S2_basis(b)
        phi_s, theta_s = local_grids(d, b_S2)
        g = Field(d, bases=(b_S2,), dtype=ComplexF64)
        g["g"] = @. (5 * cos(theta_s)^2 - 1) * sin(theta_s) * exp(1im * phi_s)

        h = Field(d, bases=(radial_basis(b),), dtype=ComplexF64)
        preset_scales!(h, dealias)
        h["g"] = @. r2^2 - 0.5 * r2^3
        f = evaluate(g * h)
        @test isapprox(f["g"], f0["g"], atol=1e-12)
    end

    # ========================================================================
    # 2. S2-radial vector * scalar multiplication
    # ========================================================================
    @testset "S2 radial vector scalar multiplication Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = build_shell(Nphi, Ntheta, Nr, dealias, ComplexF64)
        c_S2 = S2coordsys(c)
        v0 = VectorField(d, c, bases=(b,), dtype=ComplexF64)
        preset_scales!(v0, dealias)
        phi2, theta2, r2 = local_grids(d, b, scales=dealias)
        v0["g"][1, :, :, :] = @. (r2^2 - 0.5 * r2^3) * (-1im * sin(theta2) * exp(-2im * phi2))
        v0["g"][2, :, :, :] = @. (r2^2 - 0.5 * r2^3) * (cos(theta2) * sin(theta2) * exp(-2im * phi2))
        v0["g"][3, :, :, :] = @. (r2^2 - 0.5 * r2^3) * (5 * cos(theta2)^2 - 1) * sin(theta2) * exp(1im * phi2)

        b_S2 = S2_basis(b)
        phi_s, theta_s = local_grids(d, b_S2)
        u = VectorField(d, c, bases=(b_S2,), dtype=ComplexF64)
        u["g"][1, :, :] = @. -1im * sin(theta_s) * exp(-2im * phi_s)
        u["g"][2, :, :] = @. cos(theta_s) * sin(theta_s) * exp(-2im * phi_s)
        u["g"][3, :, :] = @. (5 * cos(theta_s)^2 - 1) * sin(theta_s) * exp(1im * phi_s)

        h = Field(d, bases=(radial_basis(b),), dtype=ComplexF64)
        preset_scales!(h, dealias)
        h["g"] = @. r2^2 - 0.5 * r2^3
        v = evaluate(h * u)
        @test isapprox(v["g"], v0["g"], atol=1e-12)
    end

    # ========================================================================
    # 3. Cross product (ball and shell)
    # ========================================================================
    @testset "cross product $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, ComplexF64)
        f = Field(d, bases=(b,), dtype=ComplexF64)
        f["g"] = z
        ez = evaluate(Gradient(f, c))
        u = VectorField(d, c, bases=(b,), dtype=ComplexF64)
        u["g"][3, :, :, :] = @. (6 * x^2 + 4 * y * z) / r
        u["g"][2, :, :, :] = @. -2 * (y^3 + x^2 * (y - 3 * z) - y * z^2) / (r^2 * sin(theta))
        u["g"][1, :, :, :] = @. 2 * x * (-3 * y + z) / (r * sin(theta))
        h = evaluate(CrossProduct(ez, u))
        hg = zeros(ComplexF64, size(h["g"]))
        hg[1, :, :, :] = @. -ez["g"][2, :, :, :] * u["g"][3, :, :, :] + ez["g"][3, :, :, :] * u["g"][2, :, :, :]
        hg[2, :, :, :] = @. -ez["g"][3, :, :, :] * u["g"][1, :, :, :] + ez["g"][1, :, :, :] * u["g"][3, :, :, :]
        hg[3, :, :, :] = @. -ez["g"][1, :, :, :] * u["g"][2, :, :, :] + ez["g"][2, :, :, :] * u["g"][1, :, :, :]
        @test isapprox(h["g"], hg, atol=1e-12)
    end

    # ========================================================================
    # 4. Dot product vector . vector
    # ========================================================================
    @testset "dot product vector vector $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, ComplexF64)
        f = Field(d, bases=(b,), dtype=ComplexF64)
        f["g"] = z
        ez = evaluate(Gradient(f, c))
        u = VectorField(d, c, bases=(b,), dtype=ComplexF64)
        u["g"][3, :, :, :] = @. (6 * x^2 + 4 * y * z) / r
        u["g"][2, :, :, :] = @. -2 * (y^3 + x^2 * (y - 3 * z) - y * z^2) / (r^2 * sin(theta))
        u["g"][1, :, :, :] = @. 2 * x * (-3 * y + z) / (r * sin(theta))
        h = evaluate(DotProduct(ez, u))
        hg = sum(ez["g"] .* u["g"]; dims=1)
        @test isapprox(h["g"], dropdims(hg; dims=1), atol=1e-12)
    end

    # ========================================================================
    # 5. Dot product tensor . vector
    # ========================================================================
    @testset "dot product tensor vector $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, ComplexF64)
        u = VectorField(d, c, bases=(b,), dtype=ComplexF64)
        u["g"][3, :, :, :] = @. (6 * x^2 + 4 * y * z) / r
        u["g"][2, :, :, :] = @. -2 * (y^3 + x^2 * (y - 3 * z) - y * z^2) / (r^2 * sin(theta))
        u["g"][1, :, :, :] = @. 2 * x * (-3 * y + z) / (r * sin(theta))
        T = TensorField(d, (c, c), bases=(b,), dtype=ComplexF64)
        T["g"][3, 3, :, :, :] = @. (6 * x^2 + 4 * y * z) / r^2
        T["g"][3, 2, :, :, :] = @. -2 * (y^3 + x^2 * (y - 3 * z) - y * z^2) / (r^3 * sin(theta))
        T["g"][2, 3, :, :, :] = T["g"][3, 2, :, :, :]
        T["g"][3, 1, :, :, :] = @. 2 * x * (z - 3 * y) / (r^2 * sin(theta))
        T["g"][1, 3, :, :, :] = T["g"][3, 1, :, :, :]
        T["g"][2, 2, :, :, :] = @. 6 * x^2 / (r^2 * sin(theta)^2) - (6 * x^2 + 4 * y * z) / r^2
        T["g"][2, 1, :, :, :] = @. -2 * x * (x^2 + y^2 + 3 * y * z) / (r^3 * sin(theta)^2)
        T["g"][1, 2, :, :, :] = T["g"][2, 1, :, :, :]
        T["g"][1, 1, :, :, :] = @. 6 * y^2 / (x^2 + y^2)
        v = evaluate(DotProduct(T, u))
        # vg[i,...] = sum_j T[i,j,...] * u[j,...]
        # In Python: vg = np.sum(T['g']*u['g'][:,None,:,:,:],axis=0)
        # In Julia with 1-based: T is [3,3,Nphi,Ntheta,Nr], u is [3,Nphi,Ntheta,Nr]
        # We need vg[i,Nphi,Ntheta,Nr] = sum_j T[i,j,...] * u[j,...]
        sz = size(T["g"])
        vg = zeros(ComplexF64, 3, sz[3], sz[4], sz[5])
        for j in 1:3
            vg .+= T["g"][:, j, :, :, :] .* reshape(u["g"][j, :, :, :], 1, sz[3], sz[4], sz[5])
        end
        @test isapprox(v["g"], vg, atol=1e-12)
    end

    # ========================================================================
    # 6. Multiply number * scalar
    # ========================================================================
    @testset "multiply number scalar $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, ComplexF64)
        f = Field(d, bases=(b,), dtype=ComplexF64)
        f["g"] = @. x^3 + 2 * y^3 + 3 * z^3
        h = evaluate(2 * f)
        phi2, theta2, r2 = local_grids(d, b, scales=dealias)
        x2, y2, z2 = cartesian(SphericalCoordinates, phi2, theta2, r2)
        hg = @. 2 * (x2^3 + 2 * y2^3 + 3 * z2^3)
        @test isapprox(h["g"], hg, atol=1e-12)
    end

    # ========================================================================
    # 7. Multiply scalar * number
    # ========================================================================
    @testset "multiply scalar number $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, ComplexF64)
        f = Field(d, bases=(b,), dtype=ComplexF64)
        f["g"] = @. x^3 + 2 * y^3 + 3 * z^3
        h = evaluate(f * 2)
        phi2, theta2, r2 = local_grids(d, b, scales=dealias)
        x2, y2, z2 = cartesian(SphericalCoordinates, phi2, theta2, r2)
        hg = @. 2 * (x2^3 + 2 * y2^3 + 3 * z2^3)
        @test isapprox(h["g"], hg, atol=1e-12)
    end

    # ========================================================================
    # 8. Multiply scalar * scalar
    # ========================================================================
    @testset "multiply scalar scalar $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, ComplexF64)
        f = Field(d, bases=(b,), dtype=ComplexF64)
        f["g"] = @. x^3 + 2 * y^3 + 3 * z^3
        h = evaluate(f * f)
        phi2, theta2, r2 = local_grids(d, b, scales=dealias)
        x2, y2, z2 = cartesian(SphericalCoordinates, phi2, theta2, r2)
        hg = @. (x2^3 + 2 * y2^3 + 3 * z2^3)^2
        @test isapprox(h["g"], hg, atol=1e-12)
    end

    # ========================================================================
    # 9. Multiply scalar * vector
    # ========================================================================
    @testset "multiply scalar vector $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, ComplexF64)
        phi2, theta2, r2 = local_grids(d, b, scales=dealias)
        x2, y2, z2 = cartesian(SphericalCoordinates, phi2, theta2, r2)
        f = Field(d, bases=(b,), dtype=ComplexF64)
        preset_scales!(f, dealias)
        f["g"] = @. x2^3 + 2 * y2^3 + 3 * z2^3
        u = evaluate(Gradient(f, c))
        v = evaluate(f * u)
        # vg = f['g'][None,...] * u['g']  =>  broadcast scalar over component dim
        vg = reshape(f["g"], 1, size(f["g"])...) .* u["g"]
        @test isapprox(v["g"], vg, atol=1e-12)
    end

    # ========================================================================
    # 10. Multiply vector * vector (outer product)
    # ========================================================================
    @testset "multiply vector vector $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, ComplexF64)
        f = Field(d, bases=(b,), dtype=ComplexF64)
        f["g"] = @. x^3 + 2 * y^3 + 3 * z^3
        u = evaluate(Gradient(f, c))
        T = evaluate(u * u)
        # Python: Tg = u['g'][None,...] * u['g'][:,None,...]
        # u is [3, Nphi, Ntheta, Nr], result is [3, 3, Nphi, Ntheta, Nr]
        sz = size(u["g"])  # (3, Nphi, Ntheta, Nr)
        Tg = reshape(u["g"], 1, sz...) .* reshape(u["g"], sz[1], 1, sz[2:end]...)
        @test isapprox(T["g"], Tg, atol=1e-12)
    end

    # ========================================================================
    # 11. Multiply vector * tensor (outer product)
    # ========================================================================
    @testset "multiply vector tensor $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, ComplexF64)
        f = Field(d, bases=(b,), dtype=ComplexF64)
        f["g"] = @. x^3 + 2 * y^3 + 3 * z^3
        u = evaluate(Gradient(f, c))
        T = evaluate(Gradient(u, c))
        Q = evaluate(u * T)
        # Python: Qg = u['g'][:,None,None,...] * T['g'][None,...]
        # u is [3, Nphi, Ntheta, Nr], T is [3, 3, Nphi, Ntheta, Nr]
        # Q is [3, 3, 3, Nphi, Ntheta, Nr]
        sz_u = size(u["g"])  # (3, np, nt, nr)
        sz_T = size(T["g"])  # (3, 3, np, nt, nr)
        Qg = reshape(u["g"], sz_u[1], 1, 1, sz_u[2:end]...) .* reshape(T["g"], 1, sz_T...)
        @test isapprox(Q["g"], Qg, atol=1e-12)
    end

    # ========================================================================
    # 12. Multiply tensor * tensor (outer product)
    # ========================================================================
    @testset "multiply tensor tensor $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr dealias=$dealias" for
            (bname, basis_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = basis_fn(Nphi, Ntheta, Nr, dealias, ComplexF64)
        f = Field(d, bases=(b,), dtype=ComplexF64)
        f["g"] = @. x^3 + 2 * y^3 + 3 * z^3
        u = evaluate(Gradient(f, c))
        T = evaluate(Gradient(u, c))
        Q = evaluate(T * T)
        # Python: Qg = T['g'][:,:,None,None,...] * T['g'][None,None,...]
        # T is [3, 3, Nphi, Ntheta, Nr]
        # Q is [3, 3, 3, 3, Nphi, Ntheta, Nr]
        sz_T = size(T["g"])  # (3, 3, np, nt, nr)
        Qg = reshape(T["g"], sz_T[1], sz_T[2], 1, 1, sz_T[3:end]...) .* reshape(T["g"], 1, 1, sz_T...)
        @test isapprox(Q["g"], Qg, atol=1e-12)
    end

end
